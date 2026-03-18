"""
Training Data Collection Server
Captures still images from 3 cameras via camera proxy burst endpoint, uploads to S3.

Previous approach: 60 individual HTTP requests (20 frames × 3 cameras),
  batched 2 cameras at a time, ~20s per camera sequentially.

Current approach: 3 HTTP requests (one burst per camera), all concurrent.
  The proxy collects frames server-side and returns tar.gz directly.
  Total capture time ≈ (NUM_IMAGES × CAPTURE_INTERVAL_MS/1000) ≈ 30s,
  all 3 cameras in parallel. Tar.gz uploads directly to S3 (no re-tarring).
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import requests as http_requests
import os
import shutil
import json
import re
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)

from dotenv import load_dotenv
load_dotenv("/home/pi/goatdev/pi/.env")

# ============================================================
# CONFIGURATION
# ============================================================

CAMERAS = {
    'side': '/dev/camera_side',
    'top': '/dev/camera_top',
    'front': '/dev/camera_front'
}

S3_TRAINING_BUCKET = os.environ.get('S3_TRAINING_BUCKET')

# Capture settings — these are passed to the proxy's burst endpoint
NUM_IMAGES = 20               # Frames per camera
CAPTURE_INTERVAL_MS = 1500    # 1.5s between frames for pose variation
IMAGE_WIDTH = 4656
IMAGE_HEIGHT = 3496

# Camera proxy
PROXY_URL = "http://127.0.0.1:8080"
PROXY_TIMEOUT_SEC = 10

# Burst timeout: enough time for all frames + overhead
# 20 frames × 1.5s = 30s + 15s margin
BURST_TIMEOUT_SEC = (NUM_IMAGES * CAPTURE_INTERVAL_MS / 1000) + 15

# Thresholds
MIN_DISK_MB = 1000          # Require 1GB free
MAX_GOAT_ID_LEN = 50        # Prevent path traversal

# ============================================================
# LOGGING
# ============================================================

import sys
sys.path.insert(0, '/home/pi/goatdev/pi')
from logger.pi_cloudwatch import Logger

log = Logger('pi/training')

# ============================================================
# STATE
# ============================================================

_state_lock = threading.Lock()
recording_state = {
    'active': False,
    'goat_id': None,
    'started_at': None,
    'progress': None,
    'last_error': None,
    'last_result': None
}

def set_state(**kwargs):
    with _state_lock:
        recording_state.update(kwargs)

def get_state():
    with _state_lock:
        return recording_state.copy()

def reset_state():
    with _state_lock:
        recording_state.update({
            'active': False,
            'goat_id': None,
            'started_at': None,
            'progress': None
        })

# ============================================================
# HELPERS
# ============================================================

_s3_client = None

def get_s3():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client('s3')
    return _s3_client


def s3_upload_check(bucket: str, would_upload_keys: list[str]) -> dict:
    """
    Verify we *could* upload without actually uploading.
    - head_bucket verifies bucket exists + creds allow access
    - get_caller_identity verifies creds are valid (optional)
    """
    result = {
        "ok": False,
        "bucket": bucket,
        "error": None,
        "would_upload": would_upload_keys[:10],
    }

    try:
        s3 = get_s3()
        s3.head_bucket(Bucket=bucket)

        try:
            import boto3
            sts = boto3.client("sts")
            ident = sts.get_caller_identity()
            result["aws_account"] = ident.get("Account")
            result["aws_arn"] = ident.get("Arn")
        except Exception as e:
            result["sts_warning"] = str(e)

        result["ok"] = True
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def sanitize_goat_id(goat_id) -> str:
    """Sanitize goat_id to prevent path traversal and weirdness."""
    goat_id = str(goat_id)
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', goat_id)
    if not sanitized:
        raise ValueError("goat_id must contain alphanumeric characters")
    if len(sanitized) > MAX_GOAT_ID_LEN:
        raise ValueError(f"goat_id too long (max {MAX_GOAT_ID_LEN})")
    return sanitized


def check_disk_space() -> tuple[bool, int]:
    """Check /tmp has enough space. Returns (ok, free_mb)."""
    try:
        _, _, free = shutil.disk_usage('/tmp')
        free_mb = free // (1024 * 1024)
        return free_mb >= MIN_DISK_MB, free_mb
    except Exception as e:
        log.error('disk', 'Failed to check disk space', error=str(e))
        return False, 0


def check_proxy_health() -> dict:
    """Quick check: is the proxy running and which cameras are connected?"""
    result = {'ok': False, 'cameras': {}, 'error': None}

    try:
        resp = http_requests.get(f'{PROXY_URL}/status', timeout=5)
        if resp.status_code != 200:
            result['error'] = f'Proxy returned {resp.status_code}'
            return result

        data = resp.json()
        for name in CAMERAS:
            cam = data.get('cameras', {}).get(name, {})
            result['cameras'][name] = {
                'connected': cam.get('connected', False),
                'stale': cam.get('stale', False),
                'error': cam.get('last_error'),
            }

        result['ok'] = True
        return result

    except http_requests.exceptions.ConnectionError:
        result['error'] = 'Camera proxy not running'
        return result
    except Exception as e:
        result['error'] = str(e)
        return result


def cleanup_temp_files(goat_id: str):
    """Remove any temp files for this goat_id."""
    try:
        for f in os.listdir('/tmp'):
            if f.startswith(f'{goat_id}_') and (f.endswith('.jpg') or f.endswith('.json') or f.endswith('.tar.gz')):
                os.remove(f'/tmp/{f}')
                log.debug('cleanup', f'Removed temp file', file=f)
    except Exception as e:
        log.warn('cleanup', 'Temp file cleanup failed', error=str(e))


# ============================================================
# CAPTURE LOGIC
# ============================================================

def capture_camera_burst(name: str, goat_id: str) -> dict:
    """
    Capture images from a single camera via the proxy's burst endpoint.

    One HTTP request replaces the previous 20-request loop. The proxy
    collects NUM_IMAGES frames at CAPTURE_INTERVAL_MS spacing, deduplicates
    by timestamp, and returns a tar.gz. We save it to /tmp for S3 upload.
    """
    result = {
        'name': name,
        'success': False,
        'tar_path': None,
        'total_size_bytes': 0,
        'image_count': 0,
        'error': None,
        'fix': None
    }

    log.info(f'camera:{name}', 'Starting burst capture via proxy',
             num_images=NUM_IMAGES, interval_ms=CAPTURE_INTERVAL_MS)

    try:
        resp = http_requests.post(
            f'{PROXY_URL}/capture/burst/{name}',
            json={
                'count': NUM_IMAGES,
                'interval_ms': CAPTURE_INTERVAL_MS,
                'prefix': goat_id,
            },
            timeout=BURST_TIMEOUT_SEC,
        )

        if resp.status_code == 503:
            try:
                error_data = resp.json()
            except Exception:
                error_data = {}
            result['error'] = error_data.get('error', f'Camera {name} not available')
            result['fix'] = f'Check USB connection for {name} camera'
            log.error(f'camera:{name}', 'Burst capture failed (503)',
                      error=result['error'])
            return result

        if resp.status_code != 200:
            result['error'] = f'Proxy returned {resp.status_code}'
            result['fix'] = 'Check camera proxy logs'
            log.error(f'camera:{name}', 'Burst capture failed',
                      status=resp.status_code)
            return result

        frame_count = int(resp.headers.get('X-Frame-Count', 0))
        if frame_count == 0:
            result['error'] = 'Proxy returned empty burst (0 frames)'
            result['fix'] = 'Camera may be frozen or disconnected'
            log.error(f'camera:{name}', 'Empty burst response')
            return result

        # Save tar.gz directly — this is what gets uploaded to S3
        tar_path = f'/tmp/{goat_id}_{name}.tar.gz'
        with open(tar_path, 'wb') as f:
            f.write(resp.content)

        result['success'] = True
        result['tar_path'] = tar_path
        result['total_size_bytes'] = len(resp.content)
        result['image_count'] = frame_count

        log.info(f'camera:{name}', 'Burst capture complete',
                 images=frame_count, requested=NUM_IMAGES,
                 tar_size_bytes=len(resp.content))

    except http_requests.exceptions.ConnectionError:
        result['error'] = 'Camera proxy not running'
        result['fix'] = 'Run: sudo systemctl restart camera-proxy'
        log.error(f'camera:{name}', 'Proxy connection refused')

    except http_requests.exceptions.Timeout:
        result['error'] = f'Burst capture timed out after {BURST_TIMEOUT_SEC}s'
        result['fix'] = 'Camera may be slow or frozen. Check proxy logs.'
        log.error(f'camera:{name}', 'Burst timed out',
                  timeout_sec=BURST_TIMEOUT_SEC)

    except Exception as e:
        result['error'] = str(e)
        log.error(f'camera:{name}', 'Unexpected capture error', error=str(e))

    return result


def do_capture(goat_id: str, goat_data: dict, is_test: bool):
    """
    Main capture workflow. Runs in background thread.

    All 3 cameras capture concurrently — the proxy handles USB controller
    contention internally via pair rotation. No batching needed here.

    REAL: all cameras must succeed and upload to S3.
    TEST: capture what is available, S3 capability check only (no upload).
    """
    start_time = time.time()
    results = {}
    results_lock = threading.Lock()

    try:
        set_state(progress='capturing')

        # === CAPTURE ALL 3 CAMERAS CONCURRENTLY ===
        # No batching needed — proxy handles USB pair rotation internally.
        # Each burst call is a single HTTP request that blocks for ~30s while
        # the proxy collects frames. 3 concurrent threads, 3 HTTP connections.
        threads = {}
        for name in CAMERAS:
            def _capture(n=name):
                result = capture_camera_burst(n, goat_id)
                with results_lock:
                    results[n] = result

            t = threading.Thread(target=_capture, name=f'burst-{name}')
            threads[name] = t
            t.start()

        # Wait for all threads (burst timeout + margin for tar packaging)
        for name, t in threads.items():
            t.join(timeout=BURST_TIMEOUT_SEC + 10)
            if t.is_alive():
                log.error(f'camera:{name}', 'Burst thread did not complete in time')
                with results_lock:
                    if name not in results:
                        results[name] = {
                            'name': name,
                            'success': False,
                            'tar_path': None,
                            'image_count': 0,
                            'error': 'Capture thread hung',
                            'fix': 'Restart camera-proxy: sudo systemctl restart camera-proxy'
                        }

        # === EVALUATE RESULTS ===
        successful = [r for r in results.values() if r.get('success')]
        failed = [r for r in results.values() if not r.get('success')]

        camera_status = {}
        for cam in CAMERAS.keys():
            r = results.get(cam)
            if not r:
                camera_status[cam] = "missing"
            elif r.get("success"):
                camera_status[cam] = f"captured ({r.get('image_count', 0)} images)"
            else:
                camera_status[cam] = f"failed: {r.get('error', 'unknown')}"

        total_images = sum(r.get('image_count', 0) for r in successful)

        # REAL mode: strict (any failure => batch fail)
        if (not is_test) and failed:
            # Clean up successful captures
            for r in successful:
                tar_path = r.get('tar_path')
                if tar_path and os.path.exists(tar_path):
                    os.remove(tar_path)
                log.info(f"camera:{r['name']}", 'Cleaned up tar (batch failed)')

            error_summary = '; '.join([f"{r['name']}: {r.get('error', 'unknown')}" for r in failed])

            set_state(
                last_error=error_summary,
                last_result={
                    'goat_id': goat_id,
                    'success': False,
                    'uploaded': [],
                    'total_images': 0,
                    'camera_status': camera_status,
                    'errors': [{'camera': r['name'], 'error': r.get('error'), 'fix': r.get('fix')} for r in failed],
                    'duration_sec': round(time.time() - start_time, 2)
                }
            )

            log.error('capture', 'Capture batch failed',
                     goat_id=goat_id,
                     failed_cameras=','.join([r['name'] for r in failed]),
                     error=error_summary)
            return

        # TEST mode: permissive (require at least 1 camera success)
        if is_test and not successful:
            error_summary = '; '.join([f"{r['name']}: {r.get('error', 'unknown')}" for r in failed]) or "No cameras captured"
            set_state(
                last_error=error_summary,
                last_result={
                    'goat_id': goat_id,
                    'success': False,
                    'test': True,
                    'uploaded': [],
                    'total_images': 0,
                    'camera_status': camera_status,
                    'errors': [{'camera': r['name'], 'error': r.get('error'), 'fix': r.get('fix')} for r in failed],
                    'duration_sec': round(time.time() - start_time, 2)
                }
            )
            log.error('capture', 'Test capture failed (no cameras succeeded)',
                     goat_id=goat_id, error=error_summary)
            return

        # === TEST MODE: S3 capability check only (no upload) ===
        if is_test:
            set_state(progress='checking_s3')

            would_upload = []
            for r in successful:
                would_upload.append(f'{goat_id}/{r["name"]}/images.tar.gz')
            if goat_data:
                would_upload.append(f'{goat_id}/goat_data.json')

            s3_check = s3_upload_check(S3_TRAINING_BUCKET, would_upload)

            # Clean up local tar files
            for r in successful:
                tar_path = r.get('tar_path')
                if tar_path and os.path.exists(tar_path):
                    os.remove(tar_path)

            total_time = round(time.time() - start_time, 2)

            if not s3_check.get("ok"):
                set_state(
                    last_error="S3 upload check failed",
                    last_result={
                        'goat_id': goat_id,
                        'success': False,
                        'test': True,
                        'uploaded': [],
                        'total_images': total_images,
                        'camera_status': camera_status,
                        's3_check': s3_check,
                        'errors': [{'camera': 's3', 'error': s3_check.get('error')}],
                        'duration_sec': total_time
                    }
                )
                log.error('capture', 'Test complete but S3 check failed',
                         goat_id=goat_id, error=s3_check.get("error"))
            else:
                set_state(
                    last_error=None,
                    last_result={
                        'goat_id': goat_id,
                        'success': True,
                        'test': True,
                        'uploaded': [],
                        'total_images': total_images,
                        'camera_status': camera_status,
                        's3_check': s3_check,
                        'errors': [],
                        'duration_sec': total_time
                    }
                )
                log.info('capture', 'Test capture complete (S3 check passed; no upload)',
                        goat_id=goat_id, total_images=total_images,
                        cameras=len(successful), duration_sec=total_time)
            return

        # === UPLOAD TO S3 (REAL) ===
        # Tar.gz files come directly from the proxy — no re-tarring needed.
        # One S3 PUT per camera instead of the old tar-then-upload cycle.
        set_state(progress='uploading')
        uploaded = []
        upload_errors = []

        for r in successful:
            cam_name = r['name']
            tar_path = r.get('tar_path')

            if not tar_path or not os.path.exists(tar_path):
                upload_errors.append({
                    'camera': cam_name,
                    'error': 'Tar file missing',
                    'fix': 'Capture may have been interrupted'
                })
                continue

            s3_key = f'{goat_id}/{cam_name}/images.tar.gz'

            try:
                tar_size = os.path.getsize(tar_path)
                log.info(f's3:{cam_name}', 'Uploading tar',
                        key=s3_key, size_bytes=tar_size,
                        image_count=r.get('image_count', 0))

                get_s3().upload_file(tar_path, S3_TRAINING_BUCKET, s3_key)
                uploaded.append(s3_key)

                log.info(f's3:{cam_name}', 'Upload complete', key=s3_key)

            except Exception as e:
                error_msg = str(e)
                if 'NoSuchBucket' in error_msg:
                    fix = f'Bucket {S3_TRAINING_BUCKET} does not exist'
                elif 'AccessDenied' in error_msg:
                    fix = 'Check IAM permissions for S3 upload'
                elif 'timeout' in error_msg.lower():
                    fix = 'Network issue - check internet connection'
                else:
                    fix = 'Check AWS credentials and network'

                upload_errors.append({'camera': cam_name, 'error': error_msg, 'fix': fix})
                log.error(f's3:{cam_name}', 'Upload failed',
                         error=error_msg, fix=fix, key=s3_key)

            finally:
                if os.path.exists(tar_path):
                    os.remove(tar_path)

        # Upload goat_data.json
        if goat_data and uploaded:
            try:
                json_data = {
                    'goat_id': goat_id,
                    'timestamp': datetime.now().isoformat(),
                    'cameras': [r['name'] for r in successful],
                    'images_per_camera': NUM_IMAGES,
                    'capture_interval_ms': CAPTURE_INTERVAL_MS,
                    'resolution': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
                    **{k: v for k, v in goat_data.items() if v}
                }

                json_key = f'{goat_id}/goat_data.json'
                json_path = f'/tmp/{goat_id}_goat_data.json'

                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)

                get_s3().upload_file(json_path, S3_TRAINING_BUCKET, json_key)
                uploaded.append(json_key)
                log.info('s3:metadata', 'Uploaded goat_data.json', key=json_key)

                os.remove(json_path)

            except Exception as e:
                log.error('s3:metadata', 'Failed to upload goat_data.json', error=str(e))
                upload_errors.append({'camera': 'metadata', 'error': str(e)})

        # === FINAL RESULT ===
        total_time = round(time.time() - start_time, 2)

        if upload_errors:
            set_state(
                last_error=f'Upload failed for {len(upload_errors)} files',
                last_result={
                    'goat_id': goat_id,
                    'success': False,
                    'uploaded': uploaded,
                    'total_images': total_images,
                    'camera_status': camera_status,
                    'errors': upload_errors,
                    'duration_sec': total_time
                }
            )
            log.error('capture', 'Capture complete but upload failed',
                     goat_id=goat_id, uploaded=len(uploaded),
                     errors=len(upload_errors))
        else:
            set_state(
                last_error=None,
                last_result={
                    'goat_id': goat_id,
                    'success': True,
                    'uploaded': uploaded,
                    'total_images': total_images,
                    'camera_status': camera_status,
                    'errors': [],
                    'duration_sec': total_time
                }
            )
            log.info('capture', 'Capture complete',
                    goat_id=goat_id, total_images=total_images,
                    uploaded=len(uploaded), duration_sec=total_time)

    except Exception as e:
        log.exception('capture', 'Unexpected error in capture workflow', error=str(e))
        set_state(
            last_error=str(e),
            last_result={
                'goat_id': goat_id,
                'success': False,
                'errors': [{'camera': 'system', 'error': str(e)}]
            }
        )
        cleanup_temp_files(goat_id)

    finally:
        reset_state()


# ============================================================
# ROUTES
# ============================================================

@app.route('/health')
def health():
    """Quick health check."""
    disk_ok, disk_mb = check_disk_space()
    proxy = check_proxy_health()

    cameras = {}
    for name in CAMERAS:
        cam = proxy.get('cameras', {}).get(name, {})
        cameras[name] = cam.get('connected', False)

    return jsonify({
        'status': 'ok' if all(cameras.values()) and disk_ok else 'degraded',
        'cameras': cameras,
        'disk_free_mb': disk_mb,
        'recording_active': get_state()['active']
    })


@app.route('/diagnostics')
def diagnostics():
    """Detailed diagnostics for troubleshooting."""
    log.info('diag', 'Running diagnostics')

    diag = {
        'timestamp': datetime.now().isoformat(),
        'cameras': {},
        'disk': {},
        's3': {},
        'proxy': {},
        'capture_config': {
            'num_images': NUM_IMAGES,
            'capture_interval_ms': CAPTURE_INTERVAL_MS,
            'burst_timeout_sec': BURST_TIMEOUT_SEC,
            'resolution': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
            'proxy_url': PROXY_URL,
        },
        'state': get_state()
    }

    # Proxy health (includes per-camera status)
    proxy = check_proxy_health()
    diag['proxy'] = proxy
    diag['cameras'] = proxy.get('cameras', {})

    # Disk
    disk_ok, disk_mb = check_disk_space()
    diag['disk'] = {'ok': disk_ok, 'free_mb': disk_mb, 'required_mb': MIN_DISK_MB}

    # S3
    try:
        get_s3().head_bucket(Bucket=S3_TRAINING_BUCKET)
        diag['s3'] = {'ok': True, 'bucket': S3_TRAINING_BUCKET}
    except Exception as e:
        diag['s3'] = {'ok': False, 'bucket': S3_TRAINING_BUCKET, 'error': str(e)}

    log.info('diag', 'Diagnostics complete',
            proxy_ok=proxy['ok'],
            disk_ok=disk_ok,
            s3_ok=diag['s3']['ok'])

    return jsonify(diag)


@app.route('/record', methods=['POST'])
def record():
    """Start a capture.

    REAL (default): requires all 3 cameras. Returns error if any missing.
    TEST (is_test=true or goat_id starts with 'test_'): captures available
    cameras only, does S3 capability check without uploading.
    """

    # Check if already capturing
    state = get_state()
    if state['active']:
        log.warn('capture', 'Capture already in progress',
                current_goat_id=state['goat_id'],
                started_at=state['started_at'])
        return jsonify({
            'status': 'error',
            'error_code': 'CAPTURE_IN_PROGRESS',
            'message': f'Capture already in progress for goat {state["goat_id"]}. Please wait.',
            'current_capture': {
                'goat_id': state['goat_id'],
                'started_at': state['started_at'],
                'progress': state['progress']
            }
        }), 409

    # Parse and validate input
    try:
        data = request.get_json(force=True, silent=True) or {}
    except:
        data = {}

    try:
        goat_id = sanitize_goat_id(data.get('goat_id', f'goat_{int(time.time())}'))
    except ValueError as e:
        log.warn('capture', 'Invalid goat_id', error=str(e),
                raw_goat_id=str(data.get('goat_id'))[:50])
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_GOAT_ID',
            'message': str(e)
        }), 400

    goat_data = data.get('goat_data', {})
    if not isinstance(goat_data, dict):
        goat_data = {}

    is_test = bool(data.get('is_test', False))
    require_all_cameras = bool(data.get('require_all_cameras', True))

    if str(goat_id).startswith('test_'):
        is_test = True
        require_all_cameras = False

    # Pre-flight checks
    log.info('capture', 'Capture requested',
            goat_id=goat_id, num_images=NUM_IMAGES,
            interval_ms=CAPTURE_INTERVAL_MS,
            resolution=f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
            is_test=is_test)

    # Check disk space
    disk_ok, disk_mb = check_disk_space()
    if not disk_ok:
        log.error('capture', 'Insufficient disk space',
                 free_mb=disk_mb, required_mb=MIN_DISK_MB)
        return jsonify({
            'status': 'error',
            'error_code': 'LOW_DISK_SPACE',
            'message': f'Only {disk_mb}MB free, need {MIN_DISK_MB}MB',
            'fix': 'Run: sudo rm -rf /tmp/*.jpg /tmp/*.tar.gz'
        }), 507

    # Quick proxy health check — fast-fail if proxy is down or cameras missing
    proxy = check_proxy_health()
    if not proxy['ok']:
        return jsonify({
            'status': 'error',
            'error_code': 'PROXY_UNAVAILABLE',
            'message': proxy.get('error', 'Camera proxy not reachable'),
            'fix': 'Run: sudo systemctl restart camera-proxy'
        }), 503

    connected = {n: c for n, c in proxy['cameras'].items() if c.get('connected')}
    missing = {n: c for n, c in proxy['cameras'].items() if not c.get('connected')}

    if require_all_cameras and missing:
        return jsonify({
            'status': 'error',
            'error_code': 'MISSING_CAMERAS',
            'message': f'{len(missing)} camera(s) not ready',
            'cameras_missing': list(missing.keys()),
            'cameras_available': list(connected.keys()),
            'cameras': proxy['cameras'],
        }), 503

    if not connected:
        return jsonify({
            'status': 'error',
            'error_code': 'NO_CAMERAS',
            'message': 'No cameras connected'
        }), 503

    # Start capture
    set_state(
        active=True,
        goat_id=goat_id,
        started_at=datetime.now().isoformat(),
        progress='starting',
        last_error=None
    )

    thread = threading.Thread(target=do_capture, args=(goat_id, goat_data, is_test))
    thread.start()

    estimated_sec = round(NUM_IMAGES * CAPTURE_INTERVAL_MS / 1000)

    log.info('capture', 'Capture started',
            goat_id=goat_id, num_images=NUM_IMAGES,
            interval_ms=CAPTURE_INTERVAL_MS,
            estimated_sec=estimated_sec)

    return jsonify({
        'status': 'capture_started',
        'goat_id': goat_id,
        'num_images': NUM_IMAGES,
        'capture_interval_ms': CAPTURE_INTERVAL_MS,
        'estimated_duration_sec': estimated_sec,
        'resolution': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
        'is_test': bool(is_test),
        'require_all_cameras': bool(require_all_cameras),
        'cameras_expected': list(CAMERAS.keys()),
        'cameras_available': list(connected.keys()),
        'cameras_missing': list(missing.keys()),
        'warning': None if not missing else f"Missing cameras: {', '.join(missing.keys())}"
    })


@app.route('/status')
def status():
    """Get current capture state."""
    return jsonify(get_state())


@app.route('/cancel', methods=['POST'])
def cancel():
    """Emergency cancel - reset state and clean up."""
    log.warn('cancel', 'Emergency cancel requested')

    state = get_state()
    if state['goat_id']:
        cleanup_temp_files(state['goat_id'])

    reset_state()
    set_state(last_error='Capture cancelled by user')

    log.info('cancel', 'Capture cancelled and state reset')

    return jsonify({'status': 'cancelled'})


# ============================================================
# STARTUP
# ============================================================

if __name__ == '__main__':
    log.info('startup', '=' * 50)
    log.info('startup', 'TRAINING CAPTURE SERVER STARTING')

    log.info('startup', 'Configuration',
            bucket=S3_TRAINING_BUCKET,
            num_images=NUM_IMAGES,
            capture_interval_ms=CAPTURE_INTERVAL_MS,
            burst_timeout_sec=round(BURST_TIMEOUT_SEC),
            resolution=f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
            min_disk_mb=MIN_DISK_MB,
            proxy_url=PROXY_URL)

    # Check proxy
    proxy = check_proxy_health()
    if proxy['ok']:
        for cam, health in proxy['cameras'].items():
            if health.get('connected'):
                log.info(f'startup:camera:{cam}', 'Camera ready via proxy')
            else:
                log.error(f'startup:camera:{cam}', 'Not connected',
                         error=health.get('error'))
    else:
        log.warn('startup:proxy', 'Camera proxy not reachable',
                error=proxy.get('error'),
                fix='Will retry when capture is requested')

    # Check disk
    disk_ok, disk_mb = check_disk_space()
    if not disk_ok:
        log.error('startup:disk', 'Low disk space',
                 free_mb=disk_mb, required_mb=MIN_DISK_MB,
                 fix='Run: sudo rm -rf /tmp/*.jpg /tmp/*.tar.gz')
    else:
        log.info('startup:disk', 'Disk OK', free_mb=disk_mb)

    # Check S3
    try:
        get_s3().head_bucket(Bucket=S3_TRAINING_BUCKET)
        log.info('startup:s3', 'S3 connection OK', bucket=S3_TRAINING_BUCKET)
    except Exception as e:
        log.error('startup:s3', 'S3 connection failed',
                 error=str(e), bucket=S3_TRAINING_BUCKET,
                 fix='Check AWS credentials in ~/.aws/credentials')

    log.info('startup', 'Server listening', host='0.0.0.0', port=5001)
    log.info('startup', '=' * 50)

    app.run(host='0.0.0.0', port=5001)
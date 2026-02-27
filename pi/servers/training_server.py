"""
Training Data Collection Server
Captures still images from 3 cameras via camera proxy, uploads to S3
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import threading
import requests as http_requests
import os
import shutil
import json
import re
import time
import tarfile
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

# Capture settings
NUM_IMAGES = 20             # Number of images to capture per camera
CAPTURE_INTERVAL_SEC = 1.0  # 1 image per second
IMAGE_WIDTH = 4656
IMAGE_HEIGHT = 3496

# Camera proxy
PROXY_URL = "http://127.0.0.1:8080"
PROXY_TIMEOUT_SEC = 10

# Thresholds
MIN_DISK_MB = 1000          # Require 1GB free
MIN_IMAGE_BYTES = 50000     # 50KB minimum valid image
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


def check_camera(name: str, path: str) -> dict:
    """Check camera health via proxy."""
    result = {
        'name': name,
        'path': path,
        'exists': False,
        'readable': False,
        'in_use': False,
        'error': None,
        'fix': None
    }

    try:
        resp = http_requests.get(f'{PROXY_URL}/status', timeout=5)
        if resp.status_code != 200:
            result['error'] = 'Camera proxy returned error'
            result['fix'] = 'Run: sudo systemctl restart camera-proxy'
            return result

        data = resp.json()
        cam_health = data.get('cameras', {}).get(name)

        if not cam_health:
            result['error'] = 'Camera not found in proxy'
            return result

        result['exists'] = cam_health.get('device_exists', False)
        result['readable'] = cam_health.get('connected', False)

        if not result['exists']:
            result['error'] = 'Camera not connected'
            result['fix'] = f'Check USB connection for {name} camera'
        elif not result['readable']:
            result['error'] = cam_health.get('last_error', 'Camera not ready')
            result['fix'] = 'Check camera proxy logs'
        elif cam_health.get('stale', False):
            result['error'] = f"Frame stale ({cam_health.get('frame_age_sec')}s old)"
            result['fix'] = 'Camera may be frozen. Check USB connection.'

    except http_requests.exceptions.ConnectionError:
        result['error'] = 'Camera proxy not running'
        result['fix'] = 'Run: sudo systemctl restart camera-proxy'

    except Exception as e:
        result['error'] = f'Proxy check failed: {e}'

    return result


def cleanup_temp_files(goat_id: str):
    """Remove any temp files for this goat_id."""
    try:
        for f in os.listdir('/tmp'):
            if f.startswith(f'{goat_id}_') and (f.endswith('.jpg') or f.endswith('.json')):
                os.remove(f'/tmp/{f}')
                log.debug('cleanup', f'Removed temp file', file=f)
    except Exception as e:
        log.warn('cleanup', 'Temp file cleanup failed', error=str(e))


# ============================================================
# CAPTURE LOGIC
# ============================================================

def capture_single_camera(name: str, path: str, goat_id: str) -> dict:
    """
    Capture NUM_IMAGES still images from a camera via the camera proxy.
    Grabs one frame per second from the proxy's full-res buffer.
    """
    result = {
        'name': name,
        'success': False,
        'filepaths': [],
        'total_size_bytes': 0,
        'image_count': 0,
        'error': None,
        'fix': None
    }

    log.info(f'camera:{name}', 'Starting capture via proxy',
            num_images=NUM_IMAGES, interval_sec=CAPTURE_INTERVAL_SEC)

    valid_files = []
    total_size = 0
    last_timestamp = None

    for i in range(1, NUM_IMAGES + 1):
        filepath = f'/tmp/{goat_id}_{name}_{i:02d}.jpg'

        try:
            resp = http_requests.get(
                f'{PROXY_URL}/capture/{name}',
                timeout=PROXY_TIMEOUT_SEC
            )

            if resp.status_code == 503:
                error_data = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {}
                log.warn(f'camera:{name}', f'Frame {i} unavailable',
                        status=503, error=error_data.get('error'))
                # Wait and retry once
                time.sleep(0.5)
                continue

            if resp.status_code != 200:
                log.warn(f'camera:{name}', f'Frame {i} unexpected status',
                        status=resp.status_code)
                continue

            # Check we got a new frame (not the same one again)
            frame_ts = resp.headers.get('X-Frame-Timestamp')
            if frame_ts and frame_ts == last_timestamp:
                log.warn(f'camera:{name}', f'Frame {i} is duplicate, waiting')
                time.sleep(0.2)
                # Retry
                resp = http_requests.get(
                    f'{PROXY_URL}/capture/{name}',
                    timeout=PROXY_TIMEOUT_SEC
                )
                if resp.status_code != 200:
                    continue
                frame_ts = resp.headers.get('X-Frame-Timestamp')
            last_timestamp = frame_ts

            # Validate size
            if len(resp.content) < MIN_IMAGE_BYTES:
                log.warn(f'camera:{name}', f'Frame {i} too small',
                        size=len(resp.content), min=MIN_IMAGE_BYTES)
                continue

            with open(filepath, 'wb') as f:
                f.write(resp.content)

            file_size = os.path.getsize(filepath)
            total_size += file_size
            valid_files.append(filepath)

        except http_requests.exceptions.ConnectionError:
            result['error'] = 'Camera proxy not running'
            result['fix'] = 'Run: sudo systemctl restart camera-proxy'
            log.error(f'camera:{name}', 'Proxy connection refused')
            return result

        except http_requests.exceptions.Timeout:
            log.warn(f'camera:{name}', f'Frame {i} timeout',
                    timeout=PROXY_TIMEOUT_SEC)
            continue

        except Exception as e:
            log.warn(f'camera:{name}', f'Frame {i} error', error=str(e))
            continue

        # Wait for next frame interval
        if i < NUM_IMAGES:
            time.sleep(CAPTURE_INTERVAL_SEC)

    if not valid_files:
        if not result['error']:
            result['error'] = 'No valid images captured from proxy'
            result['fix'] = 'Check camera proxy status: curl http://127.0.0.1:8080/status'
        log.error(f'camera:{name}', 'All frames failed')
        return result

    if len(valid_files) < NUM_IMAGES:
        log.warn(f'camera:{name}', 'Some frames missed',
                valid=len(valid_files), expected=NUM_IMAGES)

    result['success'] = True
    result['filepaths'] = valid_files
    result['total_size_bytes'] = total_size
    result['image_count'] = len(valid_files)

    log.info(f'camera:{name}', 'Capture complete',
            image_count=len(valid_files), total_size_bytes=total_size)
    return result


def do_capture(goat_id: str, goat_data: dict, is_test: bool):
    """
    Main capture workflow. Runs in background thread.
    REAL: all cameras must succeed and upload.
    TEST: capture what is available, do S3 capability check only (no upload).
    """
    start_time = time.time()
    results = {}
    # Lock for thread-safe writes to the shared results dict.
    results_lock = threading.Lock()

    try:
        set_state(progress='capturing')

        # Pre-check all cameras via proxy
        available_cameras = {}
        for name, path in CAMERAS.items():
            check = check_camera(name, path)
            if check['error']:
                with results_lock:
                    results[name] = {
                        'name': name,
                        'success': False,
                        'filepaths': [],
                        'image_count': 0,
                        'error': check['error'],
                        'fix': check['fix']
                    }
            else:
                available_cameras[name] = path

        # Capture cameras in batches of 2 to avoid overloading proxy
        MAX_PARALLEL = 2
        camera_list = list(available_cameras.items())
        batches = [camera_list[i:i + MAX_PARALLEL] for i in range(0, len(camera_list), MAX_PARALLEL)]

        for batch_num, batch in enumerate(batches, 1):
            log.info('capture', f'Starting batch {batch_num}/{len(batches)}',
                    cameras=','.join(name for name, _ in batch))

            threads = {}
            for name, path in batch:
                # Thread-safe write to results dict via lock.
                # Was: target=lambda n=name, p=path: results.update({n: capture_single_camera(n, p, goat_id)})
                def _capture_and_store(n=name, p=path):
                    result = capture_single_camera(n, p, goat_id)
                    with results_lock:
                        results[n] = result

                t = threading.Thread(target=_capture_and_store)
                threads[name] = t
                t.start()

            for name, t in threads.items():
                # Each camera takes ~20s (NUM_IMAGES * CAPTURE_INTERVAL_SEC)
                t.join(timeout=(NUM_IMAGES * CAPTURE_INTERVAL_SEC) + 30)
                if t.is_alive():
                    log.error(f'camera:{name}', 'Thread did not complete in time')
                    with results_lock:
                        results[name] = {
                            'name': name,
                            'success': False,
                            'filepaths': [],
                            'image_count': 0,
                            'error': 'Capture thread hung',
                            'fix': 'Restart the service: sudo systemctl restart goat-training'
                        }

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
                if r.get("error") in ("Camera not connected", "Camera proxy not running"):
                    camera_status[cam] = "missing"
                else:
                    camera_status[cam] = "failed"

        total_images = sum(r.get('image_count', 0) for r in successful)

        # REAL mode: strict (any failure => batch fail)
        if (not is_test) and failed:
            for r in successful:
                for fp in r.get('filepaths', []):
                    if os.path.exists(fp):
                        os.remove(fp)
                log.info(f"camera:{r['name']}", 'Cleaned up files (batch failed)')

            error_summary = '; '.join([f"{r['name']}: {r.get('error', 'unknown')}" for r in failed])
            fix_summary = ' | '.join([r.get('fix', '') for r in failed if r.get('fix')])

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
                     error=error_summary,
                     fix=fix_summary)
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
            log.error('capture', 'Test capture failed (no cameras succeeded)', goat_id=goat_id, error=error_summary)
            return

        # TEST: do NOT upload; do S3 capability check only
        if is_test:
            set_state(progress='checking_s3')

            would_upload = []
            for r in successful:
                for fp in r.get('filepaths', []):
                    filename = os.path.basename(fp)
                    would_upload.append(f'{goat_id}/{r["name"]}/{filename}')
            if goat_data:
                would_upload.append(f'{goat_id}/goat_data.json')

            s3_check = s3_upload_check(S3_TRAINING_BUCKET, would_upload)

            # Clean up local files
            for r in successful:
                for fp in r.get('filepaths', []):
                    if os.path.exists(fp):
                        os.remove(fp)

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
                log.error('capture', 'Test complete but S3 upload check failed', goat_id=goat_id, error=s3_check.get("error"))
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
                log.info('capture', 'Test capture complete (upload check passed; no upload)',
                        goat_id=goat_id, total_images=total_images, cameras=len(successful), duration_sec=total_time)
            return

        # Upload to S3 (REAL) - tar per camera for fewer S3 round trips
        set_state(progress='uploading')
        uploaded = []
        upload_errors = []

        for r in successful:
            cam_name = r['name']
            filepaths = r.get('filepaths', [])
            if not filepaths:
                continue

            tar_path = f'/tmp/{goat_id}_{cam_name}.tar.gz'
            s3_key = f'{goat_id}/{cam_name}/images.tar.gz'

            try:
                # Create tar.gz of all images for this camera
                with tarfile.open(tar_path, 'w:gz') as tar:
                    for fp in filepaths:
                        tar.add(fp, arcname=os.path.basename(fp))

                tar_size = os.path.getsize(tar_path)
                log.info(f's3:{cam_name}', 'Uploading tar',
                        key=s3_key, size_bytes=tar_size,
                        image_count=len(filepaths))

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
                # Clean up local files
                for fp in filepaths:
                    if os.path.exists(fp):
                        os.remove(fp)
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

        # Final result
        total_time = round(time.time() - start_time, 2)

        if upload_errors:
            set_state(
                last_error=f'Upload failed for {len(upload_errors)} files',
                last_result={
                    'goat_id': goat_id,
                    'success': False,
                    'uploaded': uploaded,
                    'total_images': total_images,
                    'errors': upload_errors,
                    'duration_sec': total_time
                }
            )
            log.error('capture', 'Capture complete but upload failed',
                     goat_id=goat_id, uploaded=len(uploaded), errors=len(upload_errors))
        else:
            set_state(
                last_error=None,
                last_result={
                    'goat_id': goat_id,
                    'success': True,
                    'uploaded': uploaded,
                    'total_images': total_images,
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
    cameras = {}
    disk_ok, disk_mb = check_disk_space()

    try:
        resp = http_requests.get(f'{PROXY_URL}/status', timeout=5)
        if resp.status_code == 200:
            proxy_data = resp.json()
            for name in CAMERAS:
                cam_health = proxy_data.get('cameras', {}).get(name, {})
                cameras[name] = cam_health.get('connected', False)
        else:
            for name in CAMERAS:
                cameras[name] = False
    except Exception:
        for name in CAMERAS:
            cameras[name] = False

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
            'capture_interval_sec': CAPTURE_INTERVAL_SEC,
            'resolution': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
            'proxy_url': PROXY_URL,
        },
        'state': get_state()
    }

    # Check each camera via proxy
    for name, path in CAMERAS.items():
        diag['cameras'][name] = check_camera(name, path)

    # Proxy health
    try:
        resp = http_requests.get(f'{PROXY_URL}/status', timeout=5)
        if resp.status_code == 200:
            diag['proxy'] = {'ok': True, 'data': resp.json()}
        else:
            diag['proxy'] = {'ok': False, 'status_code': resp.status_code}
    except Exception as e:
        diag['proxy'] = {'ok': False, 'error': str(e)}

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
            cameras_ok=sum(1 for c in diag['cameras'].values() if not c.get('error')),
            disk_ok=disk_ok,
            s3_ok=diag['s3']['ok'],
            proxy_ok=diag['proxy']['ok'])

    return jsonify(diag)


@app.route('/record', methods=['POST'])
def record():
    """Start a capture. REAL requires all 3 cameras; TEST can proceed with missing cameras."""

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
        }), 409  # Conflict

    # Parse and validate input
    try:
        data = request.get_json(force=True, silent=True) or {}
    except:
        data = {}

    try:
        goat_id = sanitize_goat_id(data.get('goat_id', f'goat_{int(time.time())}'))
    except ValueError as e:
        log.warn('capture', 'Invalid goat_id', error=str(e), raw_goat_id=str(data.get('goat_id'))[:50])
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

    # Back-compat: goat_id naming convention forces test behavior
    if str(goat_id).startswith('test_'):
        is_test = True
        require_all_cameras = False

    # Pre-flight checks
    log.info('capture', 'Capture requested',
            goat_id=goat_id, num_images=NUM_IMAGES,
            interval_sec=CAPTURE_INTERVAL_SEC,
            resolution=f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}')

    # Check disk space
    disk_ok, disk_mb = check_disk_space()
    if not disk_ok:
        log.error('capture', 'Insufficient disk space',
                 free_mb=disk_mb, required_mb=MIN_DISK_MB,
                 fix='Clear /tmp or reboot Pi')
        return jsonify({
            'status': 'error',
            'error_code': 'LOW_DISK_SPACE',
            'message': f'Only {disk_mb}MB free, need {MIN_DISK_MB}MB',
            'fix': 'Run: sudo rm -rf /tmp/*.jpg'
        }), 507  # Insufficient Storage

    # Check cameras via proxy
    camera_checks = {}
    for name, path in CAMERAS.items():
        camera_checks[name] = check_camera(name, path)

    failed_cameras = {name: check for name, check in camera_checks.items() if check.get('error')}
    working_cameras = {name: check for name, check in camera_checks.items() if not check.get('error')}

    if failed_cameras:
        log.warn('capture', 'Some cameras not ready',
                goat_id=goat_id,
                missing=','.join(failed_cameras.keys()),
                available=','.join(working_cameras.keys()))

    # Strict mode: reject if ANY camera missing/not ready
    if require_all_cameras and failed_cameras:
        errors = [f"{name}: {check['error']}" for name, check in failed_cameras.items()]
        fixes = [f"{name}: {check['fix']}" for name, check in failed_cameras.items() if check.get('fix')]

        return jsonify({
            'status': 'error',
            'error_code': 'MISSING_CAMERAS',
            'message': f'{len(failed_cameras)} camera(s) not ready',
            'cameras_missing': list(failed_cameras.keys()),
            'cameras_available': list(working_cameras.keys()),
            'cameras': camera_checks,
            'errors': errors,
            'fixes': fixes
        }), 503

    if not working_cameras:
        log.error('capture', 'No cameras available at all', goat_id=goat_id)
        return jsonify({
            'status': 'error',
            'error_code': 'NO_CAMERAS',
            'message': 'No cameras connected'
        }), 503

    # Set state and start capture
    set_state(
        active=True,
        goat_id=goat_id,
        started_at=datetime.now().isoformat(),
        progress='starting',
        last_error=None
    )

    thread = threading.Thread(target=do_capture, args=(goat_id, goat_data, is_test))
    thread.start()

    log.info('capture', 'Capture started',
            goat_id=goat_id, num_images=NUM_IMAGES,
            cameras=','.join(CAMERAS.keys()))

    return jsonify({
        'status': 'capture_started',
        'goat_id': goat_id,
        'num_images': NUM_IMAGES,
        'capture_interval_sec': CAPTURE_INTERVAL_SEC,
        'resolution': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
        'is_test': bool(is_test),
        'require_all_cameras': bool(require_all_cameras),
        'cameras_expected': list(CAMERAS.keys()),
        'cameras_available': list(working_cameras.keys()),
        'cameras_missing': list(failed_cameras.keys()),
        'warning': None if (not failed_cameras) else f"Missing/not-ready cameras: {', '.join(failed_cameras.keys())}"
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

    # Config
    log.info('startup', 'Configuration',
            bucket=S3_TRAINING_BUCKET,
            num_images=NUM_IMAGES,
            capture_interval_sec=CAPTURE_INTERVAL_SEC,
            resolution=f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
            min_disk_mb=MIN_DISK_MB,
            proxy_url=PROXY_URL)

    # Check proxy
    try:
        resp = http_requests.get(f'{PROXY_URL}/status', timeout=5)
        if resp.status_code == 200:
            proxy_status = resp.json()
            for cam, health in proxy_status.get('cameras', {}).items():
                if health.get('connected'):
                    log.info(f'startup:camera:{cam}', 'Camera ready via proxy',
                            fps=health.get('fps'))
                else:
                    log.error(f'startup:camera:{cam}', 'Not connected',
                             error=health.get('last_error'))
        else:
            log.warn('startup:proxy', 'Proxy returned error', status=resp.status_code)
    except http_requests.exceptions.ConnectionError:
        log.warn('startup:proxy', 'Camera proxy not yet running',
                fix='Will retry when capture is requested')
    except Exception as e:
        log.warn('startup:proxy', 'Proxy check failed', error=str(e))

    # Check disk
    disk_ok, disk_mb = check_disk_space()
    if not disk_ok:
        log.error('startup:disk', 'Low disk space',
                 free_mb=disk_mb, required_mb=MIN_DISK_MB,
                 fix='Run: sudo rm -rf /tmp/*.jpg')
    else:
        log.info('startup:disk', 'Disk OK', free_mb=disk_mb)

    # Check S3
    s3_ok = False
    try:
        get_s3().head_bucket(Bucket=S3_TRAINING_BUCKET)
        s3_ok = True
        log.info('startup:s3', 'S3 connection OK', bucket=S3_TRAINING_BUCKET)
    except Exception as e:
        log.error('startup:s3', 'S3 connection failed',
                 error=str(e), bucket=S3_TRAINING_BUCKET,
                 fix='Check AWS credentials in ~/.aws/credentials')

    # Summary
    if disk_ok and s3_ok:
        log.info('startup', 'All systems ready')
    else:
        log.warn('startup', 'Starting with errors - some features may not work')

    log.info('startup', 'Server listening', host='0.0.0.0', port=5001)
    log.info('startup', '=' * 50)

    app.run(host='0.0.0.0', port=5001)
"""
Training Data Collection Server
Captures still images from 3 cameras, uploads to S3
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import threading
import os
import shutil
import json
import re
import time
import tarfile
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================

CAMERAS = {
    'side': '/dev/camera_side',
    'top': '/dev/camera_top',
    'front': '/dev/camera_front'
}

S3_TRAINING_BUCKET = os.environ.get('S3_TRAINING_BUCKET', 'training-937249941844')

# Capture settings
NUM_IMAGES = 20             # Number of images to capture per camera
CAPTURE_FPS = 1             # 1 picture per second (1 second apart)
IMAGE_WIDTH = 4656
IMAGE_HEIGHT = 3496

# Thresholds
MIN_DISK_MB = 1000          # Require 1GB free
MIN_IMAGE_BYTES = 50000     # 50KB minimum valid image
FFMPEG_TIMEOUT_SEC = 30     # Kill ffmpeg after this
MAX_GOAT_ID_LEN = 50        # Prevent path traversal

# ============================================================
# LOGGING
# ============================================================

import sys
sys.path.insert(0, '/home/pi/goat-capture')
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
    """Check a single camera's status."""
    result = {
        'name': name,
        'path': path,
        'exists': os.path.exists(path),
        'readable': False,
        'in_use': False,
        'error': None,
        'fix': None
    }

    if not result['exists']:
        result['error'] = 'Camera not connected'
        result['fix'] = f'Check USB connection for {name} camera'
        return result

    result['readable'] = os.access(path, os.R_OK)
    if not result['readable']:
        result['error'] = 'Camera not readable (permission denied)'
        result['fix'] = f'Run: sudo chmod 666 {path}'
        return result

    # Check if in use
    try:
        fuser = subprocess.run(['fuser', path], capture_output=True, text=True, timeout=5)
        if fuser.stdout.strip():
            result['in_use'] = True
            result['error'] = f'Camera in use by PID {fuser.stdout.strip()}'
            result['fix'] = f'Run: sudo kill -9 {fuser.stdout.strip()}'
    except:
        pass  # fuser check is optional

    return result


def kill_stale_ffmpeg():
    """Kill any orphaned ffmpeg processes from previous runs."""
    try:
        result = subprocess.run(['pgrep', '-f', 'ffmpeg.*v4l2'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            log.warn('cleanup', f'Killing {len(pids)} stale ffmpeg processes', pids=','.join(pids))
            subprocess.run(['pkill', '-9', '-f', 'ffmpeg.*v4l2'], timeout=5)
            time.sleep(1)  # Let cameras release
    except Exception as e:
        log.warn('cleanup', 'Failed to check for stale ffmpeg', error=str(e))


def cleanup_temp_files(goat_id: str):
    """Remove any temp files for this goat_id."""
    try:
        for f in os.listdir('/tmp'):
            if f.startswith(f'{goat_id}_') and (f.endswith('.jpg') or f.endswith('.json')):
                os.remove(f'/tmp/{f}')
                log.debug('cleanup', f'Removed temp file', file=f)
    except Exception as e:
        log.warn('cleanup', 'Temp file cleanup failed', error=str(e))


def validate_image(filepath: str) -> tuple[bool, str]:
    """Check if image file is valid."""
    if not os.path.exists(filepath):
        return False, 'File does not exist'

    size = os.path.getsize(filepath)
    if size < MIN_IMAGE_BYTES:
        return False, f'File too small ({size} bytes)'

    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_type,width,height',
            '-of', 'json',
            filepath
        ], capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            return False, f'ffprobe error: {result.stderr[:100]}'

        probe = json.loads(result.stdout)
        if not probe.get('streams'):
            return False, 'No image stream found'

        stream = probe['streams'][0]
        width = stream.get('width', 0)
        height = stream.get('height', 0)
        if width < IMAGE_WIDTH or height < IMAGE_HEIGHT:
            return False, f'Resolution too low ({width}x{height}, expected {IMAGE_WIDTH}x{IMAGE_HEIGHT})'

        return True, f'OK ({width}x{height}, {size} bytes)'

    except subprocess.TimeoutExpired:
        return False, 'ffprobe timeout'
    except Exception as e:
        return False, f'Validation error: {str(e)}'


# ============================================================
# CAPTURE LOGIC
# ============================================================

def capture_single_camera(name: str, path: str, goat_id: str) -> dict:
    """
    Capture still images from a single camera. Returns result dict.
    Uses -codec:v copy to pass through raw MJPEG frames without
    decoding/re-encoding, keeping RAM usage minimal (~50-100MB vs ~3GB).
    """
    output_pattern = f'/tmp/{goat_id}_{name}_%02d.jpg'
    result = {
        'name': name,
        'success': False,
        'filepaths': [],
        'total_size_bytes': 0,
        'image_count': 0,
        'error': None,
        'fix': None
    }

    cmd = [
        'ffmpeg', '-y',
        '-f', 'v4l2',
        '-input_format', 'mjpeg',
        '-framerate', str(CAPTURE_FPS),
        '-video_size', f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
        '-i', path,
        '-frames:v', str(NUM_IMAGES),
        '-codec:v', 'copy',
        output_pattern
    ]

    log.info(f'camera:{name}', 'Starting capture',
            path=path, num_images=NUM_IMAGES, fps=CAPTURE_FPS,
            resolution=f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}')

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SEC
        )

        if proc.returncode != 0:
            stderr = proc.stderr
            if 'No such file or directory' in stderr:
                result['error'] = 'Camera disconnected during capture'
                result['fix'] = f'Check USB cable for {name} camera'
            elif 'Device or resource busy' in stderr:
                result['error'] = 'Camera is busy (another process using it)'
                result['fix'] = 'Run: sudo pkill -9 ffmpeg'
            elif 'Invalid argument' in stderr:
                result['error'] = f'Camera does not support {IMAGE_WIDTH}x{IMAGE_HEIGHT} MJPEG'
                result['fix'] = 'Check camera resolution support with: v4l2-ctl --list-formats-ext'
            else:
                result['error'] = f'ffmpeg error (code {proc.returncode})'
                result['fix'] = f'Check logs, stderr: {stderr[:200]}'

            log.error(f'camera:{name}', 'Capture failed',
                     error=result['error'], fix=result['fix'], returncode=proc.returncode)
            return result

        # Collect and validate output files
        valid_files = []
        total_size = 0
        validation_failures = []

        for i in range(1, NUM_IMAGES + 1):
            filepath = f'/tmp/{goat_id}_{name}_{i:02d}.jpg'
            if not os.path.exists(filepath):
                validation_failures.append(f'Image {i:02d} missing')
                continue

            valid, msg = validate_image(filepath)
            if not valid:
                validation_failures.append(f'Image {i:02d}: {msg}')
                os.remove(filepath)
                continue

            size = os.path.getsize(filepath)
            total_size += size
            valid_files.append(filepath)

        if not valid_files:
            result['error'] = f'No valid images captured. Failures: {"; ".join(validation_failures[:5])}'
            result['fix'] = 'Camera may be malfunctioning, try unplugging and replugging'
            log.error(f'camera:{name}', 'All images failed validation',
                     failures=validation_failures[:5])
            return result

        if len(valid_files) < NUM_IMAGES:
            log.warn(f'camera:{name}', 'Some images failed validation',
                    valid=len(valid_files), expected=NUM_IMAGES,
                    failures=validation_failures[:5])

        result['success'] = True
        result['filepaths'] = valid_files
        result['total_size_bytes'] = total_size
        result['image_count'] = len(valid_files)

        log.info(f'camera:{name}', 'Capture complete',
                image_count=len(valid_files), total_size_bytes=total_size)

    except subprocess.TimeoutExpired:
        result['error'] = f'Capture timed out after {FFMPEG_TIMEOUT_SEC}s'
        result['fix'] = 'Camera may be frozen. Unplug USB hub, wait 5s, replug.'
        log.error(f'camera:{name}', 'Capture timeout',
                 timeout_sec=FFMPEG_TIMEOUT_SEC, fix=result['fix'])

        subprocess.run(['pkill', '-9', '-f', f'ffmpeg.*{path}'], timeout=5)

    except Exception as e:
        result['error'] = f'Unexpected error: {str(e)}'
        log.exception(f'camera:{name}', 'Capture exception', error=str(e))

    return result


def do_capture(goat_id: str, goat_data: dict, is_test: bool):
    """
    Main capture workflow. Runs in background thread.
    REAL: all cameras must succeed and upload.
    TEST: capture what is available, do S3 capability check only (no upload).
    """
    start_time = time.time()
    results = {}

    try:
        set_state(progress='capturing')

        # Capture all cameras in parallel
        threads = {}
        for name, path in CAMERAS.items():
            check = check_camera(name, path)
            if check['error']:
                results[name] = {
                    'name': name,
                    'success': False,
                    'filepaths': [],
                    'image_count': 0,
                    'error': check['error'],
                    'fix': check['fix']
                }
                continue

            t = threading.Thread(
                target=lambda n=name, p=path: results.update({n: capture_single_camera(n, p, goat_id)})
            )
            threads[name] = t
            t.start()

        # Wait for all captures
        log.info('capture', f'Waiting for {len(threads)} cameras', cameras=','.join(threads.keys()))
        for name, t in threads.items():
            t.join(timeout=FFMPEG_TIMEOUT_SEC + 5)
            if t.is_alive():
                log.error(f'camera:{name}', 'Thread did not complete in time')
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
                if r.get("error") in ("Camera not connected", "Camera not readable (permission denied)"):
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
                    'capture_fps': CAPTURE_FPS,
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
    cameras = {name: os.path.exists(path) for name, path in CAMERAS.items()}
    disk_ok, disk_mb = check_disk_space()

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
        'capture_config': {
            'num_images': NUM_IMAGES,
            'capture_fps': CAPTURE_FPS,
            'resolution': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
            'format': 'mjpeg'
        },
        'state': get_state()
    }

    # Check each camera
    for name, path in CAMERAS.items():
        diag['cameras'][name] = check_camera(name, path)

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
            s3_ok=diag['s3']['ok'])

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
            goat_id=goat_id, num_images=NUM_IMAGES, fps=CAPTURE_FPS,
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

    # Check cameras before starting
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

    # Kill any stale ffmpeg processes
    kill_stale_ffmpeg()

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
        'capture_fps': CAPTURE_FPS,
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
    """Emergency cancel - kill all ffmpeg and reset state."""
    log.warn('cancel', 'Emergency cancel requested')

    kill_stale_ffmpeg()

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
            capture_fps=CAPTURE_FPS,
            resolution=f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
            min_disk_mb=MIN_DISK_MB,
            ffmpeg_timeout=FFMPEG_TIMEOUT_SEC)

    # Cleanup any stale state from previous run
    kill_stale_ffmpeg()

    # Check cameras
    all_cameras_ok = True
    for name, path in CAMERAS.items():
        check = check_camera(name, path)
        if check['error']:
            log.error(f'startup:camera:{name}', 'Camera not ready',
                     error=check['error'], fix=check['fix'])
            all_cameras_ok = False
        else:
            log.info(f'startup:camera:{name}', 'Camera ready', path=path)

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
    if all_cameras_ok and disk_ok and s3_ok:
        log.info('startup', 'All systems ready')
    else:
        log.warn('startup', 'Starting with errors - some features may not work')

    log.info('startup', 'Server listening', host='0.0.0.0', port=5001)
    log.info('startup', '=' * 50)

    app.run(host='0.0.0.0', port=5001)
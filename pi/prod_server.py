"""
Pi Production Server
Captures images from 3 cameras, sends to EC2 API for grading.
Also provides connectivity testing and health checks.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import threading
import requests
import time
import os
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================

EC2_IP = os.environ.get('EC2_IP', '3.16.96.182')
EC2_API = f'http://{EC2_IP}:8000'

# Camera paths (udev symlinks)
CAMERAS = {
    'side': '/dev/camera_side',
    'top': '/dev/camera_top',
    'front': '/dev/camera_front'
}

# Capture settings
CAMERA_NATIVE_FPS = 10      # Camera's native FPS at capture resolution
IMAGE_WIDTH = 4656
IMAGE_HEIGHT = 3496
WARMUP_FRAMES = 10           # Skip first N frames for autofocus/white balance (1s at 10fps)
CAPTURE_FRAME = 15           # Which frame to keep (after warmup settles)

# Timeouts
FFMPEG_TIMEOUT_SEC = 15      # Single capture timeout
EC2_TIMEOUT_SEC = 30         # EC2 API request timeout
PING_TIMEOUT_SEC = 15
REQUEST_TIMEOUT_SEC = 10

# Validation
MIN_IMAGE_BYTES = 50000      # 50KB minimum valid image
MAX_SERIAL_ID_LEN = 50

# ============================================================
# LOGGING
# ============================================================

import sys
sys.path.insert(0, '/home/pi/goat-capture/pi')
from logger.pi_cloudwatch import Logger

log = Logger('pi/prod')

# ============================================================
# STATE
# ============================================================

_state_lock = threading.Lock()
capture_state = {
    'active': False,
    'serial_id': None,
    'started_at': None,
    'progress': None,
    'last_error': None,
    'last_result': None
}

def set_state(**kwargs):
    with _state_lock:
        capture_state.update(kwargs)

def get_state():
    with _state_lock:
        return capture_state.copy()

def reset_state():
    with _state_lock:
        capture_state.update({
            'active': False,
            'serial_id': None,
            'started_at': None,
            'progress': None
        })

# ============================================================
# CAMERA HELPERS
# ============================================================

def check_camera(name: str, path: str) -> dict:
    """Check a single camera's status with detailed diagnostics."""
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
        # Check if the symlink target exists
        if os.path.islink(path):
            target = os.readlink(path)
            result['error'] = f'Symlink exists but target {target} missing'
            result['fix'] = f'Camera may have disconnected. Check USB and run: ls -la {path}'
            log.warn(f'camera:{name}', 'Dangling symlink', path=path, target=target)
        else:
            result['error'] = 'Camera not connected'
            result['fix'] = f'Check USB connection for {name} camera. Verify udev rule exists.'
        return result

    result['readable'] = os.access(path, os.R_OK)
    if not result['readable']:
        result['error'] = 'Camera not readable (permission denied)'
        result['fix'] = f'Run: sudo chmod 666 {path}'
        return result

    # Check if in use by another process
    try:
        fuser = subprocess.run(['fuser', path], capture_output=True, text=True, timeout=5)
        if fuser.stdout.strip():
            pids = fuser.stdout.strip()
            result['in_use'] = True
            result['error'] = f'Camera in use by PID {pids}'
            result['fix'] = f'Run: sudo kill -9 {pids}'
            log.warn(f'camera:{name}', 'Camera busy', pids=pids)
    except Exception:
        pass  # fuser check is optional

    # Query camera capabilities for logging
    try:
        v4l2 = subprocess.run(
            ['v4l2-ctl', '-d', path, '--list-formats-ext'],
            capture_output=True, text=True, timeout=5
        )
        supports_target = f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}' in v4l2.stdout
        if not supports_target:
            result['error'] = f'Camera may not support {IMAGE_WIDTH}x{IMAGE_HEIGHT} MJPEG'
            result['fix'] = f'Check: v4l2-ctl -d {path} --list-formats-ext'
            log.warn(f'camera:{name}', 'Resolution not confirmed',
                    target=f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}')
    except Exception:
        pass  # v4l2 check is optional

    return result


def kill_stale_ffmpeg():
    """Kill any orphaned ffmpeg processes from previous runs."""
    try:
        result = subprocess.run(['pgrep', '-f', 'ffmpeg.*v4l2'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            log.warn('cleanup', f'Killing {len(pids)} stale ffmpeg processes', pids=','.join(pids))
            subprocess.run(['pkill', '-9', '-f', 'ffmpeg.*v4l2'], timeout=5)
            time.sleep(1)
    except Exception as e:
        log.warn('cleanup', 'Failed to check for stale ffmpeg', error=str(e))


def capture_single_image(name: str, path: str, serial_id: str) -> dict:
    """
    Capture a single image from one camera with autofocus warmup.

    Reads at the camera's native FPS (10fps), skips WARMUP_FRAMES for
    autofocus/white balance, then keeps the next good frame.
    Uses -threads 1 to limit memory (~500MB per process).

    Returns dict with 'success', 'filepath', 'error', 'fix', and diagnostics.
    """
    filepath = f'/tmp/{serial_id}_{name}.jpg'
    result = {
        'name': name,
        'success': False,
        'filepath': None,
        'file_size_bytes': 0,
        'error': None,
        'fix': None,
        'capture_time_sec': None
    }

    # Use select filter to skip warmup frames then grab 1 frame
    # gt(n,WARMUP_FRAMES-1) skips first N frames
    # not(mod(n,1)) keeps every frame after that (we only need 1)
    frame_skip = WARMUP_FRAMES

    cmd = [
        'ffmpeg', '-y',
        '-f', 'v4l2',
        '-input_format', 'mjpeg',
        '-framerate', str(CAMERA_NATIVE_FPS),
        '-video_size', f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
        '-i', path,
        '-vf', f'select=gt(n\\,{frame_skip - 1})',
        '-fps_mode', 'vfr',
        '-frames:v', '1',
        '-threads', '1',
        '-qmin', '1',
        '-q:v', '1',
        filepath
    ]

    log.info(f'camera:{name}', 'Starting capture',
            path=path, warmup_frames=frame_skip,
            resolution=f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
            native_fps=CAMERA_NATIVE_FPS)

    capture_start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SEC
        )

        capture_time = round(time.time() - capture_start, 2)
        result['capture_time_sec'] = capture_time

        if proc.returncode != 0:
            stderr = proc.stderr

            # Parse specific errors
            if 'No such file or directory' in stderr:
                result['error'] = 'Camera disconnected during capture'
                result['fix'] = f'Check USB cable for {name} camera'
            elif 'Device or resource busy' in stderr:
                result['error'] = 'Camera is busy (another process using it)'
                result['fix'] = 'Run: sudo pkill -9 ffmpeg'
            elif 'Invalid argument' in stderr:
                result['error'] = f'Camera does not support {IMAGE_WIDTH}x{IMAGE_HEIGHT} MJPEG'
                result['fix'] = f'Check: v4l2-ctl -d {path} --list-formats-ext'
            elif proc.returncode == -9:
                result['error'] = 'ffmpeg killed (likely OOM)'
                result['fix'] = 'Check memory with: free -h'
            else:
                result['error'] = f'ffmpeg error (code {proc.returncode})'
                result['fix'] = f'stderr: {stderr[:300]}'

            log.error(f'camera:{name}', 'Capture failed',
                     error=result['error'],
                     returncode=proc.returncode,
                     capture_time_sec=capture_time,
                     stderr_snippet=stderr[:200] if stderr else None)
            return result

        # Validate output
        if not os.path.exists(filepath):
            result['error'] = 'No output file produced'
            result['fix'] = 'Camera may not be streaming. Check connection.'
            log.error(f'camera:{name}', 'No output file', filepath=filepath)
            return result

        file_size = os.path.getsize(filepath)
        if file_size < MIN_IMAGE_BYTES:
            result['error'] = f'Image too small ({file_size} bytes, min {MIN_IMAGE_BYTES})'
            result['fix'] = 'Camera may be producing blank frames. Check lens cap and lighting.'
            log.error(f'camera:{name}', 'Image too small',
                     size_bytes=file_size, min_bytes=MIN_IMAGE_BYTES)
            os.remove(filepath)
            return result

        result['success'] = True
        result['filepath'] = filepath
        result['file_size_bytes'] = file_size

        log.info(f'camera:{name}', 'Capture complete',
                size_bytes=file_size,
                capture_time_sec=capture_time)

    except subprocess.TimeoutExpired:
        result['error'] = f'Capture timed out after {FFMPEG_TIMEOUT_SEC}s'
        result['fix'] = 'Camera may be frozen. Unplug USB, wait 5s, replug.'
        result['capture_time_sec'] = FFMPEG_TIMEOUT_SEC
        log.error(f'camera:{name}', 'Capture timeout',
                 timeout_sec=FFMPEG_TIMEOUT_SEC)
        subprocess.run(['pkill', '-9', '-f', f'ffmpeg.*{path}'], timeout=5)

    except Exception as e:
        result['error'] = f'Unexpected error: {str(e)}'
        log.exception(f'camera:{name}', 'Capture exception', error=str(e))

    return result


# ============================================================
# GRADING WORKFLOW
# ============================================================

def do_grade(serial_id: str, live_weight: float):
    """
    Full capture → grade workflow. Runs in background thread.
    Captures 1 image per camera (2 parallel + 1), sends all 3 to EC2 API.
    """
    start_time = time.time()
    results = {}

    try:
        # Phase 1: Capture
        set_state(progress='capturing')

        # Pre-check all cameras
        available_cameras = {}
        for name, path in CAMERAS.items():
            check = check_camera(name, path)
            if check['error']:
                results[name] = {
                    'name': name,
                    'success': False,
                    'filepath': None,
                    'error': check['error'],
                    'fix': check['fix']
                }
                log.error(f'camera:{name}', 'Pre-check failed',
                         error=check['error'], fix=check['fix'])
            else:
                available_cameras[name] = path

        # All 3 cameras required for grading
        if len(available_cameras) < 3:
            missing = [n for n in CAMERAS if n not in available_cameras]
            error_msg = f'Missing cameras: {", ".join(missing)}'
            set_state(
                last_error=error_msg,
                last_result={
                    'serial_id': serial_id,
                    'success': False,
                    'error': error_msg,
                    'camera_results': results,
                    'duration_sec': round(time.time() - start_time, 2)
                }
            )
            log.error('grade', 'Not all cameras available',
                     serial_id=serial_id, missing=','.join(missing))
            return

        # Kill stale ffmpeg
        kill_stale_ffmpeg()

        # Capture in batches of 2 to stay within 4GB RAM
        # Each ffmpeg decode/encode of 4656x3496 uses ~500MB
        MAX_PARALLEL = 2
        camera_list = list(available_cameras.items())
        batches = [camera_list[i:i + MAX_PARALLEL] for i in range(0, len(camera_list), MAX_PARALLEL)]

        for batch_num, batch in enumerate(batches, 1):
            batch_names = [n for n, _ in batch]
            log.info('capture', f'Batch {batch_num}/{len(batches)}',
                    serial_id=serial_id, cameras=','.join(batch_names))

            threads = {}
            for name, path in batch:
                t = threading.Thread(
                    target=lambda n=name, p=path: results.update(
                        {n: capture_single_image(n, p, serial_id)}
                    )
                )
                threads[name] = t
                t.start()

            for name, t in threads.items():
                t.join(timeout=FFMPEG_TIMEOUT_SEC + 5)
                if t.is_alive():
                    log.error(f'camera:{name}', 'Capture thread hung',
                             serial_id=serial_id)
                    results[name] = {
                        'name': name,
                        'success': False,
                        'filepath': None,
                        'error': 'Capture thread hung',
                        'fix': 'Restart service: sudo systemctl restart goat-prod'
                    }

        # Check all captures succeeded
        failed = [n for n, r in results.items() if not r.get('success')]
        if failed:
            error_msg = '; '.join(f"{n}: {results[n].get('error', 'unknown')}" for n in failed)
            # Clean up any captured files
            for r in results.values():
                fp = r.get('filepath')
                if fp and os.path.exists(fp):
                    os.remove(fp)

            set_state(
                last_error=error_msg,
                last_result={
                    'serial_id': serial_id,
                    'success': False,
                    'error': error_msg,
                    'camera_results': {n: {k: v for k, v in r.items() if k != 'filepath'}
                                       for n, r in results.items()},
                    'duration_sec': round(time.time() - start_time, 2)
                }
            )
            log.error('grade', 'Capture failed',
                     serial_id=serial_id, failed=','.join(failed), error=error_msg)
            return

        capture_time = round(time.time() - start_time, 2)
        total_size = sum(r.get('file_size_bytes', 0) for r in results.values())
        log.info('capture', 'All cameras captured',
                serial_id=serial_id,
                total_size_bytes=total_size,
                capture_time_sec=capture_time,
                per_camera={n: r.get('capture_time_sec') for n, r in results.items()})

        # Phase 2: Send to EC2
        set_state(progress='grading')
        log.info('grade', 'Sending to EC2',
                serial_id=serial_id, ec2_api=EC2_API,
                total_size_bytes=total_size)

        try:
            files = {
                'side_image': (f'{serial_id}_side.jpg', open(results['side']['filepath'], 'rb'), 'image/jpeg'),
                'top_image': (f'{serial_id}_top.jpg', open(results['top']['filepath'], 'rb'), 'image/jpeg'),
                'front_image': (f'{serial_id}_front.jpg', open(results['front']['filepath'], 'rb'), 'image/jpeg'),
            }
            data = {
                'serial_id': serial_id,
                'live_weight': str(live_weight),
            }

            ec2_start = time.time()
            response = requests.post(
                f'{EC2_API}/analyze',
                files=files,
                data=data,
                timeout=EC2_TIMEOUT_SEC
            )
            ec2_time = round(time.time() - ec2_start, 2)

            # Close file handles
            for f in files.values():
                f[1].close()

            if response.status_code == 200:
                grade_result = response.json()
                total_time = round(time.time() - start_time, 2)

                set_state(
                    last_error=None,
                    last_result={
                        'serial_id': serial_id,
                        'success': True,
                        'grade': grade_result.get('grade'),
                        'measurements': grade_result.get('measurements'),
                        'confidence_scores': grade_result.get('confidence_scores'),
                        'all_views_successful': grade_result.get('all_views_successful'),
                        'warnings': grade_result.get('warnings'),
                        'timing': {
                            'capture_sec': capture_time,
                            'ec2_sec': ec2_time,
                            'total_sec': total_time
                        },
                        'duration_sec': total_time
                    }
                )
                log.info('grade', 'Grading complete',
                        serial_id=serial_id,
                        grade=grade_result.get('grade'),
                        all_views_ok=grade_result.get('all_views_successful'),
                        capture_sec=capture_time,
                        ec2_sec=ec2_time,
                        total_sec=total_time)
            else:
                error_body = response.text[:500]
                set_state(
                    last_error=f'EC2 returned {response.status_code}',
                    last_result={
                        'serial_id': serial_id,
                        'success': False,
                        'error': f'EC2 API error ({response.status_code})',
                        'ec2_response': error_body,
                        'duration_sec': round(time.time() - start_time, 2)
                    }
                )
                log.error('grade', 'EC2 API error',
                         serial_id=serial_id,
                         status_code=response.status_code,
                         response=error_body)

        except requests.exceptions.ConnectTimeout:
            set_state(
                last_error='EC2 connection timeout',
                last_result={
                    'serial_id': serial_id,
                    'success': False,
                    'error': 'EC2 connection timeout',
                    'fix': 'Check EC2 is running and port 8000 is open',
                    'duration_sec': round(time.time() - start_time, 2)
                }
            )
            log.error('grade', 'EC2 connection timeout',
                     serial_id=serial_id, ec2_api=EC2_API)

        except requests.exceptions.ConnectionError:
            set_state(
                last_error='EC2 connection refused',
                last_result={
                    'serial_id': serial_id,
                    'success': False,
                    'error': 'EC2 connection refused',
                    'fix': 'Check EC2 API container is running: docker ps',
                    'duration_sec': round(time.time() - start_time, 2)
                }
            )
            log.error('grade', 'EC2 connection refused',
                     serial_id=serial_id, ec2_api=EC2_API)

        except Exception as e:
            set_state(
                last_error=str(e),
                last_result={
                    'serial_id': serial_id,
                    'success': False,
                    'error': f'EC2 request failed: {str(e)}',
                    'duration_sec': round(time.time() - start_time, 2)
                }
            )
            log.exception('grade', 'EC2 request failed',
                         serial_id=serial_id, error=str(e))

        finally:
            # Clean up captured images
            for r in results.values():
                fp = r.get('filepath')
                if fp and os.path.exists(fp):
                    os.remove(fp)

    except Exception as e:
        log.exception('grade', 'Unexpected error in grade workflow',
                     serial_id=serial_id, error=str(e))
        set_state(
            last_error=str(e),
            last_result={
                'serial_id': serial_id,
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
        )

    finally:
        reset_state()


# ============================================================
# CONNECTIVITY HELPERS
# ============================================================

def ping_host(host: str, count: int = 3) -> dict:
    """Ping a host and return result with latency."""
    try:
        result = subprocess.run(
            ['ping', '-c', str(count), '-W', '5', host],
            capture_output=True, text=True, timeout=PING_TIMEOUT_SEC
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'avg' in line:
                    latency = line.split('/')[4]
                    return {'status': 'ok', 'latency_ms': latency}
            return {'status': 'ok', 'latency_ms': 'unknown'}
        else:
            return {'status': 'error', 'error': 'Ping failed',
                    'stderr': result.stderr[:200] if result.stderr else None}
    except subprocess.TimeoutExpired:
        return {'status': 'error', 'error': 'Timeout'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def check_ec2_api() -> dict:
    """Check EC2 API health endpoint."""
    try:
        response = requests.get(f'{EC2_API}/health', timeout=REQUEST_TIMEOUT_SEC)
        if response.status_code == 200:
            return {'status': 'ok', 'response': response.json()}
        else:
            return {'status': 'error', 'status_code': response.status_code,
                    'error': response.text[:100]}
    except requests.exceptions.ConnectTimeout:
        return {'status': 'error', 'error': 'Connection timeout'}
    except requests.exceptions.ConnectionError:
        return {'status': 'error', 'error': 'Connection refused'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def get_system_info() -> dict:
    """Get Pi system diagnostics."""
    info = {}

    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            info['cpu_temp_c'] = round(int(f.read().strip()) / 1000, 1)
    except:
        info['cpu_temp_c'] = None

    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    info['mem_total_mb'] = int(line.split()[1]) // 1024
                if 'MemAvailable' in line:
                    info['mem_available_mb'] = int(line.split()[1]) // 1024
    except:
        info['mem_available_mb'] = None

    try:
        info['load_avg'] = round(os.getloadavg()[0], 2)
    except:
        info['load_avg'] = None

    try:
        with open('/proc/uptime', 'r') as f:
            uptime_sec = float(f.read().split()[0])
            info['uptime_hours'] = round(uptime_sec / 3600, 1)
    except:
        info['uptime_hours'] = None

    # Disk space
    try:
        import shutil
        _, _, free = shutil.disk_usage('/tmp')
        info['disk_free_mb'] = free // (1024 * 1024)
    except:
        info['disk_free_mb'] = None

    return info


# ============================================================
# ROUTES
# ============================================================

@app.route('/health')
def health():
    """Quick health check with camera status."""
    cameras = {}
    for name, path in CAMERAS.items():
        cameras[name] = os.path.exists(path)

    sys_info = get_system_info()

    return jsonify({
        'status': 'ok' if all(cameras.values()) else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'cameras': cameras,
        'ec2_target': EC2_API,
        'mem_available_mb': sys_info.get('mem_available_mb'),
        'cpu_temp_c': sys_info.get('cpu_temp_c'),
        'capture_active': get_state()['active']
    })


@app.route('/diagnostics')
def diagnostics():
    """Detailed system and camera diagnostics."""
    log.info('diag', 'Running diagnostics')

    diag = {
        'timestamp': datetime.now().isoformat(),
        'system': get_system_info(),
        'cameras': {},
        'ec2': {},
        'capture_config': {
            'image_resolution': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
            'native_fps': CAMERA_NATIVE_FPS,
            'warmup_frames': WARMUP_FRAMES,
            'capture_frame': CAPTURE_FRAME,
            'max_parallel_captures': 2,
            'ffmpeg_timeout_sec': FFMPEG_TIMEOUT_SEC,
            'ec2_timeout_sec': EC2_TIMEOUT_SEC
        },
        'state': get_state()
    }

    # Check each camera in detail
    for name, path in CAMERAS.items():
        diag['cameras'][name] = check_camera(name, path)

    # Check EC2
    ec2_health = check_ec2_api()
    diag['ec2'] = {
        'api': EC2_API,
        'health': ec2_health
    }

    cameras_ok = sum(1 for c in diag['cameras'].values() if not c.get('error'))
    log.info('diag', 'Complete',
            cameras_ok=cameras_ok,
            cameras_total=len(CAMERAS),
            ec2_ok=ec2_health.get('status') == 'ok',
            mem_mb=diag['system'].get('mem_available_mb'),
            cpu_temp=diag['system'].get('cpu_temp_c'))

    return jsonify(diag)


@app.route('/grade', methods=['POST'])
def grade():
    """
    Capture images and send to EC2 for grading.

    POST body:
    {
        "serial_id": "GOAT001",
        "live_weight": 85.5
    }
    """
    state = get_state()
    if state['active']:
        log.warn('grade', 'Grade already in progress',
                current_serial=state['serial_id'])
        return jsonify({
            'status': 'error',
            'error_code': 'GRADE_IN_PROGRESS',
            'message': f'Grade in progress for {state["serial_id"]}. Please wait.',
            'current': {
                'serial_id': state['serial_id'],
                'started_at': state['started_at'],
                'progress': state['progress']
            }
        }), 409

    # Parse input
    try:
        data = request.get_json(force=True, silent=True) or {}
    except:
        data = {}

    serial_id = data.get('serial_id', '').strip()
    live_weight = data.get('live_weight')

    # Validate serial_id
    if not serial_id:
        return jsonify({
            'status': 'error',
            'error_code': 'MISSING_SERIAL_ID',
            'message': 'serial_id is required'
        }), 400

    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', serial_id)
    if not sanitized or len(sanitized) > MAX_SERIAL_ID_LEN:
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_SERIAL_ID',
            'message': 'serial_id must be alphanumeric, max 50 chars'
        }), 400

    # Validate weight
    if live_weight is None:
        return jsonify({
            'status': 'error',
            'error_code': 'MISSING_WEIGHT',
            'message': 'live_weight is required'
        }), 400

    try:
        live_weight = float(live_weight)
        if live_weight <= 0 or live_weight > 500:
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_WEIGHT',
            'message': 'live_weight must be a number between 0 and 500 lbs'
        }), 400

    # Pre-check cameras
    camera_checks = {}
    for name, path in CAMERAS.items():
        camera_checks[name] = check_camera(name, path)

    failed = {n: c for n, c in camera_checks.items() if c.get('error')}
    if failed:
        errors = [f"{n}: {c['error']}" for n, c in failed.items()]
        fixes = [f"{n}: {c['fix']}" for n, c in failed.items() if c.get('fix')]
        log.error('grade', 'Cameras not ready',
                 serial_id=sanitized, missing=','.join(failed.keys()))
        return jsonify({
            'status': 'error',
            'error_code': 'CAMERAS_NOT_READY',
            'message': f'{len(failed)} camera(s) not ready',
            'errors': errors,
            'fixes': fixes,
            'cameras': camera_checks
        }), 503

    # Start grading
    set_state(
        active=True,
        serial_id=sanitized,
        started_at=datetime.now().isoformat(),
        progress='starting',
        last_error=None
    )

    thread = threading.Thread(target=do_grade, args=(sanitized, live_weight))
    thread.start()

    log.info('grade', 'Grade started',
            serial_id=sanitized, live_weight=live_weight)

    return jsonify({
        'status': 'grade_started',
        'serial_id': sanitized,
        'live_weight': live_weight,
        'ec2_target': EC2_API
    })


@app.route('/status')
def status():
    """Get current capture/grade state."""
    return jsonify(get_state())


@app.route('/grade/test', methods=['POST'])
def grade_test():
    """
    Test grading with local image files instead of cameras.
    Runs the same EC2 submission pipeline as /grade but skips capture.

    Usage:
        curl -X POST http://localhost:5000/grade/test \
          -F "serial_id=GOAT123" \
          -F "live_weight=85.5" \
          -F "side_image=@/tmp/side_17.jpg" \
          -F "top_image=@/tmp/top_17.jpg" \
          -F "front_image=@/tmp/front_17.jpg"
    """
    serial_id = request.form.get('serial_id', '').strip()
    live_weight = request.form.get('live_weight')

    # Validate
    if not serial_id:
        return jsonify({'status': 'error', 'error_code': 'MISSING_SERIAL_ID',
                       'message': 'serial_id is required'}), 400

    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', serial_id)
    if not sanitized:
        return jsonify({'status': 'error', 'error_code': 'INVALID_SERIAL_ID',
                       'message': 'serial_id must be alphanumeric'}), 400

    try:
        live_weight = float(live_weight)
        if live_weight <= 0 or live_weight > 500:
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'error_code': 'INVALID_WEIGHT',
                       'message': 'live_weight must be a number between 0 and 500'}), 400

    # Check all 3 images provided
    for view in ['side_image', 'top_image', 'front_image']:
        if view not in request.files:
            return jsonify({'status': 'error', 'error_code': 'MISSING_IMAGE',
                           'message': f'{view} is required'}), 400
        f = request.files[view]
        if not f.filename:
            return jsonify({'status': 'error', 'error_code': 'EMPTY_IMAGE',
                           'message': f'{view} has no filename'}), 400

    log.info('grade:test', 'Test grade request',
            serial_id=sanitized, live_weight=live_weight,
            side_size=request.files['side_image'].content_length,
            top_size=request.files['top_image'].content_length,
            front_size=request.files['front_image'].content_length)

    # Forward to EC2
    start_time = time.time()
    try:
        files = {
            'side_image': (
                request.files['side_image'].filename,
                request.files['side_image'].stream,
                request.files['side_image'].content_type or 'image/jpeg'
            ),
            'top_image': (
                request.files['top_image'].filename,
                request.files['top_image'].stream,
                request.files['top_image'].content_type or 'image/jpeg'
            ),
            'front_image': (
                request.files['front_image'].filename,
                request.files['front_image'].stream,
                request.files['front_image'].content_type or 'image/jpeg'
            ),
        }
        data = {
            'serial_id': sanitized,
            'live_weight': str(live_weight),
        }

        ec2_start = time.time()
        response = requests.post(
            f'{EC2_API}/analyze',
            files=files,
            data=data,
            timeout=EC2_TIMEOUT_SEC
        )
        ec2_time = round(time.time() - ec2_start, 2)
        total_time = round(time.time() - start_time, 2)

        if response.status_code == 200:
            grade_result = response.json()
            log.info('grade:test', 'Test grade complete',
                    serial_id=sanitized,
                    grade=grade_result.get('grade'),
                    ec2_sec=ec2_time, total_sec=total_time)
            return jsonify({
                'status': 'success',
                'test': True,
                'serial_id': sanitized,
                'grade': grade_result.get('grade'),
                'measurements': grade_result.get('measurements'),
                'confidence_scores': grade_result.get('confidence_scores'),
                'all_views_successful': grade_result.get('all_views_successful'),
                'warnings': grade_result.get('warnings'),
                'timing': {
                    'ec2_sec': ec2_time,
                    'total_sec': total_time
                }
            })
        else:
            error_body = response.text[:500]
            log.error('grade:test', 'EC2 error',
                     serial_id=sanitized,
                     status_code=response.status_code,
                     response=error_body)
            return jsonify({
                'status': 'error',
                'error_code': 'EC2_ERROR',
                'message': f'EC2 returned {response.status_code}',
                'ec2_response': error_body
            }), 502

    except requests.exceptions.ConnectTimeout:
        log.error('grade:test', 'EC2 timeout', serial_id=sanitized)
        return jsonify({
            'status': 'error',
            'error_code': 'EC2_TIMEOUT',
            'message': 'EC2 connection timeout',
            'fix': 'Check EC2 is running and port 8000 is open'
        }), 504

    except requests.exceptions.ConnectionError:
        log.error('grade:test', 'EC2 connection refused', serial_id=sanitized)
        return jsonify({
            'status': 'error',
            'error_code': 'EC2_UNREACHABLE',
            'message': 'Cannot reach EC2 API',
            'fix': 'Check EC2 container is running: docker ps'
        }), 502

    except Exception as e:
        log.exception('grade:test', 'Unexpected error', serial_id=sanitized, error=str(e))
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': str(e)
        }), 500


@app.route('/cancel', methods=['POST'])
def cancel():
    """Emergency cancel - kill ffmpeg and reset."""
    log.warn('cancel', 'Cancel requested')
    kill_stale_ffmpeg()

    state = get_state()
    if state['serial_id']:
        # Clean up any temp files
        for name in CAMERAS:
            fp = f'/tmp/{state["serial_id"]}_{name}.jpg'
            if os.path.exists(fp):
                os.remove(fp)

    reset_state()
    set_state(last_error='Cancelled by user')

    return jsonify({'status': 'cancelled'})


@app.route('/test')
def test_connectivity():
    """Full connectivity test to internet and EC2."""
    log.info('test', 'Connectivity test started')
    start = time.time()

    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Test 1: Internet
    log.info('test:internet', 'Pinging 8.8.8.8')
    internet = ping_host('8.8.8.8')
    results['tests']['internet'] = internet
    if internet['status'] == 'ok':
        log.info('test:internet', 'OK', latency_ms=internet.get('latency_ms'))
    else:
        internet['fix'] = 'Check ethernet cable or WiFi connection'
        log.error('test:internet', 'Failed', error=internet.get('error'))

    # Test 2: EC2 Ping
    log.info('test:ec2_ping', f'Pinging EC2 {EC2_IP}')
    ec2_ping = ping_host(EC2_IP)
    results['tests']['ec2_ping'] = ec2_ping
    if ec2_ping['status'] == 'ok':
        log.info('test:ec2_ping', 'OK', latency_ms=ec2_ping.get('latency_ms'))
    else:
        ec2_ping['fix'] = 'Check EC2 is running and security group allows ICMP'
        log.error('test:ec2_ping', 'Failed', error=ec2_ping.get('error'))

    # Test 3: EC2 API
    log.info('test:ec2_api', f'Checking EC2 API {EC2_API}')
    ec2_api = check_ec2_api()
    results['tests']['ec2_api'] = ec2_api
    if ec2_api['status'] == 'ok':
        log.info('test:ec2_api', 'OK')
    else:
        if ec2_api.get('error') == 'Connection refused':
            ec2_api['fix'] = 'EC2 API container may be down. SSH and run: docker ps'
        elif ec2_api.get('error') == 'Connection timeout':
            ec2_api['fix'] = 'Check security group allows port 8000'
        else:
            ec2_api['fix'] = 'Check EC2 logs in CloudWatch'
        log.error('test:ec2_api', 'Failed', error=ec2_api.get('error'))

    # Test 4: Camera check
    for name, path in CAMERAS.items():
        check = check_camera(name, path)
        key = f'camera_{name}'
        if check.get('error'):
            results['tests'][key] = {
                'status': 'error',
                'error': check['error'],
                'fix': check.get('fix')
            }
        else:
            results['tests'][key] = {'status': 'ok', 'path': path}

    all_ok = all(t.get('status') == 'ok' for t in results['tests'].values())
    results['summary'] = 'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'
    results['duration_sec'] = round(time.time() - start, 2)

    log.info('test', 'Complete',
            summary=results['summary'],
            passed=sum(1 for t in results['tests'].values() if t.get('status') == 'ok'),
            total=len(results['tests']),
            duration_sec=results['duration_sec'])

    return jsonify(results)


# ============================================================
# STARTUP
# ============================================================

def run_startup_checks():
    """Run all startup checks and log results."""
    log.info('startup', '=' * 50)
    log.info('startup', 'PROD PI SERVER STARTING')
    log.info('startup', 'Configuration',
            ec2_ip=EC2_IP, ec2_api=EC2_API,
            resolution=f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
            native_fps=CAMERA_NATIVE_FPS,
            warmup_frames=WARMUP_FRAMES)

    # System info
    sys_info = get_system_info()
    log.info('startup:system', 'System info',
            cpu_temp=sys_info.get('cpu_temp_c'),
            mem_total_mb=sys_info.get('mem_total_mb'),
            mem_available_mb=sys_info.get('mem_available_mb'),
            load=sys_info.get('load_avg'),
            uptime_hrs=sys_info.get('uptime_hours'),
            disk_free_mb=sys_info.get('disk_free_mb'))

    if sys_info.get('cpu_temp_c') and sys_info['cpu_temp_c'] > 70:
        log.warn('startup:system', 'CPU temperature high',
                temp_c=sys_info['cpu_temp_c'],
                fix='Check Pi ventilation')

    if sys_info.get('mem_available_mb') and sys_info['mem_available_mb'] < 500:
        log.warn('startup:system', 'Low memory',
                available_mb=sys_info['mem_available_mb'],
                fix='Restart Pi or check for memory leaks. Need ~1GB for dual capture.')

    # Kill stale ffmpeg from previous run
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

    # Check internet
    log.info('startup:network', 'Testing internet connectivity')
    internet = ping_host('8.8.8.8', count=1)
    if internet['status'] == 'ok':
        log.info('startup:network', 'Internet OK', latency_ms=internet.get('latency_ms'))
    else:
        log.error('startup:network', 'No internet connection',
                 error=internet.get('error'),
                 fix='Check ethernet cable or WiFi')

    # Check EC2
    log.info('startup:ec2', 'Testing EC2 connectivity')
    ec2_ping = ping_host(EC2_IP, count=1)
    if ec2_ping['status'] == 'ok':
        log.info('startup:ec2', 'EC2 ping OK', latency_ms=ec2_ping.get('latency_ms'))
    else:
        log.warn('startup:ec2', 'EC2 ping failed',
                error=ec2_ping.get('error'), ip=EC2_IP)

    ec2_api = check_ec2_api()
    ec2_ok = ec2_api['status'] == 'ok'
    if ec2_ok:
        log.info('startup:ec2', 'EC2 API OK', url=EC2_API)
    else:
        log.warn('startup:ec2', 'EC2 API not reachable',
                error=ec2_api.get('error'), url=EC2_API,
                fix='May be OK if EC2 is still starting')

    # Summary
    if all_cameras_ok and ec2_ok:
        log.info('startup', 'All systems ready')
    else:
        issues = []
        if not all_cameras_ok:
            issues.append('cameras')
        if not ec2_ok:
            issues.append('ec2')
        log.warn('startup', 'Starting with issues',
                issues=','.join(issues))

    log.info('startup', 'Server listening', host='0.0.0.0', port=5000)
    log.info('startup', '=' * 50)


if __name__ == '__main__':
    run_startup_checks()
    app.run(host='0.0.0.0', port=5000)
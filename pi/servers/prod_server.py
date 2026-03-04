"""
Pi Production Server
Captures images from 3 cameras via camera proxy, sends to EC2 API for grading.
Also provides connectivity testing and health checks.

S3 ARCHIVAL NOTE:
The Pi does NOT write to S3. EC2 owns all S3 archival (raw images to
goat-captures, debug images + result.json to goat-processed). This keeps
the Pi thin — it only captures and relays.
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

from dotenv import load_dotenv
load_dotenv("/home/pi/goatdev/pi/.env")

# ============================================================
# CONFIGURATION
# ============================================================

EC2_IP = os.environ.get('EC2_IP')
EC2_API = f'http://{EC2_IP}:8000'
EC2_API_KEY = os.environ.get('EC2_API_KEY')

# Camera paths (for reference / diagnostics)
CAMERAS = {
    'side': '/dev/camera_side',
    'top': '/dev/camera_top',
    'front': '/dev/camera_front'
}

# Camera proxy
PROXY_URL = "http://127.0.0.1:8080"
PROXY_TIMEOUT_SEC = 10

# Capture settings
IMAGE_WIDTH = 4656
IMAGE_HEIGHT = 3496

# Timeouts
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
sys.path.insert(0, '/home/pi/goatdev/pi')
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
    """Check a single camera's status via proxy."""
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
        resp = requests.get(f'{PROXY_URL}/status', timeout=5)
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

    except requests.exceptions.ConnectionError:
        result['error'] = 'Camera proxy not running'
        result['fix'] = 'Run: sudo systemctl restart camera-proxy'

    except Exception as e:
        result['error'] = f'Proxy check failed: {e}'

    return result


def capture_single_image(name: str, path: str, serial_id: str) -> dict:
    """Capture a single full-res image from camera proxy."""
    result = {
        'name': name,
        'success': False,
        'filepath': None,
        'file_size_bytes': 0,
        'error': None,
        'fix': None,
        'capture_time_sec': None
    }

    filepath = f'/tmp/{serial_id}_{name}.jpg'
    capture_start = time.time()

    log.info(f'camera:{name}', 'Requesting frame from proxy')

    try:
        resp = requests.get(
            f'{PROXY_URL}/capture/{name}',
            timeout=PROXY_TIMEOUT_SEC
        )

        capture_time = round(time.time() - capture_start, 2)
        result['capture_time_sec'] = capture_time

        if resp.status_code == 503:
            error_data = {}
            try:
                if resp.headers.get('content-type', '').startswith('application/json'):
                    error_data = resp.json()
            except Exception:
                pass
            result['error'] = error_data.get('error', 'Camera not available')
            result['fix'] = 'Check camera proxy: curl http://127.0.0.1:8080/status'
            log.error(f'camera:{name}', 'Proxy 503', error=result['error'])
            return result

        if resp.status_code != 200:
            result['error'] = f'Proxy returned {resp.status_code}'
            log.error(f'camera:{name}', 'Proxy error', status=resp.status_code)
            return result

        if len(resp.content) < MIN_IMAGE_BYTES:
            result['error'] = f'Image too small ({len(resp.content)} bytes, min {MIN_IMAGE_BYTES})'
            result['fix'] = 'Camera may be producing blank frames'
            log.error(f'camera:{name}', 'Image too small',
                     size=len(resp.content), min=MIN_IMAGE_BYTES)
            return result

        with open(filepath, 'wb') as f:
            f.write(resp.content)

        result['success'] = True
        result['filepath'] = filepath
        result['file_size_bytes'] = len(resp.content)

        log.info(f'camera:{name}', 'Capture complete',
                size_bytes=len(resp.content), capture_time_sec=capture_time)

    except requests.exceptions.ConnectionError:
        result['error'] = 'Camera proxy not running'
        result['fix'] = 'Run: sudo systemctl restart camera-proxy'
        log.error(f'camera:{name}', 'Proxy connection refused')

    except requests.exceptions.Timeout:
        result['error'] = f'Proxy timeout after {PROXY_TIMEOUT_SEC}s'
        result['fix'] = 'Camera proxy may be overloaded'
        result['capture_time_sec'] = PROXY_TIMEOUT_SEC
        log.error(f'camera:{name}', 'Proxy timeout')

    except Exception as e:
        result['error'] = f'Unexpected error: {e}'
        log.exception(f'camera:{name}', 'Capture exception', error=str(e))

    return result


# ============================================================
# GRADING WORKFLOW
# ============================================================

def do_grade(serial_id: str, live_weight: float):
    """
    Full capture → grade workflow. Runs in background thread.
    Captures 1 image per camera via proxy, sends all 3 to EC2 API.
    EC2 handles S3 archival on success — Pi just captures and relays.
    """
    start_time = time.time()
    results = {}
    results_lock = threading.Lock()

    try:
        # Phase 1: Capture
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

        # Capture in batches of 2 to avoid overloading proxy
        MAX_PARALLEL = 2
        camera_list = list(available_cameras.items())
        batches = [camera_list[i:i + MAX_PARALLEL] for i in range(0, len(camera_list), MAX_PARALLEL)]

        for batch_num, batch in enumerate(batches, 1):
            batch_names = [n for n, _ in batch]
            log.info('capture', f'Batch {batch_num}/{len(batches)}',
                    serial_id=serial_id, cameras=','.join(batch_names))

            threads = {}
            for name, path in batch:
                def _capture_and_store(n=name, p=path):
                    result = capture_single_image(n, p, serial_id)
                    with results_lock:
                        results[n] = result

                t = threading.Thread(target=_capture_and_store)
                threads[name] = t
                t.start()

            for name, t in threads.items():
                t.join(timeout=PROXY_TIMEOUT_SEC + 10)
                if t.is_alive():
                    log.error(f'camera:{name}', 'Capture thread hung',
                             serial_id=serial_id)
                    with results_lock:
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
        # EC2 handles grading, S3 archival of raw images, debug images, and result.json.
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
                headers={'X-API-Key': EC2_API_KEY} if EC2_API_KEY else {},
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
            # Clean up captured images from /tmp
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

    try:
        resp = requests.get(f'{PROXY_URL}/status', timeout=5)
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

    sys_info = get_system_info()

    return jsonify({
        'status': 'ok' if all(cameras.values()) else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'cameras': cameras,
        'ec2_target': EC2_API,  # type: ignore
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
        'proxy': {},
        'ec2': {},
        'capture_config': {
            'image_resolution': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
            'proxy_url': PROXY_URL,
            'ec2_timeout_sec': EC2_TIMEOUT_SEC
        },
        'state': get_state()
    }

    # Check each camera via proxy
    for name, path in CAMERAS.items():
        diag['cameras'][name] = check_camera(name, path)

    # Proxy health
    try:
        resp = requests.get(f'{PROXY_URL}/status', timeout=5)
        if resp.status_code == 200:
            diag['proxy'] = {'ok': True, 'data': resp.json()}
        else:
            diag['proxy'] = {'ok': False, 'status_code': resp.status_code}
    except Exception as e:
        diag['proxy'] = {'ok': False, 'error': str(e)}

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
            proxy_ok=diag['proxy'].get('ok', False),
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

    Checks EC2 reachability BEFORE starting the capture.
    EC2 handles all S3 archival on successful grades.
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

    # Check EC2 reachability BEFORE capturing images.
    ec2_check = check_ec2_api()
    if ec2_check['status'] != 'ok':
        log.error('grade', 'EC2 not reachable, aborting before capture',
                 serial_id=sanitized,
                 ec2_api=EC2_API,
                 error=ec2_check.get('error'))
        return jsonify({
            'status': 'error',
            'error_code': 'EC2_UNREACHABLE',
            'message': 'EC2 API not reachable, aborting before capture',
            'ec2_api': EC2_API,
            'ec2_error': ec2_check.get('error'),
            'fix': 'Check EC2 is running and port 8000 is open'
        }), 503

    # Pre-check cameras via proxy
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
    Test grading with uploaded image files instead of cameras.
    Forwards directly to EC2 /analyze — EC2 handles grading and S3 archival.

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

    # Forward to EC2 (EC2 handles grading + S3 archival)
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
            headers={'X-API-Key': EC2_API_KEY} if EC2_API_KEY else {},
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
    """Emergency cancel - reset state and clean up."""
    log.warn('cancel', 'Cancel requested')

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

    # Test 4: Camera proxy check
    try:
        resp = requests.get(f'{PROXY_URL}/status', timeout=5)
        if resp.status_code == 200:
            proxy_data = resp.json()
            results['tests']['camera_proxy'] = {'status': 'ok'}
            for name in CAMERAS:
                cam_health = proxy_data.get('cameras', {}).get(name, {})
                key = f'camera_{name}'
                if cam_health.get('connected'):
                    results['tests'][key] = {
                        'status': 'ok',
                        'fps': cam_health.get('fps'),
                        'frame_age': cam_health.get('frame_age_sec')
                    }
                else:
                    results['tests'][key] = {
                        'status': 'error',
                        'error': cam_health.get('last_error', 'Not connected'),
                        'fix': f'Check USB connection for {name} camera'
                    }
        else:
            results['tests']['camera_proxy'] = {
                'status': 'error',
                'error': f'Proxy returned {resp.status_code}',
                'fix': 'Run: sudo systemctl restart camera-proxy'
            }
    except requests.exceptions.ConnectionError:
        results['tests']['camera_proxy'] = {
            'status': 'error',
            'error': 'Camera proxy not running',
            'fix': 'Run: sudo systemctl restart camera-proxy'
        }
    except Exception as e:
        results['tests']['camera_proxy'] = {
            'status': 'error',
            'error': str(e)
        }

    all_ok = all(t.get('status') == 'ok' for t in results['tests'].values())
    results['summary'] = 'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'
    results['duration_sec'] = round(time.time() - start, 2)

    log.info('test', 'Complete',
            summary=results['summary'],
            passed=sum(1 for t in results['tests'].values() if t.get('status') == 'ok'),
            total=len(results['tests']),
            duration_sec=results['duration_sec'])

    return jsonify(results)


@app.route('/debug/<serial_id>/<view>')
def proxy_debug_image(serial_id, view):
    """
    Proxy a debug image from EC2.
    
    GET /debug/{serial_id}/{view}  ->  EC2 GET /debug/{serial_id}/{view}
    """
    if view not in ('side', 'top', 'front'):
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_VIEW',
            'message': 'view must be one of: side, top, front'
        }), 400

    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', serial_id.strip())
    if not sanitized or len(sanitized) > MAX_SERIAL_ID_LEN:
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_SERIAL_ID',
            'message': 'serial_id must be alphanumeric, max 50 chars'
        }), 400

    try:
        resp = requests.get(
            f'{EC2_API}/debug/{sanitized}/{view}',
            headers={'X-API-Key': EC2_API_KEY} if EC2_API_KEY else {},
            timeout=REQUEST_TIMEOUT_SEC
        )

        if resp.status_code == 200:
            from flask import Response
            return Response(
                resp.content,
                mimetype='image/jpeg',
                headers={'Content-Disposition': f'inline; filename={sanitized}_{view}_debug.jpg'}
            )
        elif resp.status_code == 404:
            return jsonify({
                'status': 'error',
                'error_code': 'NOT_FOUND',
                'message': f'No debug image for {sanitized}/{view}'
            }), 404
        else:
            return jsonify({
                'status': 'error',
                'error_code': 'EC2_ERROR',
                'message': f'EC2 returned {resp.status_code}'
            }), 502

    except requests.exceptions.ConnectionError:
        return jsonify({
            'status': 'error',
            'error_code': 'EC2_UNREACHABLE',
            'message': 'Cannot reach EC2 API',
            'fix': 'Check EC2 container is running'
        }), 502

    except Exception as e:
        log.exception('debug_proxy', 'Failed to fetch debug image',
                     serial_id=sanitized, view=view, error=str(e))
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': str(e)
        }), 500


@app.route('/debug/<serial_id>')
def proxy_debug_list(serial_id):
    """
    Proxy the debug image listing from EC2.
    
    GET /debug/{serial_id}  ->  EC2 GET /debug/{serial_id}
    """
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', serial_id.strip())
    if not sanitized or len(sanitized) > MAX_SERIAL_ID_LEN:
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_SERIAL_ID',
            'message': 'serial_id must be alphanumeric, max 50 chars'
        }), 400

    try:
        resp = requests.get(
            f'{EC2_API}/debug/{sanitized}',
            headers={'X-API-Key': EC2_API_KEY} if EC2_API_KEY else {},
            timeout=REQUEST_TIMEOUT_SEC
        )

        if resp.status_code == 200:
            data = resp.json()
            # Rewrite URLs to point at this Pi proxy instead of EC2
            for img in data.get('debug_images', []):
                img['url'] = f'/debug/{sanitized}/{img["view"]}'
            return jsonify(data)
        elif resp.status_code == 404:
            return jsonify({
                'status': 'error',
                'error_code': 'NOT_FOUND',
                'message': f'No debug images for {sanitized}'
            }), 404
        else:
            return jsonify({
                'status': 'error',
                'error_code': 'EC2_ERROR',
                'message': f'EC2 returned {resp.status_code}'
            }), 502

    except requests.exceptions.ConnectionError:
        return jsonify({
            'status': 'error',
            'error_code': 'EC2_UNREACHABLE',
            'message': 'Cannot reach EC2 API'
        }), 502

    except Exception as e:
        log.exception('debug_proxy', 'Failed to list debug images',
                     serial_id=sanitized, error=str(e))
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': str(e)
        }), 500


# ============================================================
# STARTUP
# ============================================================

def run_startup_checks():
    """Run all startup checks and log results."""
    log.info('startup', '=' * 50)
    log.info('startup', 'PROD PI SERVER STARTING')
    log.info('startup', 'Configuration',
            ec2_ip=EC2_IP, ec2_api=EC2_API,
            ec2_api_key_set=bool(EC2_API_KEY),
            resolution=f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
            proxy_url=PROXY_URL)

    if not EC2_API_KEY:
        log.critical('startup', 'EC2_API_KEY not set — EC2 will reject grading requests',
                    fix='Set EC2_API_KEY in .env to match API_KEY on EC2')

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
                fix='Restart Pi or check for memory leaks')

    # Check camera proxy
    try:
        resp = requests.get(f'{PROXY_URL}/status', timeout=5)
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
    except requests.exceptions.ConnectionError:
        log.warn('startup:proxy', 'Camera proxy not yet running',
                fix='Will retry when grade is requested')
    except Exception as e:
        log.warn('startup:proxy', 'Proxy check failed', error=str(e))

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
    if ec2_ok:
        log.info('startup', 'All systems ready')
    else:
        log.warn('startup', 'Starting with issues',
                issues='ec2')

    log.info('startup', 'Server listening', host='0.0.0.0', port=5000)
    log.info('startup', '=' * 50)


if __name__ == '__main__':
    run_startup_checks()
    app.run(host='0.0.0.0', port=5000)
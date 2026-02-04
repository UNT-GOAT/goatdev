"""
Training Data Collection Server
Records video from 3 cameras, uploads to S3
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

S3_TRAINING_BUCKET = os.environ.get('S3_TRAINING_BUCKET', 'temp-training-937249941844')

# Thresholds
MIN_DISK_MB = 1000          # Require 1GB free
MIN_VIDEO_BYTES = 100000    # 100KB minimum valid video
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


def sanitize_goat_id(goat_id) -> str:
    """Sanitize goat_id to prevent path traversal and weirdness."""
    goat_id = str(goat_id)
    # Allow alphanumeric, underscore, hyphen only
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
            if f.startswith(f'{goat_id}_') and f.endswith('.mp4'):
                os.remove(f'/tmp/{f}')
                log.debug('cleanup', f'Removed temp file', file=f)
    except Exception as e:
        log.warn('cleanup', 'Temp file cleanup failed', error=str(e))


def validate_video(filepath: str) -> tuple[bool, str]:
    """Check if video file is valid using ffprobe."""
    if not os.path.exists(filepath):
        return False, 'File does not exist'
    
    size = os.path.getsize(filepath)
    if size < MIN_VIDEO_BYTES:
        return False, f'File too small ({size} bytes)'
    
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_type,width,height,nb_frames',
            '-of', 'json',
            filepath
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return False, f'ffprobe error: {result.stderr[:100]}'
        
        probe = json.loads(result.stdout)
        if not probe.get('streams'):
            return False, 'No video stream found'
        
        return True, f'OK ({size} bytes)'
        
    except subprocess.TimeoutExpired:
        return False, 'ffprobe timeout'
    except Exception as e:
        return False, f'Validation error: {str(e)}'


# ============================================================
# RECORDING LOGIC
# ============================================================

def record_single_camera(name: str, path: str, goat_id: str, duration: int) -> dict:
    """
    Record from a single camera. Returns result dict.
    Runs in a thread.
    """
    filepath = f'/tmp/{goat_id}_{name}.mp4'
    result = {
        'name': name,
        'success': False,
        'filepath': None,
        'size_bytes': 0,
        'error': None,
        'fix': None
    }
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'v4l2',
        '-framerate', '30',
        '-video_size', '1920x1080',
        '-i', path,
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        filepath
    ]
    
    log.info(f'camera:{name}', 'Starting capture', path=path, duration=duration)
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SEC
        )
        
        if proc.returncode != 0:
            # Parse common ffmpeg errors
            stderr = proc.stderr
            if 'No such file or directory' in stderr:
                result['error'] = 'Camera disconnected during recording'
                result['fix'] = f'Check USB cable for {name} camera'
            elif 'Device or resource busy' in stderr:
                result['error'] = 'Camera is busy (another process using it)'
                result['fix'] = 'Run: sudo pkill -9 ffmpeg'
            elif 'Invalid argument' in stderr:
                result['error'] = 'Camera does not support 1920x1080@30fps'
                result['fix'] = 'Check camera model compatibility'
            else:
                result['error'] = f'ffmpeg error (code {proc.returncode})'
                result['fix'] = f'Check logs, stderr: {stderr[:200]}'
            
            log.error(f'camera:{name}', 'Capture failed',
                     error=result['error'], fix=result['fix'], returncode=proc.returncode)
            return result
        
        # Validate the output
        valid, validation_msg = validate_video(filepath)
        if not valid:
            result['error'] = f'Video validation failed: {validation_msg}'
            result['fix'] = 'Camera may be malfunctioning, try unplugging and replugging'
            log.error(f'camera:{name}', 'Video validation failed',
                     error=validation_msg, file=filepath)
            # Clean up bad file
            if os.path.exists(filepath):
                os.remove(filepath)
            return result
        
        result['success'] = True
        result['filepath'] = filepath
        result['size_bytes'] = os.path.getsize(filepath)
        
        log.info(f'camera:{name}', 'Capture complete',
                size_bytes=result['size_bytes'], file=filepath)
        
    except subprocess.TimeoutExpired:
        result['error'] = f'Recording timed out after {FFMPEG_TIMEOUT_SEC}s'
        result['fix'] = 'Camera may be frozen. Unplug USB hub, wait 5s, replug.'
        log.error(f'camera:{name}', 'Capture timeout',
                 timeout_sec=FFMPEG_TIMEOUT_SEC, fix=result['fix'])
        
        # Kill the hung process
        subprocess.run(['pkill', '-9', '-f', f'ffmpeg.*{path}'], timeout=5)
        
    except Exception as e:
        result['error'] = f'Unexpected error: {str(e)}'
        log.exception(f'camera:{name}', 'Capture exception', error=str(e))
    
    return result


def do_recording(goat_id: str, duration: int, goat_data: dict):
    """
    Main recording workflow. Runs in background thread.
    All cameras must succeed or entire batch fails.
    """
    start_time = time.time()
    results = {}
    
    try:
        set_state(progress='recording')
        
        # Record all cameras in parallel
        threads = {}
        for name, path in CAMERAS.items():
            # Only record from cameras that exist
            check = check_camera(name, path)
            if check['error']:
                results[name] = {
                    'name': name,
                    'success': False,
                    'error': check['error'],
                    'fix': check['fix']
                }
                continue
            
            t = threading.Thread(
                target=lambda n=name, p=path: results.update({n: record_single_camera(n, p, goat_id, duration)})
            )
            threads[name] = t
            t.start()
        
        # Wait for all recordings
        log.info('record', f'Waiting for {len(threads)} cameras', cameras=','.join(threads.keys()))
        for name, t in threads.items():
            t.join(timeout=FFMPEG_TIMEOUT_SEC + 5)
            if t.is_alive():
                log.error(f'camera:{name}', 'Thread did not complete in time')
                results[name] = {
                    'name': name,
                    'success': False,
                    'error': 'Recording thread hung',
                    'fix': 'Restart the service: sudo systemctl restart goat-training'
                }
        
        # Check results - ALL must succeed
        successful = [r for r in results.values() if r.get('success')]
        failed = [r for r in results.values() if not r.get('success')]
        
        if failed:
            # Batch fails - clean up any successful recordings
            for r in successful:
                if r.get('filepath') and os.path.exists(r['filepath']):
                    os.remove(r['filepath'])
                    log.info(f"camera:{r['name']}", 'Cleaned up file (batch failed)')
            
            error_summary = '; '.join([f"{r['name']}: {r.get('error', 'unknown')}" for r in failed])
            fix_summary = ' | '.join([r.get('fix', '') for r in failed if r.get('fix')])
            
            set_state(
                last_error=error_summary,
                last_result={
                    'goat_id': goat_id,
                    'success': False,
                    'uploaded': [],
                    'errors': [{'camera': r['name'], 'error': r.get('error'), 'fix': r.get('fix')} for r in failed],
                    'duration_sec': round(time.time() - start_time, 2)
                }
            )
            
            log.error('record', 'Recording batch failed',
                     goat_id=goat_id,
                     failed_cameras=','.join([r['name'] for r in failed]),
                     error=error_summary,
                     fix=fix_summary)
            return
        
        # All cameras succeeded - check if test
        is_test = str(goat_id).startswith('test_')
        
        if is_test:
            log.info('record', 'Test recording complete - skipping upload',
                    goat_id=goat_id, cameras=len(successful))
            
            # Cleanup test files
            for r in successful:
                if r.get('filepath') and os.path.exists(r['filepath']):
                    os.remove(r['filepath'])
            
            set_state(last_result={
                'goat_id': goat_id,
                'success': True,
                'uploaded': [],
                'test': True,
                'cameras_recorded': len(successful),
                'duration_sec': round(time.time() - start_time, 2)
            })
            return
        
        # Upload to S3
        set_state(progress='uploading')
        uploaded = []
        upload_errors = []
        
        for r in successful:
            s3_key = f'{goat_id}/{goat_id}_{r["name"]}.mp4'
            
            try:
                log.info(f's3:{r["name"]}', 'Uploading', 
                        key=s3_key, size_bytes=r['size_bytes'])
                
                get_s3().upload_file(r['filepath'], S3_TRAINING_BUCKET, s3_key)
                uploaded.append(s3_key)
                
                log.info(f's3:{r["name"]}', 'Upload complete', key=s3_key)
                
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
                
                upload_errors.append({'camera': r['name'], 'error': error_msg, 'fix': fix})
                log.error(f's3:{r["name"]}', 'Upload failed',
                         error=error_msg, fix=fix, key=s3_key)
            
            finally:
                # Always clean up local file
                if os.path.exists(r['filepath']):
                    os.remove(r['filepath'])
        
        # Upload goat_data.json
        if goat_data and uploaded:  # Only if we uploaded videos
            try:
                json_data = {
                    'goat_id': goat_id,
                    'timestamp': datetime.now().isoformat(),
                    'cameras': [r['name'] for r in successful],
                    **{k: v for k, v in goat_data.items() if v}  # Only non-empty fields
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
                    'errors': upload_errors,
                    'duration_sec': total_time
                }
            )
            log.error('record', 'Recording complete but upload failed',
                     goat_id=goat_id, uploaded=len(uploaded), errors=len(upload_errors))
        else:
            set_state(
                last_error=None,
                last_result={
                    'goat_id': goat_id,
                    'success': True,
                    'uploaded': uploaded,
                    'errors': [],
                    'duration_sec': total_time
                }
            )
            log.info('record', 'Recording complete',
                    goat_id=goat_id, uploaded=len(uploaded), duration_sec=total_time)
    
    except Exception as e:
        log.exception('record', 'Unexpected error in recording workflow', error=str(e))
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
    """Start a recording. All 3 cameras must succeed."""
    
    # Check if already recording
    state = get_state()
    if state['active']:
        log.warn('record', 'Recording already in progress',
                current_goat_id=state['goat_id'],
                started_at=state['started_at'])
        return jsonify({
            'status': 'error',
            'error_code': 'RECORDING_IN_PROGRESS',
            'message': f'Recording already in progress for goat {state["goat_id"]}. Please wait.',
            'current_recording': {
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
        log.warn('record', 'Invalid goat_id', error=str(e), raw_goat_id=str(data.get('goat_id'))[:50])
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_GOAT_ID',
            'message': str(e)
        }), 400
    
    try:
        duration = int(data.get('duration', 5))
        if duration < 1 or duration > 10:
            raise ValueError('Duration must be 1-10 seconds')
    except (ValueError, TypeError) as e:
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_DURATION',
            'message': str(e)
        }), 400
    
    goat_data = data.get('goat_data', {})
    if not isinstance(goat_data, dict):
        goat_data = {}
    
    # Pre-flight checks
    log.info('record', 'Recording requested', goat_id=goat_id, duration=duration)
    
    # Check disk space
    disk_ok, disk_mb = check_disk_space()
    if not disk_ok:
        log.error('record', 'Insufficient disk space',
                 free_mb=disk_mb, required_mb=MIN_DISK_MB,
                 fix='Clear /tmp or reboot Pi')
        return jsonify({
            'status': 'error',
            'error_code': 'LOW_DISK_SPACE',
            'message': f'Only {disk_mb}MB free, need {MIN_DISK_MB}MB',
            'fix': 'Run: sudo rm -rf /tmp/*.mp4'
        }), 507  # Insufficient Storage
    
    # Check ALL cameras before starting
    camera_checks = {}
    for name, path in CAMERAS.items():
        camera_checks[name] = check_camera(name, path)
    
    failed_cameras = {name: check for name, check in camera_checks.items() if check.get('error')}
    
    if failed_cameras:
        errors = []
        fixes = []
        for name, check in failed_cameras.items():
            errors.append(f"{name}: {check['error']}")
            if check.get('fix'):
                fixes.append(f"{name}: {check['fix']}")
            log.error(f'camera:{name}', 'Pre-flight check failed',
                     error=check['error'], fix=check.get('fix'))
        
        return jsonify({
            'status': 'error',
            'error_code': 'CAMERA_ERROR',
            'message': f'{len(failed_cameras)} camera(s) not ready',
            'cameras': camera_checks,
            'errors': errors,
            'fixes': fixes
        }), 503  # Service Unavailable
    
    # Kill any stale ffmpeg processes
    kill_stale_ffmpeg()
    
    # Set state and start recording
    set_state(
        active=True,
        goat_id=goat_id,
        started_at=datetime.now().isoformat(),
        progress='starting',
        last_error=None
    )
    
    thread = threading.Thread(target=do_recording, args=(goat_id, duration, goat_data))
    thread.start()
    
    log.info('record', 'Recording started',
            goat_id=goat_id, duration=duration, cameras=','.join(CAMERAS.keys()))
    
    return jsonify({
        'status': 'recording_started',
        'goat_id': goat_id,
        'duration': duration,
        'cameras': list(CAMERAS.keys())
    })


@app.route('/status')
def status():
    """Get current recording state."""
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
    set_state(last_error='Recording cancelled by user')
    
    log.info('cancel', 'Recording cancelled and state reset')
    
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
                 fix='Run: sudo rm -rf /tmp/*.mp4')
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
    
    # Summary
    if all_cameras_ok and disk_ok:
        log.info('startup', 'All systems ready')
    else:
        log.warn('startup', 'Starting with errors - some features may not work')
    
    log.info('startup', 'Server listening', host='0.0.0.0', port=5001)
    log.info('startup', '=' * 50)
    
    app.run(host='0.0.0.0', port=5001)
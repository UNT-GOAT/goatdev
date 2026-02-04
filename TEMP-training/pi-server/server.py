"""
Temporary Training Data Collection Server
DELETE THIS ENTIRE FOLDER WHEN TRAINING IS COMPLETE
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import threading
import os
import shutil
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Camera device paths (set by udev rules)
CAMERAS = {
    'side': '/dev/camera_side',
    'top': '/dev/camera_top',
    'front': '/dev/camera_front'
}

# S3 config
S3_TRAINING_BUCKET = os.environ.get('S3_TRAINING_BUCKET', 'goat-training-ACCOUNTID')
s3 = None

import sys
sys.path.insert(0, '/home/pi/goat-capture/logger')
from logger.pi_cloudwatch import SimpleLogger

log = SimpleLogger('pi/training')

def get_s3():
    global s3
    if s3 is None:
        import boto3
        s3 = boto3.client('s3')
    return s3

def get_system_info():
    """Get system diagnostics"""
    info = {}
    
    # Disk space
    try:
        total, used, free = shutil.disk_usage('/tmp')
        info['disk_tmp_free_mb'] = free // (1024 * 1024)
    except:
        info['disk_tmp_free_mb'] = 'unknown'
    
    # Memory
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemAvailable' in line:
                    info['mem_available_mb'] = int(line.split()[1]) // 1024
                    break
    except:
        info['mem_available_mb'] = 'unknown'
    
    # CPU temp (Pi specific)
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            info['cpu_temp_c'] = int(f.read().strip()) / 1000
    except:
        info['cpu_temp_c'] = 'unknown'
    
    # Load average
    try:
        info['load_avg'] = os.getloadavg()[0]
    except:
        info['load_avg'] = 'unknown'
    
    return info

def check_camera_details(path):
    """Get detailed camera info using v4l2-ctl"""
    details = {'exists': os.path.exists(path)}
    
    if not details['exists']:
        return details
    
    # Check if device is readable
    try:
        details['readable'] = os.access(path, os.R_OK)
    except:
        details['readable'] = False
    
    # Get camera capabilities
    try:
        result = subprocess.run(
            ['v4l2-ctl', '-d', path, '--all'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            details['v4l2_info'] = result.stdout[:500]
        else:
            details['v4l2_error'] = result.stderr[:200]
    except FileNotFoundError:
        details['v4l2_info'] = 'v4l2-ctl not installed'
    except subprocess.TimeoutExpired:
        details['v4l2_info'] = 'timeout'
    except Exception as e:
        details['v4l2_info'] = f'error: {str(e)}'
    
    # Check if camera is in use
    try:
        result = subprocess.run(
            ['fuser', path],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            details['in_use_by_pid'] = result.stdout.strip()
        else:
            details['in_use_by_pid'] = None
    except:
        details['in_use_by_pid'] = 'unknown'
    
    return details

def test_s3_connection():
    """Test S3 connectivity"""
    try:
        get_s3().head_bucket(Bucket=S3_TRAINING_BUCKET)
        return {'status': 'ok', 'bucket': S3_TRAINING_BUCKET}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'bucket': S3_TRAINING_BUCKET}

# Recording state
recording_state = {
    'active': False,
    'progress': None,
    'last_error': None,
    'last_result': None
}


@app.route('/health')
def health():
    """Basic health check"""
    available_cameras = {name: os.path.exists(path) for name, path in CAMERAS.items()}
    return jsonify({
        'status': 'ok',
        'cameras': available_cameras,
        'bucket': S3_TRAINING_BUCKET
    })


@app.route('/diagnostics')
def diagnostics():
    """Detailed system diagnostics for troubleshooting"""
    log("Running diagnostics...")
    
    diag = {
        'timestamp': datetime.now().isoformat(),
        'system': get_system_info(),
        'cameras': {},
        's3': test_s3_connection(),
        'recording_state': recording_state,
        'environment': {
            'S3_TRAINING_BUCKET': S3_TRAINING_BUCKET,
            'pwd': os.getcwd(),
            'user': os.environ.get('USER', 'unknown')
        }
    }
    
    # Check each camera in detail
    for name, path in CAMERAS.items():
        diag['cameras'][name] = check_camera_details(path)
    
    # Check ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        diag['ffmpeg'] = {'installed': True, 'version': result.stdout.split('\n')[0]}
    except FileNotFoundError:
        diag['ffmpeg'] = {'installed': False, 'error': 'ffmpeg not found'}
    except Exception as e:
        diag['ffmpeg'] = {'installed': False, 'error': str(e)}
    
    # List video devices
    try:
        video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
        diag['video_devices'] = video_devices
    except:
        diag['video_devices'] = []
    
    # Check symlinks
    diag['symlinks'] = {}
    for name, path in CAMERAS.items():
        if os.path.islink(path):
            diag['symlinks'][name] = {'target': os.readlink(path), 'exists': os.path.exists(path)}
        else:
            diag['symlinks'][name] = {'target': None, 'exists': os.path.exists(path)}
    
    log(f"Diagnostics complete")
    return jsonify(diag)


@app.route('/record', methods=['POST'])
def record():
    """Record 5s video from all cameras simultaneously"""
    global recording_state
    
    if recording_state['active']:
        log("ERROR: Recording already in progress")
        return jsonify({
            'status': 'error',
            'error_code': 'RECORDING_IN_PROGRESS',
            'message': 'A recording is already in progress. Please wait for it to finish.'
        }), 400
    
    data = request.json or {}
    goat_id = data.get('goat_id', 1)
    duration = min(int(data.get('duration', 5)), 10)
    goat_data = data.get('goat_data', {})
    
    log(f"=== Starting recording: goat_id={goat_id}, duration={duration}s ===")
    if goat_data:
        log(f"  Goat data: {goat_data}")
    
    # Check disk space (need at least 500MB for temp files)
    try:
        _, _, free = shutil.disk_usage('/tmp')
        free_mb = free // (1024 * 1024)
        if free_mb < 500:
            log(f"ERROR: Low disk space - only {free_mb}MB free")
            return jsonify({
                'status': 'error',
                'error_code': 'LOW_DISK_SPACE',
                'message': f'Not enough disk space. Only {free_mb}MB available, need at least 500MB.',
                'details': {'free_mb': free_mb}
            }), 400
    except Exception as e:
        log(f"Warning: Could not check disk space: {e}")
    
    # Check cameras before starting
    available = []
    missing = []
    not_readable = []
    for name, path in CAMERAS.items():
        if os.path.exists(path):
            if os.access(path, os.R_OK):
                available.append(name)
                log(f"  Camera '{name}' found at {path}")
            else:
                not_readable.append(name)
                log(f"  Camera '{name}' EXISTS but NOT READABLE at {path}")
        else:
            missing.append(name)
            log(f"  Camera '{name}' NOT FOUND at {path}")
    
    if not_readable:
        log(f"ERROR: Camera permission issue!")
        return jsonify({
            'status': 'error',
            'error_code': 'CAMERA_PERMISSION_DENIED',
            'message': f'Camera(s) found but not readable: {", ".join(not_readable)}. Check permissions.',
            'details': {
                'not_readable': not_readable,
                'available': available,
                'missing': missing
            }
        }), 400
    
    if not available:
        log("ERROR: No cameras available!")
        return jsonify({
            'status': 'error',
            'error_code': 'NO_CAMERAS',
            'message': 'No cameras connected. Please check that cameras are plugged in and try again.',
            'details': {
                'expected': list(CAMERAS.keys()),
                'missing': missing
            }
        }), 400
    
    recording_state['active'] = True
    recording_state['progress'] = 'starting'
    recording_state['last_error'] = None
    recording_state['last_result'] = None
    
    def do_record():
        global recording_state
        uploaded = []
        errors = []
        
        try:
            recording_state['progress'] = 'recording'
            log("Recording started...")
            
            threads = []
            results = {}
            thread_errors = {}
            
            def record_camera(name, path):
                filename = f'{goat_id}_{name}.mp4'
                filepath = f'/tmp/{filename}'
                
                log(f"  [{name}] Starting ffmpeg capture...")
                
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
                
                log(f"  [{name}] Command: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    thread_errors[name] = f"ffmpeg error (code {result.returncode}): {result.stderr[:500]}"
                    log(f"  [{name}] FFMPEG ERROR: {result.stderr[:200]}")
                    return
                
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    if size > 0:
                        results[name] = filepath
                        log(f"  [{name}] Recorded successfully: {filepath} ({size} bytes)")
                    else:
                        thread_errors[name] = "File created but empty"
                        log(f"  [{name}] ERROR: File is empty")
                        os.remove(filepath)
                else:
                    thread_errors[name] = "File not created"
                    log(f"  [{name}] ERROR: File not created")
            
            # Start recording threads
            for name, path in CAMERAS.items():
                if os.path.exists(path):
                    t = threading.Thread(target=record_camera, args=(name, path))
                    threads.append(t)
                    t.start()
            
            log(f"Waiting for {len(threads)} recording threads...")
            for t in threads:
                t.join()
            
            log(f"Recording complete. Success: {list(results.keys())}, Failed: {list(thread_errors.keys())}")
            
            # Check if this is a test recording
            is_test = str(goat_id).startswith('test_')
            
            # Upload to S3 (skip for test requests)
            if results and not is_test:
                recording_state['progress'] = 'uploading'
                log(f"Uploading {len(results)} files to S3...")
                
                # Upload videos
                for name, filepath in results.items():
                    filename = os.path.basename(filepath)
                    s3_key = f'{goat_id}/{filename}'
                    
                    try:
                        log(f"  [{name}] Uploading to s3://{S3_TRAINING_BUCKET}/{s3_key}")
                        get_s3().upload_file(filepath, S3_TRAINING_BUCKET, s3_key)
                        uploaded.append(s3_key)
                        log(f"  [{name}] Upload complete")
                    except Exception as e:
                        error_msg = f"S3 upload failed: {str(e)}"
                        errors.append(f"{name}: {error_msg}")
                        log(f"  [{name}] ERROR: {error_msg}")
                    finally:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            log(f"  [{name}] Cleaned up temp file")
                
                # Upload goat_data.json
                if goat_data:
                    try:
                        json_data = {
                            'goat_id': goat_id,
                            'timestamp': datetime.now().isoformat(),
                        }
                        # Only include non-empty fields
                        if goat_data.get('description'):
                            json_data['description'] = goat_data['description']
                        if goat_data.get('live_weight'):
                            json_data['live_weight'] = goat_data['live_weight']
                        if goat_data.get('grade'):
                            json_data['grade'] = goat_data['grade']
                        
                        json_key = f'{goat_id}/goat_data.json'
                        json_filepath = f'/tmp/{goat_id}_goat_data.json'
                        
                        with open(json_filepath, 'w') as f:
                            json.dump(json_data, f, indent=2)
                        
                        log(f"  [goat_data] Uploading to s3://{S3_TRAINING_BUCKET}/{json_key}")
                        get_s3().upload_file(json_filepath, S3_TRAINING_BUCKET, json_key)
                        uploaded.append(json_key)
                        log(f"  [goat_data] Upload complete")
                        
                        if os.path.exists(json_filepath):
                            os.remove(json_filepath)
                    except Exception as e:
                        error_msg = f"goat_data.json upload failed: {str(e)}"
                        errors.append(error_msg)
                        log(f"  [goat_data] ERROR: {error_msg}")
            else:
                # Clean up temp files for test recordings
                for name, filepath in results.items():
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        log(f"  [{name}] Cleaned up temp file (test)")
                
                if is_test:
                    log("Test detected; skipping upload.")
                else:
                    log("No files to upload.")
            
            # Compile final result
            errors.extend([f"{k}: {v}" for k, v in thread_errors.items()])
            
            # Build camera-by-camera status
            camera_status = {}
            for name in CAMERAS.keys():
                if name in [os.path.basename(u).split('_')[1].replace('.mp4', '') for u in uploaded]:
                    camera_status[name] = 'uploaded'
                elif name in thread_errors:
                    camera_status[name] = 'failed'
                elif name in missing:
                    camera_status[name] = 'not_connected'
                else:
                    camera_status[name] = 'unknown'
            
            recording_state['last_result'] = {
                'goat_id': goat_id,
                'uploaded': uploaded,
                'errors': errors,
                'camera_status': camera_status,
                'success': len(uploaded) > 0 or (is_test and len(thread_errors) == 0)
            }
            
            log(f"=== Recording finished: {len(uploaded)} uploaded, {len(errors)} errors ===")
            
        except Exception as e:
            error_msg = f"Recording error: {str(e)}"
            log(f"CRITICAL ERROR: {error_msg}")
            recording_state['last_error'] = error_msg
            import traceback
            traceback.print_exc()
        finally:
            recording_state['active'] = False
            recording_state['progress'] = None
    
    thread = threading.Thread(target=do_record)
    thread.start()
    
    return jsonify({
        'status': 'recording_started',
        'goat_id': goat_id,
        'duration': duration,
        'cameras_available': available,
        'cameras_missing': missing,
        'warning': f'Only {len(available)}/3 cameras connected: {", ".join(available)}' if missing else None
    })


@app.route('/status')
def status():
    return jsonify(recording_state)


if __name__ == '__main__':
    log("=" * 60)
    log("TRAINING CAPTURE SERVER STARTING")
    log("=" * 60)
    
    # Environment
    log(f"S3 bucket: {S3_TRAINING_BUCKET}")
    log(f"Working directory: {os.getcwd()}")
    log(f"User: {os.environ.get('USER', 'unknown')}")
    
    # System info
    sys_info = get_system_info()
    log(f"System: disk_free={sys_info.get('disk_tmp_free_mb')}MB, "
        f"mem_available={sys_info.get('mem_available_mb')}MB, "
        f"cpu_temp={sys_info.get('cpu_temp_c')}C, "
        f"load={sys_info.get('load_avg')}")
    
    # Check ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        log(f"ffmpeg: {result.stdout.split(chr(10))[0]}")
    except FileNotFoundError:
        log("ERROR: ffmpeg not installed! Run: sudo apt install ffmpeg")
    except Exception as e:
        log(f"ERROR checking ffmpeg: {e}")
    
    # List all video devices
    try:
        video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
        log(f"Video devices found: {video_devices}")
    except Exception as e:
        log(f"ERROR listing video devices: {e}")
    
    # Check cameras
    log("Camera status:")
    for name, path in CAMERAS.items():
        exists = os.path.exists(path)
        is_link = os.path.islink(path)
        target = os.readlink(path) if is_link else None
        readable = os.access(path, os.R_OK) if exists else False
        
        status = []
        if exists:
            status.append("EXISTS")
        else:
            status.append("MISSING")
        if is_link:
            status.append(f"symlink->{target}")
        if readable:
            status.append("readable")
        elif exists:
            status.append("NOT READABLE")
        
        log(f"  {name}: {path} [{', '.join(status)}]")
    
    # Test S3 connection
    log("Testing S3 connection...")
    s3_test = test_s3_connection()
    if s3_test['status'] == 'ok':
        log(f"  S3: OK - bucket '{S3_TRAINING_BUCKET}' accessible")
    else:
        log(f"  S3: ERROR - {s3_test.get('error')}")
    
    log("=" * 60)
    log("Server ready on http://0.0.0.0:5001")
    log("Endpoints:")
    log("  GET  /health      - Quick health check")
    log("  GET  /diagnostics - Detailed system diagnostics")
    log("  GET  /status      - Current recording state")
    log("  POST /record      - Start recording")
    log("=" * 60)
    
    app.run(host='0.0.0.0', port=5001)
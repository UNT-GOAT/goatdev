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

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

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
            'message': 'Recording already in progress'
        }), 400
    
    data = request.json or {}
    goat_id = data.get('goat_id', 1)
    duration = min(int(data.get('duration', 5)), 10)
    
    log(f"=== Starting recording: goat_id={goat_id}, duration={duration}s ===")
    
    # Check cameras before starting
    available = []
    missing = []
    for name, path in CAMERAS.items():
        if os.path.exists(path):
            available.append(name)
            log(f"  Camera '{name}' found at {path}")
        else:
            missing.append(name)
            log(f"  Camera '{name}' NOT FOUND at {path}")
    
    if not available:
        log("ERROR: No cameras available!")
        return jsonify({
            'status': 'error',
            'message': f'No cameras found. Expected: {list(CAMERAS.values())}'
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
            
            # Upload to S3
            if results:
                recording_state['progress'] = 'uploading'
                log(f"Uploading {len(results)} files to S3...")
                
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
            else:
                log("No files to upload")
            
            # Compile final result
            errors.extend([f"{k}: {v}" for k, v in thread_errors.items()])
            
            recording_state['last_result'] = {
                'goat_id': goat_id,
                'uploaded': uploaded,
                'errors': errors,
                'success': len(uploaded) > 0
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
        'cameras_missing': missing
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
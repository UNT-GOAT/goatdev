"""
Temporary Training Data Collection Server
DELETE THIS ENTIRE FOLDER WHEN TRAINING IS COMPLETE
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import threading
import os
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

def get_s3():
    global s3
    if s3 is None:
        import boto3
        s3 = boto3.client('s3')
    return s3

# Recording state
recording_state = {
    'active': False,
    'progress': None
}


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/record', methods=['POST'])
def record():
    """Record 5s video from all cameras simultaneously"""
    global recording_state
    
    if recording_state['active']:
        return jsonify({
            'status': 'error', 
            'message': 'Recording already in progress'
        }), 400
    
    data = request.json or {}
    goat_id = data.get('goat_id', 1)
    duration = min(int(data.get('duration', 5)), 10)
    
    recording_state['active'] = True
    recording_state['progress'] = 'starting'
    
    def do_record():
        global recording_state
        uploaded = []
        
        try:
            recording_state['progress'] = 'recording'
            
            threads = []
            results = {}
            
            def record_camera(name, path):
                filename = f'{goat_id}_{name}.mp4'
                filepath = f'/tmp/{filename}'
                
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
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    results[name] = filepath
                else:
                    print(f"Failed to record {name}: {result.stderr}")
            
            # Start recording threads
            for name, path in CAMERAS.items():
                if os.path.exists(path):
                    t = threading.Thread(target=record_camera, args=(name, path))
                    threads.append(t)
                    t.start()
            
            for t in threads:
                t.join()
            
            # Upload to S3
            recording_state['progress'] = 'uploading'
            
            for name, filepath in results.items():
                filename = os.path.basename(filepath)
                s3_key = f'{goat_id}/{filename}'
                
                try:
                    get_s3().upload_file(filepath, S3_TRAINING_BUCKET, s3_key)
                    uploaded.append(s3_key)
                except Exception as e:
                    print(f"Failed to upload {name}: {e}")
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)
            
        except Exception as e:
            print(f"Recording error: {e}")
        finally:
            recording_state['active'] = False
            recording_state['progress'] = None
    
    thread = threading.Thread(target=do_record)
    thread.start()
    
    return jsonify({
        'status': 'recording_started',
        'goat_id': goat_id,
        'duration': duration
    })


@app.route('/status')
def status():
    return jsonify(recording_state)


if __name__ == '__main__':
    print(f"Training capture server starting...")
    print(f"S3 bucket: {S3_TRAINING_BUCKET}")
    app.run(host='0.0.0.0', port=5001)

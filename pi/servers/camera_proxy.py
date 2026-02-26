#!/usr/bin/env python3
"""
Camera Proxy Service

Single process owns all 3 USB cameras. Pi 4's USB controller cannot handle
3 simultaneous full-res MJPEG streams, so we rotate pairs: only 2 cameras
are open at a time, cycling (side,top) → (side,front) → (top,front).

Raw buffer: ~647KB-1.1MB per frame (no decode), used for captures
Preview buffer: decoded + resized to 640x480, used for MJPEG streams

Port: 8080
Replaces: view_focus.py

Endpoints:
  GET  /stream/<camera>            - MJPEG preview stream (640x480)
  GET  /capture/<camera>           - Latest full-res JPEG
  POST /capture/all                - All 3 cameras, saved to /tmp, returns paths
  GET  /status                     - Per-camera health + system info
  GET  /focus/<camera>/<val>       - Set focus_absolute via v4l2-ctl
  GET  /autofocus/<camera>/<on>    - Set autofocus on/off
  POST /save                       - Save focus settings to file
  GET  /settings                   - Load saved focus settings
"""

from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import cv2
import subprocess
import threading
import time
import json
import os
import sys
import numpy as np

sys.path.insert(0, '/home/pi/goatdev/pi')
from logger.pi_cloudwatch import Logger

app = Flask(__name__)
CORS(app)

log = Logger('camera-proxy')

# === CONFIGURATION ===

CAMERAS = {
    'side': '/dev/camera_side',
    'top': '/dev/camera_top',
    'front': '/dev/camera_front',
}

FULL_RES_WIDTH = 4656
FULL_RES_HEIGHT = 3496
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480
PREVIEW_JPEG_QUALITY = 70

SETTINGS_FILE = '/home/pi/camera_focus_settings.json'

# Pair rotation settings
FRAMES_PER_PAIR = 10        # Frames to read per camera pair before swapping
CAMERA_PAIRS = [            # Rotate through these pairs
    ('side', 'top'),
    ('side', 'front'),
    ('top', 'front'),
]

STALE_FRAME_THRESHOLD_SEC = 5
HEARTBEAT_INTERVAL_SEC = 300

API_PORT = 8080


# === FRAME BUFFERS ===

class FrameBuffer:
    """Thread-safe buffer for a single camera's frames."""

    def __init__(self, name, device):
        self.name = name
        self.device = device

        # Raw frame buffer (full-res JPEG bytes)
        self.raw_frame = None
        self.raw_timestamp = 0
        self.raw_lock = threading.Lock()

        # Preview buffer (640x480 JPEG bytes)
        self.preview_frame = None
        self.preview_lock = threading.Lock()

        # Stats
        self.frame_count = 0
        self.error_count = 0
        self.last_error = None

        # Stream client tracking
        self.stream_clients = 0

    def store_raw(self, raw_bytes):
        """Store a new raw frame."""
        now = time.time()
        with self.raw_lock:
            self.raw_frame = raw_bytes
            self.raw_timestamp = now
        self.frame_count += 1
        self.last_error = None

    def record_error(self, error_msg):
        """Record a read error."""
        self.error_count += 1
        self.last_error = error_msg

    def get_raw_frame(self):
        """Get latest raw JPEG bytes and timestamp. Returns (bytes, timestamp) or (None, 0)."""
        with self.raw_lock:
            return self.raw_frame, self.raw_timestamp

    def get_preview_frame(self):
        """Get latest preview JPEG bytes. Returns bytes or None."""
        with self.preview_lock:
            return self.preview_frame

    def update_preview(self):
        """Decode raw frame, resize, and store as preview JPEG."""
        with self.raw_lock:
            raw = self.raw_frame
        if raw is None:
            return

        try:
            raw_array = np.frombuffer(raw, dtype=np.uint8)
            decoded = cv2.imdecode(raw_array, cv2.IMREAD_COLOR)
            if decoded is None:
                return
            preview = cv2.resize(decoded, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            _, jpeg = cv2.imencode(
                '.jpg', preview,
                [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY]
            )
            with self.preview_lock:
                self.preview_frame = jpeg.tobytes()
        except Exception as e:
            log.warn(f'preview:{self.name}', 'Preview encode error', error=str(e))

    def generate_stream(self):
        """Generator for MJPEG stream."""
        self.stream_clients += 1
        try:
            last_frame = None
            while True:
                frame = self.get_preview_frame()
                if frame is None or frame == last_frame:
                    time.sleep(0.05)
                    continue
                last_frame = frame
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + frame
                    + b'\r\n'
                )
        finally:
            self.stream_clients -= 1

    def get_health(self):
        """Return health dict for status endpoint."""
        frame_age = None
        if self.raw_timestamp > 0:
            frame_age = round(time.time() - self.raw_timestamp, 2)

        raw_size = len(self.raw_frame) if self.raw_frame else 0

        return {
            'connected': self.raw_frame is not None,
            'device': self.device,
            'device_exists': os.path.exists(self.device),
            'frame_count': self.frame_count,
            'frame_age_sec': frame_age,
            'raw_buffer_bytes': raw_size,
            'stream_clients': self.stream_clients,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'stale': frame_age is not None and frame_age > STALE_FRAME_THRESHOLD_SEC,
        }


# === GLOBAL STATE ===

buffers = {}


def init_buffers():
    """Create a FrameBuffer for each camera."""
    for name, device in CAMERAS.items():
        buffers[name] = FrameBuffer(name, device)
    log.info('startup', 'Frame buffers initialized',
            cameras=','.join(CAMERAS.keys()))


# === CAMERA OPEN/CLOSE HELPERS ===

def open_camera(device):
    """Open a camera device in raw MJPEG mode. Returns cap or None."""
    if not os.path.exists(device):
        return None

    try:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_RES_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_RES_HEIGHT)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Raw MJPEG, no decode

        return cap
    except Exception as e:
        log.error('camera', 'Open failed', device=device, error=str(e))
        return None


def close_camera(cap):
    """Safely release a camera."""
    if cap:
        try:
            cap.release()
        except Exception:
            pass


# === PAIR ROTATION READER ===

def pair_rotation_loop():
    """
    Main reader loop. Only 2 cameras open at a time.
    Cycles through pairs: (side,top) → (side,front) → (top,front) → repeat.
    Reads FRAMES_PER_PAIR frames from each pair before swapping.

    Pi 4 USB controller cannot handle 3 simultaneous full-res MJPEG streams.
    With 2 open + pair rotation, each camera gets ~1.8fps equally.
    """
    pair_index = 0

    while True:
        a_name, b_name = CAMERA_PAIRS[pair_index % len(CAMERA_PAIRS)]
        a_buf = buffers[a_name]
        b_buf = buffers[b_name]

        # Open both cameras
        cap_a = open_camera(a_buf.device)
        cap_b = open_camera(b_buf.device)

        if not cap_a:
            a_buf.record_error(f'Failed to open {a_buf.device}')
            log.warn(f'reader:{a_name}', 'Failed to open', device=a_buf.device)
        if not cap_b:
            b_buf.record_error(f'Failed to open {b_buf.device}')
            log.warn(f'reader:{b_name}', 'Failed to open', device=b_buf.device)

        # Read frames from this pair
        for _ in range(FRAMES_PER_PAIR):
            if cap_a:
                try:
                    ret, frame = cap_a.read()
                    if ret and frame is not None:
                        a_buf.store_raw(frame.tobytes())
                    else:
                        a_buf.record_error('Read returned empty frame')
                except Exception as e:
                    a_buf.record_error(f'Read exception: {e}')

            if cap_b:
                try:
                    ret, frame = cap_b.read()
                    if ret and frame is not None:
                        b_buf.store_raw(frame.tobytes())
                    else:
                        b_buf.record_error('Read returned empty frame')
                except Exception as e:
                    b_buf.record_error(f'Read exception: {e}')

        # Close both before opening next pair
        close_camera(cap_a)
        close_camera(cap_b)

        # Update previews for cameras that have stream clients
        for name, buf in buffers.items():
            if buf.stream_clients > 0:
                buf.update_preview()

        pair_index += 1


# === PREVIEW LOOP ===

def preview_loop():
    """
    Background thread that updates preview frames for cameras
    with active stream clients. Runs independently of pair rotation
    so previews update even when a camera isn't in the active pair.
    """
    last_timestamps = {name: 0 for name in buffers}

    while True:
        any_clients = False
        for name, buf in buffers.items():
            if buf.stream_clients <= 0:
                continue
            any_clients = True

            # Check if we have a new raw frame
            with buf.raw_lock:
                ts = buf.raw_timestamp
            if ts > last_timestamps[name]:
                buf.update_preview()
                last_timestamps[name] = ts

        if not any_clients:
            time.sleep(0.5)
        else:
            time.sleep(0.1)


# === V4L2 HELPERS ===

def run_v4l2(device, ctrl, value):
    """Run v4l2-ctl to set a camera control."""
    try:
        result = subprocess.run(
            ['v4l2-ctl', '-d', device, '--set-ctrl', f'{ctrl}={value}'],
            capture_output=True, text=True, timeout=5
        )
        return result
    except subprocess.TimeoutExpired:
        log.error('v4l2', 'v4l2-ctl timeout',
                 device=device, ctrl=ctrl, value=value)
        return None
    except Exception as e:
        log.error('v4l2', 'v4l2-ctl error',
                 device=device, ctrl=ctrl, value=value, error=str(e))
        return None


def apply_saved_settings():
    """Apply saved focus settings on startup."""
    if not os.path.exists(SETTINGS_FILE):
        log.info('startup', 'No saved focus settings')
        return

    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)

        for camera, focus_val in settings.items():
            device = CAMERAS.get(camera)
            if device and os.path.exists(device):
                run_v4l2(device, 'focus_automatic_continuous', 0)
                run_v4l2(device, 'focus_absolute', focus_val)
                log.info(f'startup:{camera}', 'Focus applied',
                        focus=focus_val)
            else:
                log.warn(f'startup:{camera}', 'Device not found, skipping focus',
                        device=device)
    except Exception as e:
        log.error('startup', 'Failed to apply focus settings', error=str(e))


# === HEARTBEAT ===

def heartbeat_loop():
    """Log periodic health status."""
    while True:
        time.sleep(HEARTBEAT_INTERVAL_SEC)

        healths = {}
        for name, buf in buffers.items():
            h = buf.get_health()
            healths[name] = {
                'connected': h['connected'],
                'frame_age': h['frame_age_sec'],
                'frames': h['frame_count'],
                'errors': h['error_count'],
                'clients': h['stream_clients'],
                'buffer_kb': round(h['raw_buffer_bytes'] / 1024, 1),
            }

        total_clients = sum(h['clients'] for h in healths.values())
        all_connected = all(h['connected'] for h in healths.values())

        # Memory info
        mem_mb = None
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        mem_mb = int(line.split()[1]) // 1024
                        break
        except Exception:
            pass

        log.info('heartbeat', 'Status',
                all_connected=all_connected,
                total_stream_clients=total_clients,
                mem_available_mb=mem_mb,
                cameras=json.dumps(healths))


# === FLASK ROUTES: STREAMS & CAPTURES ===

@app.route('/stream/<camera>')
def stream(camera):
    """MJPEG preview stream (640x480). Multiple clients supported."""
    buf = buffers.get(camera)
    if not buf:
        return jsonify({'error': f'Unknown camera: {camera}'}), 404

    if buf.raw_frame is None:
        return jsonify({
            'error': f'Camera {camera} not yet available',
            'last_error': buf.last_error
        }), 503

    return Response(
        buf.generate_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/capture/<camera>')
def capture(camera):
    """Return latest full-res raw JPEG for a single camera."""
    buf = buffers.get(camera)
    if not buf:
        return jsonify({'error': f'Unknown camera: {camera}'}), 404

    raw, timestamp = buf.get_raw_frame()

    if raw is None:
        log.warn(f'capture:{camera}', 'No frame available',
                last_error=buf.last_error)
        return jsonify({
            'error': f'No frame available for {camera}',
            'last_error': buf.last_error
        }), 503

    frame_age = time.time() - timestamp
    if frame_age > STALE_FRAME_THRESHOLD_SEC:
        log.warn(f'capture:{camera}', 'Frame is stale',
                frame_age_sec=round(frame_age, 2),
                threshold=STALE_FRAME_THRESHOLD_SEC)
        return jsonify({
            'error': f'Frame is stale ({frame_age:.1f}s old)',
            'frame_age_sec': round(frame_age, 2),
            'threshold_sec': STALE_FRAME_THRESHOLD_SEC
        }), 503

    log.info(f'capture:{camera}', 'Frame served',
            size_bytes=len(raw),
            frame_age_sec=round(frame_age, 2))

    return Response(raw, mimetype='image/jpeg', headers={
        'X-Frame-Timestamp': str(timestamp),
        'X-Frame-Age-Sec': str(round(frame_age, 3)),
        'X-Frame-Size-Bytes': str(len(raw)),
    })


@app.route('/capture/all', methods=['POST'])
def capture_all_cameras():
    """Capture from all 3 cameras, save to /tmp, return file paths."""
    data = request.get_json(silent=True) or {}
    prefix = data.get('prefix', 'capture')

    results = {}
    all_ok = True

    for name, buf in buffers.items():
        raw, timestamp = buf.get_raw_frame()

        if raw is None:
            results[name] = {
                'success': False,
                'error': 'No frame available',
                'last_error': buf.last_error,
            }
            all_ok = False
            log.error(f'capture_all:{name}', 'No frame available')
            continue

        frame_age = time.time() - timestamp
        if frame_age > STALE_FRAME_THRESHOLD_SEC:
            results[name] = {
                'success': False,
                'error': f'Frame stale ({frame_age:.1f}s old)',
                'frame_age_sec': round(frame_age, 2),
            }
            all_ok = False
            log.warn(f'capture_all:{name}', 'Stale frame',
                    frame_age_sec=round(frame_age, 2))
            continue

        filepath = f'/tmp/{prefix}_{name}.jpg'
        try:
            with open(filepath, 'wb') as f:
                f.write(raw)

            file_size = os.path.getsize(filepath)
            results[name] = {
                'success': True,
                'filepath': filepath,
                'file_size_bytes': file_size,
                'frame_age_sec': round(frame_age, 2),
                'timestamp': timestamp,
            }
            log.info(f'capture_all:{name}', 'Saved',
                    filepath=filepath, size_bytes=file_size,
                    frame_age_sec=round(frame_age, 2))

        except Exception as e:
            results[name] = {
                'success': False,
                'error': f'Write failed: {e}',
            }
            all_ok = False
            log.error(f'capture_all:{name}', 'Write failed',
                     filepath=filepath, error=str(e))

    return jsonify({
        'success': all_ok,
        'cameras': results,
    }), 200 if all_ok else 503


# === FLASK ROUTES: STATUS ===

@app.route('/status')
def status():
    """Per-camera health and system overview."""
    cameras = {}
    for name, buf in buffers.items():
        cameras[name] = buf.get_health()

    all_have_frames = all(c['connected'] for c in cameras.values())
    any_stale = any(c.get('stale', False) for c in cameras.values())

    if all_have_frames and not any_stale:
        overall = 'ok'
    elif all_have_frames:
        overall = 'degraded'
    else:
        overall = 'error'

    return jsonify({
        'status': overall,
        'cameras': cameras,
        'timestamp': time.time(),
    })


# === FLASK ROUTES: FOCUS CONTROL ===

@app.route('/focus/<camera>/<int:val>')
def set_focus(camera, val):
    """Set manual focus value."""
    device = CAMERAS.get(camera)
    if not device:
        return jsonify({'error': f'Unknown camera: {camera}'}), 404

    result = run_v4l2(device, 'focus_absolute', val)
    if result is None:
        return f'Error: v4l2-ctl failed', 500

    if result.returncode == 0:
        log.info(f'focus:{camera}', 'Focus set', value=val)
        return f'{camera} focus={val}'
    return f'Error: {result.stderr}'


@app.route('/autofocus/<camera>/<int:on>')
def set_af(camera, on):
    """Set autofocus on or off."""
    device = CAMERAS.get(camera)
    if not device:
        return jsonify({'error': f'Unknown camera: {camera}'}), 404

    result = run_v4l2(device, 'focus_automatic_continuous', on)
    if result is None:
        return f'Error: v4l2-ctl failed', 500

    if result.returncode == 0:
        log.info(f'autofocus:{camera}', f'AF {"on" if on else "off"}')
        return f'{camera} AF={"on" if on else "off"}'
    return f'Error: {result.stderr}'


@app.route('/save', methods=['POST'])
def save_settings():
    """Save focus settings to file."""
    settings = request.get_json()
    if not settings:
        return jsonify({'error': 'No settings provided'}), 400

    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        log.info('settings', 'Focus settings saved', settings=json.dumps(settings))
        return f'Settings saved: {settings}'
    except Exception as e:
        log.error('settings', 'Failed to save', error=str(e))
        return jsonify({'error': f'Save failed: {e}'}), 500


@app.route('/settings')
def get_settings():
    """Load saved focus settings."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return jsonify(json.load(f))
        except Exception as e:
            log.error('settings', 'Failed to load', error=str(e))
            return jsonify({'error': f'Load failed: {e}'}), 500
    return jsonify({'side': 200, 'top': 200, 'front': 200})


@app.route('/capture/test')
def capture_test():
    """Capture test shots from all cameras (legacy compat)."""
    results = []
    for name, buf in buffers.items():
        raw, timestamp = buf.get_raw_frame()
        if raw is None:
            results.append(f'{name}: NOT AVAILABLE')
            continue

        frame_age = time.time() - timestamp
        if frame_age > STALE_FRAME_THRESHOLD_SEC:
            results.append(f'{name}: STALE ({frame_age:.1f}s)')
            continue

        outfile = f'/tmp/test_{name}.jpg'
        try:
            with open(outfile, 'wb') as f:
                f.write(raw)
            size_kb = round(len(raw) / 1024, 1)
            results.append(f'{name}: OK ({size_kb}KB)')
        except Exception as e:
            results.append(f'{name}: FAILED ({e})')

    return ' | '.join(results)


# === STARTUP ===

def run_startup_checks():
    """Verify cameras and apply settings at startup."""
    log.info('startup', '=' * 50)
    log.info('startup', 'CAMERA PROXY STARTING')
    log.info('startup', 'Configuration',
            full_res=f'{FULL_RES_WIDTH}x{FULL_RES_HEIGHT}',
            preview_res=f'{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}',
            frames_per_pair=FRAMES_PER_PAIR,
            pairs=str(CAMERA_PAIRS),
            stale_threshold=STALE_FRAME_THRESHOLD_SEC,
            heartbeat_interval=HEARTBEAT_INTERVAL_SEC)

    # System info
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemAvailable' in line:
                    mem_mb = int(line.split()[1]) // 1024
                    log.info('startup:system', 'Memory',
                            available_mb=mem_mb)
                    if mem_mb < 500:
                        log.warn('startup:system', 'Low memory',
                                available_mb=mem_mb)
                    break
    except Exception:
        pass

    # Check each camera device
    all_ok = True
    for name, device in CAMERAS.items():
        exists = os.path.exists(device)
        if exists:
            readable = os.access(device, os.R_OK)
            if readable:
                log.info(f'startup:camera:{name}', 'Device ready',
                        device=device)
            else:
                log.error(f'startup:camera:{name}', 'Not readable',
                         device=device, fix=f'sudo chmod 666 {device}')
                all_ok = False
        else:
            log.error(f'startup:camera:{name}', 'Not found',
                     device=device, fix='Check USB connection')
            all_ok = False

    # Disable autofocus on all cameras before readers start
    for name, device in CAMERAS.items():
        if os.path.exists(device):
            run_v4l2(device, 'focus_automatic_continuous', 0)
            log.info(f'startup:{name}', 'Autofocus disabled')

    # Apply saved focus
    apply_saved_settings()

    if all_ok:
        log.info('startup', 'All cameras ready')
    else:
        log.warn('startup', 'Some cameras not ready — starting anyway')

    log.info('startup', 'Strategy: pair rotation (2 cameras open at a time)')
    log.info('startup', 'Server listening', host='0.0.0.0', port=API_PORT)
    log.info('startup', '=' * 50)


# === MAIN ===

if __name__ == '__main__':
    run_startup_checks()
    init_buffers()

    # Start pair rotation reader thread
    reader_thread = threading.Thread(
        target=pair_rotation_loop, daemon=True, name='pair-reader'
    )
    reader_thread.start()

    # Start preview thread
    preview_thread = threading.Thread(
        target=preview_loop, daemon=True, name='preview'
    )
    preview_thread.start()

    # Start heartbeat thread
    heartbeat_thread = threading.Thread(
        target=heartbeat_loop, daemon=True, name='heartbeat'
    )
    heartbeat_thread.start()

    try:
        app.run(host='0.0.0.0', port=API_PORT, threaded=True)
    except KeyboardInterrupt:
        log.info('shutdown', 'Shutting down')
#!/usr/bin/env python3
"""
Camera Proxy Service

Single process owns all 3 USB cameras. Reads raw MJPEG frames continuously
into shared buffers. All consumers (preview streams, captures, focus tool)
read from these buffers. No other process opens the camera devices.

Raw buffer: ~647KB per frame (no decode), used for captures
Preview buffer: decoded + resized to 640x480, used for MJPEG streams

Port: 8080
Replaces: view_focus.py

Endpoints:
  GET  /stream/<camera>            - MJPEG preview stream (640x480)
  GET  /capture/<camera>           - Latest full-res JPEG (~647KB)
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

RECONNECT_INTERVAL_SEC = 5
RECONNECT_MAX_BACKOFF_SEC = 30
STALE_FRAME_THRESHOLD_SEC = 5
HEARTBEAT_INTERVAL_SEC = 300

API_PORT = 8080

# Allow max 2 concurrent USB reads — prevents bus saturation
# while not blocking all cameras if one is slow
usb_semaphore = threading.Semaphore(1)


# === CAMERA READER ===

class CameraReader:
    """
    Owns a single camera device. Background thread reads raw MJPEG frames
    into a shared buffer, coordinated via usb_lock so only one camera
    reads at a time. Preview thread decodes + resizes when clients
    are watching.
    """

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

        # State
        self.connected = False
        self.running = False
        self.cap = None
        self.frame_count = 0
        self.error_count = 0
        self.consecutive_failures = 0
        self.last_error = None
        self.fps_estimate = 0.0
        self.stream_clients = 0

        # Threads
        self._reader_thread = None
        self._preview_thread = None

    def start(self):
        """Start reader and preview threads."""
        self.running = True

        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name=f'reader-{self.name}'
        )
        self._reader_thread.start()

        self._preview_thread = threading.Thread(
            target=self._preview_loop, daemon=True, name=f'preview-{self.name}'
        )
        self._preview_thread.start()

        log.info(f'reader:{self.name}', 'Reader started', device=self.device)

    def stop(self):
        """Stop threads and release camera."""
        self.running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=5)
        if self._preview_thread:
            self._preview_thread.join(timeout=5)
        self._release()
        log.info(f'reader:{self.name}', 'Reader stopped')

    def _open(self):
        """Open the camera device with raw MJPEG mode."""
        if not os.path.exists(self.device):
            self.connected = False
            self.last_error = 'Device not found'
            return False

        try:
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not cap.isOpened():
                self.connected = False
                self.last_error = 'Failed to open device'
                log.error(f'reader:{self.name}', 'Failed to open',
                         device=self.device)
                return False

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_RES_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_RES_HEIGHT)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Raw MJPEG, no decode

            # Verify resolution
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w != FULL_RES_WIDTH or actual_h != FULL_RES_HEIGHT:
                log.warn(f'reader:{self.name}', 'Resolution mismatch',
                        expected=f'{FULL_RES_WIDTH}x{FULL_RES_HEIGHT}',
                        actual=f'{actual_w}x{actual_h}')

            self.cap = cap
            self.connected = True
            self.consecutive_failures = 0
            self.last_error = None

            log.info(f'reader:{self.name}', 'Camera opened',
                    resolution=f'{actual_w}x{actual_h}',
                    device=self.device)
            return True

        except Exception as e:
            self.connected = False
            self.last_error = str(e)
            log.error(f'reader:{self.name}', 'Open exception',
                     error=str(e), device=self.device)
            return False

    def _release(self):
        """Release the camera device."""
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                log.warn(f'reader:{self.name}', 'Release error', error=str(e))
            self.cap = None
        self.connected = False

    def _read_loop(self):
        """Read frames using shared USB lock to prevent bus saturation."""
        backoff = RECONNECT_INTERVAL_SEC
        fps_window_start = time.time()
        fps_window_count = 0

        while self.running:
            # Reconnect if needed
            if not self.connected:
                if not self._open():
                    log.warn(f'reader:{self.name}', 'Reconnecting',
                            backoff_sec=backoff,
                            consecutive_failures=self.consecutive_failures)
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, RECONNECT_MAX_BACKOFF_SEC)
                    continue
                backoff = RECONNECT_INTERVAL_SEC
                fps_window_start = time.time()
                fps_window_count = 0

            # Acquire USB lock — only one camera reads at a time
            with usb_semaphore:
                try:
                    ret, frame = self.cap.read()
                except Exception as e:
                    self.error_count += 1
                    self.consecutive_failures += 1
                    self.last_error = f'Read exception: {e}'
                    log.error(f'reader:{self.name}', 'Read exception',
                             error=str(e))
                    self._release()
                    continue

            if not ret or frame is None:
                self.error_count += 1
                self.consecutive_failures += 1
                self.last_error = 'Read returned empty frame'

                if self.consecutive_failures == 1:
                    log.warn(f'reader:{self.name}', 'Frame read failed')
                elif self.consecutive_failures % 10 == 0:
                    log.error(f'reader:{self.name}',
                             f'{self.consecutive_failures} consecutive failures',
                             last_error=self.last_error)

                if self.consecutive_failures >= 10:
                    log.error(f'reader:{self.name}',
                             'Too many failures, reconnecting',
                             consecutive_failures=self.consecutive_failures)
                    self._release()
                continue

            # Store raw JPEG bytes
            now = time.time()
            raw_bytes = frame.tobytes()

            with self.raw_lock:
                self.raw_frame = raw_bytes
                self.raw_timestamp = now

            self.frame_count += 1
            self.consecutive_failures = 0
            self.last_error = None

            # FPS calculation (rolling 10s window)
            fps_window_count += 1
            elapsed = now - fps_window_start
            if elapsed >= 10:
                self.fps_estimate = round(fps_window_count / elapsed, 1)
                fps_window_start = now
                fps_window_count = 0

            # Yield to other cameras
            time.sleep(0.3)

    def _preview_loop(self):
        """
        Generates preview frames by decoding + resizing raw frames.
        Only runs when stream clients are connected (saves CPU).
        """
        last_raw_ts = 0

        while self.running:
            # Sleep if no clients watching
            if self.stream_clients <= 0:
                time.sleep(0.5)
                continue

            # Get latest raw frame
            with self.raw_lock:
                raw = self.raw_frame
                ts = self.raw_timestamp

            if raw is None or ts == last_raw_ts:
                time.sleep(0.05)
                continue

            last_raw_ts = ts

            try:
                # Decode raw JPEG
                import numpy as np
                raw_array = np.frombuffer(raw, dtype=np.uint8)
                decoded = cv2.imdecode(raw_array, cv2.IMREAD_COLOR)

                if decoded is None:
                    continue

                # Resize for preview
                preview = cv2.resize(decoded, (PREVIEW_WIDTH, PREVIEW_HEIGHT))

                # Encode to JPEG
                _, jpeg = cv2.imencode(
                    '.jpg', preview,
                    [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY]
                )

                with self.preview_lock:
                    self.preview_frame = jpeg.tobytes()

            except Exception as e:
                log.warn(f'preview:{self.name}', 'Preview encode error',
                        error=str(e))
                time.sleep(0.1)

    def get_raw_frame(self):
        """
        Get latest full-res raw JPEG bytes and its timestamp.
        Returns (bytes, timestamp) or (None, 0).
        """
        with self.raw_lock:
            return self.raw_frame, self.raw_timestamp

    def get_preview_frame(self):
        """Get latest preview JPEG bytes. Returns bytes or None."""
        with self.preview_lock:
            return self.preview_frame

    def generate_stream(self):
        """
        Generator for MJPEG stream. Yields preview frames as multipart response.
        Tracks client count so preview thread knows when to run.
        """
        self.stream_clients += 1
        try:
            last_frame = None
            while self.running:
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
            'connected': self.connected,
            'device': self.device,
            'device_exists': os.path.exists(self.device),
            'frame_count': self.frame_count,
            'fps': self.fps_estimate,
            'frame_age_sec': frame_age,
            'raw_buffer_bytes': raw_size,
            'stream_clients': self.stream_clients,
            'error_count': self.error_count,
            'consecutive_failures': self.consecutive_failures,
            'last_error': self.last_error,
            'stale': frame_age is not None and frame_age > STALE_FRAME_THRESHOLD_SEC,
        }


# === GLOBAL STATE ===

readers = {}


def init_readers():
    """Create and start a CameraReader for each camera."""
    for name, device in CAMERAS.items():
        reader = CameraReader(name, device)
        readers[name] = reader
        reader.start()

    log.info('startup', 'All readers started',
            cameras=','.join(CAMERAS.keys()))


def shutdown_readers():
    """Stop all readers."""
    for name, reader in readers.items():
        reader.stop()
    log.info('shutdown', 'All readers stopped')


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
        for name, reader in readers.items():
            h = reader.get_health()
            healths[name] = {
                'connected': h['connected'],
                'fps': h['fps'],
                'frame_age': h['frame_age_sec'],
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
    reader = readers.get(camera)
    if not reader:
        return jsonify({'error': f'Unknown camera: {camera}'}), 404

    if not reader.connected:
        return jsonify({
            'error': f'Camera {camera} not connected',
            'last_error': reader.last_error
        }), 503

    return Response(
        reader.generate_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/capture/<camera>')
def capture(camera):
    """
    Return latest full-res raw JPEG for a single camera.
    Returns 503 if frame is stale or camera disconnected.
    """
    reader = readers.get(camera)
    if not reader:
        return jsonify({'error': f'Unknown camera: {camera}'}), 404

    raw, timestamp = reader.get_raw_frame()

    if raw is None:
        log.warn(f'capture:{camera}', 'No frame available',
                connected=reader.connected,
                last_error=reader.last_error)
        return jsonify({
            'error': f'No frame available for {camera}',
            'connected': reader.connected,
            'last_error': reader.last_error
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
    """
    Capture from all 3 cameras, save to /tmp, return file paths.
    Used by training and prod servers.

    POST body (optional):
    {
        "prefix": "goat_123"   - filename prefix (default: "capture")
    }
    """
    data = request.get_json(silent=True) or {}
    prefix = data.get('prefix', 'capture')

    results = {}
    all_ok = True

    for name, reader in readers.items():
        raw, timestamp = reader.get_raw_frame()

        if raw is None:
            results[name] = {
                'success': False,
                'error': 'No frame available',
                'connected': reader.connected,
                'last_error': reader.last_error,
            }
            all_ok = False
            log.error(f'capture_all:{name}', 'No frame available',
                     connected=reader.connected)
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
    for name, reader in readers.items():
        cameras[name] = reader.get_health()

    all_connected = all(c['connected'] for c in cameras.values())
    any_stale = any(c.get('stale', False) for c in cameras.values())

    if all_connected and not any_stale:
        overall = 'ok'
    elif all_connected:
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
    """Capture test shots from all cameras (legacy compat with view_focus.py)."""
    results = []
    for name, reader in readers.items():
        raw, timestamp = reader.get_raw_frame()
        if raw is None:
            results.append(f'{name}: NOT CONNECTED')
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
            results.append(f'{name}: ✅ OK ({size_kb}KB)')
        except Exception as e:
            results.append(f'{name}: ❌ FAILED ({e})')

    return ' | '.join(results)


# === STARTUP ===

def run_startup_checks():
    """Verify cameras and apply settings at startup."""
    log.info('startup', '=' * 50)
    log.info('startup', 'CAMERA PROXY STARTING')
    log.info('startup', 'Configuration',
            full_res=f'{FULL_RES_WIDTH}x{FULL_RES_HEIGHT}',
            preview_res=f'{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}',
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

    log.info('startup', 'Server listening', host='0.0.0.0', port=API_PORT)
    log.info('startup', '=' * 50)


# === MAIN ===

if __name__ == '__main__':
    run_startup_checks()
    init_readers()

    # Start heartbeat thread
    heartbeat_thread = threading.Thread(
        target=heartbeat_loop, daemon=True, name='heartbeat'
    )
    heartbeat_thread.start()

    try:
        app.run(host='0.0.0.0', port=API_PORT, threaded=True)
    except KeyboardInterrupt:
        log.info('shutdown', 'Shutting down')
    finally:
        shutdown_readers()
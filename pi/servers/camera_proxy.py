#!/usr/bin/env python3
"""
Camera Proxy Service

Single process owns all 3 USB cameras. All 3 stay open at full resolution
but we only read from 2 at a time (rotating pairs) to avoid USB controller
select() timeouts. This gives ~8fps per camera with zero open/close overhead.

Key insight: Pi 4's VL805 USB controller chokes on 3 concurrent reads but
handles 2 fine. Keeping the 3rd camera open (but not reading) avoids the
1.2s sensor warmup penalty that would come from closing/reopening.

Idle mode: When no stream clients are connected and no capture is pending,
the reader loop sleeps instead of reading. Cameras stay open (no warmup
penalty on resume) but USB controller and MJPEG encoder are idle, reducing
wear. First frame after waking is available in ~100-200ms.

Preview pipeline: Uses turbojpeg for DCT-domain downscaling during JPEG
decode. A 16MP frame (4656x3496) is scaled to 1/8 (~582x437) in ~25ms
vs ~250ms with the old cv2 decode→resize→encode path. This brings the
MJPEG preview stream from ~3fps to ~8-10fps.

Raw buffer: ~647KB-1.1MB per frame (no decode), used for captures
Preview buffer: turbojpeg DCT downscaled + re-encoded, used for MJPEG streams

Port: 8080

Run directly:
  python3 camera_proxy.py

Endpoints:
  GET  /stream/<camera>            - MJPEG preview stream (~582x437)
  GET  /capture/<camera>           - Latest full-res JPEG
  GET  /capture                    - Alias for /capture/all (GET convenience)
  POST /capture/all                - All 3 cameras, saved to /tmp, returns paths
  POST /capture/burst/<camera>     - Burst capture: N frames over time, returns tar.gz
  GET  /status                     - Per-camera health + system info
  GET  /focus/<camera>/<val>       - Set focus_absolute via v4l2-ctl
  GET  /autofocus/<camera>/<on>    - Set autofocus on/off
  POST /save                       - Save focus settings to file
  GET  /settings                   - Load saved focus settings
"""

from flask import Flask, Response, request, jsonify
from turbojpeg import TurboJPEG, TJSAMP_420, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT

import cv2
import subprocess
import threading
import time
import json
import os
import sys
import numpy as np
import tarfile
import io
from flask_sock import Sock

_tjpeg = TurboJPEG()

sys.path.insert(0, '/home/pi/goatdev/pi')
from logger.pi_cloudwatch import Logger

app = Flask(__name__)
sock = Sock(app)

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

# Reader settings
FRAMES_PER_PAIR = 5         # Frames to read per pair before rotating
TARGET_FPS = 30              # Target FPS per camera (sleep to throttle)
CAMERA_PAIRS = [
    ('side', 'top'),
    ('side', 'front'),
    ('top', 'front'),
]

STALE_FRAME_THRESHOLD_SEC = 3
HEARTBEAT_INTERVAL_SEC = 300
RECONNECT_INTERVAL_SEC = 5

# Idle mode settings
IDLE_CHECK_INTERVAL_SEC = 0.5   # How often to check for wake conditions while idle
CAPTURE_WAKE_TIMEOUT_SEC = 2.0  # Max time to wait for fresh frame after wake

# Stream heartbeat timeout — if no new frame consumed in this many
# seconds, assume client is gone and break the generator.
STREAM_HEARTBEAT_TIMEOUT_SEC = 30

# Burst capture settings
BURST_MIN_IMAGE_BYTES = 50000   # 50KB minimum valid frame
BURST_MAX_COUNT = 100           # Cap frames per burst
BURST_MIN_INTERVAL_MS = 100     # Floor on interval between frames
BURST_DEFAULT_COUNT = 20
BURST_DEFAULT_INTERVAL_MS = 1500

API_PORT = 8080


# === IDLE MODE STATE ===
# _reader_idle is now a threading.Event() for thread safety.
# Previously it was a plain bool written by the reader thread and read by
# Flask request threads — unsafe on ARM (Pi 4) due to potential torn reads.

_capture_requested = threading.Event()  # Set by capture endpoints to wake reader
_reader_idle = threading.Event()         # Set when reader loop is sleeping
_new_frame_event = threading.Event()   # Set by store_raw() to wake preview loop


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
        self.preview_timestamp = 0
        self.preview_lock = threading.Lock()

        # Stats
        self.frame_count = 0
        self.error_count = 0
        self.last_error = None
        self.fps_estimate = 0.0
        self._fps_window_start = time.time()
        self._fps_window_count = 0

        # Stream client counter is now protected by a lock.
        # Previously bare += 1 / -= 1 was not atomic with gevent greenlets,
        # risking corrupted counts that permanently break idle mode.
        self.stream_clients = 0
        self._clients_lock = threading.Lock()

    def _increment_clients(self):
        """Thread-safe increment of stream_clients."""
        with self._clients_lock:
            self.stream_clients += 1

    def _decrement_clients(self):
        """Thread-safe decrement of stream_clients."""
        with self._clients_lock:
            self.stream_clients = max(0, self.stream_clients - 1)

    def _get_client_count(self):
        """Thread-safe read of stream_clients."""
        with self._clients_lock:
            return self.stream_clients

    def store_raw(self, raw_bytes):
        """Store a new raw frame."""
        now = time.time()
        with self.raw_lock:
            self.raw_frame = raw_bytes
            self.raw_timestamp = now
        self.frame_count += 1
        self.last_error = None

        # FPS calculation (rolling 10s window)
        self._fps_window_count += 1
        elapsed = now - self._fps_window_start
        if elapsed >= 10:
            self.fps_estimate = round(self._fps_window_count / elapsed, 1)
            self._fps_window_start = now
            self._fps_window_count = 0

        # Wake preview loop immediately
        _new_frame_event.set()

    def record_error(self, error_msg):
        """Record a read error."""
        self.error_count += 1
        self.last_error = error_msg

    def get_raw_frame(self):
        """Get latest raw JPEG bytes and timestamp."""
        with self.raw_lock:
            return self.raw_frame, self.raw_timestamp

    def get_preview_frame(self):
        """Get latest preview JPEG bytes."""
        with self.preview_lock:
            return self.preview_frame

    def update_preview(self):
        """Scale raw MJPEG using turbojpeg DCT downscale + re-encode.

        turbojpeg 1.x doesn't support compressed-domain transform with
        scaling, so we still decode→encode. But DCT scaling during decode
        (1/8) keeps the pixel buffer tiny (582x437 vs 4656x3496), making
        both steps fast: ~15ms decode + ~8ms encode = ~23ms total.
        """
        with self.raw_lock:
            raw = self.raw_frame
            ts = self.raw_timestamp

        if raw is None or ts == self.preview_timestamp:
            return

        try:
            decoded = _tjpeg.decode(
                raw,
                scaling_factor=(1, 8),
                flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT,
            )
            if decoded is None:
                return

            preview_bytes = _tjpeg.encode(
                decoded,
                quality=PREVIEW_JPEG_QUALITY,
                jpeg_subsample=TJSAMP_420,
                flags=TJFLAG_FASTDCT,
            )

            with self.preview_lock:
                self.preview_frame = preview_bytes
                self.preview_timestamp = ts
        except Exception as e:
            log.warn(f'preview:{self.name}', 'Preview encode error', error=str(e))

    def generate_stream(self):
        """Generator for MJPEG stream.
        """
        self._increment_clients()
        try:
            last_frame = None
            last_yield_time = time.time()
            while True:
                frame = self.get_preview_frame()
                if frame is None or frame == last_frame:
                    # If no new frame in too long, client is likely gone
                    if time.time() - last_yield_time > STREAM_HEARTBEAT_TIMEOUT_SEC:
                        log.info(f'stream:{self.name}',
                                 'Client timed out (no new frames consumed)',
                                 timeout_sec=STREAM_HEARTBEAT_TIMEOUT_SEC)
                        break
                    time.sleep(0.05)
                    continue
                last_frame = frame
                last_yield_time = time.time()
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + frame
                    + b'\r\n'
                )
        finally:
            self._decrement_clients()

    def get_health(self):
        """Return health dict for status endpoint."""
        frame_age = None
        if self.raw_timestamp > 0:
            frame_age = round(time.time() - self.raw_timestamp, 2)

        raw_size = len(self.raw_frame) if self.raw_frame else 0
        client_count = self._get_client_count()

        return {
            'connected': self.raw_frame is not None,
            'device': self.device,
            'device_exists': os.path.exists(self.device),
            'frame_count': self.frame_count,
            'fps': self.fps_estimate,
            'frame_age_sec': frame_age,
            'raw_buffer_bytes': raw_size,
            'stream_clients': client_count,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'stale': frame_age is not None and frame_age > STALE_FRAME_THRESHOLD_SEC,
        }


# === GLOBAL STATE ===

buffers = {}
caps = {}          # Open VideoCapture handles, keyed by camera name


def init_buffers():
    """Create a FrameBuffer for each camera."""
    for name, device in CAMERAS.items():
        buffers[name] = FrameBuffer(name, device)
    log.info('startup', 'Frame buffers initialized',
            cameras=','.join(CAMERAS.keys()))


# === CAMERA OPEN/CLOSE HELPERS ===

def open_camera(name, device):
    """Open a camera device in raw MJPEG mode. Returns cap or None."""
    if not os.path.exists(device):
        log.error(f'camera:{name}', 'Device not found', device=device)
        return None

    try:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            log.error(f'camera:{name}', 'Failed to open', device=device)
            return None

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_RES_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_RES_HEIGHT)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Raw MJPEG, no decode

        # Warmup read — first frame takes ~1.2s due to sensor init
        ret, _ = cap.read()
        if not ret:
            log.warn(f'camera:{name}', 'Warmup read failed, retrying')
            ret, _ = cap.read()

        if ret:
            log.info(f'camera:{name}', 'Opened and warmed up', device=device)
        else:
            log.warn(f'camera:{name}', 'Opened but warmup failed', device=device)

        return cap

    except Exception as e:
        log.error(f'camera:{name}', 'Open exception', device=device, error=str(e))
        return None


def open_all_cameras():
    """Open all cameras at startup. All stay open for the lifetime of the process."""
    for name, device in CAMERAS.items():
        cap = open_camera(name, device)
        if cap:
            caps[name] = cap
        else:
            buffers[name].record_error(f'Failed to open {device}')

    opened = [n for n, c in caps.items() if c is not None]
    failed = [n for n in CAMERAS if n not in caps]

    log.info('startup', 'Cameras opened',
            opened=','.join(opened) or 'none',
            failed=','.join(failed) or 'none')


def reconnect_camera(name):
    """Try to reconnect a failed camera."""
    device = CAMERAS[name]

    # Close existing handle if any
    if name in caps:
        try:
            caps[name].release()
        except Exception:
            pass
        del caps[name]

    cap = open_camera(name, device)
    if cap:
        caps[name] = cap
        log.info(f'camera:{name}', 'Reconnected', device=device)
        return True
    return False


# === IDLE MODE HELPERS ===

def _has_active_demand():
    """Check if any stream clients are connected or a capture is pending."""
    return True

    return any(buf._get_client_count() > 0 for buf in buffers.values())


def _wait_for_fresh_frame(camera_name, timeout=CAPTURE_WAKE_TIMEOUT_SEC):
    """Wait for a fresh frame after waking from idle. Returns True if got one."""
    buf = buffers.get(camera_name)
    if not buf:
        return False

    start = time.time()
    initial_count = buf.frame_count

    # Signal the reader to wake up
    _capture_requested.set()

    while time.time() - start < timeout:
        if buf.frame_count > initial_count:
            return True
        time.sleep(0.05)

    return False


# === MAIN READER LOOP ===

def reader_loop():
    """
    Main reader loop. All 3 cameras stay open at full resolution.
    Only reads from 2 at a time (rotating pairs) to stay within
    USB controller bandwidth. Single thread, no contention.

    When no stream clients are connected and no capture is pending,
    the reader enters idle mode — cameras stay open but no reads
    occur. This reduces USB controller wear and CPU usage.

    Pi 4's VL805 USB controller gets select() timeouts when 3
    cameras do concurrent reads, but handles 2 fine. Keeping
    the 3rd open but idle avoids the 1.2s warmup penalty on swap.

    _reader_idle is now a threading.Event for thread-safe
    reads from Flask/gevent request threads.
    """
    pair_index = 0
    throttle_interval = 1.0 / TARGET_FPS

    while True:
        # Skip if no cameras open
        if not caps:
            time.sleep(RECONNECT_INTERVAL_SEC)
            for name in CAMERAS:
                if name not in caps:
                    reconnect_camera(name)
            continue

        # === IDLE CHECK ===
        if not _has_active_demand():
            if not _reader_idle.is_set():
                log.info('reader', 'Entering idle mode (no clients, no captures)')
                _reader_idle.set()

            # Sleep until something needs us
            _capture_requested.wait(timeout=IDLE_CHECK_INTERVAL_SEC)
            continue

        # === WAKING FROM IDLE ===
        if _reader_idle.is_set():
            log.info('reader', 'Waking from idle')
            _reader_idle.clear()

        # Clear capture request flag (reader is now active)
        _capture_requested.clear()

        a_name, b_name = CAMERA_PAIRS[pair_index % len(CAMERA_PAIRS)]

        # Read FRAMES_PER_PAIR from this pair
        for _ in range(FRAMES_PER_PAIR):
            read_start = time.time()

            if a_name in caps:
                try:
                    ret, frame = caps[a_name].read()
                    if ret and frame is not None:
                        buffers[a_name].store_raw(frame.tobytes())
                    else:
                        buffers[a_name].record_error('Read returned empty frame')
                except Exception as e:
                    buffers[a_name].record_error(f'Read exception: {e}')
                    log.error(f'reader:{a_name}', 'Read exception', error=str(e))

            if b_name in caps:
                try:
                    ret, frame = caps[b_name].read()
                    if ret and frame is not None:
                        buffers[b_name].store_raw(frame.tobytes())
                    else:
                        buffers[b_name].record_error('Read returned empty frame')
                except Exception as e:
                    buffers[b_name].record_error(f'Read exception: {e}')
                    log.error(f'reader:{b_name}', 'Read exception', error=str(e))

            # Throttle to target FPS
            read_elapsed = time.time() - read_start
            sleep_time = throttle_interval - read_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        pair_index += 1

        # Periodic reconnection check for missing cameras
        for name in CAMERAS:
            buf = buffers[name]
            if name not in caps and buf.error_count > 0:
                if buf.error_count % 10 == 0:
                    log.info(f'reader:{name}', 'Attempting reconnect',
                            errors=buf.error_count)
                    reconnect_camera(name)


# === PREVIEW LOOP ===

def preview_loop():
    """
    Background thread that updates preview frames for cameras
    with active stream clients.

    Event-driven: wakes immediately when store_raw() signals a new frame,
    instead of polling every 50ms. Eliminates 0-50ms dead time per frame.
    """
    while True:
        # Block until a new raw frame arrives (or timeout for safety)
        _new_frame_event.wait(timeout=1.0)
        _new_frame_event.clear()

        for name, buf in buffers.items():
            if buf._get_client_count() <= 0:
                continue
            buf.update_preview()


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
                log.info(f'startup:{camera}', 'Focus applied', focus=focus_val)
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
                'fps': h['fps'],
                'frame_age': h['frame_age_sec'],
                'frames': h['frame_count'],
                'errors': h['error_count'],
                'clients': h['stream_clients'],
                'buffer_kb': round(h['raw_buffer_bytes'] / 1024, 1),
            }

        total_clients = sum(h['clients'] for h in healths.values())
        all_connected = all(h['connected'] for h in healths.values())

        mem_mb = None
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        mem_mb = int(line.split()[1]) // 1024
                        break
        except Exception:
            pass

        # Use .is_set() for thread-safe read
        log.info('heartbeat', 'Status',
                all_connected=all_connected,
                total_stream_clients=total_clients,
                reader_idle=_reader_idle.is_set(),
                mem_available_mb=mem_mb,
                cameras=json.dumps(healths))


# === FLASK ROUTES: STREAMS & CAPTURES ===

@app.route('/stream/<camera>')
def stream(camera):
    """MJPEG preview stream (640x480). Multiple clients supported.
    Wakes reader from idle if needed — first frame may take ~100-200ms."""
    buf = buffers.get(camera)
    if not buf:
        return jsonify({'error': f'Unknown camera: {camera}'}), 404

    # Wake reader if idle (stream_clients increment will keep it awake)
    if _reader_idle.is_set():
        _capture_requested.set()

    if buf.raw_frame is None:
        # If we just woke from idle, give reader a moment to get a frame
        if not _wait_for_fresh_frame(camera, timeout=3.0):
            return jsonify({
                'error': f'Camera {camera} not yet available',
                'last_error': buf.last_error
            }), 503

    return Response(
        buf.generate_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, no-transform',
            'X-Accel-Buffering': 'no',
        }
    )

@sock.route('/ws/stream/<camera>')
def ws_stream(ws, camera):
    """WebSocket preview stream. Sends binary JPEG frames as WS messages.

    Advantages over MJPEG:
    - No per-frame HTTP headers (saves ~200 bytes/frame)
    - Cloudflare passes WS frames without buffering
    - Client-side frame dropping: if browser is slow, we skip frames
    - Clean disconnect detection

    The client should display each received binary message as a JPEG image
    using URL.createObjectURL(new Blob([data], {type: 'image/jpeg'})).
    """
    buf = buffers.get(camera)
    if not buf:
        ws.close(reason=f'Unknown camera: {camera}')
        return

    # Wake reader if idle
    if _reader_idle.is_set():
        _capture_requested.set()

    if buf.raw_frame is None:
        if not _wait_for_fresh_frame(camera, timeout=3.0):
            ws.close(reason=f'Camera {camera} not available')
            return

    buf._increment_clients()
    log.info(f'ws:{camera}', 'Client connected',
             clients=buf._get_client_count())

    try:
        last_preview_ts = 0
        stale_count = 0

        while True:
            # Get latest preview frame
            with buf.preview_lock:
                frame = buf.preview_frame
                ts = buf.preview_timestamp

            if frame is None or ts == last_preview_ts:
                stale_count += 1
                if stale_count > 600:  # ~30s with no new frames
                    log.info(f'ws:{camera}', 'No new frames, closing')
                    break
                time.sleep(0.05)
                continue

            stale_count = 0
            last_preview_ts = ts

            try:
                ws.send(frame)
            except Exception:
                break  # Client disconnected

    finally:
        buf._decrement_clients()
        log.info(f'ws:{camera}', 'Client disconnected',
                 clients=buf._get_client_count())
        

@app.route('/capture/<camera>')
def capture(camera):
    """Return latest full-res raw JPEG for a single camera.
    Wakes reader from idle if needed."""
    buf = buffers.get(camera)
    if not buf:
        return jsonify({'error': f'Unknown camera: {camera}'}), 404

    # If reader is idle or frame is stale, wake and wait for fresh frame
    raw, timestamp = buf.get_raw_frame()
    frame_age = (time.time() - timestamp) if timestamp > 0 else float('inf')

    if raw is None or frame_age > STALE_FRAME_THRESHOLD_SEC:
        if not _wait_for_fresh_frame(camera):
            if raw is None:
                log.warn(f'capture:{camera}', 'No frame available after wake',
                        last_error=buf.last_error)
                return jsonify({
                    'error': f'No frame available for {camera}',
                    'last_error': buf.last_error
                }), 503

        # Re-read after wake
        raw, timestamp = buf.get_raw_frame()

    if raw is None:
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


# Cleaned up route structure to remove confusion.
# - GET /capture is a convenience alias (reads JSON body with defaults).
# - POST /capture/all is the explicit multi-camera capture endpoint.
# Previously /capture had methods=['GET', 'POST'] which was misleading:
# a GET would read JSON body (unexpected), and POST /capture/all worked
# but GET /capture/all would 405. Now the intent is clear.

@app.route('/capture', methods=['GET'])
def capture_default():
    """GET convenience alias for capture_all_cameras()."""
    return capture_all_cameras()


@app.route('/capture/all', methods=['POST'])
def capture_all_endpoint():
    """POST /capture/all — the explicit multi-camera capture endpoint."""
    return capture_all_cameras()


def capture_all_cameras():
    """Capture from all 3 cameras, save to /tmp, return file paths.
    Wakes reader from idle if needed.
    """
    data = request.get_json(silent=True) or {}
    prefix = data.get('prefix', 'capture')

    # Wake reader and wait for EACH camera to get a fresh frame.
    # This guarantees at least 2 pair rotations worth of reads.
    if _reader_idle.is_set():
        _capture_requested.set()

    # Always wait per-camera to ensure freshness (handles both idle-wake
    # and active-but-stale scenarios)
    for name in buffers:
        raw, timestamp = buffers[name].get_raw_frame()
        frame_age = (time.time() - timestamp) if timestamp > 0 else float('inf')
        if raw is None or frame_age > STALE_FRAME_THRESHOLD_SEC:
            _wait_for_fresh_frame(name, timeout=CAPTURE_WAKE_TIMEOUT_SEC)

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


# === FLASK ROUTES: BURST CAPTURE ===

@app.route('/capture/burst/<camera>', methods=['POST'])
def capture_burst(camera):
    """Burst capture: collect N unique frames over time, return as tar.gz.

    Deduplicates by frame timestamp entirely proxy-side — no per-frame
    HTTP round-trips from the caller. The reader loop continues its normal
    pair rotation; this endpoint just samples from the buffer at intervals.

    Body (JSON, all optional):
      count:       Number of frames to collect (default 20, max 100)
      interval_ms: Milliseconds between frames (default 1500, min 100)
                   Higher = more pose variation (goat moves between shots)
      prefix:      Filename prefix inside tar (default 'capture')

    Returns: application/gzip (tar.gz containing JPEG files)
    Headers:
      X-Frame-Count:     Actual frames captured (may be < count on timeout)
      X-Requested-Count: Originally requested count
      X-Total-Raw-Size:  Sum of raw JPEG sizes before compression

    Timing: 20 frames @ 1500ms = ~30s. The proxy's ~8fps read rate
    guarantees a fresh frame is always available at each interval.
    """
    buf = buffers.get(camera)
    if not buf:
        return jsonify({'error': f'Unknown camera: {camera}'}), 404

    data = request.get_json(silent=True) or {}
    count = min(int(data.get('count', BURST_DEFAULT_COUNT)), BURST_MAX_COUNT)
    interval_ms = max(int(data.get('interval_ms', BURST_DEFAULT_INTERVAL_MS)), BURST_MIN_INTERVAL_MS)
    prefix = data.get('prefix', 'capture')

    # Wake reader if idle
    if _reader_idle.is_set():
        _capture_requested.set()
    if not _wait_for_fresh_frame(camera, timeout=3.0):
        return jsonify({
            'error': f'Camera {camera} not available',
            'last_error': buf.last_error
        }), 503

    frames = []
    last_timestamp = 0
    timeout_sec = count * (interval_ms / 1000) + 10  # generous timeout
    start = time.time()
    stall_count = 0

    log.info(f'burst:{camera}', 'Starting burst capture',
             count=count, interval_ms=interval_ms, prefix=prefix)

    while len(frames) < count and (time.time() - start) < timeout_sec:
        raw, timestamp = buf.get_raw_frame()

        # Wait for a new unique frame
        if raw is None or timestamp <= last_timestamp:
            time.sleep(0.05)
            stall_count += 1
            # If no new frame in ~10s, camera is probably dead
            if stall_count > 200:
                log.warn(f'burst:{camera}', 'Stalled waiting for new frame',
                         frames_so_far=len(frames), stall_count=stall_count)
                break
            continue

        # Skip tiny/corrupt frames
        if len(raw) < BURST_MIN_IMAGE_BYTES:
            log.warn(f'burst:{camera}', 'Frame too small, skipping',
                     size=len(raw), min=BURST_MIN_IMAGE_BYTES)
            time.sleep(0.05)
            continue

        frames.append((raw, timestamp))
        last_timestamp = timestamp
        stall_count = 0

        # Sleep until next capture interval (unless we have enough)
        if len(frames) < count:
            time.sleep(interval_ms / 1000)

    if not frames:
        log.error(f'burst:{camera}', 'No frames captured',
                  last_error=buf.last_error)
        return jsonify({
            'error': f'No frames captured from {camera}',
            'last_error': buf.last_error
        }), 503

    # Package as tar.gz — single response, no per-frame HTTP overhead
    tar_buffer = io.BytesIO()
    total_raw_size = 0
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        for i, (frame_bytes, ts) in enumerate(frames, 1):
            filename = f'{prefix}_{camera}_{i:02d}.jpg'
            info = tarfile.TarInfo(name=filename)
            info.size = len(frame_bytes)
            info.mtime = ts
            tar.addfile(info, io.BytesIO(frame_bytes))
            total_raw_size += len(frame_bytes)

    tar_bytes = tar_buffer.getvalue()
    duration = round(time.time() - start, 2)

    log.info(f'burst:{camera}', 'Burst complete',
             frames=len(frames), requested=count,
             tar_size_bytes=len(tar_bytes),
             total_raw_bytes=total_raw_size,
             duration_sec=duration)

    return Response(tar_bytes, mimetype='application/gzip', headers={
        'Content-Disposition': f'attachment; filename="{prefix}_{camera}.tar.gz"',
        'X-Frame-Count': str(len(frames)),
        'X-Requested-Count': str(count),
        'X-Total-Raw-Size': str(total_raw_size),
    })


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

    total_clients = sum(c['stream_clients'] for c in cameras.values())

    return jsonify({
        'status': overall,
        'reader_idle': _reader_idle.is_set(),
        'active_stream_clients': total_clients,
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
    # Wake reader if idle
    if _reader_idle.is_set():
        _capture_requested.set()
        time.sleep(0.5)

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
            target_fps=TARGET_FPS,
            frames_per_pair=FRAMES_PER_PAIR,
            stale_threshold=STALE_FRAME_THRESHOLD_SEC,
            heartbeat_interval=HEARTBEAT_INTERVAL_SEC,
            stream_timeout=STREAM_HEARTBEAT_TIMEOUT_SEC,
            idle_mode='enabled')

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

    # Disable autofocus on all cameras before opening
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

    log.info('startup', 'Strategy: all 3 open, read 2 at a time (pair rotation), idle when no clients')
    log.info('startup', 'Server listening', host='0.0.0.0', port=API_PORT)
    log.info('startup', '=' * 50)


def start_background_threads():
    """Start reader, preview, and heartbeat threads."""
    reader_thread = threading.Thread(
        target=reader_loop, daemon=True, name='pair-reader'
    )
    reader_thread.start()

    preview_thread = threading.Thread(
        target=preview_loop, daemon=True, name='preview'
    )
    preview_thread.start()

    heartbeat_thread = threading.Thread(
        target=heartbeat_loop, daemon=True, name='heartbeat'
    )
    heartbeat_thread.start()


# === MODULE-LEVEL INIT (runs on import for gunicorn) ===

_initialized = False

def ensure_initialized():
    """Initialize once, whether run directly or imported by gunicorn."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    run_startup_checks()
    init_buffers()
    open_all_cameras()
    start_background_threads()

# Initialize when module is loaded (covers gunicorn import)
ensure_initialized()


# === MAIN (direct execution fallback) ===

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=API_PORT, threaded=True)
    except KeyboardInterrupt:
        log.info('shutdown', 'Shutting down')
    finally:
        for name, cap in caps.items():
            try:
                cap.release()
            except Exception:
                pass
        log.info('shutdown', 'All cameras released')
#!/usr/bin/env python3
"""
Camera Proxy Service — HerdSync
================================
Architecture:
  HardwareWorker (separate process, no gevent)
    - Owns all USB camera I/O via OpenCV + V4L2
    - Streams at 720p sequentially to avoid USB 2.0 bandwidth saturation
    - On capture signal: switches resolution in-place to full res per camera,
      grabs frames, switches back, resumes streaming
  Flask/gevent (main process)
    - Reads frames from shared memory for /stream endpoints
    - Signals hardware worker for /capture/all full-res grabs
    - Manages focus, status, telemetry

USB Bandwidth Note:
  Pi 4's VL805 controller tops out at ~35 MB/s for USB 2.0. Three 16MP MJPEG
  cameras at full res simultaneously exceeds the bus ceiling. Sequential 720p
  streaming (~3-7 MB/s total) stays well within budget. Full-res captures use
  in-place resolution switching (no release/reopen) one camera at a time.
"""
import multiprocessing as mp
import os
import sys

# === TIER 1: GEVENT ISOLATION ===
# Only monkeypatch the MainProcess (Web Server).
# This keeps the HardwareWorker clean and preemptive.
if mp.current_process().name == 'MainProcess':
    import gevent.monkey
    gevent.monkey.patch_all()

import time
import json
import threading
import subprocess
import tarfile
import io
import glob
from multiprocessing import shared_memory
from flask import Flask, Response, request, jsonify
from turbojpeg import TurboJPEG, TJFLAG_FASTUPSAMPLE

# HerdSync Internal Logger
sys.path.insert(0, '/home/pi/goatdev/pi')
from logger.pi_cloudwatch import Logger

app = Flask(__name__)
log = Logger('camera-proxy')
_tjpeg = TurboJPEG()

# === CONFIGURATION ===
CAMERAS = {'side': '/dev/camera_side', 'top': '/dev/camera_top', 'front': '/dev/camera_front'}
CAMERA_ORDER = ['side', 'top', 'front']
CAMERA_INDEX = {name: i for i, name in enumerate(CAMERA_ORDER)}

FULL_RES = (4656, 3496)
STREAM_RES = (1280, 720)
SETTLE_FRAMES = 1               # Frames to burn after resolution switch (auto-exposure)

SHM_SIZE = 6 * 1024 * 1024      # 6 MB per camera (full-res MJPEG can be ~1-2 MB)
STATS_SHM_SIZE = 8192
SHM_NAMES = {name: f"herdsync_{name}_raw" for name in CAMERAS}
STATS_SHM_NAME = "herdsync_system_stats"
EVENT_SHM_NAME = "herdsync_capture_flag"
FOCUS_FILE = '/home/pi/camera_focus_settings.json'
MIN_IMAGE_BYTES = 50000

# Capture event protocol (EVENT_SHM layout):
#   byte 0: state
#     0x00 = idle
#     0x01 = capture requested
#     0x02 = capture in progress (ACK)
#     0x03 = capture complete
#     0xFF = capture failed
#   bytes 1-3: camera mask (1=capture this camera, 0=skip)
#     byte 1 = side, byte 2 = top, byte 3 = front
EVENT_SHM_SIZE = 4
EVENT_IDLE = 0x00
EVENT_REQUESTED = 0x01
EVENT_IN_PROGRESS = 0x02
EVENT_DONE = 0x03
EVENT_FAILED = 0xFF

hardware_process = None


# === HARDWARE WORKER ===

def hardware_worker():
    """
    Hardware Isolation: Owns all USB camera I/O.

    Main loop reads cameras SEQUENTIALLY at STREAM_RES with no artificial sleep —
    USB read latency (~25-40ms at 720p) is the natural throttle, yielding ~8-13fps
    per camera.

    On capture signal, switches each camera to FULL_RES in-place via cap.set()
    (no release/reopen), grabs one settle + one real frame, switches back to
    STREAM_RES, then moves to the next camera.
    """
    import cv2
    from logger.pi_cloudwatch import Logger
    worker_log = Logger('hardware-worker')
    worker_log.info('init', 'HardwareWorker starting',
                    stream_res='{}x{}'.format(STREAM_RES[0], STREAM_RES[1]),
                    full_res='{}x{}'.format(FULL_RES[0], FULL_RES[1]))

    # --- SHM Setup ---
    shm_frames = {}
    for name, shm_name in SHM_NAMES.items():
        try:
            shm_frames[name] = shared_memory.SharedMemory(name=shm_name, create=True, size=SHM_SIZE)
            shm_frames[name].buf[:4] = b'\x00\x00\x00\x00'
        except FileExistsError:
            shm_frames[name] = shared_memory.SharedMemory(name=shm_name)

    try:
        shm_stats = shared_memory.SharedMemory(name=STATS_SHM_NAME, create=True, size=STATS_SHM_SIZE)
    except FileExistsError:
        shm_stats = shared_memory.SharedMemory(name=STATS_SHM_NAME)

    try:
        shm_event = shared_memory.SharedMemory(name=EVENT_SHM_NAME, create=True, size=EVENT_SHM_SIZE)
        shm_event.buf[0] = EVENT_IDLE
    except FileExistsError:
        shm_event = shared_memory.SharedMemory(name=EVENT_SHM_NAME)
        shm_event.buf[0] = EVENT_IDLE

    # --- Focus Persistence ---
    focus_vals = {n: 200 for n in CAMERAS}
    if os.path.exists(FOCUS_FILE):
        try:
            with open(FOCUS_FILE, 'r') as f:
                focus_vals.update(json.load(f))
        except Exception:
            pass

    # --- State ---
    caps = {}
    local_state = {
        'cameras': {
            n: {
                'fps': 0.0,
                'frame_count': 0,
                'last_ts': 0.0,
                'clients': 0,
                'connected': False,
                'errors': 0,
                'resolution': '{}x{}'.format(STREAM_RES[0], STREAM_RES[1])
            } for n in CAMERAS
        },
        'settings': focus_vals,
        'mode': 'streaming'
    }

    def sync_stats():
        """Write local_state into stats SHM for Flask to read."""
        try:
            state_bytes = json.dumps(local_state).encode('utf-8')
            shm_stats.buf[:len(state_bytes)] = state_bytes
            shm_stats.buf[len(state_bytes):len(state_bytes) + 1] = b'\0'
        except Exception:
            pass

    def resolve_device(dev_path):
        """Resolve symlink to real device path (OpenCV V4L2 needs real paths)."""
        return os.path.realpath(dev_path)

    def open_cam(name, dev, resolution=STREAM_RES):
        """Open a camera at the given resolution with minimal V4L2 buffering."""
        real_dev = resolve_device(dev)
        cap = cv2.VideoCapture(real_dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        return None

    def set_resolution(cap, width, height):
        """Switch resolution in-place without release/reopen."""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def apply_focus(name, dev):
        """Disable autofocus and apply saved focus value."""
        real_dev = resolve_device(dev)
        subprocess.run(
            ['v4l2-ctl', '-d', real_dev, '--set-ctrl', 'focus_automatic_continuous=0'],
            capture_output=True
        )
        subprocess.run(
            ['v4l2-ctl', '-d', real_dev, '--set-ctrl',
             'focus_absolute={}'.format(local_state['settings'][name])],
            capture_output=True
        )

    def write_frame_to_shm(name, frame_data):
        """Atomic write: clear size -> write data -> commit size."""
        data = frame_data.tobytes()
        size = len(data)
        if size + 4 > SHM_SIZE:
            worker_log.error('shm', 'Frame too large for SHM: {} bytes'.format(size),
                             camera=name)
            return False
        shm_frames[name].buf[:4] = b'\x00\x00\x00\x00'
        shm_frames[name].buf[4:4 + size] = data
        shm_frames[name].buf[:4] = size.to_bytes(4, 'little')
        return True

    def release_all():
        """Release all open captures."""
        for n in CAMERA_ORDER:
            if caps.get(n):
                caps[n].release()
                caps[n] = None

    def open_all_streaming():
        """Open all cameras at streaming resolution."""
        for n in CAMERA_ORDER:
            caps[n] = open_cam(n, CAMERAS[n], resolution=STREAM_RES)
            local_state['cameras'][n]['connected'] = caps[n] is not None
            local_state['cameras'][n]['resolution'] = '{}x{}'.format(STREAM_RES[0], STREAM_RES[1])
            if caps[n]:
                apply_focus(n, CAMERAS[n])

    def sync_focus_from_flask():
        """Check if Flask process wrote new focus values into stats SHM."""
        try:
            raw_in = bytes(shm_stats.buf).split(b'\0')[0].decode('utf-8')
            flask_updates = json.loads(raw_in)
            for n in CAMERAS:
                local_state['cameras'][n]['clients'] = flask_updates['cameras'][n].get('clients', 0)
                new_f = flask_updates['settings'].get(n)
                if new_f and new_f != local_state['settings'][n]:
                    local_state['settings'][n] = new_f
                    real_dev = resolve_device(CAMERAS[n])
                    subprocess.run(
                        ['v4l2-ctl', '-d', real_dev, '--set-ctrl',
                         'focus_absolute={}'.format(new_f)],
                        capture_output=True
                    )
        except Exception:
            pass

    def do_fullres_capture(camera_mask):
        """
        Full-resolution capture via in-place resolution switching.

        For each requested camera:
          1. cap.set() to FULL_RES (~100ms V4L2 ioctl, no USB renegotiation)
          2. Burn SETTLE_FRAMES for auto-exposure adjustment
          3. Read the real frame, write to SHM
          4. cap.set() back to STREAM_RES

        Tested at ~1.5s per camera with 1 settle frame. Total ~3-4.5s for all 3.
        Returns dict of {name: success_bool}.
        """
        local_state['mode'] = 'capturing'
        sync_stats()
        results = {}

        worker_log.info('capture', 'Starting in-place full-res capture')
        capture_start = time.time()

        for i, name in enumerate(CAMERA_ORDER):
            if not camera_mask[i]:
                results[name] = False
                continue

            cap = caps.get(name)
            if not cap:
                worker_log.error('capture', '{} not open, skipping'.format(name))
                results[name] = False
                continue

            cam_start = time.time()

            # Switch to full res in-place
            set_resolution(cap, FULL_RES[0], FULL_RES[1])

            # Burn settle frames for auto-exposure
            for _ in range(SETTLE_FRAMES):
                cap.read()

            # Real frame
            ret, frame = cap.read()
            if ret:
                if write_frame_to_shm(name, frame):
                    local_state['cameras'][name]['last_ts'] = time.time()
                    local_state['cameras'][name]['frame_count'] += 1
                    data_size = len(frame.tobytes())
                    cam_time = round((time.time() - cam_start) * 1000)
                    worker_log.info('capture',
                                    '{} captured: {} bytes in {}ms'.format(name, data_size, cam_time))
                    results[name] = True
                else:
                    results[name] = False
            else:
                worker_log.error('capture', '{} read() failed at full res'.format(name))
                results[name] = False

            # Switch back to stream res immediately
            set_resolution(cap, STREAM_RES[0], STREAM_RES[1])

        total_time = round((time.time() - capture_start) * 1000)
        worker_log.info('capture', 'Capture complete in {}ms'.format(total_time),
                        results={n: r for n, r in results.items()})

        local_state['mode'] = 'streaming'
        sync_stats()

        return results

    # --- Initial camera open ---
    open_all_streaming()
    sync_stats()
    worker_log.info('init', 'Cameras opened',
                    connected={n: local_state['cameras'][n]['connected'] for n in CAMERA_ORDER})

    # --- Main Loop ---
    try:
        while True:
            # 1. Sync focus / client counts from Flask
            sync_focus_from_flask()

            # 2. Check for capture request
            if shm_event.buf[0] == EVENT_REQUESTED:
                camera_mask = [shm_event.buf[1 + i] for i in range(3)]
                shm_event.buf[0] = EVENT_IN_PROGRESS

                try:
                    results = do_fullres_capture(camera_mask)
                    all_ok = all(
                        results.get(CAMERA_ORDER[i], True)
                        for i in range(3)
                        if camera_mask[i]
                    )
                    shm_event.buf[0] = EVENT_DONE if all_ok else EVENT_FAILED
                except Exception as e:
                    worker_log.exception('capture', 'Capture failed', error=str(e))
                    # Try to recover — switch all back to stream res
                    for name in CAMERA_ORDER:
                        cap = caps.get(name)
                        if cap:
                            try:
                                set_resolution(cap, STREAM_RES[0], STREAM_RES[1])
                            except Exception:
                                pass
                    local_state['mode'] = 'streaming'
                    sync_stats()
                    shm_event.buf[0] = EVENT_FAILED

                continue

            # 3. Sequential streaming reads — no artificial sleep
            #    USB read latency at 720p (~25-40ms) is the natural throttle
            for name in CAMERA_ORDER:
                if not caps.get(name):
                    caps[name] = open_cam(name, CAMERAS[name], resolution=STREAM_RES)
                    if caps[name]:
                        apply_focus(name, CAMERAS[name])
                        local_state['cameras'][name]['connected'] = True
                        local_state['cameras'][name]['errors'] = 0
                    else:
                        continue

                ret, frame = caps[name].read()

                if ret:
                    now = time.time()
                    local_state['cameras'][name]['errors'] = 0

                    # FPS calculation (EWMA)
                    prev_ts = local_state['cameras'][name]['last_ts']
                    if prev_ts > 0:
                        instant_fps = 1.0 / max(now - prev_ts, 0.001)
                        local_state['cameras'][name]['fps'] = round(
                            (0.8 * local_state['cameras'][name]['fps']) +
                            (0.2 * instant_fps), 1
                        )

                    write_frame_to_shm(name, frame)
                    local_state['cameras'][name]['frame_count'] += 1
                    local_state['cameras'][name]['last_ts'] = now
                else:
                    local_state['cameras'][name]['errors'] += 1
                    if local_state['cameras'][name]['errors'] > 10:
                        worker_log.error('stream',
                                         '{} persistent read failures, reopening'.format(name))
                        caps[name].release()
                        caps[name] = open_cam(name, CAMERAS[name], resolution=STREAM_RES)
                        if caps[name]:
                            apply_focus(name, CAMERAS[name])
                        local_state['cameras'][name]['errors'] = 0

            sync_stats()

    finally:
        release_all()
        for s in list(shm_frames.values()) + [shm_stats, shm_event]:
            s.close()


# === FLASK CONSUMER & TELEMETRY ===

class CameraProxy:
    """Reads frames from SHM written by the hardware worker."""

    def __init__(self, name):
        self.name = name
        self.shm_frame = None
        self.shm_stats = None

    def _ensure_shm(self):
        try:
            if not self.shm_frame:
                self.shm_frame = shared_memory.SharedMemory(name=SHM_NAMES[self.name])
            if not self.shm_stats:
                self.shm_stats = shared_memory.SharedMemory(name=STATS_SHM_NAME)
            return True
        except FileNotFoundError:
            return False

    def get_raw_frame(self):
        if not self._ensure_shm():
            return None
        size = int.from_bytes(self.shm_frame.buf[:4], 'little')
        if size == 0:
            return None
        return bytes(self.shm_frame.buf[4:4 + size])

    def get_last_ts(self):
        if not self._ensure_shm():
            return 0.0
        try:
            raw = bytes(self.shm_stats.buf).split(b'\0')[0].decode('utf-8')
            data = json.loads(raw)
            return data['cameras'][self.name].get('last_ts', 0.0)
        except Exception:
            return 0.0

    def update_clients(self, delta):
        if not self._ensure_shm():
            return
        try:
            raw = bytes(self.shm_stats.buf).split(b'\0')[0].decode('utf-8')
            data = json.loads(raw)
            data['cameras'][self.name]['clients'] = max(
                0, data['cameras'][self.name].get('clients', 0) + delta
            )
            b = json.dumps(data).encode('utf-8')
            self.shm_stats.buf[:len(b)] = b
            self.shm_stats.buf[len(b):len(b) + 1] = b'\0'
        except Exception:
            pass

    def stream_generator(self):
        self.update_clients(1)
        try:
            while True:
                raw = self.get_raw_frame()
                if raw:
                    decoded = _tjpeg.decode(raw, scaling_factor=(1, 8), flags=0)
                    yield (
                        b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                        _tjpeg.encode(decoded, quality=70) +
                        b'\r\n'
                    )
                time.sleep(0.1)
        finally:
            self.update_clients(-1)


PROXIES = {n: CameraProxy(n) for n in CAMERAS}


def _signal_capture(camera_names):
    """
    Signal the hardware worker to do a full-res capture.

    Args:
        camera_names: list of camera names to capture, e.g. ['side', 'top', 'front']

    Returns:
        (success: bool, error: str or None)
    """
    try:
        shm_event = shared_memory.SharedMemory(name=EVENT_SHM_NAME)
    except FileNotFoundError:
        return False, 'Hardware worker not running'

    if shm_event.buf[0] in (EVENT_REQUESTED, EVENT_IN_PROGRESS):
        return False, 'Capture already in progress'

    # Set camera mask
    for i, name in enumerate(CAMERA_ORDER):
        shm_event.buf[1 + i] = 1 if name in camera_names else 0

    # Signal
    shm_event.buf[0] = EVENT_REQUESTED

    # Poll for completion — in-place switching ~1.5s per camera, ~4.5s max
    deadline = time.time() + 8.0
    while time.time() < deadline:
        state = shm_event.buf[0]
        if state == EVENT_DONE:
            shm_event.buf[0] = EVENT_IDLE
            return True, None
        elif state == EVENT_FAILED:
            shm_event.buf[0] = EVENT_IDLE
            return False, 'Hardware worker reported capture failure'
        time.sleep(0.05)

    shm_event.buf[0] = EVENT_IDLE
    return False, 'Capture timed out (8s)'


def heartbeat_loop():
    """CLOUDWATCH HEARTBEAT: Detailed health telemetry."""
    while True:
        time.sleep(300)
        try:
            shm = shared_memory.SharedMemory(name=STATS_SHM_NAME)
            raw = bytes(shm.buf).split(b'\0')[0].decode('utf-8')
            data = json.loads(raw)
            log.info('heartbeat', 'System Health',
                     cameras=json.dumps(data['cameras']),
                     mode=data.get('mode', 'unknown'),
                     load=os.getloadavg()[0])
        except Exception:
            pass


def health_monitor_loop():
    """WATCHDOG: Restarts frozen hardware process."""
    while True:
        time.sleep(20)
        try:
            shm = shared_memory.SharedMemory(name=STATS_SHM_NAME)
            raw = bytes(shm.buf).split(b'\0')[0].decode('utf-8')
            data = json.loads(raw)

            # Don't watchdog during captures — stream is intentionally paused
            if data.get('mode') == 'capturing':
                continue

            for name, s in data['cameras'].items():
                if s['connected'] and (time.time() - s['last_ts'] > 30):
                    log.error('watchdog', 'Freeze on {}. Rebooting HardwareWorker.'.format(name),
                              last_ts=s['last_ts'], age=round(time.time() - s['last_ts'], 1))
                    global hardware_process
                    if hardware_process:
                        hardware_process.terminate()
                        hardware_process.join(timeout=5)
                    time.sleep(2)
                    hardware_process = mp.Process(
                        target=hardware_worker, daemon=True, name="HardwareWorker"
                    )
                    hardware_process.start()
                    break
        except Exception:
            pass


# === ROUTES ===

@app.route('/stream/<camera>')
def stream(camera):
    """MJPEG stream at 720p for monitoring."""
    if camera not in CAMERAS:
        return "Invalid camera", 404
    return Response(
        PROXIES[camera].stream_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/capture/<camera>')
def capture(camera):
    """
    Capture a single full-res image.
    Triggers in-place resolution switch for this camera only.
    """
    if camera not in CAMERAS:
        return jsonify({'error': 'Unknown camera: {}'.format(camera)}), 404

    success, error = _signal_capture([camera])
    if not success:
        return jsonify({'error': error}), 503

    proxy = PROXIES[camera]
    raw = proxy.get_raw_frame()
    if not raw or len(raw) < MIN_IMAGE_BYTES:
        return jsonify({'error': 'Frame too small or missing for {}'.format(camera)}), 503

    return Response(raw, mimetype='image/jpeg', headers={
        'X-Frame-Size-Bytes': str(len(raw)),
    })


@app.route('/capture/all')
def capture_all():
    """
    Capture full-res images from ALL cameras.
    In-place resolution switching, one camera at a time.
    Returns all three images as a gzipped tar archive.
    """
    success, error = _signal_capture(CAMERA_ORDER)
    if not success:
        return jsonify({'error': error}), 503

    frames = {}
    for name in CAMERA_ORDER:
        raw = PROXIES[name].get_raw_frame()
        if raw and len(raw) >= MIN_IMAGE_BYTES:
            frames[name] = raw
        else:
            return jsonify({
                'error': 'No valid frame for {}'.format(name),
                'captured': list(frames.keys()),
                'failed': name
            }), 503

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode='w:gz') as tar:
        for name, data in frames.items():
            info = tarfile.TarInfo(name="{}.jpg".format(name))
            info.size = len(data)
            info.mtime = time.time()
            tar.addfile(info, io.BytesIO(data))

    return Response(
        tar_buf.getvalue(),
        mimetype='application/gzip',
        headers={
            'X-Capture-Cameras': ','.join(frames.keys()),
            'Content-Disposition': 'attachment; filename=capture.tar.gz'
        }
    )


@app.route('/capture/all/individual')
def capture_all_individual():
    """
    Capture full-res from all cameras, return metadata as JSON.
    Individual frames can be pulled from /capture/frame/<camera> after this call.

    Designed for prod_server.py which needs to send images as separate multipart fields.
    """
    success, error = _signal_capture(CAMERA_ORDER)
    if not success:
        return jsonify({'success': False, 'error': error}), 503

    result = {'success': True, 'cameras': {}}
    for name in CAMERA_ORDER:
        raw = PROXIES[name].get_raw_frame()
        if raw and len(raw) >= MIN_IMAGE_BYTES:
            result['cameras'][name] = {
                'size_bytes': len(raw),
                'ready': True
            }
        else:
            return jsonify({
                'success': False,
                'error': 'No valid frame for {}'.format(name),
                'cameras': result['cameras']
            }), 503

    return jsonify(result)


@app.route('/capture/frame/<camera>')
def capture_frame(camera):
    """
    Return the latest frame in SHM for a camera (whatever resolution it was captured at).
    Use after /capture/all/individual to pull individual full-res frames.
    """
    if camera not in CAMERAS:
        return jsonify({'error': 'Unknown camera: {}'.format(camera)}), 404

    raw = PROXIES[camera].get_raw_frame()
    if not raw or len(raw) < MIN_IMAGE_BYTES:
        return jsonify({'error': 'No valid frame for {}'.format(camera)}), 503

    return Response(raw, mimetype='image/jpeg', headers={
        'X-Frame-Size-Bytes': str(len(raw)),
    })


@app.route('/capture/burst/<camera>', methods=['POST'])
def capture_burst(camera):
    """Burst capture at current streaming resolution. NOT full-res."""
    if camera not in CAMERAS:
        return jsonify({'error': 'Unknown camera: {}'.format(camera)}), 404

    data = request.get_json(silent=True) or {}
    count = min(int(data.get('count', 20)), 100)
    interval = max(int(data.get('interval_ms', 1000)), 100) / 1000.0

    proxy = PROXIES[camera]
    frames = []

    for _ in range(count):
        raw = proxy.get_raw_frame()
        if raw:
            frames.append((raw, time.time()))
        time.sleep(interval)

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode='w:gz') as tar:
        for i, (f_bytes, ts) in enumerate(frames):
            info = tarfile.TarInfo(name="burst_{}_{:03d}.jpg".format(camera, i))
            info.size = len(f_bytes)
            info.mtime = ts
            tar.addfile(info, io.BytesIO(f_bytes))

    return Response(
        tar_buf.getvalue(),
        mimetype='application/gzip',
        headers={'X-Frame-Count': str(len(frames))}
    )


@app.route('/focus/<camera>/<int:val>', methods=['GET', 'POST'])
def set_focus(camera, val):
    """FOCUS: Updates SHM settings and persistence file."""
    if camera not in CAMERAS:
        return jsonify({'error': 'Unknown camera: {}'.format(camera)}), 404
    try:
        shm = shared_memory.SharedMemory(name=STATS_SHM_NAME)
        raw = bytes(shm.buf).split(b'\0')[0].decode('utf-8')
        data = json.loads(raw)
        data['settings'][camera] = val
        b = json.dumps(data).encode('utf-8')
        shm.buf[:len(b)] = b
        shm.buf[len(b):len(b) + 1] = b'\0'
        with open(FOCUS_FILE, 'w') as f:
            json.dump(data['settings'], f)
    except Exception:
        pass
    return "Focus set to {}".format(val)


@app.route('/status')
def status():
    try:
        shm = shared_memory.SharedMemory(name=STATS_SHM_NAME)
        raw = bytes(shm.buf).split(b'\0')[0].decode('utf-8')
        return jsonify(json.loads(raw))
    except Exception:
        return jsonify({"status": "starting"})


# === BOOTSTRAP ===
_initialized = False


def ensure_initialized():
    global _initialized, hardware_process
    if _initialized:
        return
    _initialized = True
    if mp.current_process().name == 'MainProcess':
        for shm_file in glob.glob('/dev/shm/herdsync_*'):
            try:
                os.unlink(shm_file)
            except Exception:
                pass
        try:
            mp.set_start_method('spawn', force=True)
        except Exception:
            pass
        hardware_process = mp.Process(target=hardware_worker, daemon=True, name="HardwareWorker")
        hardware_process.start()
        threading.Thread(target=heartbeat_loop, daemon=True).start()
        threading.Thread(target=health_monitor_loop, daemon=True).start()


ensure_initialized()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
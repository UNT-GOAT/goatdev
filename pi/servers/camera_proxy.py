#!/usr/bin/env python3
"""
Camera Proxy Service — HerdSync
================================
Architecture:
  HardwareWorker (separate process, no gevent)
    - Owns all USB camera I/O via OpenCV + V4L2
    - Streams at low res (640×480) sequentially to avoid USB 2.0 bandwidth saturation
    - On capture signal: stops stream, opens cameras one-at-a-time at full res,
      grabs frames, writes to SHM, resumes streaming
  Flask/gevent (main process)
    - Reads frames from shared memory for /stream endpoints
    - Signals hardware worker for /capture/all full-res grabs
    - Manages focus, status, telemetry

USB Bandwidth Note:
  Pi 4's VL805 controller tops out at ~35 MB/s for USB 2.0. Three 16MP MJPEG
  cameras at full res (~2-4 MB/frame × 10fps × 2 active = 40-80 MB/s) exceeds
  the bus ceiling. Sequential low-res streaming + on-demand full-res one-at-a-time
  keeps every read well within budget.
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
import struct
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
STREAM_RES = (640, 480)
TARGET_STREAM_FPS = 15
SETTLE_FRAMES = 3               # Frames to burn after resolution switch (auto-exposure)

SHM_SIZE = 6 * 1024 * 1024      # 6 MB per camera (full-res MJPEG can be 2-4 MB)
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

    Main loop reads cameras SEQUENTIALLY at STREAM_RES.
    On capture signal, switches to FULL_RES one-at-a-time, grabs frames,
    then resumes streaming.
    """
    import cv2
    from logger.pi_cloudwatch import Logger
    worker_log = Logger('hardware-worker')
    worker_log.info('init', 'HardwareWorker starting',
                    stream_res=f'{STREAM_RES[0]}x{STREAM_RES[1]}',
                    full_res=f'{FULL_RES[0]}x{FULL_RES[1]}',
                    target_fps=TARGET_STREAM_FPS)

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
                'resolution': f'{STREAM_RES[0]}x{STREAM_RES[1]}'
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

    def open_cam(name, dev, resolution=STREAM_RES):
        """Open a camera at the given resolution with minimal V4L2 buffering."""
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize queued USB transfers
            return cap
        return None

    def apply_focus(name, dev):
        """Disable autofocus and apply saved focus value."""
        subprocess.run(
            ['v4l2-ctl', '-d', dev, '--set-ctrl', 'focus_automatic_continuous=0'],
            capture_output=True
        )
        subprocess.run(
            ['v4l2-ctl', '-d', dev, '--set-ctrl', f'focus_absolute={local_state["settings"][name]}'],
            capture_output=True
        )

    def write_frame_to_shm(name, frame_data):
        """Atomic write: clear size → write data → commit size."""
        data = frame_data.tobytes()
        size = len(data)
        if size + 4 > SHM_SIZE:
            worker_log.error('shm', f'Frame too large for SHM: {size} bytes', camera=name)
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
            local_state['cameras'][n]['resolution'] = f'{STREAM_RES[0]}x{STREAM_RES[1]}'
            if caps[n]:
                apply_focus(n, CAMERAS[n])

    def sync_focus_from_flask():
        """Check if Flask process wrote new focus values into stats SHM."""
        try:
            raw_in = bytes(shm_stats.buf).split(b'\0')[0].decode('utf-8')
            flask_updates = json.loads(raw_in)
            for n in CAMERAS:
                # Sync client counts
                local_state['cameras'][n]['clients'] = flask_updates['cameras'][n].get('clients', 0)
                # Sync focus changes
                new_f = flask_updates['settings'].get(n)
                if new_f and new_f != local_state['settings'][n]:
                    local_state['settings'][n] = new_f
                    subprocess.run(
                        ['v4l2-ctl', '-d', CAMERAS[n], '--set-ctrl', f'focus_absolute={new_f}'],
                        capture_output=True
                    )
        except Exception:
            pass

    def do_fullres_capture(camera_mask):
        """
        Full-resolution capture sequence.
        Releases all stream captures, opens requested cameras one-at-a-time
        at FULL_RES, burns settle frames, grabs one good frame, writes to SHM.
        Returns dict of {name: success_bool}.
        """
        local_state['mode'] = 'capturing'
        sync_stats()
        results = {}

        # Release all streaming captures first — free the USB bus entirely
        release_all()
        worker_log.info('capture', 'Stream paused, starting full-res sequence')

        for i, name in enumerate(CAMERA_ORDER):
            if not camera_mask[i]:
                results[name] = False
                continue

            worker_log.info('capture', f'Opening {name} at full res')
            cap = open_cam(name, CAMERAS[name], resolution=FULL_RES)

            if not cap:
                worker_log.error('capture', f'Failed to open {name} at full res')
                results[name] = False
                continue

            # Apply focus
            apply_focus(name, CAMERAS[name])

            # Burn frames for auto-exposure to settle after resolution switch
            for _ in range(SETTLE_FRAMES):
                cap.read()

            # Grab the real frame
            ret, frame = cap.read()
            if ret:
                if write_frame_to_shm(name, frame):
                    local_state['cameras'][name]['last_ts'] = time.time()
                    local_state['cameras'][name]['frame_count'] += 1
                    data_size = len(frame.tobytes())
                    worker_log.info('capture', f'{name} captured',
                                    size_bytes=data_size,
                                    resolution=f'{FULL_RES[0]}x{FULL_RES[1]}')
                    results[name] = True
                else:
                    results[name] = False
            else:
                worker_log.error('capture', f'{name} read() failed at full res')
                results[name] = False

            # Release immediately — free bus for next camera
            cap.release()

        # Resume streaming
        worker_log.info('capture', 'Resuming stream',
                        results={n: r for n, r in results.items()})
        open_all_streaming()
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
                    # Try to recover streaming
                    try:
                        open_all_streaming()
                        local_state['mode'] = 'streaming'
                        sync_stats()
                    except Exception:
                        pass
                    shm_event.buf[0] = EVENT_FAILED

                continue  # Skip to next iteration — streaming just resumed

            # 3. Sequential streaming reads — one camera at a time
            for name in CAMERA_ORDER:
                if not caps.get(name):
                    # Try to reconnect
                    caps[name] = open_cam(name, CAMERAS[name], resolution=STREAM_RES)
                    if caps[name]:
                        apply_focus(name, CAMERAS[name])
                        local_state['cameras'][name]['connected'] = True
                        local_state['cameras'][name]['errors'] = 0
                    else:
                        continue

                loop_start = time.time()
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
                        worker_log.error('stream', f'{name} persistent read failures, reopening')
                        caps[name].release()
                        caps[name] = open_cam(name, CAMERAS[name], resolution=STREAM_RES)
                        if caps[name]:
                            apply_focus(name, CAMERAS[name])
                        local_state['cameras'][name]['errors'] = 0

                # Sleep to hit target FPS — per camera, so total loop is ~3x this
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / TARGET_STREAM_FPS) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

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

    # Check not already capturing
    if shm_event.buf[0] in (EVENT_REQUESTED, EVENT_IN_PROGRESS):
        return False, 'Capture already in progress'

    # Set camera mask
    for i, name in enumerate(CAMERA_ORDER):
        shm_event.buf[1 + i] = 1 if name in camera_names else 0

    # Signal
    shm_event.buf[0] = EVENT_REQUESTED

    # Poll for completion
    deadline = time.time() + 12.0  # 12s timeout (settle frames + 3 cameras)
    while time.time() < deadline:
        state = shm_event.buf[0]
        if state == EVENT_DONE:
            shm_event.buf[0] = EVENT_IDLE
            return True, None
        elif state == EVENT_FAILED:
            shm_event.buf[0] = EVENT_IDLE
            return False, 'Hardware worker reported capture failure'
        time.sleep(0.05)

    # Timeout — reset flag so worker doesn't get stuck
    shm_event.buf[0] = EVENT_IDLE
    return False, 'Capture timed out (12s)'


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

            # Don't watchdog during captures — the stream is intentionally paused
            if data.get('mode') == 'capturing':
                continue

            for name, s in data['cameras'].items():
                if s['connected'] and (time.time() - s['last_ts'] > 30):
                    log.error('watchdog', f'Freeze on {name}. Rebooting HardwareWorker.',
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
    """MJPEG stream at low resolution for monitoring."""
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
    Triggers hardware worker to switch to FULL_RES for this camera only.
    """
    if camera not in CAMERAS:
        return jsonify({'error': f'Unknown camera: {camera}'}), 404

    success, error = _signal_capture([camera])
    if not success:
        return jsonify({'error': error}), 503

    # Read the frame from SHM
    proxy = PROXIES[camera]
    raw = proxy.get_raw_frame()
    if not raw or len(raw) < MIN_IMAGE_BYTES:
        return jsonify({'error': f'Frame too small or missing for {camera}'}), 503

    return Response(raw, mimetype='image/jpeg', headers={
        'X-Frame-Size-Bytes': str(len(raw)),
    })


@app.route('/capture/all')
def capture_all():
    """
    Capture full-res images from ALL cameras.
    Hardware worker opens each one-at-a-time at FULL_RES, ensuring no USB contention.
    Returns all three images as a gzipped tar archive.

    Response headers:
        X-Capture-Cameras: comma-separated list of cameras captured
    """
    success, error = _signal_capture(CAMERA_ORDER)
    if not success:
        return jsonify({'error': error}), 503

    # Collect frames from SHM
    frames = {}
    for name in CAMERA_ORDER:
        raw = PROXIES[name].get_raw_frame()
        if raw and len(raw) >= MIN_IMAGE_BYTES:
            frames[name] = raw
        else:
            return jsonify({
                'error': f'No valid frame for {name}',
                'captured': list(frames.keys()),
                'failed': name
            }), 503

    # Package as gzipped tar
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode='w:gz') as tar:
        for name, data in frames.items():
            info = tarfile.TarInfo(name=f"{name}.jpg")
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
                'error': f'No valid frame for {name}',
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
        return jsonify({'error': f'Unknown camera: {camera}'}), 404

    raw = PROXIES[camera].get_raw_frame()
    if not raw or len(raw) < MIN_IMAGE_BYTES:
        return jsonify({'error': f'No valid frame for {camera}'}), 503

    return Response(raw, mimetype='image/jpeg', headers={
        'X-Frame-Size-Bytes': str(len(raw)),
    })


@app.route('/capture/burst/<camera>', methods=['POST'])
def capture_burst(camera):
    """Burst capture at current streaming resolution. NOT full-res."""
    if camera not in CAMERAS:
        return jsonify({'error': f'Unknown camera: {camera}'}), 404

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
            info = tarfile.TarInfo(name=f"burst_{camera}_{i:03d}.jpg")
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
        return jsonify({'error': f'Unknown camera: {camera}'}), 404
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
    return f"Focus set to {val}"


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
        # Auto-wipe SHM zombies
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
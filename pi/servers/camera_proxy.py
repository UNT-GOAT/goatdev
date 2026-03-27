#!/usr/bin/env python3
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
FULL_RES = (4656, 3496)
TARGET_FPS = 10
FRAMES_PER_PAIR = 10
SHM_SIZE = 2 * 1024 * 1024  
STATS_SHM_SIZE = 8192       
SHM_NAMES = {name: f"herdsync_{name}_raw" for name in CAMERAS}
STATS_SHM_NAME = "herdsync_system_stats"
EVENT_SHM_NAME = "herdsync_capture_flag"
FOCUS_FILE = '/home/pi/camera_focus_settings.json'

hardware_process = None

# === HARDWARE WORKER (RECOVERY ENGINE) ===

def hardware_worker():
    """Hardware Isolation: Handles USB logic and Focus persistence."""
    import cv2
    from logger.pi_cloudwatch import Logger
    worker_log = Logger('hardware-worker')

    # 1. SHM Setup
    shm_frames = {}
    for name, shm_name in SHM_NAMES.items():
        try:
            shm_frames[name] = shared_memory.SharedMemory(name=shm_name, create=True, size=SHM_SIZE)
            shm_frames[name].buf[:4] = b'\x00\x00\x00\x00'
        except FileExistsError:
            shm_frames[name] = shared_memory.SharedMemory(name=shm_name)

    try: shm_stats = shared_memory.SharedMemory(name=STATS_SHM_NAME, create=True, size=STATS_SHM_SIZE)
    except FileExistsError: shm_stats = shared_memory.SharedMemory(name=STATS_SHM_NAME)
    
    try:
        shm_event = shared_memory.SharedMemory(name=EVENT_SHM_NAME, create=True, size=1)
        shm_event.buf[0] = 0
    except FileExistsError:
        shm_event = shared_memory.SharedMemory(name=EVENT_SHM_NAME)

    # 2. Focus Persistence
    focus_vals = {n: 200 for n in CAMERAS}
    if os.path.exists(FOCUS_FILE):
        try:
            with open(FOCUS_FILE, 'r') as f: focus_vals.update(json.load(f))
        except: pass

    caps = {}
    local_state = {
        'cameras': {n: {'fps': 0.0, 'frame_count': 0, 'last_ts': 0.0, 'clients': 0, 'connected': False, 'errors': 0} for n in CAMERAS},
        'settings': focus_vals
    }

    def sync_stats():
        try:
            state_bytes = json.dumps(local_state).encode('utf-8')
            shm_stats.buf[:len(state_bytes)] = state_bytes
            shm_stats.buf[len(state_bytes):len(state_bytes)+1] = b'\0'
        except: pass

    def open_cam(name, dev):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_RES[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_RES[1])
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            return cap
        return None

    # Initial Open & Focus Apply
    for n, d in CAMERAS.items():
        caps[n] = open_cam(n, d)
        local_state['cameras'][n]['connected'] = caps[n] is not None
        if caps[n]:
            subprocess.run(['v4l2-ctl', '-d', d, '--set-ctrl', 'focus_automatic_continuous=0'], capture_output=True)
            subprocess.run(['v4l2-ctl', '-d', d, '--set-ctrl', f'focus_absolute={local_state["settings"][n]}'], capture_output=True)
    sync_stats()

    pairs = [('side', 'top'), ('side', 'front'), ('top', 'front')]
    
    try:
        while True:
            # Sync clients and focus updates from Flask process
            try:
                raw_in = bytes(shm_stats.buf).split(b'\0')[0].decode('utf-8')
                flask_updates = json.loads(raw_in)
                for n in CAMERAS:
                    local_state['cameras'][n]['clients'] = flask_updates['cameras'][n].get('clients', 0)
                    new_f = flask_updates['settings'].get(n)
                    if new_f and new_f != local_state['settings'][n]:
                        local_state['settings'][n] = new_f
                        subprocess.run(['v4l2-ctl', '-d', CAMERAS[n], '--set-ctrl', f'focus_absolute={new_f}'], capture_output=True)
            except: pass

            has_demand = any(local_state['cameras'][n]['clients'] > 0 for n in CAMERAS) or shm_event.buf[0] == 1
            if not has_demand:
                time.sleep(0.5); continue

            for pair in pairs:
                for _ in range(FRAMES_PER_PAIR):
                    loop_start = time.time()
                    for name in pair:
                        if caps.get(name):
                            ret, frame = caps[name].read()
                            if ret:
                                now = time.time()
                                local_state['cameras'][name]['errors'] = 0
                                # FPS Math
                                prev_ts = local_state['cameras'][name]['last_ts']
                                if prev_ts > 0:
                                    instant_fps = 1.0 / (now - prev_ts)
                                    local_state['cameras'][name]['fps'] = round((0.8 * local_state['cameras'][name]['fps']) + (0.2 * instant_fps), 1)

                                # ATOMIC WRITE: Clear size, write data, then commit size
                                shm_frames[name].buf[:4] = b'\x00\x00\x00\x00'
                                data = frame.tobytes(); size = len(data)
                                shm_frames[name].buf[4:4+size] = data
                                shm_frames[name].buf[:4] = size.to_bytes(4, 'little') 
                                
                                local_state['cameras'][name]['frame_count'] += 1
                                local_state['cameras'][name]['last_ts'] = now
                            else:
                                # USB Reset Logic
                                local_state['cameras'][name]['errors'] += 1
                                if local_state['cameras'][name]['errors'] > 10:
                                    caps[name].release()
                                    caps[name] = open_cam(name, CAMERAS[name])
                                    local_state['cameras'][name]['errors'] = 0
                    
                    sync_stats()
                    sleep_time = (1.0 / TARGET_FPS) - (time.time() - loop_start)
                    if sleep_time > 0: time.sleep(sleep_time)
    finally:
        for c in caps.values(): (c.release() if c else None)
        for s in list(shm_frames.values()) + [shm_stats, shm_event]: s.close()

# === FLASK CONSUMER & TELEMETRY ===

class CameraProxy:
    def __init__(self, name):
        self.name = name
        self.shm_frame = None
        self.shm_stats = None

    def _ensure_shm(self):
        try:
            if not self.shm_frame: self.shm_frame = shared_memory.SharedMemory(name=SHM_NAMES[self.name])
            if not self.shm_stats: self.shm_stats = shared_memory.SharedMemory(name=STATS_SHM_NAME)
            return True
        except FileNotFoundError: return False

    def get_raw_frame(self):
        if not self._ensure_shm(): return None
        size = int.from_bytes(self.shm_frame.buf[:4], 'little')
        if size == 0: return None
        return bytes(self.shm_frame.buf[4:4+size])

    def update_clients(self, delta):
        if not self._ensure_shm(): return
        try:
            raw = bytes(self.shm_stats.buf).split(b'\0')[0].decode('utf-8')
            data = json.loads(raw)
            data['cameras'][self.name]['clients'] = max(0, data['cameras'][self.name].get('clients', 0) + delta)
            b = json.dumps(data).encode('utf-8')
            self.shm_stats.buf[:len(b)] = b; self.shm_stats.buf[len(b):len(b)+1] = b'\0'
        except: pass

    def stream_generator(self):
        self.update_clients(1)
        try:
            while True:
                raw = self.get_raw_frame()
                if raw:
                    decoded = _tjpeg.decode(raw, scaling_factor=(1, 8), flags=0)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + _tjpeg.encode(decoded, quality=70) + b'\r\n')
                time.sleep(0.1)
        finally:
            self.update_clients(-1)

PROXIES = {n: CameraProxy(n) for n in CAMERAS}

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
                     load=os.getloadavg()[0])
        except: pass

def health_monitor_loop():
    """WATCHDOG: Restarts frozen hardware process."""
    while True:
        time.sleep(20)
        try:
            shm = shared_memory.SharedMemory(name=STATS_SHM_NAME)
            raw = bytes(shm.buf).split(b'\0')[0].decode('utf-8')
            data = json.loads(raw)
            for name, s in data['cameras'].items():
                if s['clients'] > 0 and (time.time() - s['last_ts'] > 15):
                    log.error('watchdog', f'Freeze on {name}. Rebooting HardwareWorker.')
                    global hardware_process
                    if hardware_process: hardware_process.terminate()
                    time.sleep(2)
                    hardware_process = mp.Process(target=hardware_worker, daemon=True, name="HardwareWorker")
                    hardware_process.start()
                    break
        except: pass

@app.route('/stream/<camera>')
def stream(camera):
    if camera not in CAMERAS: return "Invalid camera", 404
    return Response(PROXIES[camera].stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture/<camera>')
def capture(camera):
    """Return latest full-res raw JPEG for a single camera."""
    if camera not in CAMERAS:
        return jsonify({'error': f'Unknown camera: {camera}'}), 404
    proxy = PROXIES[camera]
    # Wake hardware worker
    try:
        shm_ev = shared_memory.SharedMemory(name=EVENT_SHM_NAME)
        shm_ev.buf[0] = 1
    except: pass
    # Wait for a frame (up to 3s)
    raw = None
    for _ in range(60):
        raw = proxy.get_raw_frame()
        if raw and len(raw) > 50000:
            break
        time.sleep(0.05)
    # Clear wake flag
    try:
        shm_ev = shared_memory.SharedMemory(name=EVENT_SHM_NAME)
        shm_ev.buf[0] = 0
    except: pass
    if not raw:
        return jsonify({'error': f'No frame available for {camera}'}), 503
    return Response(raw, mimetype='image/jpeg', headers={
        'X-Frame-Size-Bytes': str(len(raw)),
    })

@app.route('/capture/burst/<camera>', methods=['POST'])
def capture_burst(camera):
    """BURST: Samples SHM and returns tar.gz."""
    data = request.get_json(silent=True) or {}
    count = min(int(data.get('count', 20)), 100)
    interval = max(int(data.get('interval_ms', 1000)), 100) / 1000.0
    proxy = PROXIES[camera]
    frames = []
    try:
        shm_ev = shared_memory.SharedMemory(name=EVENT_SHM_NAME)
        shm_ev.buf[0] = 1 # Wake
        for _ in range(count):
            raw = proxy.get_raw_frame()
            if raw: frames.append((raw, time.time()))
            time.sleep(interval)
        shm_ev.buf[0] = 0
    except: pass
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode='w:gz') as tar:
        for i, (f_bytes, ts) in enumerate(frames):
            info = tarfile.TarInfo(name=f"burst_{camera}_{i:03d}.jpg")
            info.size = len(f_bytes); info.mtime = ts
            tar.addfile(info, io.BytesIO(f_bytes))
    return Response(tar_buf.getvalue(), mimetype='application/gzip')

@app.route('/focus/<camera>/<int:val>', methods=['GET', 'POST'])
def set_focus(camera, val):
    """FOCUS: Updates SHM settings and persistence file."""
    try:
        shm = shared_memory.SharedMemory(name=STATS_SHM_NAME)
        raw = bytes(shm.buf).split(b'\0')[0].decode('utf-8')
        data = json.loads(raw)
        data['settings'][camera] = val
        b = json.dumps(data).encode('utf-8')
        shm.buf[:len(b)] = b; shm.buf[len(b):len(b)+1] = b'\0'
        with open(FOCUS_FILE, 'w') as f: json.dump(data['settings'], f)
    except: pass
    return f"Focus set to {val}"

@app.route('/status')
def status():
    try:
        shm = shared_memory.SharedMemory(name=STATS_SHM_NAME)
        raw = bytes(shm.buf).split(b'\0')[0].decode('utf-8')
        return jsonify(json.loads(raw))
    except: return jsonify({"status": "starting"})

# === BOOTSTRAP ===
_initialized = False
def ensure_initialized():
    global _initialized, hardware_process
    if _initialized: return
    _initialized = True
    if mp.current_process().name == 'MainProcess':
        # Auto-wipe SHM zombies
        for shm_file in glob.glob('/dev/shm/herdsync_*'):
            try: os.unlink(shm_file)
            except: pass
        try: mp.set_start_method('spawn', force=True)
        except: pass 
        hardware_process = mp.Process(target=hardware_worker, daemon=True, name="HardwareWorker")
        hardware_process.start()
        threading.Thread(target=heartbeat_loop, daemon=True).start()
        threading.Thread(target=health_monitor_loop, daemon=True).start()

ensure_initialized()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
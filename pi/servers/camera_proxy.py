#!/usr/bin/env python3
import multiprocessing as mp
import os
import sys

# ONLY patch gevent if we are the main web process.
# This keeps the HardwareWorker clean and preemptive.
if mp.current_process().name == 'MainProcess':
    import gevent.monkey
    gevent.monkey.patch_all()

import os
import sys
import time
import json
import threading
import subprocess
import tarfile
import io
import multiprocessing as mp
from multiprocessing import shared_memory
from flask import Flask, Response, request, jsonify
from turbojpeg import TurboJPEG, TJSAMP_420, TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE

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
SHM_SIZE = 2 * 1024 * 1024  # 2MB per camera
STATS_SHM_SIZE = 8192       # 8KB for stats/settings JSON
SHM_NAMES = {name: f"herdsync_{name}_raw" for name in CAMERAS}
STATS_SHM_NAME = "herdsync_system_stats"
EVENT_SHM_NAME = "herdsync_capture_flag"

# === HARDWARE WORKER (ISOLATED PROCESS) ===

def hardware_worker():
    """Hardware Process: Clean, Preemptive, and Isolated."""
    import cv2
    import os
    import subprocess
    import json
    import time
    from multiprocessing import shared_memory

    # 1. Setup Memory Blocks
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

    # 2. Camera Logic & Local State
    caps = {}  # CRITICAL: Fix for the NameError
    local_state = {
        'cameras': {n: {'fps': 0.0, 'frame_count': 0, 'last_ts': 0.0, 'clients': 0, 'connected': False} for n in CAMERAS},
        'settings': {n: 200 for n in CAMERAS}
    }

    # Helper to write stats to SHM
    def sync_stats():
        try:
            state_bytes = json.dumps(local_state).encode('utf-8')
            shm_stats.buf[:len(state_bytes)] = state_bytes
            shm_stats.buf[len(state_bytes):len(state_bytes)+1] = b'\0'
        except: pass

    # 3. Initial Status Write (Tells Flask we are alive but warming up)
    sync_stats()

    def open_cam(name, dev):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_RES[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_RES[1])
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            return cap
        return None

    # Initial Hardware Open
    for n, d in CAMERAS.items():
        caps[n] = open_cam(n, d)
        local_state['cameras'][n]['connected'] = caps[n] is not None
        sync_stats() # Update status as each camera connects
        if caps[n]:
            subprocess.run(['v4l2-ctl', '-d', d, '--set-ctrl', 'focus_automatic_continuous=0'], capture_output=True)
            subprocess.run(['v4l2-ctl', '-d', d, '--set-ctrl', f'focus_absolute=200'], capture_output=True)

    pairs = [('side', 'top'), ('side', 'front'), ('top', 'front')]
    
    try:
        while True:
            # Sync client demand from Flask
            try:
                raw_in = bytes(shm_stats.buf).split(b'\0')[0].decode('utf-8')
                flask_updates = json.loads(raw_in)
                for n in CAMERAS: local_state['cameras'][n]['clients'] = flask_updates['cameras'][n].get('clients', 0)
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
                                # FPS Calculation (Smoothed)
                                prev_ts = local_state['cameras'][name]['last_ts']
                                if prev_ts > 0:
                                    instant_fps = 1.0 / (now - prev_ts)
                                    local_state['cameras'][name]['fps'] = round((0.8 * local_state['cameras'][name]['fps']) + (0.2 * instant_fps), 1)

                                data = frame.tobytes(); size = len(data)
                                shm_frames[name].buf[:4] = size.to_bytes(4, 'little')
                                shm_frames[name].buf[4:4+size] = data
                                local_state['cameras'][name]['frame_count'] += 1
                                local_state['cameras'][name]['last_ts'] = now
                    
                    sync_stats()
                    sleep_time = (1.0 / TARGET_FPS) - (time.time() - loop_start)
                    if sleep_time > 0: time.sleep(sleep_time)
    finally:
        for c in caps.values(): (c.release() if c else None)
        for s in list(shm_frames.values()) + [shm_stats, shm_event]: s.close()

# === FLASK CONSUMER LOGIC ===

class CameraProxy:
    def __init__(self, name):
        self.name = name
        self.shm_frame = None
        self.shm_stats = None
        self.shm_event = None
        self.last_ts = 0

    def _ensure_shm(self):
        """Lazy connection to memory blocks to avoid race conditions."""
        try:
            if not self.shm_frame: self.shm_frame = shared_memory.SharedMemory(name=SHM_NAMES[self.name])
            if not self.shm_stats: self.shm_stats = shared_memory.SharedMemory(name=STATS_SHM_NAME)
            if not self.shm_event: self.shm_event = shared_memory.SharedMemory(name=EVENT_SHM_NAME)
            return True
        except FileNotFoundError:
            return False

    def get_raw(self):
        if not self._ensure_shm(): return None
        size = int.from_bytes(self.shm_frame.buf[:4], 'little')
        return bytes(self.shm_frame.buf[4:4+size]) if size > 0 else None

    def update_clients(self, delta):
        if not self._ensure_shm(): return
        try:
            raw = bytes(self.shm_stats.buf).split(b'\0')[0].decode('utf-8')
            data = json.loads(raw)
            data['cameras'][self.name]['clients'] = max(0, data['cameras'][self.name].get('clients', 0) + delta)
            b = json.dumps(data).encode('utf-8')
            self.shm_stats.buf[:len(b)] = b
            self.shm_stats.buf[len(b):len(b)+1] = b'\0'
        except: pass

    def stream_generator(self):
        self.update_clients(1)
        try:
            while True:
                raw = self.get_raw()
                if raw:
                    decoded = _tjpeg.decode(raw, scaling_factor=(1, 8), flags=TJFLAG_FASTUPSAMPLE)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + _tjpeg.encode(decoded, quality=70) + b'\r\n')
                time.sleep(0.1)
        finally:
            self.update_clients(-1)

PROXIES = {n: None for n in CAMERAS}
def get_proxy(name):
    if not PROXIES[name]: PROXIES[name] = CameraProxy(name)
    return PROXIES[name]

@app.route('/stream/<camera>')
def stream(camera):
    if camera not in CAMERAS: return "Invalid camera", 404
    return Response(get_proxy(camera).stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    try:
        shm = shared_memory.SharedMemory(name=STATS_SHM_NAME)
        raw = bytes(shm.buf).split(b'\0')[0].decode('utf-8')
        return jsonify(json.loads(raw))
    except: return jsonify({"status": "starting", "message": "Hardware initializing..."})

@app.route('/capture/all', methods=['POST', 'GET'])
def capture_all():
    p = get_proxy('side') # Use any proxy to get the event handle
    if p._ensure_shm(): p.shm_event.buf[0] = 1
    time.sleep(1.0) # Wake time
    results = {}
    for name in CAMERAS:
        raw = get_proxy(name).get_raw()
        if raw:
            path = f"/tmp/capture_{name}.jpg"
            with open(path, 'wb') as f: f.write(raw)
            results[name] = {"success": True, "path": path}
    if p.shm_event: p.shm_event.buf[0] = 0
    return jsonify(results)

@app.route('/focus/<camera>/<int:val>')
def set_focus(camera, val):
    subprocess.run(['v4l2-ctl', '-d', CAMERAS[camera], '--set-ctrl', f'focus_absolute={val}'])
    return f"Focus command sent for {camera}"

# === MODULE-LEVEL BOOTSTRAP ===

_initialized = False 

def ensure_initialized():
    """Triggers the Hardware Process spawn safely for Gunicorn + Spawn."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    # When using 'spawn', the child process re-imports this file.
    # This check prevents the child from trying to spawn its own child.
    if mp.current_process().name != 'MainProcess':
        return

    try:
        # Use 'spawn' to ensure the Hardware process is clean of gevent
        mp.set_start_method('spawn', force=True)
    except (RuntimeError, AttributeError):
        pass 

    p = mp.Process(target=hardware_worker, daemon=True, name="HardwareWorker")
    p.start()
    log.info('startup', f'Hardware Isolation Process Started | PID: {p.pid}')

# Trigger the boot
ensure_initialized()
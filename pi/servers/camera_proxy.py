#!/usr/bin/env python3
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

# HerdSync internal logger
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
SHM_SIZE = 2 * 1024 * 1024  # 2MB for JPEGs
STATS_SHM_SIZE = 4096       # 4KB for stats JSON
EVENT_SHM_NAME = "herdsync_capture_flag"
STATS_SHM_NAME = "herdsync_stats_json"
SHM_NAMES = {name: f"herdsync_{name}_raw" for name in CAMERAS}

# === HARDWARE WORKER (ISOLATED PROCESS) ===

def hardware_worker():
    """Hardware Process: Create memory blocks BEFORE anything else."""
    import cv2
    
    # Create SHM blocks immediately so the 'files' exist for Flask
    shm_frames = {}
    for name, shm_name in SHM_NAMES.items():
        try:
            shm_frames[name] = shared_memory.SharedMemory(name=shm_name, create=True, size=SHM_SIZE)
            # Initialize with zero size to prevent Flask from reading junk
            shm_frames[name].buf[:4] = b'\x00\x00\x00\x00'
        except FileExistsError:
            shm_frames[name] = shared_memory.SharedMemory(name=shm_name)

    caps = {}
    local_stats = {n: {'fps': 0.0, 'frame_count': 0, 'last_ts': 0, 'clients': 0, 'connected': False} for n in CAMERAS}

    def open_cam(name, dev):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_RES[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_RES[1])
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            return cap
        return None

    # Initial Open
    for n, d in CAMERAS.items():
        caps[n] = open_cam(n, d)
        local_stats[n]['connected'] = caps[n] is not None

    pairs = [('side', 'top'), ('side', 'front'), ('top', 'front')]
    
    try:
        while True:
            # Sync clients/demand from Flask (Read stats block)
            try:
                raw_stats = bytes(shm_stats.buf).split(b'\0')[0].decode('utf-8')
                flask_updates = json.loads(raw_stats)
                for n in CAMERAS: local_stats[n]['clients'] = flask_updates[n].get('clients', 0)
            except: pass

            has_demand = any(local_stats[n]['clients'] > 0 for n in CAMERAS) or shm_event.buf[0] == 1
            
            if not has_demand:
                time.sleep(0.5); continue

            for pair in pairs:
                for _ in range(FRAMES_PER_PAIR):
                    start = time.time()
                    for name in pair:
                        if caps.get(name):
                            ret, frame = caps[name].read()
                            if ret:
                                data = frame.tobytes(); size = len(data)
                                shm_frames[name].buf[:4] = size.to_bytes(4, 'little')
                                shm_frames[name].buf[4:4+size] = data
                                local_stats[name]['frame_count'] += 1
                                local_stats[name]['last_ts'] = time.time()
                    
                    # Update Stats SHM
                    stat_bytes = json.dumps(local_stats).encode('utf-8')
                    shm_stats.buf[:len(stat_bytes)] = stat_bytes
                    shm_stats.buf[len(stat_bytes):len(stat_bytes)+1] = b'\0'
                    
                    sleep_time = (1.0 / TARGET_FPS) - (time.time() - start)
                    if sleep_time > 0: time.sleep(sleep_time)
    finally:
        for c in caps.values(): (c.release() if c else None)
        for s in list(shm_frames.values()) + [shm_stats, shm_event]: s.close()

# === FLASK PROCESS ===

class CameraProxy:
    def __init__(self, name):
        self.name = name
        self.shm = None
        self.last_ts = 0

    def get_raw(self):
        # Attempt to connect only when needed
        if self.shm is None:
            try:
                self.shm = shared_memory.SharedMemory(name=SHM_NAMES[self.name])
            except FileNotFoundError:
                return None # Hardware isn't ready yet, return empty
        
        size = int.from_bytes(self.shm.buf[:4], 'little')
        return bytes(self.shm.buf[4:4+size]) if size > 0 else None

    def update_flask_stats(self, delta):
        try:
            raw = bytes(self.shm_stats.buf).split(b'\0')[0].decode('utf-8')
            data = json.loads(raw)
            data[self.name]['clients'] = max(0, data[self.name].get('clients', 0) + delta)
            b = json.dumps(data).encode('utf-8')
            self.shm_stats.buf[:len(b)] = b
            self.shm_stats.buf[len(b):len(b)+1] = b'\0'
        except: pass

    def stream_generator(self):
        self.update_flask_stats(1)
        try:
            while True:
                raw = self.get_raw()
                if raw:
                    decoded = _tjpeg.decode(raw, scaling_factor=(1, 8), flags=TJFLAG_FASTUPSAMPLE)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + _tjpeg.encode(decoded, quality=70) + b'\r\n')
                time.sleep(0.1)
        finally:
            self.update_flask_stats(-1)

PROXIES = {n: None for n in CAMERAS}
def get_proxy(name):
    if not PROXIES[name]: PROXIES[name] = CameraProxy(name)
    return PROXIES[name]

@app.route('/stream/<camera>')
def stream(camera):
    return Response(get_proxy(camera).stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    try:
        shm = shared_memory.SharedMemory(name=STATS_SHM_NAME)
        raw = bytes(shm.buf).split(b'\0')[0].decode('utf-8')
        return jsonify(json.loads(raw))
    except: return jsonify({"error": "Stats unavailable"})

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # Ensure clean memory space for Pi 4
    p = mp.Process(target=hardware_worker, daemon=True)
    p.start()
    app.run(host='0.0.0.0', port=8080)
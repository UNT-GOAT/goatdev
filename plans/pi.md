# Raspberry Pi Setup Guide

## Overview

The Pi runs two separate services:

| Service       | Purpose                              | Port | Path                         |
| ------------- | ------------------------------------ | ---- | ---------------------------- |
| goat-capture  | Production - single photos on demand | 5000 | `/home/pi/goat-capture/`     |
| goat-training | Temporary - 5s video recording       | 5001 | `/home/pi/training-capture/` |

```
/home/pi/
├── goat-capture/           # PRODUCTION
│   ├── server.py
│   └── requirements.txt
│
└── training-capture/       # TEMPORARY - delete when done
    ├── server.py
    └── requirements.txt
```

---

## Hardware

- Raspberry Pi 4 (4GB)
- 32GB MicroSD card
- USB-C power supply (5V 3A)
- 3x USB cameras + extenders

---

## Initial Pi Setup

### 1. Flash OS

1. Download **Raspberry Pi OS Lite (64-bit)**
2. Flash with **Raspberry Pi Imager**

In imager settings:

- Hostname: `goat-pi`
- Enable SSH
- Username/password: `pi` / `<password>` - Capstone123
- Configure WiFi

### 2. First Boot

```bash
ssh pi@goat-pi.local

sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv ffmpeg git
```

### 3. Install Tailscale

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up


ssh pi@100.110.194.126
```

Note the Tailscale IP (100.x.x.x) - this is how you'll access the Pi remotely.

### 4. Camera udev Rules

Create `/etc/udev/rules.d/99-cameras.rules`:

```bash
sudo nano /etc/udev/rules.d/99-cameras.rules
```

```
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="xxxx", KERNELS=="*-1", SYMLINK+="camera_side"
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="xxxx", KERNELS=="*-2", SYMLINK+="camera_top"
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="xxxx", KERNELS=="*-3", SYMLINK+="camera_front"
```

Find vendor/product IDs:

```bash
lsusb
```

Reload rules:

```bash
sudo udevadm control --reload-rules
sudo reboot
```

Verify:

```bash
ls -la /dev/camera_*
```

Final Cam USB Layout:

    2.0     3.0

|--------|--------|
| 1.3 - top | 1.1 - usb hub |
|--------|--------|
| 1.4 - front | 1.2 - side |
|--------|--------|

### 5. AWS Credentials

```bash
mkdir -p ~/.aws
nano ~/.aws/credentials
```

```ini
[default]
aws_access_key_id = <KEY>
aws_secret_access_key = <SECRET>
region = us-east-2
```

---

## Production Service (goat-capture)

### Install

```bash
mkdir -p /home/pi/goat-capture
cd /home/pi/goat-capture

# Create server.py
cat > server.py << 'EOF'
from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import boto3
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

CAMERAS = {
    'side': '/dev/camera_side',
    'top': '/dev/camera_top',
    'front': '/dev/camera_front'
}

S3_BUCKET = os.environ.get('S3_BUCKET', 'goat-captures-ACCOUNTID')
s3 = boto3.client('s3')

# Initialize cameras
cams = {}
for name, path in CAMERAS.items():
    if os.path.exists(path):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cams[name] = cap


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'cameras': list(cams.keys())})


@app.route('/capture', methods=['POST'])
def capture():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    uploaded = []

    for name, cam in cams.items():
        ret, frame = cam.read()
        if not ret:
            continue

        filename = f'{timestamp}_{name}.jpg'
        filepath = f'/tmp/{filename}'
        cv2.imwrite(filepath, frame)

        s3_key = f'captures/{timestamp}/{filename}'
        s3.upload_file(filepath, S3_BUCKET, s3_key)
        uploaded.append(s3_key)
        os.remove(filepath)

    return jsonify({
        'status': 'success',
        'timestamp': timestamp,
        'images': uploaded
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
EOF

# Create requirements
cat > requirements.txt << 'EOF'
flask
flask-cors
opencv-python-headless
boto3
EOF

# Install dependencies
pip3 install -r requirements.txt
```

### Systemd Service

```bash
sudo tee /etc/systemd/system/goat-capture.service << 'EOF'
[Unit]
Description=Goat Capture Server (Production)
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/goat-capture/server.py
WorkingDirectory=/home/pi/goat-capture
User=pi
Restart=always
RestartSec=5
Environment=S3_BUCKET=goat-captures-ACCOUNTID

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable goat-capture
sudo systemctl start goat-capture
```

### Verify

```bash
curl http://localhost:5000/health
```

---

## Training Service (goat-training) - TEMPORARY

### Install

```bash
mkdir -p /home/pi/training-capture
cd /home/pi/training-capture

# Create server.py
cat > server.py << 'EOF'
from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import threading
import os

app = Flask(__name__)
CORS(app)

CAMERAS = {
    'side': '/dev/camera_side',
    'top': '/dev/camera_top',
    'front': '/dev/camera_front'
}

S3_TRAINING_BUCKET = os.environ.get('S3_TRAINING_BUCKET', 'goat-training-ACCOUNTID')
s3 = None

def get_s3():
    global s3
    if s3 is None:
        import boto3
        s3 = boto3.client('s3')
    return s3

recording_state = {'active': False, 'progress': None}


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/record', methods=['POST'])
def record():
    global recording_state

    if recording_state['active']:
        return jsonify({'status': 'error', 'message': 'Recording in progress'}), 400

    data = request.json or {}
    goat_id = data.get('goat_id', 1)
    duration = min(int(data.get('duration', 5)), 10)

    recording_state['active'] = True
    recording_state['progress'] = 'starting'

    def do_record():
        global recording_state
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

                subprocess.run(cmd, capture_output=True)

                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    results[name] = filepath

            for name, path in CAMERAS.items():
                if os.path.exists(path):
                    t = threading.Thread(target=record_camera, args=(name, path))
                    threads.append(t)
                    t.start()

            for t in threads:
                t.join()

            recording_state['progress'] = 'uploading'

            for name, filepath in results.items():
                filename = os.path.basename(filepath)
                s3_key = f'{goat_id}/{filename}'
                try:
                    get_s3().upload_file(filepath, S3_TRAINING_BUCKET, s3_key)
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)
        finally:
            recording_state['active'] = False
            recording_state['progress'] = None

    threading.Thread(target=do_record).start()

    return jsonify({'status': 'recording_started', 'goat_id': goat_id})


@app.route('/status')
def status():
    return jsonify(recording_state)


if __name__ == '__main__':
    print(f"Training server starting on port 5001")
    app.run(host='0.0.0.0', port=5001)
EOF

# Create requirements
cat > requirements.txt << 'EOF'
flask
flask-cors
boto3
EOF

# Install dependencies
pip3 install -r requirements.txt
```

### Systemd Service

```bash
sudo tee /etc/systemd/system/goat-training.service << 'EOF'
[Unit]
Description=Goat Training Capture Server (TEMPORARY)
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/training-capture/server.py
WorkingDirectory=/home/pi/training-capture
User=pi
Restart=always
RestartSec=5
Environment=S3_TRAINING_BUCKET=goat-training-ACCOUNTID

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable goat-training
sudo systemctl start goat-training
```

### Verify

```bash
curl http://localhost:5001/health
```

---

## Remove Training Service (When Done)

```bash
sudo systemctl stop goat-training
sudo systemctl disable goat-training
sudo rm /etc/systemd/system/goat-training.service
sudo systemctl daemon-reload
rm -rf /home/pi/training-capture
```

---

## GitHub Actions Deployment

### Secrets Required

| Secret          | Value           |
| --------------- | --------------- |
| PI_TAILSCALE_IP | 100.x.x.x       |
| PI_SSH_KEY      | SSH private key |

### Production Deploy (.github/workflows/deploy-pi.yml)

```yaml
name: Deploy to Pi

on:
  push:
    branches: [main]
    paths: ["pi/**"]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.PI_TAILSCALE_IP }}
          username: pi
          key: ${{ secrets.PI_SSH_KEY }}
          script: |
            cd /home/pi/goat-capture
            git pull origin main 2>/dev/null || git clone https://github.com/OWNER/REPO.git .
            cp -r pi/* /home/pi/goat-capture/
            pip3 install -r requirements.txt
            sudo systemctl restart goat-capture
```

### Training Deploy (.github/workflows/deploy-training.yml)

```yaml
name: Deploy Training to Pi

on:
  push:
    branches: [main]
    paths: ["training/pi-server/**"]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.PI_TAILSCALE_IP }}
          username: pi
          key: ${{ secrets.PI_SSH_KEY }}
          script: |
            mkdir -p /home/pi/training-capture
            cd /home/pi/training-capture
            git pull origin main 2>/dev/null || git clone https://github.com/OWNER/REPO.git .
            cp -r training/pi-server/* /home/pi/training-capture/
            pip3 install -r requirements.txt
            sudo systemctl restart goat-training
```

---

## Quick Reference

| Task               | Command                                |
| ------------------ | -------------------------------------- |
| SSH                | `ssh pi@100.x.x.x`                     |
| Production logs    | `journalctl -u goat-capture -f`        |
| Training logs      | `journalctl -u goat-training -f`       |
| Restart production | `sudo systemctl restart goat-capture`  |
| Restart training   | `sudo systemctl restart goat-training` |
| Check cameras      | `ls -la /dev/camera_*`                 |
| Test production    | `curl http://localhost:5000/health`    |
| Test training      | `curl http://localhost:5001/health`    |

---

## S3 Buckets

| Bucket                  | Purpose           | Service       |
| ----------------------- | ----------------- | ------------- |
| goat-captures-ACCOUNTID | Production photos | goat-capture  |
| goat-training-ACCOUNTID | Training videos   | goat-training |

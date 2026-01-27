# Raspberry Pi Setup and Maintainence Plan

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Setup Steps](#setup-steps)
  - [1. Flash OS](#1-flash-os)
  - [2. First Boot Setup](#2-first-boot-setup)
  - [3. Install Tailscale](#3-install-tailscale)
  - [4. Camera udev Rules](#4-camera-udev-rules)
  - [5. Capture Server Code](#5-capture-server-code)
  - [6. AWS Credentials](#6-aws-credentials)
  - [7. Systemd Service](#7-systemd-service)
  - [8. GitHub Actions Deploy](#8-github-actions-deploy)
  - [9. Monitoring](#9-monitoring)
- [Quick Reference](#quick-reference)

---

## Hardware Requirements

**Goal:** Minimum hardware to capture 3 camera angles and upload to cloud.

- Raspberry Pi 4 (4GB)
- 32GB MicroSD card
- USB-C power supply (5V 3A)
- 3x USB extenders for cameras

---

## Setup Steps

### 1. Flash OS

**Goal:** Get a lightweight, headless Linux system on the Pi that's ready for SSH on first boot.

1. Download **Raspberry Pi OS Lite (64-bit)**
2. Flash with **Raspberry Pi Imager**

**In imager settings:**

- Set hostname: `goat-pi`
- Enable SSH
- Set username/password: `pi` / `your-password`
- Configure WiFi

---

### 2. First Boot Setup

**Goal:** Install everything needed to run the capture server and talk to AWS.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-opencv libatlas-base-dev
pip3 install flask boto3 opencv-python-headless
mkdir -p /home/pi/goat-capture
```

---

### 3. Install Tailscale

**Goal:** Remote SSH access from anywhere without dealing with port forwarding or static IPs at the facility.

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

> **Note:** Note the Tailscale IP (100.x.x.x) - this is how you'll SSH in remotely.

---

### 4. Camera udev Rules

**Goal:** Ensure each camera always maps to the same device name (`camera_side`, etc.) regardless of boot order or USB port detection order.

Create `/etc/udev/rules.d/99-cameras.rules`:

```bash
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="xxxx", KERNELS=="*-1", SYMLINK+="camera_side"
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="xxxx", KERNELS=="*-2", SYMLINK+="camera_top"
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="xxxx", KERNELS=="*-3", SYMLINK+="camera_front"
```

**Find vendor/product IDs:**

```bash
lsusb
```

**Reload rules:**

```bash
sudo udevadm control --reload-rules
```

---

### 5. Capture Server Code

**Goal:** Flask server that serves live preview frames (1/sec) and handles capture-to-S3 on demand from the tablet.

#### `/home/pi/goat-capture/server.py`

```python
from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import base64
import boto3
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

CAMERAS = {
    'side': '/dev/camera_side',
    'top': '/dev/camera_top',
    'front': '/dev/camera_front'
}

cams = {}
for name, path in CAMERAS.items():
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cams[name] = cap

S3_BUCKET = os.environ.get('S3_BUCKET', 'your-goat-bucket')
s3 = boto3.client('s3')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/preview')
def preview():
    previews = {}
    for name, cam in cams.items():
        ret, frame = cam.read()
        if not ret:
            previews[name] = None
            continue
        small = cv2.resize(frame, (320, 240))
        _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 50])
        previews[name] = base64.b64encode(buf).decode('utf-8')
    return jsonify(previews)


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
```

#### `/home/pi/goat-capture/requirements.txt`

```txt
flask
flask-cors
opencv-python-headless
boto3
```

---

### 6. AWS Credentials

**Goal:** Allow Pi to upload images to S3 without hardcoding secrets in code.

Create `/home/pi/.aws/credentials`:

```ini
[default]
aws_access_key_id = YOUR_KEY
aws_secret_access_key = YOUR_SECRET
region = us-east-1
```

---

### 7. Systemd Service

**Goal:** Capture server starts automatically on boot and restarts if it crashes. No manual intervention needed after power outage.

Create `/etc/systemd/system/goat-capture.service`:

```ini
[Unit]
Description=Goat Capture Server
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/goat-capture/server.py
WorkingDirectory=/home/pi/goat-capture
User=pi
Restart=always
RestartSec=5
Environment=S3_BUCKET=goat-bucket-name

[Install]
WantedBy=multi-user.target
```

**Enable and start the service:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable goat-capture
sudo systemctl start goat-capture
```

---

### 8. GitHub Actions Deploy

**Goal:** Push code to GitHub → automatically deploys to Pi. No manual SSH needed for updates.

Create `.github/workflows/deploy-pi.yml`:

```yaml
name: Deploy to Pi

on:
  push:
    branches: [main]
    paths: ["pi/**"]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.PI_TAILSCALE_IP }}
          username: pi
          key: ${{ secrets.PI_SSH_KEY }}
          script: |
            cd /home/pi/goat-capture
            git pull origin main
            pip3 install -r requirements.txt
            sudo systemctl restart goat-capture
```

**GitHub Secrets to add:**

- `PI_TAILSCALE_IP`: 100.x.x.x
- `PI_SSH_KEY`: Pi's private SSH key

---

### 9. Monitoring

**Goal:** Get alerted if the Pi goes offline or the capture service crashes.

Add Pi heartbeat to **UptimeRobot** or **Healthchecks.io**:

```
http://100.x.x.x:5000/health
```

- Pings every x minutes
- Emails you if it fails

---

## Quick Reference

| Task            | Command                                                                |
| --------------- | ---------------------------------------------------------------------- |
| SSH in          | `ssh pi@100.x.x.x`                                                     |
| View logs       | `journalctl -u goat-capture -f`                                        |
| Restart service | `sudo systemctl restart goat-capture`                                  |
| Check cameras   | `ls -la /dev/video*`                                                   |
| Manual update   | `cd ~/goat-capture && git pull && sudo systemctl restart goat-capture` |
| Reboot          | `sudo reboot`                                                          |

---

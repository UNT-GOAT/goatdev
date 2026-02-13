# Raspberry Pi Capture System

## Architecture Overview

<img src="../readme_assets/pi1.png" alt="Raspberry Pi" width="50%"/>

<img src="../readme_assets/pi2.png" alt="Raspberry Pi with case" width="50%"/>

The Pi runs two Flask services for goat image capture, both accessible remotely via Tailscale VPN.

```
┌─────────────────────────────────────────────────────────────┐
│                      Raspberry Pi 5 (4GB)                   │
│                                                             │
│  ┌─────────────────────┐    ┌──────────────────────┐        │
│  │   goat-prod.service │    │ goat-training.service│        │
│  │       Port 5000     │    │       Port 5001      │        │
│  │                     │    │                      │        │
│  │  Single-shot capture│    │  20-image capture    │        │
│  │  + EC2 grading      │    │  + S3 upload         │        │
│  └──────────┬──────────┘    └──────────┬───────────┘        │
│             │                          │                    │
│             ▼                          ▼                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              3x USB Cameras (udev symlinks)          │   │
│  │ /dev/camera_side  /dev/camera_top  /dev/camera_front |   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐         │
│  │  CloudWatch Logger  │    │  Heartbeat Cron     │         │
│  │  (all services)     │    │  (every 2 min)      │         │
│  └─────────────────────┘    └─────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Tailscale VPN
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Remote Access                           │
│                                                                 │
│   Developer Machine ──────► ssh pi@100.xxx.xxx.xxx              │
│                      ──────► curl http://100.xxx.xxx.xxx:5000   │
│                                                                 │
│   EC2 API Server ◄────────── Pi sends images for grading        │
│   S3 Buckets     ◄────────── Pi uploads training data           │
└─────────────────────────────────────────────────────────────────┘
```

## Services

### Production Service (`goat-prod.service`) - Port 5000

Single-shot capture for live grading. Captures one image per camera, sends all three to EC2 API, returns grade.

**Workflow:**

1. Receive POST with `serial_id` and `live_weight`
2. Capture 1 image from each camera (batched 2+1 to avoid OOM)
3. POST images to EC2 `/analyze` endpoint
4. Return grade result

**Key Endpoints:**

| Endpoint       | Method | Description                               |
| -------------- | ------ | ----------------------------------------- |
| `/health`      | GET    | Quick status check                        |
| `/diagnostics` | GET    | Detailed system info                      |
| `/grade`       | POST   | Capture + grade workflow                  |
| `/grade/test`  | POST   | Grade with uploaded images (skip capture) |
| `/test`        | GET    | Full connectivity test                    |
| `/status`      | GET    | Current capture state                     |
| `/cancel`      | POST   | Emergency stop                            |

### Training Service (`goat-training.service`) - Port 5001

Multi-image capture for model training. Captures 20 images per camera at 1fps, uploads to S3 as tarballs.

**Workflow:**

1. Receive POST with `goat_id` and optional metadata
2. Capture 20 images per camera (1 second apart)
3. Tar images per camera, upload to S3
4. Upload metadata JSON

**Key Endpoints:**

| Endpoint       | Method | Description           |
| -------------- | ------ | --------------------- |
| `/health`      | GET    | Quick status check    |
| `/diagnostics` | GET    | Detailed system info  |
| `/record`      | POST   | Start capture session |
| `/status`      | GET    | Current capture state |
| `/cancel`      | POST   | Emergency stop        |

## Tailscale Setup

Tailscale provides secure remote access to the Pi without port forwarding or exposing it to the public internet.

### Network Topology

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Developer Mac   │     │   Raspberry Pi   │     │   EC2 Instance   │
│  100.xxx.xxx.1   │◄───►│  100.xxx.xxx.126 │◄───►│  (public IP)     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
         └────────────────────────┴────────────────────────┘
                         Tailscale Mesh VPN
```

### Access Methods

**SSH to Pi:**

```bash
ssh pi@100.xxx.xxx.xxx
```

**Test services:**

```bash
# Production health
curl http://100.xxx.xxx.xxx:5000/health

# Training health
curl http://100.xxx.xxx.xxx:5001/health

# Full diagnostics
curl http://100.xxx.xxx.xxx:5000/diagnostics | jq
```

### Tailscale Admin

- Dashboard: https://login.tailscale.com/admin/machines
- Pi appears as `goat-pi` in the machine list
- Can enable/disable access, view connection status, manage ACLs

## Camera Configuration

Cameras are mapped to stable device paths via udev rules:

| View  | Symlink             | USB Port |
| ----- | ------------------- | -------- |
| Side  | `/dev/camera_side`  | 1.2      |
| Top   | `/dev/camera_top`   | 1.3      |
| Front | `/dev/camera_front` | 1.4      |

**Capture Settings:**

- Resolution: 4656 × 3496 (16MP)
- Format: MJPEG
- Native FPS: 10
- Warmup: 10 frames skipped for autofocus and white balance

## Logging

All services log to AWS CloudWatch under the `/goatdev` log group:

| Stream         | Source             |
| -------------- | ------------------ |
| `pi/prod`      | Production service |
| `pi/training`  | Training service   |
| `pi/heartbeat` | Health monitoring  |

## Heartbeat Monitoring

A cron job runs every 2 minutes checking service health:

- Verifies `goat-prod` service is active
- Verifies port 5000 is listening
- Logs errors to CloudWatch only when unhealthy
- Logs single "recovered" message when health restored

## File Structure

```
/home/pi/goat-capture/
├── pi/
│   └── server.py           # Production service
├── training/
│   └── pi-server/
│       └── server.py       # Training service
├── logger/
│   ├── pi_cloudwatch.py    # Shared logging module
│   └── pi_heartbeat_cron.py
└── venv/                   # Python virtual environment
```

## Quick Reference

| Task               | Command                                |
| ------------------ | -------------------------------------- |
| SSH to Pi          | `ssh pi@100.xxx.xxx.xxx`               |
| Restart production | `sudo systemctl restart goat-prod`     |
| Restart training   | `sudo systemctl restart goat-training` |
| View prod logs     | `journalctl -u goat-prod -f`           |
| View training logs | `journalctl -u goat-training -f`       |
| Check cameras      | `ls -la /dev/camera_*`                 |
| Kill stuck ffmpeg  | `sudo pkill -9 ffmpeg`                 |
| Check memory       | `free -h`                              |
| Check disk         | `df -h /tmp`                           |
| Tailscale status   | `tailscale status`                     |

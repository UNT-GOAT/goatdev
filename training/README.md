# Goat Training Capture

Captures training images from 3 USB cameras on a Raspberry Pi and uploads them to S3 for YOLO model training.

## Overview

A Flask server running on the Pi exposes a REST API. A web UI (served separately or opened locally) connects to the API to trigger captures. Each capture session photographs a single goat from three angles (side, top, front), producing 20 images per camera — 60 total — with exactly 1-second spacing between frames.

Images are tarred per camera and uploaded to S3. An optional metadata JSON file records goat data (description, weight, grade) alongside the images.

## Hardware

- **Raspberry Pi 4B (4GB RAM)**
- **3× USB cameras** capable of 4656×3496 MJPEG at 10fps
- USB hub (powered recommended — 3 cameras draw significant current)
- Network connection (Tailscale VPN for remote access)

### Camera Mapping

Cameras are mapped to stable device paths via udev rules:

| View  | Device Path         | Position               |
| ----- | ------------------- | ---------------------- |
| Side  | `/dev/camera_side`  | Side profile of goat   |
| Top   | `/dev/camera_top`   | Overhead, looking down |
| Front | `/dev/camera_front` | Head-on front view     |

## How Capture Works

1. Client POSTs to `/record` with goat_id, goat_data, and capture mode
2. Server pre-checks: disk space, camera availability, stale processes
3. Cameras capture in **batches of 2** to stay within RAM limits:
   - Batch 1: side + top in parallel (~21s)
   - Batch 2: front alone (~21s)
   - Total: ~42s per goat
4. Each camera runs a single ffmpeg process that:
   - Opens the V4L2 device at native 10fps MJPEG
   - Skips the first 1 second (10 frames) for autofocus/white balance warmup
   - Selects every 10th frame thereafter → 1 frame per second
   - Decodes and re-encodes with `-threads 1` to limit memory (~500MB per process)
   - Writes 20 individual JPEG files to `/tmp`
5. Images are validated (minimum size check)
6. Per-camera images are tarred (`images.tar.gz`) and uploaded to S3
7. Metadata JSON uploaded alongside if goat_data was provided
8. Temp files cleaned up

### Why Not All 3 Cameras in Parallel?

Each ffmpeg process decoding 4656×3496 frames uses ~500MB RAM. Three simultaneous processes exceed 4GB and trigger the OOM killer. The 2+1 batching keeps peak memory at ~1GB for ffmpeg.

### Why Not `-codec:v copy`?

Raw MJPEG passthrough (`-codec:v copy`) uses almost no RAM, but ffmpeg can't apply the `select` filter without decoding. The select filter is required to get correct 1-second frame spacing since the cameras only support 10fps minimum at this resolution.

## S3 Structure

```
s3://training-937249941844/
├── 1/
│   ├── side/images.tar.gz      # 20 JPEGs
│   ├── top/images.tar.gz       # 20 JPEGs
│   ├── front/images.tar.gz     # 20 JPEGs
│   └── goat_data.json          # metadata
├── 2/
│   ├── side/images.tar.gz
│   ├── top/images.tar.gz
│   ├── front/images.tar.gz
│   └── goat_data.json
└── ...
```

Each tar contains files named `{goat_id}_{camera}_{01-20}.jpg`.

### goat_data.json

```json
{
  "goat_id": "17",
  "timestamp": "2026-02-10T18:30:00.000000",
  "cameras": ["side", "top", "front"],
  "images_per_camera": 20,
  "resolution": "4656x3496",
  "capture_fps": 1,
  "description": "meat",
  "live_weight": "85 lbs",
  "grade": "choice"
}
```

## API Endpoints

### `GET /health`

Quick health check. Returns camera availability, disk space, and capture state.

```json
{
  "status": "ok",
  "cameras": { "side": true, "top": true, "front": true },
  "disk_free_mb": 2500,
  "recording_active": false
}
```

### `GET /diagnostics`

Detailed diagnostics including per-camera device checks, S3 connectivity, and current capture config.

### `POST /record`

Start a capture session.

**Request body:**

```json
{
  "goat_id": "17",
  "goat_data": {
    "description": "meat",
    "live_weight": "85 lbs",
    "grade": "choice"
  },
  "is_test": false,
  "require_all_cameras": true
}
```

**Modes:**

- **Real** (`is_test: false`): All 3 cameras must succeed. Images uploaded to S3. Any camera failure aborts the entire batch.
- **Test** (`is_test: true`): Captures with whatever cameras are available. Checks S3 access but does not upload. Files are cleaned up immediately.

**Error codes:** `CAPTURE_IN_PROGRESS`, `INVALID_GOAT_ID`, `LOW_DISK_SPACE`, `MISSING_CAMERAS`, `NO_CAMERAS`

### `GET /status`

Poll capture progress. Used by the web UI during capture.

```json
{
  "active": true,
  "goat_id": "17",
  "started_at": "2026-02-10T18:30:00.000000",
  "progress": "capturing",
  "last_error": null,
  "last_result": null
}
```

Progress stages: `starting` → `capturing` → `uploading` → (done)

For test mode: `starting` → `capturing` → `checking_s3` → (done)

### `POST /cancel`

Emergency stop. Kills all ffmpeg processes, cleans up temp files, resets state.

## Web UI

![Training Capture Interface](readme_assets/training_interface.png)

`index.html` — a single-page app that connects to the Pi server over Tailscale.

Features:

- Password-protected login
- Auto-incrementing goat counter (synced from S3 bucket listing)
- Goat data form (description, weight, grade)
- Big red CAPTURE button (real mode) and smaller TEST button
- Activity log showing capture progress in real-time
- Polls `/status` every 500ms, deduplicates status messages

## Configuration

All config is at the top of `server.py`:

| Setting                     | Default                 | Description                                       |
| --------------------------- | ----------------------- | ------------------------------------------------- |
| `NUM_IMAGES`                | 20                      | Images per camera per capture                     |
| `CAPTURE_FPS`               | 1                       | Target frames per second (spacing between images) |
| `CAMERA_NATIVE_FPS`         | 10                      | Camera's actual FPS at capture resolution         |
| `IMAGE_WIDTH`               | 4656                    | Capture width in pixels                           |
| `IMAGE_HEIGHT`              | 3496                    | Capture height in pixels                          |
| `MIN_DISK_MB`               | 1000                    | Minimum free disk space to start capture          |
| `MIN_IMAGE_BYTES`           | 50000                   | Minimum valid image size (50KB)                   |
| `FFMPEG_TIMEOUT_SEC`        | 30                      | Per-ffmpeg process timeout                        |
| `CAPTURE_TOTAL_TIMEOUT_SEC` | 60                      | Total timeout for one camera's full capture       |
| `S3_TRAINING_BUCKET`        | `training-937249941844` | S3 bucket (overridable via env var)               |

## Setup

### Prerequisites

```bash
sudo apt install ffmpeg v4l-utils
pip install flask flask-cors boto3
```

### AWS Credentials

The Pi uses a static IAM user (`goat-pi-capture`) with S3 put/get/list permissions:

```bash
aws configure
# Access Key ID: ...
# Secret Access Key: ...
# Region: us-east-2
```

### Camera udev Rules

Create `/etc/udev/rules.d/99-cameras.rules` to map cameras to stable device paths based on USB port:

```
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="xxxx", ATTR{index}=="0", KERNELS=="x-x.x", SYMLINK+="camera_side"
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="xxxx", ATTR{index}=="0", KERNELS=="x-x.x", SYMLINK+="camera_top"
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="xxxx", ATTR{index}=="0", KERNELS=="x-x.x", SYMLINK+="camera_front"
```

Then reload: `sudo udevadm control --reload-rules && sudo udevadm trigger`

### Systemd Service

```ini
[Unit]
Description=Goat Training Capture Server
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/goat-capture/training
ExecStart=/home/pi/goat-capture/venv/bin/python server.py
Restart=always
RestartSec=5
Environment=S3_TRAINING_BUCKET=training-937249941844

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable goat-training
sudo systemctl start goat-training
```

### Verify

```bash
# Check service
sudo systemctl status goat-training

# Check cameras
v4l2-ctl -d /dev/camera_side --list-formats-ext

# Check S3 access
aws s3 ls s3://training-937249941844/

# Hit health endpoint
curl http://localhost:5001/health
```

## Troubleshooting

**OOM / ffmpeg killed (code -9):** The `MAX_PARALLEL = 2` batching should prevent this. If it still happens, reduce to `MAX_PARALLEL = 1` (sequential capture, ~63s per goat).

**Blurry/pink first images:** The 1-second warmup skip should handle this. If images are still blurry, increase `warmup_frames` in `capture_single_camera`.

**"Camera does not support resolution":** Check supported formats with `v4l2-ctl -d /dev/camera_side --list-formats-ext`. The camera may not support 4656×3496 MJPEG.

**S3 403 / InvalidSignature:** Check that the Pi's clock is synced (`timedatectl status`) and AWS credentials are valid (`aws sts get-caller-identity`).

**Capture timeout on client:** The web UI polls for 2 minutes (240 × 500ms). If captures + upload consistently exceed this, increase `maxAttempts` in `index.html`.

**Camera "Device or resource busy":** A previous ffmpeg process didn't exit cleanly. Run `sudo pkill -9 ffmpeg` or hit the `/cancel` endpoint.

**Stale ffmpeg processes:** The server kills orphaned ffmpeg processes on startup and before each capture. If cameras seem stuck, restart the service: `sudo systemctl restart goat-training`.

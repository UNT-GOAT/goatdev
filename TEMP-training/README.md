# Training Data Collection System

> **TEMPORARY** - Delete this entire folder once model retraining is complete.

## Overview

This system allows Becky to record short video clips of goats for model retraining. Videos are automatically uploaded to S3, then processed locally to extract frames for training.

## Architecture

```
Becky's Phone → herd-sync.com/training → Pi Server → Records 5s video → S3
                                                                         ↓
                                                                   Local machine
                                                                         ↓
                                                                 extract_frames.py
                                                                         ↓
                                                                  Roboflow → Train
```

## Components

### 1. Pi Server (`pi-server/`)

Flask server running on port 5001:

- Records 5-second videos from connected cameras
- Uploads to S3 bucket `temp-training-937249941844`
- Endpoints:
  - `GET /health` - Quick health check
  - `GET /diagnostics` - Detailed system info
  - `GET /status` - Current recording state
  - `POST /record` - Start recording

### 2. Web UI (`web/`)

Password-protected single HTML file hosted at `herd-sync.com/training`:

- Simple goat counter (persists between sessions)
- Big "RECORD" button
- Shows recording progress and status
- Auto-increments counter on successful recording

### 3. Processing Scripts (`scripts/`)

Run locally after Becky has collected videos:

- `extract_frames.py` - Downloads videos from S3, extracts frames at 5fps

## Deployment

### Pi Server

Deployed via GitHub Actions (`.github/workflows/deploy-pi.yml`):

- Triggers on changes to `pi/**` or `TEMP-training/pi-server/**`
- Restarts `goat-training.service` automatically on the pi

### Web UI

Deployed via GitHub Actions (`.github/workflows/deploy-frontend.yml`):

- Triggers on changes to `TEMP-training/web/**`
- Syncs to `s3://goat-web-937249941844/training/`

## Usage

### For Becky

1. Go to `herd-sync.com/training`
2. Enter password
3. Position goat in camera view
4. Tap RECORD
5. Wait for confirmation
6. Counter increments automatically
7. Repeat for each goat

### Processing (after collection)

```bash
cd TEMP-training/scripts
pip install -r requirements.txt
python extract_frames.py --fps 5
```

Upload `extracted_frames/` folder to Roboflow for labeling.

## Frame Yield

Per goat: 3 cameras × 5 seconds × 5fps = 75 frames
50 goats = ~3,750 training frames

## S3 Structure

```
temp-training-937249941844/
├── 1/
│   ├── 1_side.mp4
│   ├── 1_top.mp4
│   └── 1_front.mp4
├── 2/
│   └── ...
```

## Cleanup

When training is complete, delete everything:

```bash
# Remove S3 training data
aws s3 rb s3://temp-training-937249941844 --force

# Remove web UI
aws s3 rm s3://goat-web-937249941844/training --recursive

# On Pi
sudo systemctl stop goat-training
sudo systemctl disable goat-training
sudo rm /etc/systemd/system/goat-training.service

# Remove from repo
rm -rf TEMP-training/
git add -A && git commit -m "Remove training system" && git push
```

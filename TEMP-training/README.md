# Training Data Collection System

> **TEMPORARY** - Delete this entire folder once model retraining is complete.

## Overview

This system allows Becky to record short video clips of goats for model retraining. Videos are automatically uploaded to S3, then processed locally to extract frames for training.

## Architecture

```
Becky's iPad → Web UI → Pi (Tailscale) → Records 5s video → S3
                                                              ↓
                                                        local machine
                                                              ↓
                                                      extract_frames.py
                                                              ↓
                                                        Label & Train
```

## Components

### 1. Pi Server (`pi-server/`)

Flask server running on the Raspberry Pi that:

- Provides live camera previews
- Records 5-second videos from all 3 cameras simultaneously
- Uploads to S3 training bucket

### 2. Web UI (`web/`)

Single HTML file that Becky uses on her tablet:

- Shows live preview from all cameras
- Big "RECORD" button
- Optional goat ID field
- Shows recent recordings

### 3. Processing Scripts (`scripts/`)

Run these locally after Becky has collected videos:

- `extract_frames.py` - Downloads videos, extracts frames at 5fps
- `filter_blur.py` - Removes blurry frames
- `organize.py` - Creates YOLO dataset structure with train/val split

## Setup

### S3 Bucket

```bash
aws s3 mb s3://goat-training-ACCOUNTID --region us-east-2
```

### Pi Setup

1. Copy files to Pi:

```bash
scp -r pi-server/* pi@100.x.x.x:/home/pi/training-capture/
```

2. Install dependencies:

```bash
ssh pi@100.x.x.x
cd /home/pi/training-capture
pip3 install -r requirements.txt
sudo apt install ffmpeg
```

3. Install systemd service:

```bash
sudo cp goat-training.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable goat-training
sudo systemctl start goat-training
```

4. Verify it's running:

```bash
curl http://localhost:5001/health
```

### Web UI

Option A: Just open `web/index.html` directly on Becky's iPad (file://)

Option B: Host on S3:

```bash
aws s3 cp web/index.html s3://goat-training-ACCOUNTID/ui/index.html --acl public-read
# Access at: http://goat-training-ACCOUNTID.s3.amazonaws.com/ui/index.html
```

Configure the Pi's Tailscale IP in the settings panel.

## Usage

### For Becky

1. Open the web UI on tablet
2. Enter Pi's Tailscale IP in settings (first time only)
3. Position goat in camera view (check previews)
4. Optionally enter goat ID/name
5. Tap "RECORD 5 SECONDS"
6. Wait for upload confirmation
7. Repeat for each goat

### For Me (Processing)

1. After Becky has collected videos, extract frames:

```bash
cd training/scripts
pip install -r requirements.txt

# Extract frames from all videos (5 fps = 25 frames per 5s video)
python extract_frames.py --fps 5
```

2. Upload `extracted_frames/` to Roboflow

3. Label and train in Roboflow

## Frame Yield Estimate

Per goat (3 cameras × 5 seconds × 5fps):

- Raw frames: 75 per goat

50 goats = ~3,750 training frames

## Cleanup

When training is complete:

1. Delete S3 bucket:

```bash
aws s3 rb s3://goat-training-ACCOUNTID --force
```

2. Remove from Pi:

```bash
ssh pi@100.x.x.x
sudo systemctl stop goat-training
sudo systemctl disable goat-training
sudo rm /etc/systemd/system/goat-training.service
rm -rf /home/pi/training-capture
```

3. Delete this folder from repo:

```bash
rm -rf training/
git add -A && git commit -m "Remove training data collection system" && git push
```

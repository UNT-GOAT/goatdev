# Goatdev - AWS Architecture

## Table of Contents

- [Overview](#overview)
- [Architecture Components](#architecture-components)
  - [1. S3 Buckets](#1-s3-buckets)
  - [2. RDS PostgreSQL](#2-rds-postgresql)
  - [3. EC2 Instance](#3-ec2-instance)
  - [4. Web Hosting (S3 + CloudFront)](#4-web-hosting-s3--cloudfront)
  - [5. IAM Setup](#5-iam-setup)
  - [6. Processing Flow](#6-processing-flow)
  - [7. Monitoring (CloudWatch)](#7-monitoring-cloudwatch)
  - [8. GitHub Actions Deploy](#8-github-actions-deploy)
- [Cost Estimate](#cost-estimate)
- [Quick Reference](#quick-reference)

---

## Overview

```
ethan TODO: work on nicer, more in depth lucidchart layout
┌─────────────────────────────────────────────────────────────────┐
│                            AWS                                  │
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐   │
│  │ CloudFront   │      │     S3       │      │     S3       │   │
│  │ + S3 (web)   │      │ goat-captures│      │goat-processed│   │
│  └──────┬───────┘      └──────┬───────┘      └───────▲──────┘   │
│         │                     │                      │          │
│         │              ┌──────▼───────┐              │          │
│         │              │    EC2       │              │          │
│         └─────────────▶│  FastAPI +   │──────────────┘          │
│           (API calls)  │  YOLO models │                         │
│                        └──────┬───────┘                         │
│                               │                                 │
│                        ┌──────▼───────┐                         │
│                        │     RDS      │                         │
│                        │  PostgreSQL  │                         │
│                        └──────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Components

### 1. S3 Buckets

**Goal:** Store raw captures and total processed outputs separately for clean permissions and lifecycle rules.

| Bucket           | Purpose                                   | Lifecycle             |
| ---------------- | ----------------------------------------- | --------------------- |
| `goat-captures`  | Raw images from Pi                        | Glacier after 90 days |
| `goat-processed` | Annotated images, model outputs           | Glacier after 90 days |
| `goat-web`       | Static web app files (the actual website) | None                  |

#### Bucket Structure

```
goat-captures/
└── captures/
    └── {timestamp}/            - maybe the goat tag instead?
        ├── {timestamp}_side.jpg
        ├── {timestamp}_top.jpg
        └── {timestamp}_front.jpg

goat-processed/
└── results/
    └── {timestamp}/            -  same here
        ├── {timestamp}_side_annotated.jpg
        ├── {timestamp}_top_annotated.jpg
        ├── {timestamp}_front_annotated.jpg
        └── measurements.json
```

---

### 2. RDS PostgreSQL

**Goal:** Structured storage for goat records and measurements.

#### Instance Configuration

| Setting  | Value                              |
| -------- | ---------------------------------- |
| Engine   | PostgreSQL 15                      |
| Instance | db.t3.micro (~$15/mo)              |
| Storage  | 20GB gp2                           |
| Multi-AZ | No (single facility, not critical) |

SUBJECT TO CHANGE once we get building this thing

#### Database Schema

Cooper TODO: he been cookin here

---

### 3. EC2 Instance

**Goal:** Run FastAPI server with YOLO models loaded in memory for consistent, fast inference.

#### Instance Configuration

| Setting | Value                       |
| ------- | --------------------------- |
| Type    | t3.medium (2 vCPU, 4GB RAM) |
| AMI     | Ubuntu 22.04 LTS            |
| Storage | 30GB gp3                    |
| Cost    | ~$30/mo                     |

#### Security Group

| Type   | Port | Source              |
| ------ | ---- | ------------------- |
| SSH    | 22   | Your IP / Tailscale |
| HTTP   | 80   | 0.0.0.0/0           |
| HTTPS  | 443  | 0.0.0.0/0           |
| Custom | 8000 | 0.0.0.0/0 (API)     |

#### API Server Code - beginning example subject to change

`/home/ubuntu/goat-api/main.py`:

```python
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import boto3
import psycopg2
from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
models = {
    'side': YOLO('models/side/best.pt'),
    'top': YOLO('models/top/best.pt'),
    'front': YOLO('models/front/best.pt'),
}

s3 = boto3.client('s3')
CAPTURE_BUCKET = os.environ.get('CAPTURE_BUCKET', 'goat-captures')
PROCESSED_BUCKET = os.environ.get('PROCESSED_BUCKET', 'goat-processed')

def get_db():
    return psycopg2.connect(
        host=os.environ['DB_HOST'],
        database=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD']
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process/{timestamp}")
async def process_capture(timestamp: str, background_tasks: BackgroundTasks):
    """Queue processing for a capture"""
    background_tasks.add_task(run_inference, timestamp)
    return {"status": "processing", "timestamp": timestamp}


@app.get("/results/{timestamp}")
def get_results(timestamp: str):
    """Check if results are ready"""
    try:
        response = s3.get_object(
            Bucket=PROCESSED_BUCKET,
            Key=f'results/{timestamp}/measurements.json'
        )
        return json.loads(response['Body'].read())
    except s3.exceptions.NoSuchKey:
        return {"status": "processing"}


def run_inference(timestamp: str):
    """Download images, run models, save results"""
    results = {}

    for angle in ['side', 'top', 'front']:
        # Download from S3
        key = f'captures/{timestamp}/{timestamp}_{angle}.jpg'
        local_path = f'/tmp/{timestamp}_{angle}.jpg'
        s3.download_file(CAPTURE_BUCKET, key, local_path)

        # Run model
        img = cv2.imread(local_path)
        output = models[angle].predict(img)

        # Extract measurements (your existing logic)
        measurements = extract_measurements(output, angle)
        results[angle] = measurements

        # Save annotated image
        annotated = output[0].plot()
        annotated_path = f'/tmp/{timestamp}_{angle}_annotated.jpg'
        cv2.imwrite(annotated_path, annotated)
        s3.upload_file(
            annotated_path,
            PROCESSED_BUCKET,
            f'results/{timestamp}/{timestamp}_{angle}_annotated.jpg'
        )

        # Cleanup
        os.remove(local_path)
        os.remove(annotated_path)

    # Combine measurements
    final = combine_measurements(results)

    # Save to S3
    s3.put_object(
        Bucket=PROCESSED_BUCKET,
        Key=f'results/{timestamp}/measurements.json',
        Body=json.dumps(final)
    )

    # Save to DB (if goat_id provided)
    # save_to_db(final)


def extract_measurements(output, angle):
    # Existing measurement extraction logic
    # From model/side/side_yolo_measurements.py etc.
    pass


def combine_measurements(results):
    # Combine side/top/front into final measurements
    return {
        "status": "complete",
        "measurements": results,
        "processed_at": datetime.now().isoformat()
    }


# CRUD endpoints for goats
@app.get("/goats")
def list_goats():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM goats ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


@app.post("/goats")
def create_goat(tag_number: str, name: str = None, breed: str = None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO goats (tag_number, name, breed) VALUES (%s, %s, %s) RETURNING id",
        (tag_number, name, breed)
    )
    goat_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return {"id": goat_id}


@app.get("/goats/{goat_id}/measurements")
def get_goat_measurements(goat_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM measurements WHERE goat_id = %s ORDER BY timestamp DESC",
        (goat_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows
```

#### Systemd Service

`/etc/systemd/system/goat-api.service`:

```ini
[Unit]
Description=Goat API Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/goat-api
ExecStart=/home/ubuntu/goat-api/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=CAPTURE_BUCKET=goat-captures
Environment=PROCESSED_BUCKET=goat-processed
Environment=DB_HOST=your-rds-endpoint.region.rds.amazonaws.com
Environment=DB_NAME=goatdb
Environment=DB_USER=goatadmin
Environment=DB_PASSWORD=your-password

[Install]
WantedBy=multi-user.target
```

**Enable and start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable goat-api
sudo systemctl start goat-api
```

---

### 4. Web Hosting (S3 + CloudFront)

**Goal:** Serve static web app with HTTPS.

#### S3 Static Hosting

**Bucket policy for public read:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::goat-web/*"
    }
  ]
}
```

#### CloudFront Configuration

group TODO: pick a domain and configure Route 53 DNS

| Setting         | Value                                   |
| --------------- | --------------------------------------- |
| Origin          | goat-web.s3.amazonaws.com               |
| Viewer Protocol | Redirect HTTP to HTTPS                  |
| Cache Policy    | CachingOptimized                        |
| Price Class     | Use North America/Europe only (cheaper) |

---

### 5. IAM Setup

**Goal:** Minimal permissions per identity. Pi can only upload, EC2 can process, devs can manage.

#### Pi Capture User

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": "arn:aws:s3:::goat-captures/captures/*"
    }
  ]
}
```

**Create user:**

```bash
aws iam create-user --user-name pi-capture-user
aws iam put-user-policy --user-name pi-capture-user --policy-name pi-upload --policy-document file://pi-policy.json
aws iam create-access-key --user-name pi-capture-user
```

#### EC2 Instance Role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": [
        "arn:aws:s3:::goat-captures/*",
        "arn:aws:s3:::goat-processed/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["rds:*"],
      "Resource": "*"
    }
  ]
}
```

> 📝 **Note:** Attach to EC2 instance via instance profile.

#### Developer User

Full access to project resources (or use your existing admin user).

---

### 6. Processing Flow

**Goal:** Async processing so tablet doesn't block waiting for inference.

```
1. Associate taps Capture
         │
2. Pi captures 3 images
         │
3. Pi uploads to S3 (goat-captures)
         │
4. Pi calls POST /process/{timestamp}
         │
5. API returns immediately: {"status": "processing"}
         │
6. Background task runs inference (~5 sec)
         │
7. Results saved to S3 + RDS
         │
8. Tablet polls GET /results/{timestamp}
         │
9. Results ready → display to farmer
```

---

### 7. Monitoring (CloudWatch)

**Goal:** Know when things break before Becky does.

#### Alarms

| Metric                 | Threshold    | Action      |
| ---------------------- | ------------ | ----------- |
| EC2 CPU > 80%          | 5 min        | Email alert |
| EC2 StatusCheck failed | 1 min        | Email alert |
| API 5xx errors         | > 5 in 5 min | Email alert |
| RDS connections > 80%  | 5 min        | Email alert |

(Not including Pi tailscale heartbeat)

#### Logs

```bash
# EC2 API logs → CloudWatch
sudo apt install amazon-cloudwatch-agent
# Configure to ship /var/log/goat-api.log
```

---

### 8. GitHub Actions Deploy

**Goal:** Push to main deploys API automatically.

`.github/workflows/deploy-api.yml`:

```yaml
name: Deploy API

on:
  push:
    branches: [main]
    paths: ["api/**"]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to EC2
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            cd /home/ubuntu/goat-api
            git pull origin main
            source venv/bin/activate
            pip install -r requirements.txt
            sudo systemctl restart goat-api
```

---

## Cost Estimate

| Service            | Monthly Cost   |
| ------------------ | -------------- |
| EC2 t3.medium      | ~$30           |
| RDS db.t3.micro    | ~$15           |
| S3 (estimate 10GB) | ~$2            |
| CloudFront         | ~$1            |
| Data transfer      | ~$5            |
| **Total**          | **~$50-60/mo** |

---

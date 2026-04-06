# Goat Grading API - `goat-api`

AI-powered grading service that analyzes goat and lamb images using YOLO segmentation models and returns body measurements plus a grade. Runs as a Docker container on EC2 (port 8000).

## What It Does

- Receives 3 images (side, top, front views) + live weight from the Pi
- Runs YOLO instance segmentation on each view to extract the animal's body mask
- Measures heights (side view), widths at anatomical positions (top view), and chest width (front view)
- Converts pixel measurements to centimeters using per-camera calibration files
- Calculates a grade from measurements + weight
- Generates debug overlay images with measurement annotations
- Archives raw images and debug overlays to S3 on successful grades
- Serves debug images via `/debug/{serial_id}/{view}` for the dashboard's grade review modal

## Architecture Decisions

**CPU-only PyTorch** - the EC2 instance is a t3.medium (no GPU). A GPU instance would be faster but costs 10x more for this workload.

**API key auth (not JWT)** - Pi-to-EC2 is service-to-service. A shared API key in `X-API-Key` header is simpler than JWT for this use case. The key is in environment variables on both machines.

**S3 archival on success only** - failed grades (missing views, bad detections) are not archived. This prevents polluting the buckets with unusable data. Archival runs in a background thread so it never blocks the API response.

**Debug image cleanup** - keeps only the 100 most recent serial IDs worth of debug images on disk. Older directories are pruned after each new analysis.

**Thread-safe inference** - a single `_inference_lock` serializes all YOLO inference calls. The models are not thread-safe, and concurrent inference would OOM on a 4GB instance.

## Measurement Strategy

- **Side view**: mask divided into thirds (front/middle/rear). Extracts head height, withers height, and rump height from ground level to the top of each third.
- **Top view**: widths measured at anatomical positions (shoulder, waist, rump) defined as configurable percentages from the tail end. Goat orientation (head left/right) is configured in `config.py`.
- **Front view**: chest width from the bounding box of the largest detected contour.
- **Calibration**: each camera has a `pixels_per_cm` value from a calibration tool. Measurements are converted from pixels to centimeters using these values.

## Directory Structure

```
goat-api/
├── Dockerfile              # python:3.11-slim + OpenCV system deps, copies model/ from repo root
├── requirements.txt        # PyTorch CPU, ultralytics, OpenCV, FastAPI, boto3
├── README.md
└── api/
    ├── __init__.py         # Package exports
    ├── main.py             # App setup, lifespan (model loading, S3 config), /analyze and /health endpoints
    ├── grader.py           # YOLO model loading, inference lock, measurement extraction from masks
    ├── grade_calculator.py # Tier tables, CI/MDR computation, grade assignment, reasoning details
    ├── api_auth.py         # API key middleware (X-API-Key header)
    ├── config.py           # Model paths, S3 buckets, thresholds, goat orientation config
    ├── models.py           # Pydantic schemas (AnalyzeResponse, MeasurementsResponse, etc.)
    ├── s3.py               # Lazy S3 client, archive_in_background() thread spawning
    ├── image_validation.py # Image validation, serial_id sanitization
    ├── image_debug.py      # save_debug_images(), old directory pruning, /debug endpoints via APIRouter
    └── logger.py           # Structured logging ([LEVEL] [component] message | key=value)

model/                      # At repo root, NOT inside goat-api/
├── side/
│   ├── best.pt             # YOLO segmentation weights (side view)
│   └── side_calibration.json
├── top/
│   ├── best.pt             # YOLO segmentation weights (top view)
│   └── top_calibration.json
└── front/
    ├── best.pt             # YOLO segmentation weights (front view)
    └── front_calibration.json
```

**Important**: The Dockerfile must be built from the repository root (`docker build -f goat-api/Dockerfile .`) because it copies `model/` from outside the `goat-api/` directory.

## Environment Variables

| Variable              | Required | Description                                                         |
| --------------------- | -------- | ------------------------------------------------------------------- |
| `API_KEY`             | Yes      | Shared secret for Pi authentication (must match Pi's `EC2_API_KEY`) |
| `S3_CAPTURES_BUCKET`  | Yes      | Bucket for raw capture images                                       |
| `S3_PROCESSED_BUCKET` | Yes      | Bucket for debug overlays                                           |
| `AWS_REGION`          | No       | AWS region (default: `us-east-2`)                                   |

## Grade Algorithm

Grades are calculated from three inputs per animal:

**Live weight (lbs)** - entered by the operator

**Condition Index (CI)** - avg_body_width_cm / withers_height_cm - measures overall body fill. A wider animal relative to its height indicates better condition.

**Muscle Distribution Ratio (MDR)** - rump_width_cm / waist_width_cm - measures hindquarter muscle development. A higher ratio means more muscle mass in the rear, which correlates with meat yield.

Grade tiers are defined per category (species × description: meat/dairy/cross for goats, lamb/ewe for lambs). An animal must meet all three minimums (weight, CI, MDR) to qualify for a tier. Tiers are checked top-down from best to worst - first match wins. Animals below all tiers receive "No Roll."
Graceful degradation: if a view fails and a ratio can't be computed, the algorithm grades with whatever data is available rather than returning no grade at all. Missing ratios are treated as passing, so the grade is effectively weight + the available ratio.
These thresholds are preliminary - calibrated from facility operator knowledge and initial measurements.

They will be refined as more animals are graded and validated against manual grades from the facility.

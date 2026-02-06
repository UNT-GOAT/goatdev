# Goat Grading API

AI-powered goat measurement and grading system.

## Overview

This API accepts images from three camera angles (side, top, front), runs YOLO segmentation models to extract body measurements, and calculates a grade based on measurements and live weight.

## Project Structure

```
goat-api/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes
│   ├── models.py            # Pydantic schemas
│   ├── grader.py            # YOLO model wrapper
│   ├── grade_calculator.py  # Measurements → grade (TBD)
│   ├── storage.py           # Results persistence
│   ├── logger.py            # Structured logging
│   └── config.py            # Configuration
├── data/
│   └── results.json         # Local results storage
├── Dockerfile
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- Model files in `../model/` directory:
  - `model/side/best.pt`
  - `model/top/best.pt`
  - `model/front/best.pt`
  - `model/side/side_calibration.json`
  - `model/top/top_calibration.json`
  - `model/front/front_calibration.json`

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn api.main:app --reload --port 8000
```

## Docker

```bash
# Build
docker build -t goat-api .

# Run (mount model directory)
docker run -d \
  --name goat-api \
  -p 8000:8000 \
  -v $(pwd)/../model:/app/model:ro \
  goat-api
```

## API Endpoints

### Health Checks

```bash
# Basic health
GET /health

# Deep health (runs test inference)
GET /health/deep
```

### Analyze

```bash
POST /analyze
Content-Type: multipart/form-data

Fields:
- serial_id: string (required) - Unique goat identifier
- live_weight: float (required) - Weight in lbs (10-500)
- side_image: file (required) - Side view image
- top_image: file (required) - Top view image
- front_image: file (required) - Front view image
```

Example with curl:
```bash
curl -X POST http://localhost:8000/analyze \
  -F "serial_id=GOAT001" \
  -F "live_weight=75.5" \
  -F "side_image=@/path/to/side.jpg" \
  -F "top_image=@/path/to/top.jpg" \
  -F "front_image=@/path/to/front.jpg"
```

Response:
```json
{
  "serial_id": "GOAT001",
  "timestamp": "2026-02-06T12:00:00",
  "live_weight_lbs": 75.5,
  "measurements": {
    "head_height_cm": 72.73,
    "withers_height_cm": 53.09,
    "rump_height_cm": 55.47,
    "top_body_width_cm": 34.51,
    "front_body_width_cm": 32.31,
    "avg_body_width_cm": 33.41
  },
  "confidence_scores": {
    "side": 0.949,
    "top": 0.969,
    "front": 0.959
  },
  "grade": "Choice",
  "all_views_successful": true,
  "view_errors": null,
  "warnings": null,
  "success": true
}
```

### Results

```bash
# Get all results
GET /results

# Get single result
GET /results/{serial_id}

# Delete result
DELETE /results/{serial_id}
```

## Logging

Logs follow a structured format for CloudWatch:

```
[LEVEL] [component] message | key=value key2=value2
```

Examples:
```
[INFO] [startup] API ready
[INFO] [analyze] Request received | serial_id=GOAT001 weight=75.5
[ERROR] [grader:model:side] No goat detected | serial_id=GOAT001
[INFO] [analyze] Request complete | serial_id=GOAT001 grade=Choice duration_sec=2.34
```

## Configuration

Key settings in `api/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MIN_IMAGE_DIMENSION` | 100 | Minimum image width/height in pixels |
| `MAX_FILE_SIZE_BYTES` | 20MB | Maximum upload size |
| `MIN_CONFIDENCE_THRESHOLD` | 0.1 | YOLO detection threshold |
| `WARN_CONFIDENCE_THRESHOLD` | 0.3 | Warn if confidence below this |
| `MIN_WEIGHT_LBS` | 10 | Minimum valid weight |
| `MAX_WEIGHT_LBS` | 500 | Maximum valid weight |

## Grade Values

Valid grades (best to worst):
- Reserve
- CAB Prime
- Prime
- CAB Choice
- Choice
- Select
- No Roll

**Note:** Grade calculation is currently a placeholder. The actual algorithm needs to be implemented in `api/grade_calculator.py`.

## Error Handling

All errors include:
- `error`: Human-readable message
- `error_code`: Machine-readable code
- `fix`: Suggested action (when applicable)

Example error response:
```json
{
  "error": "No goat detected in side image",
  "error_code": "NO_DETECTION",
  "fix": "Ensure goat is fully visible in side camera view"
}
```

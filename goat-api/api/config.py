"""
Configuration for Goat Grading API
"""

from pathlib import Path
import os

# =============================================================
# GOAT ORIENTATION CONFIGURATION
#
# Configure which direction the goat faces in each view
# This helps with cross-view measurement alignment
#
# Options: 'left', 'right'
# - 'left' means head is on the left side of the image
# - 'right' means head is on the right side of the image
#
#
# This will be adjusted once we have cameras set up and can confirm orientations

TOP_VIEW_DIRECTION = 'right'   # Goat faces right in top view
TOP_RUMP_PCT = 0.20            # back legs ~25% from tail end
TOP_SHOULDER_PCT = 0.80        # front legs ~80% from tail end
#=============================================================

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"  # ../model relative to goat-api/
DATA_DIR = BASE_DIR / "data"

# Model paths
SIDE_MODEL_PATH = MODEL_DIR / "side" / "best.pt"
TOP_MODEL_PATH = MODEL_DIR / "top" / "best.pt"
FRONT_MODEL_PATH = MODEL_DIR / "front" / "best.pt"

# Calibration paths
SIDE_CALIBRATION_PATH = MODEL_DIR / "side" / "side_calibration.json"
TOP_CALIBRATION_PATH = MODEL_DIR / "top" / "top_calibration.json"
FRONT_CALIBRATION_PATH = MODEL_DIR / "front" / "front_calibration.json"

# Results storage
RESULTS_FILE = DATA_DIR / "results.json"

# S3 buckets
S3_CAPTURES_BUCKET = os.environ.get("S3_CAPTURES_BUCKET", "goat-captures-937249941844")
S3_PROCESSED_BUCKET = os.environ.get("S3_PROCESSED_BUCKET", "goat-processed-937249941844")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# Validation thresholds
MIN_IMAGE_DIMENSION = 100  # pixels
MAX_IMAGE_SIZE_MB = 20          # TODO: adjust once working with new cameras
MAX_FILE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

MIN_CONFIDENCE_THRESHOLD = 0.1  # YOLO detection threshold
WARN_CONFIDENCE_THRESHOLD = 0.3  # Warn if below this

# Measurement sanity bounds (cm)
MIN_MEASUREMENT_CM = 5
MAX_MEASUREMENT_CM = 200

# Weight bounds (lbs)
MIN_WEIGHT_LBS = 20
MAX_WEIGHT_LBS = 300

# Timeouts
MODEL_INFERENCE_TIMEOUT_SEC = 30
S3_UPLOAD_TIMEOUT_SEC = 60

# Valid image extensions
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Grade values (for validation)
VALID_GRADES = [
    "Reserve",
    "CAB Prime",
    "Prime",
    "CAB Choice",
    "Choice",
    "Select",
    "No Roll"
]

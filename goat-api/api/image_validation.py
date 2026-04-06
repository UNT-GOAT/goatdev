"""
Input Validation

Pure validation functions for images and serial IDs.
No app or request dependencies — just data in, result out.
"""

import re
import cv2
import numpy as np
from typing import Optional

from fastapi import UploadFile

from .logger import log
from .config import (
    MIN_IMAGE_DIMENSION, MAX_FILE_SIZE_BYTES, VALID_IMAGE_EXTENSIONS
)


def validate_image(
    file: UploadFile, view_name: str
) -> tuple[Optional[np.ndarray], Optional[bytes], Optional[dict]]:
    """
    Validate and decode an uploaded image.

    Returns:
        (image_array, raw_bytes, error_dict) — on success error_dict is None,
        on failure image_array and raw_bytes are None.

    raw_bytes is preserved for S3 archival so we only read the upload once.
    """
    # Check filename
    if not file.filename:
        return None, None, {
            "error": f"No filename for {view_name} image",
            "error_code": "MISSING_FILENAME",
            "fix": "Ensure file has a name"
        }

    # Check extension
    ext = '.' + file.filename.lower().split('.')[-1] if '.' in file.filename else ''
    if ext not in VALID_IMAGE_EXTENSIONS:
        return None, None, {
            "error": f"Invalid file type for {view_name}: {ext}",
            "error_code": "INVALID_FILE_TYPE",
            "fix": f"Use one of: {', '.join(VALID_IMAGE_EXTENSIONS)}"
        }

    # Read file content (single read — reused for both decoding and S3 archival)
    try:
        content = file.file.read()
    except Exception as e:
        return None, None, {
            "error": f"Failed to read {view_name} image: {e}",
            "error_code": "READ_ERROR",
            "fix": "Ensure file is not corrupted"
        }

    # Check size
    if len(content) > MAX_FILE_SIZE_BYTES:
        size_mb = len(content) / (1024 * 1024)
        return None, None, {
            "error": f"{view_name} image too large: {size_mb:.1f}MB",
            "error_code": "FILE_TOO_LARGE",
            "fix": f"Maximum size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB"
        }

    # Decode image
    try:
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return None, None, {
            "error": f"Failed to decode {view_name} image: {e}",
            "error_code": "DECODE_ERROR",
            "fix": "Ensure image is valid JPEG/PNG"
        }

    if image is None:
        return None, None, {
            "error": f"{view_name} image is corrupted or invalid format",
            "error_code": "INVALID_IMAGE",
            "fix": "Re-capture the image"
        }

    # Check dimensions
    height, width = image.shape[:2]
    if height < MIN_IMAGE_DIMENSION or width < MIN_IMAGE_DIMENSION:
        return None, None, {
            "error": f"{view_name} image too small: {width}x{height}",
            "error_code": "IMAGE_TOO_SMALL",
            "fix": f"Minimum dimension is {MIN_IMAGE_DIMENSION}px"
        }

    return image, content, None


def sanitize_serial_id(serial_id: str) -> tuple[Optional[str], Optional[dict]]:
    """Validate and sanitize serial_id."""
    if not serial_id:
        return None, {
            "error": "serial_id is required",
            "error_code": "MISSING_SERIAL_ID"
        }

    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', serial_id.strip())

    if not sanitized:
        return None, {
            "error": "serial_id must contain alphanumeric characters",
            "error_code": "INVALID_SERIAL_ID",
            "fix": "Use letters, numbers, underscore, or hyphen only"
        }

    if len(sanitized) > 50:
        return None, {
            "error": "serial_id too long",
            "error_code": "SERIAL_ID_TOO_LONG",
            "fix": "Maximum 50 characters"
        }

    if sanitized != serial_id.strip():
        log.warn('validate', 'serial_id sanitized',
                 original=serial_id[:50], sanitized=sanitized)

    return sanitized, None
"""
Goat Grading API

FastAPI application for analyzing goat images and calculating grades.
"""

import cv2
import numpy as np
import time
import os
import json
import shutil
import psutil
import threading
from io import BytesIO
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .logger import log
from .config import (
    MIN_IMAGE_DIMENSION, MAX_FILE_SIZE_BYTES, VALID_IMAGE_EXTENSIONS,
    MIN_WEIGHT_LBS, MAX_WEIGHT_LBS,
    S3_CAPTURES_BUCKET, S3_PROCESSED_BUCKET
)
from .models import (
    AnalyzeResponse, MeasurementsResponse, ConfidenceScores, ViewError,
    HealthResponse, DeepHealthResponse, ResultsResponse, ErrorResponse
)
from .grader import grader
from .storage import storage
from .grade_calculator import calculate_grade


# Maximum number of serial_id debug directories to keep.
# Once exceeded, oldest directories (by mtime) are pruned after each
# new analysis. Prevents /app/data/debug/ from growing unbounded.
DEBUG_DIR_BASE = '/app/data/debug'
MAX_DEBUG_SERIAL_IDS = 100


# =============================================================================
# S3 HELPER
# =============================================================================
# EC2 owns ALL S3 archival. The Pi does not write to S3.
#
# Responsibility:
#   goat-captures bucket  — raw uploaded images (as received from Pi)
#   goat-processed bucket — debug overlay images + result.json
#
# Archival only happens on successful grades (all views OK).
# Runs in a background thread so it never blocks the API response.
# =============================================================================

_s3_client = None
_s3_lock = threading.Lock()


def get_s3():
    """Lazy-init S3 client. Thread-safe."""
    global _s3_client
    if _s3_client is None:
        with _s3_lock:
            if _s3_client is None:
                import boto3
                _s3_client = boto3.client('s3')
    return _s3_client


def _s3_upload_bytes(bucket: str, key: str, data: bytes, content_type: str = 'image/jpeg'):
    """Upload bytes to S3. Logs warning on failure, never raises."""
    try:
        get_s3().put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        log.info('s3', 'Uploaded', bucket=bucket, key=key, size_bytes=len(data))
    except Exception as e:
        log.warn('s3', 'Upload failed (non-fatal)', bucket=bucket, key=key, error=str(e))


def _s3_upload_file(bucket: str, key: str, filepath: str, content_type: str = 'image/jpeg'):
    """Upload a local file to S3. Logs warning on failure, never raises."""
    try:
        get_s3().upload_file(filepath, bucket, key, ExtraArgs={'ContentType': content_type})
        log.info('s3', 'Uploaded file', bucket=bucket, key=key,
                size_bytes=os.path.getsize(filepath))
    except Exception as e:
        log.warn('s3', 'File upload failed (non-fatal)', bucket=bucket, key=key, error=str(e))


def _archive_to_s3(
    serial_id: str,
    raw_images: dict,
    debug_image_paths: dict,
    result_dict: dict
):
    """
    Archive a successful grade to S3. Runs in a background thread.

    Writes:
      goat-captures/{serial_id}/{view}.jpg        — raw images
      goat-processed/{serial_id}/{view}_debug.jpg  — debug overlays
      goat-processed/{serial_id}/result.json       — full grade result

    Args:
        serial_id: Goat identifier
        raw_images: Dict of {view_name: bytes} for the raw uploaded images
        debug_image_paths: Dict of {view_name: filepath} for debug images on disk
        result_dict: The full AnalyzeResponse as a dict
    """
    try:
        # Raw images → captures bucket
        if S3_CAPTURES_BUCKET:
            for view_name, raw_bytes in raw_images.items():
                _s3_upload_bytes(
                    S3_CAPTURES_BUCKET,
                    f'{serial_id}/{view_name}.jpg',
                    raw_bytes
                )
        else:
            log.warn('s3:captures', 'S3_CAPTURES_BUCKET not set, skipping')

        # Debug images + result → processed bucket
        if S3_PROCESSED_BUCKET:
            for view_name, debug_path in debug_image_paths.items():
                if os.path.exists(debug_path):
                    _s3_upload_file(
                        S3_PROCESSED_BUCKET,
                        f'{serial_id}/{view_name}_debug.jpg',
                        debug_path
                    )

            # Include S3 paths in result for cross-referencing
            result_with_paths = dict(result_dict)
            result_with_paths['s3'] = {
                'captures_bucket': S3_CAPTURES_BUCKET,
                'processed_bucket': S3_PROCESSED_BUCKET,
                'raw_images': {v: f'{serial_id}/{v}.jpg' for v in raw_images},
                'debug_images': {v: f'{serial_id}/{v}_debug.jpg' for v in debug_image_paths},
                'result_key': f'{serial_id}/result.json'
            }

            result_json = json.dumps(result_with_paths, indent=2, default=str)
            _s3_upload_bytes(
                S3_PROCESSED_BUCKET,
                f'{serial_id}/result.json',
                result_json.encode('utf-8'),
                content_type='application/json'
            )
        else:
            log.warn('s3:processed', 'S3_PROCESSED_BUCKET not set, skipping')

        log.info('s3', 'Archival complete', serial_id=serial_id)

    except Exception as e:
        # Catch-all so the background thread never crashes silently
        log.warn('s3', 'Archival failed (non-fatal)', serial_id=serial_id, error=str(e))


# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    log.info('startup', '=' * 50)
    log.info('startup', 'GOAT GRADING API STARTING')
    
    # Initialize storage
    storage_ok, storage_error = storage.initialize()
    if not storage_ok:
        log.critical('startup', 'Storage initialization failed',
                    error=storage_error,
                    fix='Check data directory permissions and disk space')
    else:
        log.info('startup', 'Storage initialized')
    
    # Initialize grader (loads models)
    grader_ok, grader_errors = grader.initialize()
    if not grader_ok:
        log.critical('startup', 'Grader initialization failed',
                    errors=len(grader_errors),
                    fix='Check model files exist and are valid')
        for error in grader_errors:
            log.error('startup', error)
    else:
        log.info('startup', 'Grader initialized', gpu=grader.gpu_available)
    
    # Log S3 config
    log.info('startup', 'S3 archival config',
            captures_bucket=S3_CAPTURES_BUCKET or '(not set)',
            processed_bucket=S3_PROCESSED_BUCKET or '(not set)')
    
    # Log system info
    memory = psutil.virtual_memory()
    log.info('startup', 'System info',
            memory_total_gb=round(memory.total / (1024**3), 1),
            memory_available_gb=round(memory.available / (1024**3), 1),
            cpu_count=psutil.cpu_count())
    
    log.info('startup', 'API ready')
    log.info('startup', '=' * 50)
    
    yield
    
    # Shutdown
    log.info('shutdown', 'API shutting down')


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="Goat Grading API",
    description="AI-powered goat measurement and grading system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    log.warn('http', 'HTTP error', 
            status=exc.status_code, 
            detail=str(exc.detail),
            path=request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "error_code": f"HTTP_{exc.status_code}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    log.exception('http', 'Unhandled exception',
                 path=request.url.path,
                 error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "fix": "Check server logs for details"
        }
    )


# =============================================================================
# HELPERS
# =============================================================================

def validate_image(file: UploadFile, view_name: str) -> tuple[Optional[np.ndarray], Optional[bytes], Optional[dict]]:
    """
    Validate and decode an uploaded image.
    
    Returns:
        (image_array, raw_bytes, error_dict) - on success error_dict is None,
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
            "fix": f"Maximum size is {MAX_FILE_SIZE_BYTES // (1024*1024)}MB"
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
    """Validate and sanitize serial_id"""
    if not serial_id:
        return None, {
            "error": "serial_id is required",
            "error_code": "MISSING_SERIAL_ID"
        }
    
    import re
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


def _cleanup_old_debug_dirs():
    """
    Remove oldest debug directories when count exceeds MAX_DEBUG_SERIAL_IDS.

    Previously debug images accumulated forever with no cleanup policy.
    On a busy system doing 50+ grades/day, this would fill the disk within
    weeks. Now we keep only the most recent MAX_DEBUG_SERIAL_IDS serial IDs
    worth of debug images, pruning the oldest (by directory mtime) after
    each new analysis.

    This runs synchronously after saving debug images — it's fast since
    it only does a listdir + stat on the debug base dir, and only removes
    directories when we're over the limit.
    """
    try:
        if not os.path.isdir(DEBUG_DIR_BASE):
            return

        entries = []
        for name in os.listdir(DEBUG_DIR_BASE):
            full_path = os.path.join(DEBUG_DIR_BASE, name)
            if os.path.isdir(full_path):
                try:
                    mtime = os.path.getmtime(full_path)
                    entries.append((mtime, full_path, name))
                except OSError:
                    continue

        if len(entries) <= MAX_DEBUG_SERIAL_IDS:
            return

        # Sort by mtime ascending (oldest first)
        entries.sort(key=lambda x: x[0])
        to_remove = len(entries) - MAX_DEBUG_SERIAL_IDS

        for i in range(to_remove):
            _, dir_path, dir_name = entries[i]
            try:
                shutil.rmtree(dir_path)
                log.info('debug_cleanup', 'Removed old debug dir',
                        serial_id=dir_name)
            except Exception as e:
                log.warn('debug_cleanup', 'Failed to remove debug dir',
                        serial_id=dir_name, error=str(e))

        log.info('debug_cleanup', 'Cleanup complete',
                removed=to_remove,
                remaining=MAX_DEBUG_SERIAL_IDS)

    except Exception as e:
        # Never let cleanup failure affect the grading response
        log.warn('debug_cleanup', 'Cleanup error (non-fatal)', error=str(e))


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Basic health check"""
    return HealthResponse(
        status="ok" if grader.is_initialized and storage.is_initialized else "degraded",
        timestamp=datetime.now().isoformat(),
        models_loaded=grader.is_initialized,
        side_model=grader.side_model is not None,
        top_model=grader.top_model is not None,
        front_model=grader.front_model is not None,
        storage_ok=storage.is_initialized
    )


@app.get("/health/deep", response_model=DeepHealthResponse)
async def health_deep():
    """Deep health check with inference test"""
    log.info('health', 'Running deep health check')
    
    inference_ok = False
    inference_time = None
    
    if grader.is_initialized:
        inference_ok, inference_time = grader.run_test_inference()
    
    memory = psutil.virtual_memory()
    
    return DeepHealthResponse(
        status="ok" if inference_ok else "degraded",
        timestamp=datetime.now().isoformat(),
        models_loaded=grader.is_initialized,
        side_model=grader.side_model is not None,
        top_model=grader.top_model is not None,
        front_model=grader.front_model is not None,
        storage_ok=storage.is_initialized,
        inference_test=inference_ok,
        inference_time_ms=inference_time,
        memory_usage_mb=round((memory.total - memory.available) / (1024**2), 1),
        gpu_available=grader.gpu_available
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    serial_id: str = Form(...),
    live_weight: float = Form(...),
    side_image: UploadFile = File(...),
    top_image: UploadFile = File(...),
    front_image: UploadFile = File(...)
):
    """
    Analyze goat images and return measurements + grade.
    
    Expects multipart form with:
    - serial_id: Unique goat identifier
    - live_weight: Weight in lbs
    - side_image: Side view image file
    - top_image: Top view image file
    - front_image: Front view image file
    
    On successful grades (all views OK), archives to S3 in a background thread:
    - Raw images → goat-captures bucket: {serial_id}/{view}.jpg
    - Debug overlays → goat-processed bucket: {serial_id}/{view}_debug.jpg
    - Result JSON → goat-processed bucket: {serial_id}/result.json
    
    S3 archival is skipped for failed grades to avoid polluting buckets
    with unusable data.
    """
    start_time = time.time()
    
    # Validate serial_id
    serial_id, error = sanitize_serial_id(serial_id)
    if error:
        log.warn('analyze', 'Invalid serial_id', error=error['error'])
        raise HTTPException(status_code=400, detail=error)
    
    log.info('analyze', 'Request received', serial_id=serial_id, weight=live_weight)
    
    # Check grader is ready
    if not grader.is_initialized:
        log.error('analyze', 'Grader not ready', serial_id=serial_id)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Grading service not ready",
                "error_code": "SERVICE_UNAVAILABLE",
                "fix": "Wait for models to load or restart server"
            }
        )
    
    # Validate weight
    if live_weight < MIN_WEIGHT_LBS or live_weight > MAX_WEIGHT_LBS:
        log.warn('analyze', 'Invalid weight', serial_id=serial_id, weight=live_weight)
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"live_weight must be between {MIN_WEIGHT_LBS} and {MAX_WEIGHT_LBS} lbs",
                "error_code": "INVALID_WEIGHT",
                "fix": "Check the weight value"
            }
        )
    
    # Validate and decode images.
    # validate_image returns (cv2_array, raw_bytes, error) — raw_bytes is
    # preserved for S3 archival so we only read each upload once.
    raw_images = {}
    
    side_img, side_raw, side_error = validate_image(side_image, "side")
    if side_error:
        log.error('analyze', 'Side image validation failed',
                 serial_id=serial_id, error=side_error['error'])
        raise HTTPException(status_code=400, detail=side_error)
    raw_images['side'] = side_raw
    
    top_img, top_raw, top_error = validate_image(top_image, "top")
    if top_error:
        log.error('analyze', 'Top image validation failed',
                 serial_id=serial_id, error=top_error['error'])
        raise HTTPException(status_code=400, detail=top_error)
    raw_images['top'] = top_raw
    
    front_img, front_raw, front_error = validate_image(front_image, "front")
    if front_error:
        log.error('analyze', 'Front image validation failed',
                 serial_id=serial_id, error=front_error['error'])
        raise HTTPException(status_code=400, detail=front_error)
    raw_images['front'] = front_raw
    
    log.info('analyze', 'Images validated',
            serial_id=serial_id,
            side_shape=f"{side_img.shape[1]}x{side_img.shape[0]}",
            top_shape=f"{top_img.shape[1]}x{top_img.shape[0]}",
            front_shape=f"{front_img.shape[1]}x{front_img.shape[0]}")
    
    # Process images
    grader_result = grader.process_images(side_img, top_img, front_img, serial_id)
    
    # Save debug images to disk (for /debug endpoint serving)
    debug_image_paths = {}
    if grader_result.get('debug_images'):
        debug_dir = f'{DEBUG_DIR_BASE}/{serial_id}'
        os.makedirs(debug_dir, exist_ok=True)
        
        for view_name, debug_img in grader_result['debug_images'].items():
            if debug_img is not None:
                debug_path = f'{debug_dir}/{view_name}_debug.jpg'
                try:
                    cv2.imwrite(debug_path, debug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    debug_image_paths[view_name] = debug_path
                    log.info('analyze', f'Saved debug image', 
                            serial_id=serial_id, view=view_name, path=debug_path)
                except Exception as e:
                    log.error('analyze', f'Failed to save debug image',
                             serial_id=serial_id, view=view_name, error=str(e))
        
        # Prune old debug directories
        _cleanup_old_debug_dirs()
    
    # Calculate grade
    grade = None
    if grader_result.get('all_views_successful'):
        grade = calculate_grade(
            grader_result['measurements'],
            live_weight,
            serial_id
        )
    else:
        log.warn('analyze', 'Skipping grade calculation due to view failures',
                serial_id=serial_id)
    
    # Build response
    measurements = MeasurementsResponse(
        head_height_cm=grader_result['measurements'].get('head_height_cm'),
        withers_height_cm=grader_result['measurements'].get('withers_height_cm'),
        rump_height_cm=grader_result['measurements'].get('rump_height_cm'),
        shoulder_width_cm=grader_result['measurements'].get('shoulder_width_cm'),
        waist_width_cm=grader_result['measurements'].get('waist_width_cm'),
        rump_width_cm=grader_result['measurements'].get('rump_width_cm'),
        top_body_width_cm=grader_result['measurements'].get('top_body_width_cm'),
        front_body_width_cm=grader_result['measurements'].get('front_body_width_cm'),
        avg_body_width_cm=grader_result['measurements'].get('avg_body_width_cm')
    )
    
    confidence = ConfidenceScores(
        side=grader_result['confidence_scores'].get('side'),
        top=grader_result['confidence_scores'].get('top'),
        front=grader_result['confidence_scores'].get('front')
    )
    
    view_errors = None
    if grader_result.get('view_errors'):
        view_errors = [
            ViewError(view=e['view'], error=e['error'], fix=e.get('fix'))
            for e in grader_result['view_errors']
        ]
    
    response = AnalyzeResponse(
        serial_id=serial_id,
        timestamp=datetime.now().isoformat(),
        live_weight_lbs=live_weight,
        measurements=measurements,
        confidence_scores=confidence,
        grade=grade,
        all_views_successful=grader_result['all_views_successful'],
        view_errors=view_errors,
        warnings=grader_result.get('warnings'),
        success=True
    )
    
    # Save result to local storage
    result_dict = response.model_dump()
    save_ok, save_error = storage.save_result(result_dict)
    if not save_ok:
        log.error('analyze', 'Failed to save result',
                 serial_id=serial_id, error=save_error)
    
    # =========================================================================
    # S3 ARCHIVAL (background, success-only)
    # =========================================================================
    # Only archive when grading actually produced useful results.
    # Failed grades (missing views, bad detections, etc.) are not worth
    # storing — they'd just pollute the buckets with unusable data.
    #
    # Runs in a background thread so the API response returns immediately.
    # =========================================================================
    if response.success and response.all_views_successful:
        thread = threading.Thread(
            target=_archive_to_s3,
            args=(serial_id, raw_images, debug_image_paths, result_dict),
            daemon=True
        )
        thread.start()
        log.info('analyze', 'S3 archival started in background', serial_id=serial_id)
    else:
        log.info('analyze', 'Skipping S3 archival (grade not fully successful)',
                serial_id=serial_id,
                success=response.success,
                all_views_ok=response.all_views_successful)

    duration = round(time.time() - start_time, 2)
    log.info('analyze', 'Request complete',
            serial_id=serial_id,
            grade=grade,
            all_views_ok=grader_result['all_views_successful'],
            duration_sec=duration)
    
    return response


@app.get("/results", response_model=ResultsResponse)
async def get_results():
    """Get all analysis results"""
    results = storage.get_all_results()
    
    # Convert to response models
    response_results = []
    for r in results:
        try:
            response_results.append(AnalyzeResponse(**r))
        except Exception as e:
            log.warn('results', 'Skipping invalid result', error=str(e))
    
    return ResultsResponse(
        total_results=len(response_results),
        results=response_results
    )


@app.get("/results/{serial_id}", response_model=AnalyzeResponse)
async def get_result(serial_id: str):
    """Get a single analysis result by serial_id"""
    # Sanitize
    serial_id, error = sanitize_serial_id(serial_id)
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    result = storage.get_result(serial_id)
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Result not found for serial_id: {serial_id}",
                "error_code": "NOT_FOUND"
            }
        )
    
    return AnalyzeResponse(**result)


@app.delete("/results/{serial_id}")
async def delete_result(serial_id: str):
    """Delete an analysis result"""
    serial_id, error = sanitize_serial_id(serial_id)
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    if storage.delete_result(serial_id):
        return {"status": "deleted", "serial_id": serial_id}
    else:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Result not found for serial_id: {serial_id}",
                "error_code": "NOT_FOUND"
            }
        )


@app.get("/debug/{serial_id}/{view}")
async def get_debug_image(serial_id: str, view: str):
    """
    Get debug image for a specific serial_id and view.
    
    Args:
        serial_id: Goat identifier
        view: One of 'side', 'top', 'front'
        
    Returns:
        JPEG image with measurement overlays
    """
    from fastapi.responses import FileResponse
    
    # Validate view
    if view not in ['side', 'top', 'front']:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid view: {view}",
                "error_code": "INVALID_VIEW",
                "fix": "View must be one of: side, top, front"
            }
        )
    
    # Sanitize serial_id
    serial_id, error = sanitize_serial_id(serial_id)
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    # Check if debug image exists
    debug_path = f'{DEBUG_DIR_BASE}/{serial_id}/{view}_debug.jpg'
    
    if not os.path.exists(debug_path):
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Debug image not found for {serial_id}/{view}",
                "error_code": "NOT_FOUND",
                "fix": "Ensure the goat was analyzed and debug images were generated"
            }
        )
    
    return FileResponse(
        debug_path,
        media_type="image/jpeg",
        filename=f"{serial_id}_{view}_debug.jpg"
    )


@app.get("/debug/{serial_id}")
async def list_debug_images(serial_id: str):
    """
    List available debug images for a serial_id.
    
    Returns:
        List of available views with URLs
    """
    # Sanitize serial_id
    serial_id, error = sanitize_serial_id(serial_id)
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    debug_dir = f'{DEBUG_DIR_BASE}/{serial_id}'
    
    if not os.path.exists(debug_dir):
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"No debug images found for serial_id: {serial_id}",
                "error_code": "NOT_FOUND"
            }
        )
    
    available_views = []
    for view in ['side', 'top', 'front']:
        debug_path = f'{debug_dir}/{view}_debug.jpg'
        if os.path.exists(debug_path):
            available_views.append({
                "view": view,
                "url": f"/debug/{serial_id}/{view}",
                "filename": f"{view}_debug.jpg"
            })
    
    return {
        "serial_id": serial_id,
        "debug_images": available_views,
        "count": len(available_views)
    }
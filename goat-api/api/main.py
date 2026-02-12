"""
Goat Grading API

FastAPI application for analyzing goat images and calculating grades.
"""

import cv2
import numpy as np
import time
import psutil
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
    MIN_WEIGHT_LBS, MAX_WEIGHT_LBS
)
from .models import (
    AnalyzeResponse, MeasurementsResponse, ConfidenceScores, ViewError,
    HealthResponse, DeepHealthResponse, ResultsResponse, ErrorResponse
)
from .grader import grader
from .storage import storage
from .grade_calculator import calculate_grade


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

def validate_image(file: UploadFile, view_name: str) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Validate and decode an uploaded image.
    
    Returns:
        (image_array, error_dict) - one will be None
    """
    # Check filename
    if not file.filename:
        return None, {
            "error": f"No filename for {view_name} image",
            "error_code": "MISSING_FILENAME",
            "fix": "Ensure file has a name"
        }
    
    # Check extension
    ext = '.' + file.filename.lower().split('.')[-1] if '.' in file.filename else ''
    if ext not in VALID_IMAGE_EXTENSIONS:
        return None, {
            "error": f"Invalid file type for {view_name}: {ext}",
            "error_code": "INVALID_FILE_TYPE",
            "fix": f"Use one of: {', '.join(VALID_IMAGE_EXTENSIONS)}"
        }
    
    # Read file content
    try:
        content = file.file.read()
        file.file.seek(0)  # Reset for potential re-read
    except Exception as e:
        return None, {
            "error": f"Failed to read {view_name} image: {e}",
            "error_code": "READ_ERROR",
            "fix": "Ensure file is not corrupted"
        }
    
    # Check size
    if len(content) > MAX_FILE_SIZE_BYTES:
        size_mb = len(content) / (1024 * 1024)
        return None, {
            "error": f"{view_name} image too large: {size_mb:.1f}MB",
            "error_code": "FILE_TOO_LARGE",
            "fix": f"Maximum size is {MAX_FILE_SIZE_BYTES // (1024*1024)}MB"
        }
    
    # Decode image
    try:
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return None, {
            "error": f"Failed to decode {view_name} image: {e}",
            "error_code": "DECODE_ERROR",
            "fix": "Ensure image is valid JPEG/PNG"
        }
    
    if image is None:
        return None, {
            "error": f"{view_name} image is corrupted or invalid format",
            "error_code": "INVALID_IMAGE",
            "fix": "Re-capture the image"
        }
    
    # Check dimensions
    height, width = image.shape[:2]
    if height < MIN_IMAGE_DIMENSION or width < MIN_IMAGE_DIMENSION:
        return None, {
            "error": f"{view_name} image too small: {width}x{height}",
            "error_code": "IMAGE_TOO_SMALL",
            "fix": f"Minimum dimension is {MIN_IMAGE_DIMENSION}px"
        }
    
    return image, None


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
    
    # Validate and decode images
    side_img, side_error = validate_image(side_image, "side")
    if side_error:
        log.error('analyze', 'Side image validation failed',
                 serial_id=serial_id, error=side_error['error'])
        raise HTTPException(status_code=400, detail=side_error)
    
    top_img, top_error = validate_image(top_image, "top")
    if top_error:
        log.error('analyze', 'Top image validation failed',
                 serial_id=serial_id, error=top_error['error'])
        raise HTTPException(status_code=400, detail=top_error)
    
    front_img, front_error = validate_image(front_image, "front")
    if front_error:
        log.error('analyze', 'Front image validation failed',
                 serial_id=serial_id, error=front_error['error'])
        raise HTTPException(status_code=400, detail=front_error)
    
    log.info('analyze', 'Images validated',
            serial_id=serial_id,
            side_shape=f"{side_img.shape[1]}x{side_img.shape[0]}",
            top_shape=f"{top_img.shape[1]}x{top_img.shape[0]}",
            front_shape=f"{front_img.shape[1]}x{front_img.shape[0]}")
    
    # Process images
    grader_result = grader.process_images(side_img, top_img, front_img, serial_id)
    
    # Save debug images if generated
    debug_image_paths = {}
    if grader_result.get('debug_images'):
        import os
        debug_dir = f'/app/data/debug/{serial_id}'
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
        # Heights from side view
        head_height_cm=grader_result['measurements'].get('head_height_cm'),
        withers_height_cm=grader_result['measurements'].get('withers_height_cm'),
        rump_height_cm=grader_result['measurements'].get('rump_height_cm'),
        # Widths from top view (cross-referenced with side view leg positions)
        shoulder_width_cm=grader_result['measurements'].get('shoulder_width_cm'),
        waist_width_cm=grader_result['measurements'].get('waist_width_cm'),
        rump_width_cm=grader_result['measurements'].get('rump_width_cm'),
        # Fallback/legacy
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
    
    # Save result
    result_dict = response.model_dump()
    save_ok, save_error = storage.save_result(result_dict)
    if not save_ok:
        log.error('analyze', 'Failed to save result',
                 serial_id=serial_id, error=save_error)
        # Don't fail the request, just log the error
    
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
    import os
    
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
    debug_path = f'/app/data/debug/{serial_id}/{view}_debug.jpg'
    
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
    import os
    
    # Sanitize serial_id
    serial_id, error = sanitize_serial_id(serial_id)
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    debug_dir = f'/app/data/debug/{serial_id}'
    
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

"""
Goat Grading API

FastAPI application for analyzing goat images and calculating grades.
"""

import time
import psutil
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .logger import log
from .config import (
    MIN_WEIGHT_LBS, MAX_WEIGHT_LBS,
    S3_CAPTURES_BUCKET, S3_PROCESSED_BUCKET
)
from .models import (
    AnalyzeResponse, MeasurementsResponse, ConfidenceScores, ViewError,
    HealthResponse, DeepHealthResponse
)
from .grader import grader
from .grade_calculator import calculate_grade
from .image_validation import validate_image, sanitize_serial_id
from .s3 import archive_in_background
from .image_debug import save_debug_images, router as debug_router
from .api_auth import APIKeyMiddleware, API_KEY


# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    log.info('startup', '=' * 50)
    log.info('startup', 'GOAT GRADING API STARTING')

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

    # Log API key status
    if not API_KEY:
        log.critical('startup', 'API_KEY NOT SET — all non-health requests will be rejected',
                     fix='Set API_KEY environment variable')
    else:
        log.info('startup', 'API key configured')

    # Log system info
    memory = psutil.virtual_memory()
    log.info('startup', 'System info',
             memory_total_gb=round(memory.total / (1024**3), 1),
             memory_available_gb=round(memory.available / (1024**3), 1),
             cpu_count=psutil.cpu_count())

    log.info('startup', 'API ready')
    log.info('startup', '=' * 50)

    yield

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

# API key authentication — all non-health endpoints require X-API-Key header.
# CORS middleware must be outermost (added first) so preflight OPTIONS
# requests get CORS headers before hitting auth.
app.add_middleware(APIKeyMiddleware)

# Debug image endpoints
app.include_router(debug_router)


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
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Basic health check"""
    return HealthResponse(
        status="ok" if grader.is_initialized else "degraded",
        timestamp=datetime.now().isoformat(),
        models_loaded=grader.is_initialized,
        side_model=grader.side_model is not None,
        top_model=grader.top_model is not None,
        front_model=grader.front_model is not None,
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
        inference_test=inference_ok,
        inference_time_ms=inference_time,
        memory_usage_mb=round((memory.total - memory.available) / (1024**2), 1),
        gpu_available=grader.gpu_available
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    serial_id: str = Form(...),
    live_weight: float = Form(...),
    species: str = Form("goat"),
    description: str = Form("meat"),
    side_image: UploadFile = File(...),
    top_image: UploadFile = File(...),
    front_image: UploadFile = File(...)
):
    """
    Analyze goat images and return measurements + grade.

    Expects multipart form with:
    - serial_id: Unique goat identifier
    - live_weight: Weight in lbs
    - species: 'goat' or 'lamb'
    - description: 'meat', 'dairy', 'cross', 'lamb', 'ewe'
    - side_image: Side view image file
    - top_image: Top view image file
    - front_image: Front view image file

    On successful grades (all views OK), archives to S3 in a background thread:
    - Raw images → goat-captures bucket: {serial_id}/{view}.jpg
    - Debug overlays → goat-processed bucket: {serial_id}/{view}_debug.jpg
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

    # Save debug images
    debug_image_paths = {}
    if grader_result.get('debug_images'):
        debug_image_paths = save_debug_images(serial_id, grader_result['debug_images'])

    # Calculate grade
    grade = None
    grade_details = None
    if grader_result.get('all_views_successful'):
        grade_result = calculate_grade(
            grader_result['measurements'],
            live_weight,
            serial_id,
            species=species,
            description=description,
        )
        grade = grade_result['grade']
        grade_details = grade_result['details']
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
        grade_details=grade_details,
        all_views_successful=grader_result['all_views_successful'],
        view_errors=view_errors,
        warnings=grader_result.get('warnings'),
        success=True
    )

    # S3 archival (background, success-only)
    if response.success and response.all_views_successful:
        archive_in_background(serial_id, raw_images, debug_image_paths)
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
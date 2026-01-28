"""
Goat Measurement API

FastAPI server that processes goat images through YOLO models
and extracts body measurements.

Endpoints:
    POST /process/{timestamp}?goat_id={id}  - Start async processing
    GET  /results/{timestamp}               - Poll for results
    GET  /health                            - Health check
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import os

from .services.processor import ProcessingService
from .services.s3 import S3Service
from .database import DatabaseService

app = FastAPI(
    title="Goat Measurement API",
    description="AI-powered body measurement system for goat grading",
    version="1.0.0"
)

# CORS - allow tablet app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
s3_service = S3Service(
    captures_bucket=os.environ.get("S3_CAPTURES_BUCKET", "goat-captures-937249941844"),
    processed_bucket=os.environ.get("S3_PROCESSED_BUCKET", "goat-processed-937249941844"),
    region=os.environ.get("AWS_REGION", "us-east-2")
)

db_service = DatabaseService(
    host=os.environ.get("DB_HOST"),
    database=os.environ.get("DB_NAME", "goatdb"),
    user=os.environ.get("DB_USER", "goatadmin"),
    password=os.environ.get("DB_PASSWORD")
)

processor = ProcessingService(s3_service, db_service)

# In-memory job tracking (replace with Redis in prod if needed)
processing_jobs: Dict[str, Dict[str, Any]] = {}


# ============== Response Models ==============

class ProcessingStartedResponse(BaseModel):
    status: str
    timestamp: str
    goat_id: int
    message: str


class ProcessingResultResponse(BaseModel):
    status: str  # "processing", "completed", "failed"
    timestamp: str
    goat_id: Optional[int] = None
    measurements: Optional[Dict[str, Any]] = None
    confidence_scores: Optional[Dict[str, float]] = None
    annotated_images: Optional[Dict[str, str]] = None  # Pre-signed URLs
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]


# ============== Background Processing ==============

async def process_images_background(timestamp: str, goat_id: int):
    """Background task that runs YOLO processing"""
    try:
        processing_jobs[timestamp]["status"] = "processing"
        
        # Run the actual processing
        result = await processor.process_goat_images(timestamp, goat_id)
        
        processing_jobs[timestamp] = {
            "status": "completed",
            "goat_id": goat_id,
            "result": result,
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        processing_jobs[timestamp] = {
            "status": "failed",
            "goat_id": goat_id,
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }


# ============== Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    
    # Check S3 connectivity
    s3_status = "ok" if s3_service.check_connection() else "error"
    
    # Check DB connectivity
    db_status = "ok" if db_service.check_connection() else "error"
    
    return HealthResponse(
        status="ok" if s3_status == "ok" and db_status == "ok" else "degraded",
        timestamp=datetime.now().isoformat(),
        services={
            "s3": s3_status,
            "database": db_status
        }
    )


@app.post("/process/{timestamp}", response_model=ProcessingStartedResponse)
async def start_processing(
    timestamp: str,
    goat_id: int,
    background_tasks: BackgroundTasks
):
    """
    Start async processing of goat images.
    
    Pi uploads images to s3://goat-captures/captures/{timestamp}/
    Then calls this endpoint to trigger processing.
    
    Args:
        timestamp: Folder name in S3 (e.g., "20260127_143052")
        goat_id: ID of the goat being measured
    """
    
    # Check if images exist in S3
    if not s3_service.images_exist(timestamp):
        raise HTTPException(
            status_code=404,
            detail=f"No images found for timestamp {timestamp}"
        )
    
    # Check if already processing
    if timestamp in processing_jobs:
        job = processing_jobs[timestamp]
        if job["status"] == "processing":
            raise HTTPException(
                status_code=409,
                detail=f"Already processing timestamp {timestamp}"
            )
    
    # Initialize job tracking
    processing_jobs[timestamp] = {
        "status": "queued",
        "goat_id": goat_id,
        "started_at": datetime.now().isoformat()
    }
    
    # Start background processing
    background_tasks.add_task(process_images_background, timestamp, goat_id)
    
    return ProcessingStartedResponse(
        status="queued",
        timestamp=timestamp,
        goat_id=goat_id,
        message="Processing started. Poll GET /results/{timestamp} for status."
    )


@app.get("/results/{timestamp}", response_model=ProcessingResultResponse)
async def get_results(timestamp: str):
    """
    Poll for processing results.
    
    Returns status: "processing", "completed", or "failed"
    When completed, includes measurements and annotated image URLs.
    """
    
    if timestamp not in processing_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"No processing job found for timestamp {timestamp}"
        )
    
    job = processing_jobs[timestamp]
    
    if job["status"] == "processing" or job["status"] == "queued":
        return ProcessingResultResponse(
            status=job["status"],
            timestamp=timestamp,
            goat_id=job.get("goat_id")
        )
    
    elif job["status"] == "completed":
        result = job["result"]
        
        # Generate pre-signed URLs for annotated images
        annotated_urls = s3_service.get_annotated_image_urls(timestamp)
        
        return ProcessingResultResponse(
            status="completed",
            timestamp=timestamp,
            goat_id=job.get("goat_id"),
            measurements=result.get("measurements"),
            confidence_scores=result.get("confidence_scores"),
            annotated_images=annotated_urls
        )
    
    else:  # failed
        return ProcessingResultResponse(
            status="failed",
            timestamp=timestamp,
            goat_id=job.get("goat_id"),
            error=job.get("error")
        )


@app.get("/results/{timestamp}/images/{view}")
async def get_annotated_image(timestamp: str, view: str):
    """
    Get pre-signed URL for a specific annotated image.
    
    Args:
        timestamp: Processing timestamp
        view: "side", "top", or "front"
    """
    if view not in ["side", "top", "front"]:
        raise HTTPException(
            status_code=400,
            detail="View must be 'side', 'top', or 'front'"
        )
    
    url = s3_service.get_annotated_image_url(timestamp, view)
    
    if not url:
        raise HTTPException(
            status_code=404,
            detail=f"No annotated {view} image found for timestamp {timestamp}"
        )
    
    return {"url": url, "expires_in_seconds": 3600}


# ============== Goat CRUD (placeholder for Cooper DB) ==============

@app.get("/goats")
async def list_goats(limit: int = 100, offset: int = 0):
    """List all goats"""
    # TODO: Implement with your database schema
    return db_service.get_goats(limit, offset)


@app.get("/goats/{goat_id}")
async def get_goat(goat_id: int):
    """Get a specific goat"""
    goat = db_service.get_goat(goat_id)
    if not goat:
        raise HTTPException(status_code=404, detail="Goat not found")
    return goat


@app.get("/goats/{goat_id}/measurements")
async def get_goat_measurements(goat_id: int, limit: int = 10):
    """Get measurement history for a goat"""
    return db_service.get_goat_measurements(goat_id, limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

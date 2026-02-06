"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List
from datetime import datetime
import re

from .config import MIN_WEIGHT_LBS, MAX_WEIGHT_LBS, VALID_GRADES


class MeasurementsResponse(BaseModel):
    """Measurements extracted from all three views"""
    head_height_cm: Optional[float] = None
    withers_height_cm: Optional[float] = None
    rump_height_cm: Optional[float] = None
    top_body_width_cm: Optional[float] = None
    front_body_width_cm: Optional[float] = None
    avg_body_width_cm: Optional[float] = None


class ConfidenceScores(BaseModel):
    """YOLO confidence scores for each view"""
    side: Optional[float] = None
    top: Optional[float] = None
    front: Optional[float] = None


class ViewError(BaseModel):
    """Error details for a single view"""
    view: str
    error: str
    fix: Optional[str] = None


class AnalyzeRequest(BaseModel):
    """Request validation for /analyze endpoint (metadata only, images come via form)"""
    serial_id: str = Field(..., min_length=1, max_length=50)
    live_weight: float = Field(..., gt=0)
    
    @field_validator('serial_id')
    @classmethod
    def validate_serial_id(cls, v: str) -> str:
        # Allow alphanumeric, underscore, hyphen only
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', v)
        if not sanitized:
            raise ValueError('serial_id must contain alphanumeric characters')
        if sanitized != v:
            raise ValueError('serial_id contains invalid characters (use alphanumeric, underscore, hyphen only)')
        return sanitized
    
    @field_validator('live_weight')
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if v < MIN_WEIGHT_LBS:
            raise ValueError(f'live_weight must be at least {MIN_WEIGHT_LBS} lbs')
        if v > MAX_WEIGHT_LBS:
            raise ValueError(f'live_weight must be at most {MAX_WEIGHT_LBS} lbs')
        return v


class AnalyzeResponse(BaseModel):
    """Response from /analyze endpoint"""
    serial_id: str
    timestamp: str
    live_weight_lbs: float
    measurements: MeasurementsResponse
    confidence_scores: ConfidenceScores
    grade: Optional[str] = None
    all_views_successful: bool
    view_errors: Optional[List[ViewError]] = None
    warnings: Optional[List[str]] = None
    success: bool


class HealthResponse(BaseModel):
    """Response from /health endpoint"""
    status: str
    timestamp: str
    models_loaded: bool
    side_model: bool
    top_model: bool
    front_model: bool
    storage_ok: bool


class DeepHealthResponse(HealthResponse):
    """Response from /health/deep endpoint"""
    inference_test: bool
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    gpu_available: bool


class ResultsResponse(BaseModel):
    """Response from /results endpoint"""
    total_results: int
    results: List[AnalyzeResponse]


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    error_code: str
    detail: Optional[str] = None
    fix: Optional[str] = None

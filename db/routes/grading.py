"""
Goat grading results routes.

Write path: ec2-api calls POST /grading after grading completes.
Read path: frontend calls GET /grading/{serial_id} to show history.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from routes import get_conn
import json

router = APIRouter()


class GradeResultIn(BaseModel):
    serial_id: int
    grade: Optional[str] = None
    live_weight: Optional[float] = None
    all_views_ok: Optional[bool] = None
    measurements: Optional[dict] = None

    # Raw capture S3 keys
    side_raw_s3_key: Optional[str] = None
    top_raw_s3_key: Optional[str] = None
    front_raw_s3_key: Optional[str] = None

    # Debug overlay S3 keys
    side_debug_s3_key: Optional[str] = None
    top_debug_s3_key: Optional[str] = None
    front_debug_s3_key: Optional[str] = None

    # Timing
    capture_sec: Optional[float] = None
    ec2_sec: Optional[float] = None
    total_sec: Optional[float] = None

    # Warnings
    warnings: Optional[list[str]] = None


@router.get("/{serial_id}")
async def get_grades_for_goat(request: Request, serial_id: int):
    """Get all grading results for a goat, newest first."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        # Verify goat exists
        goat = await conn.fetchrow("SELECT serial_id FROM goats WHERE serial_id = $1", serial_id)
        if not goat:
            raise HTTPException(404, "Goat not found")

        rows = await conn.fetch(
            """SELECT * FROM goat_grade_results
               WHERE serial_id = $1
               ORDER BY graded_at DESC""",
            serial_id,
        )
        results = []
        for r in rows:
            d = dict(r)
            # asyncpg returns jsonb as str, parse it
            if d.get("measurements") and isinstance(d["measurements"], str):
                d["measurements"] = json.loads(d["measurements"])
            results.append(d)
        return results


@router.get("/result/{result_id}")
async def get_grade_result(request: Request, result_id: int):
    """Get a single grading result by ID."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM goat_grade_results WHERE id = $1", result_id
        )
        if not row:
            raise HTTPException(404, "Grade result not found")
        d = dict(row)
        if d.get("measurements") and isinstance(d["measurements"], str):
            d["measurements"] = json.loads(d["measurements"])
        return d


@router.post("", status_code=201)
async def create_grade_result(request: Request, body: GradeResultIn):
    """Record a new grading result. Called by ec2-api after grading completes."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        # Verify goat exists
        goat = await conn.fetchrow(
            "SELECT serial_id FROM goats WHERE serial_id = $1", body.serial_id
        )
        if not goat:
            raise HTTPException(404, f"Goat with serial_id {body.serial_id} not found")

        measurements_json = json.dumps(body.measurements) if body.measurements else None

        row = await conn.fetchrow(
            """INSERT INTO goat_grade_results
               (serial_id, grade, live_weight, all_views_ok, measurements,
                side_raw_s3_key, top_raw_s3_key, front_raw_s3_key,
                side_debug_s3_key, top_debug_s3_key, front_debug_s3_key,
                capture_sec, ec2_sec, total_sec, warnings)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
               RETURNING *""",
            body.serial_id, body.grade, body.live_weight, body.all_views_ok,
            measurements_json,
            body.side_raw_s3_key, body.top_raw_s3_key, body.front_raw_s3_key,
            body.side_debug_s3_key, body.top_debug_s3_key, body.front_debug_s3_key,
            body.capture_sec, body.ec2_sec, body.total_sec,
            body.warnings,
        )

        # Also update the goat's grade field with the latest
        if body.grade:
            await conn.execute(
                "UPDATE goats SET grade = $1 WHERE serial_id = $2",
                body.grade, body.serial_id,
            )

        d = dict(row)
        if d.get("measurements") and isinstance(d["measurements"], str):
            d["measurements"] = json.loads(d["measurements"])
        return d


@router.get("")
async def list_recent_grades(request: Request, limit: int = 50):
    """List recent grading results across all goats."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT g.*, goats.hook_id, goats.description
               FROM goat_grade_results g
               JOIN goats ON goats.serial_id = g.serial_id
               ORDER BY g.graded_at DESC
               LIMIT $1""",
            min(limit, 200),
        )
        results = []
        for r in rows:
            d = dict(r)
            if d.get("measurements") and isinstance(d["measurements"], str):
                d["measurements"] = json.loads(d["measurements"])
            results.append(d)
        return results

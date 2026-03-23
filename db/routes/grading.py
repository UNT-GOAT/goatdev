"""
Grading results routes.

Write path: frontend calls POST /grading after operator confirms a grade.
Read path: frontend calls GET /grading/{serial_id} to show result.

One grade result per animal — re-grading replaces the existing result.

Supports both goats and lambs — the grade_results table references
animals(serial_id) instead of a species-specific table.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from routes import get_conn
import json

router = APIRouter()

GRADEABLE_SPECIES = ("goat", "lamb")


class GradeResultIn(BaseModel):
    serial_id: int
    grade: Optional[str] = None
    live_weight: Optional[float] = None
    all_views_ok: Optional[bool] = None
    measurements: Optional[dict] = None

    side_raw_s3_key: Optional[str] = None
    top_raw_s3_key: Optional[str] = None
    front_raw_s3_key: Optional[str] = None

    side_debug_s3_key: Optional[str] = None
    top_debug_s3_key: Optional[str] = None
    front_debug_s3_key: Optional[str] = None

    capture_sec: Optional[float] = None
    ec2_sec: Optional[float] = None
    total_sec: Optional[float] = None

    warnings: Optional[list[str]] = None


@router.get("/{serial_id}")
async def get_grades_for_animal(request: Request, serial_id: int):
    """Get the grading result for an animal."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        animal = await conn.fetchrow(
            "SELECT serial_id, species FROM animals WHERE serial_id = $1", serial_id
        )
        if not animal:
            raise HTTPException(404, "Animal not found")
        if animal["species"] not in GRADEABLE_SPECIES:
            raise HTTPException(400, f"Grading not supported for {animal['species']}")

        rows = await conn.fetch(
            """SELECT * FROM grade_results
               WHERE serial_id = $1
               ORDER BY graded_at DESC""",
            serial_id,
        )
        results = []
        for r in rows:
            d = dict(r)
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
            "SELECT * FROM grade_results WHERE id = $1", result_id
        )
        if not row:
            raise HTTPException(404, "Grade result not found")
        d = dict(row)
        if d.get("measurements") and isinstance(d["measurements"], str):
            d["measurements"] = json.loads(d["measurements"])
        return d


@router.delete("/result/{result_id}")
async def delete_grade_result(request: Request, result_id: int):
    """Delete a single grading result by ID."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "DELETE FROM grade_results WHERE id = $1 RETURNING id", result_id
        )
        if not row:
            raise HTTPException(404, "Grade result not found")
        return {"deleted": result_id}


@router.put("/result/{result_id}")
async def update_grade_result(request: Request, result_id: int):
    """Update fields on an existing grading result."""
    pool = await get_conn(request)
    body = await request.json()

    allowed = {"grade", "live_weight", "all_views_ok"}
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        raise HTTPException(400, "No valid fields to update")

    async with pool.acquire() as conn:
        sets = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates))
        vals = [result_id] + list(updates.values())
        row = await conn.fetchrow(
            f"UPDATE grade_results SET {sets} WHERE id = $1 RETURNING *", *vals
        )
        if not row:
            raise HTTPException(404, "Grade result not found")
        d = dict(row)
        if d.get("measurements") and isinstance(d["measurements"], str):
            d["measurements"] = json.loads(d["measurements"])
        return d


@router.post("", status_code=201)
async def create_grade_result(request: Request, body: GradeResultIn):
    """
    Record a grading result. One result per animal — if a result already
    exists for this serial_id, it is replaced (upsert).
    Called by frontend after operator confirms.
    """
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        animal = await conn.fetchrow(
            "SELECT serial_id, species FROM animals WHERE serial_id = $1", body.serial_id
        )
        if not animal:
            raise HTTPException(404, f"Animal with serial_id {body.serial_id} not found")
        if animal["species"] not in GRADEABLE_SPECIES:
            raise HTTPException(400, f"Grading not supported for {animal['species']}")

        measurements_json = json.dumps(body.measurements) if body.measurements else None

        # Check if a grade result already exists for this animal
        existing = await conn.fetchrow(
            "SELECT id FROM grade_results WHERE serial_id = $1", body.serial_id
        )

        if existing:
            # Re-grade: replace existing result
            row = await conn.fetchrow(
                """UPDATE grade_results SET
                    grade = $2, live_weight = $3, all_views_ok = $4, measurements = $5,
                    side_raw_s3_key = $6, top_raw_s3_key = $7, front_raw_s3_key = $8,
                    side_debug_s3_key = $9, top_debug_s3_key = $10, front_debug_s3_key = $11,
                    capture_sec = $12, ec2_sec = $13, total_sec = $14, warnings = $15,
                    graded_at = NOW()
                   WHERE serial_id = $1
                   RETURNING *""",
                body.serial_id, body.grade, body.live_weight, body.all_views_ok,
                measurements_json,
                body.side_raw_s3_key, body.top_raw_s3_key, body.front_raw_s3_key,
                body.side_debug_s3_key, body.top_debug_s3_key, body.front_debug_s3_key,
                body.capture_sec, body.ec2_sec, body.total_sec,
                body.warnings,
            )
        else:
            # First grade
            row = await conn.fetchrow(
                """INSERT INTO grade_results
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

        # Sync grade to species table
        if body.grade:
            species = animal["species"]
            table = species + "s"
            await conn.execute(
                f"UPDATE {table} SET grade = $1 WHERE serial_id = $2",
                body.grade, body.serial_id,
            )

        d = dict(row)
        if d.get("measurements") and isinstance(d["measurements"], str):
            d["measurements"] = json.loads(d["measurements"])
        return d


@router.get("")
async def list_recent_grades(request: Request, limit: int = 50):
    """List recent grading results across all gradeable animals."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT g.*, a.species,
                      COALESCE(goats.hook_id, NULL) as hook_id,
                      COALESCE(goats.description, lambs.description) as description
               FROM grade_results g
               JOIN animals a ON a.serial_id = g.serial_id
               LEFT JOIN goats ON goats.serial_id = g.serial_id
               LEFT JOIN lambs ON lambs.serial_id = g.serial_id
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
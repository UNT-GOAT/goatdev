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
from typing import Optional, Any
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
    grade_details: Optional[dict] = None
    manual_override_history: Optional[list[dict[str, Any]]] = None

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


def _parse_json_field(value):
    if value and isinstance(value, str):
        return json.loads(value)
    return value


def _normalize_grade_row(row) -> dict:
    d = dict(row)
    d["measurements"] = _parse_json_field(d.get("measurements"))
    d["grade_details"] = _parse_json_field(d.get("grade_details"))
    history = _parse_json_field(d.get("manual_override_history"))
    d["manual_override_history"] = history if isinstance(history, list) else []
    return d


def _validate_override_annotation(
    history,
    *,
    new_grade,
    username: str,
    from_grade: Optional[str] = None,
    allowed_contexts: tuple[str, ...] = ("pre_save_override", "saved_result_edit"),
):
    if not isinstance(history, list) or not history:
        raise HTTPException(400, "Manual grade changes require an annotation")

    entry = history[-1]
    if not isinstance(entry, dict):
        raise HTTPException(400, "Invalid manual override history entry")

    annotation = (entry.get("annotation") or "").strip()
    entry_username = (entry.get("username") or "").strip()
    entry_from_grade = entry.get("from_grade")
    to_grade = entry.get("to_grade")
    change_context = entry.get("change_context")

    if not annotation:
        raise HTTPException(400, "Manual grade changes require an annotation")
    if not entry_username:
        raise HTTPException(400, "Manual override must include username")
    if entry_username != username:
        raise HTTPException(400, "Manual override username does not match the authenticated user")
    if to_grade != new_grade:
        raise HTTPException(400, "Manual override history does not match grade change")
    if from_grade is not None and entry_from_grade != from_grade:
        raise HTTPException(400, "Manual override history does not match the prior grade")
    if change_context not in allowed_contexts:
        raise HTTPException(400, "Invalid manual override change context")
    return history


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
        return [_normalize_grade_row(r) for r in rows]


@router.get("/result/{result_id}")
async def get_grade_result(request: Request, result_id: int):
    """Get a single grading result by ID."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT g.*, a.species,
                      COALESCE(goats.description, lambs.description) as description
               FROM grade_results g
               JOIN animals a ON a.serial_id = g.serial_id
               LEFT JOIN goats ON goats.serial_id = g.serial_id
               LEFT JOIN lambs ON lambs.serial_id = g.serial_id
               WHERE g.id = $1""",
            result_id,
        )
        if not row:
            raise HTTPException(404, "Grade result not found")
        return _normalize_grade_row(row)


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

    async with pool.acquire() as conn:
        username = request.headers.get("x-auth-username", "unknown")
        existing = await conn.fetchrow(
            "SELECT * FROM grade_results WHERE id = $1",
            result_id,
        )
        if not existing:
            raise HTTPException(404, "Grade result not found")

        allowed = {"grade", "live_weight", "all_views_ok", "grade_details", "manual_override_history"}
        updates = {k: v for k, v in body.items() if k in allowed}
        if not updates:
            raise HTTPException(400, "No valid fields to update")

        existing_grade = existing.get("grade")
        if "grade" in updates:
            if updates["grade"] != existing_grade:
                history = _validate_override_annotation(
                    body.get("manual_override_history"),
                    new_grade=updates["grade"],
                    from_grade=existing_grade,
                    username=username,
                    allowed_contexts=("saved_result_edit",),
                )
                updates["manual_override_history"] = json.dumps(history)
            elif "manual_override_history" in updates:
                updates["manual_override_history"] = json.dumps(
                    updates["manual_override_history"] or []
                )
        elif "manual_override_history" in updates:
            updates["manual_override_history"] = json.dumps(
                updates["manual_override_history"] or []
            )

        if "grade_details" in updates:
            updates["grade_details"] = json.dumps(updates["grade_details"]) if updates["grade_details"] is not None else None

        sets = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates))
        vals = [result_id] + list(updates.values())
        row = await conn.fetchrow(
            f"UPDATE grade_results SET {sets} WHERE id = $1 RETURNING *", *vals
        )
        if "grade" in updates:
            serial_id = row["serial_id"]
            animal = await conn.fetchrow(
                "SELECT species FROM animals WHERE serial_id = $1",
                serial_id,
            )
            if animal and animal["species"] in GRADEABLE_SPECIES:
                table = animal["species"] + "s"
                await conn.execute(
                    f"UPDATE {table} SET grade = $1 WHERE serial_id = $2",
                    row["grade"], serial_id,
                )
        return _normalize_grade_row(row)


@router.post("", status_code=201)
async def create_grade_result(request: Request, body: GradeResultIn):
    """
    Record a grading result. One result per animal — if a result already
    exists for this serial_id, it is replaced (upsert).
    Called by frontend after operator confirms.
    """
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        username = request.headers.get("x-auth-username", "unknown")
        animal = await conn.fetchrow(
            "SELECT serial_id, species FROM animals WHERE serial_id = $1", body.serial_id
        )
        if not animal:
            raise HTTPException(404, f"Animal with serial_id {body.serial_id} not found")
        if animal["species"] not in GRADEABLE_SPECIES:
            raise HTTPException(400, f"Grading not supported for {animal['species']}")

        measurements_json = json.dumps(body.measurements) if body.measurements else None
        grade_details_json = json.dumps(body.grade_details) if body.grade_details else None
        manual_override_history = body.manual_override_history or []
        existing = await conn.fetchrow(
            "SELECT id, grade, manual_override_history FROM grade_results WHERE serial_id = $1", body.serial_id
        )
        existing_history = []
        if existing:
            existing_history = _parse_json_field(existing.get("manual_override_history")) or []
        if manual_override_history:
            last_entry = manual_override_history[-1] if isinstance(manual_override_history, list) and manual_override_history else None
            from_grade = last_entry.get("from_grade") if isinstance(last_entry, dict) else None
            allowed_contexts = ("pre_save_override", "saved_result_edit")
            _validate_override_annotation(
                manual_override_history,
                new_grade=body.grade,
                from_grade=from_grade if existing and from_grade == existing["grade"] else None,
                username=username,
                allowed_contexts=allowed_contexts,
            )
        elif existing_history:
            manual_override_history = existing_history
        manual_override_history_json = json.dumps(manual_override_history)

        if existing:
            # Re-grade: replace existing result
            row = await conn.fetchrow(
                """UPDATE grade_results SET
                    grade = $2, live_weight = $3, all_views_ok = $4, measurements = $5,
                    grade_details = $6, manual_override_history = $7,
                    side_raw_s3_key = $8, top_raw_s3_key = $9, front_raw_s3_key = $10,
                    side_debug_s3_key = $11, top_debug_s3_key = $12, front_debug_s3_key = $13,
                    capture_sec = $14, ec2_sec = $15, total_sec = $16, warnings = $17,
                    graded_at = NOW()
                   WHERE serial_id = $1
                   RETURNING *""",
                body.serial_id, body.grade, body.live_weight, body.all_views_ok,
                measurements_json,
                grade_details_json, manual_override_history_json,
                body.side_raw_s3_key, body.top_raw_s3_key, body.front_raw_s3_key,
                body.side_debug_s3_key, body.top_debug_s3_key, body.front_debug_s3_key,
                body.capture_sec, body.ec2_sec, body.total_sec,
                body.warnings,
            )
        else:
            # First grade
            row = await conn.fetchrow(
                """INSERT INTO grade_results
                   (serial_id, grade, live_weight, all_views_ok, measurements, grade_details, manual_override_history,
                    side_raw_s3_key, top_raw_s3_key, front_raw_s3_key,
                    side_debug_s3_key, top_debug_s3_key, front_debug_s3_key,
                    capture_sec, ec2_sec, total_sec, warnings)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                   RETURNING *""",
                body.serial_id, body.grade, body.live_weight, body.all_views_ok,
                measurements_json, grade_details_json, manual_override_history_json,
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

        return _normalize_grade_row(row)


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
        return [_normalize_grade_row(r) for r in rows]

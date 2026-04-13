"""
Grading results and new-animal grading session routes.

Existing-animal grading continues to write directly against a committed serial_id.
New-animal grading now stages review data under a persisted grading session and
assigns the final serial_id only when the operator saves.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes import get_conn
from routes.animal_factory import _create_species_animal
from routes.serials import (
    allocate_next_committed_serial,
    get_active_new_animal_session,
    get_next_committed_serial,
    lock_new_animal_creation_gate,
)

router = APIRouter()

GRADEABLE_SPECIES = ("goat", "lamb")
SESSION_UPDATE_STATUSES = {"capturing", "review_pending", "failed"}


class GradeResultIn(BaseModel):
    serial_id: int
    analysis_key: Optional[str] = None
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


class GradingSessionCreateIn(BaseModel):
    species: str
    description: Optional[str] = None
    live_weight: Optional[float] = None


class GradingSessionUpdateIn(BaseModel):
    status: str
    result_payload: Optional[dict[str, Any]] = None
    last_error: Optional[str] = None
    description: Optional[str] = None
    live_weight: Optional[float] = None


class GradingSessionFinalizeIn(BaseModel):
    prov_id: Optional[int] = None
    kill_date: Optional[date] = None


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


def _normalize_session_row(row) -> dict:
    d = dict(row)
    d["result_payload"] = _parse_json_field(d.get("result_payload"))
    return d


def _sanitize_analysis_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    if len(cleaned) > 120:
        raise HTTPException(400, "analysis_key must be 120 chars or fewer")
    allowed = "".join(ch for ch in cleaned if ch.isalnum() or ch in {"_", "-"})
    if allowed != cleaned:
        raise HTTPException(400, "analysis_key must be alphanumeric, underscore, or dash")
    return cleaned


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


async def _serialize_session(conn, row) -> dict:
    payload = _normalize_session_row(row)
    payload["next_serial_id"] = await get_next_committed_serial(conn)
    return payload


async def _upsert_grade_result(conn, *, username: str, animal: dict, payload: dict):
    measurements_json = json.dumps(payload.get("measurements")) if payload.get("measurements") else None
    grade_details_json = json.dumps(payload.get("grade_details")) if payload.get("grade_details") else None
    manual_override_history = payload.get("manual_override_history") or []
    existing = await conn.fetchrow(
        "SELECT id, grade, manual_override_history FROM grade_results WHERE serial_id = $1",
        payload["serial_id"],
    )
    existing_history = []
    if existing:
        existing_history = _parse_json_field(existing.get("manual_override_history")) or []

    if manual_override_history:
        last_entry = manual_override_history[-1] if isinstance(manual_override_history, list) and manual_override_history else None
        from_grade = last_entry.get("from_grade") if isinstance(last_entry, dict) else None
        _validate_override_annotation(
            manual_override_history,
            new_grade=payload.get("grade"),
            from_grade=from_grade if existing and from_grade == existing["grade"] else None,
            username=username,
            allowed_contexts=("pre_save_override", "saved_result_edit"),
        )
    elif existing_history:
        manual_override_history = existing_history

    manual_override_history_json = json.dumps(manual_override_history)
    analysis_key = _sanitize_analysis_key(payload.get("analysis_key"))

    row = await conn.fetchrow(
        """INSERT INTO grade_results
           (serial_id, analysis_key, grade, live_weight, all_views_ok, measurements, grade_details, manual_override_history,
            side_raw_s3_key, top_raw_s3_key, front_raw_s3_key,
            side_debug_s3_key, top_debug_s3_key, front_debug_s3_key,
            capture_sec, ec2_sec, total_sec, warnings)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
           ON CONFLICT (serial_id) DO UPDATE SET
               analysis_key = COALESCE(EXCLUDED.analysis_key, grade_results.analysis_key),
               grade = EXCLUDED.grade,
               live_weight = EXCLUDED.live_weight,
               all_views_ok = EXCLUDED.all_views_ok,
               measurements = EXCLUDED.measurements,
               grade_details = EXCLUDED.grade_details,
               manual_override_history = EXCLUDED.manual_override_history,
               side_raw_s3_key = EXCLUDED.side_raw_s3_key,
               top_raw_s3_key = EXCLUDED.top_raw_s3_key,
               front_raw_s3_key = EXCLUDED.front_raw_s3_key,
               side_debug_s3_key = EXCLUDED.side_debug_s3_key,
               top_debug_s3_key = EXCLUDED.top_debug_s3_key,
               front_debug_s3_key = EXCLUDED.front_debug_s3_key,
               capture_sec = EXCLUDED.capture_sec,
               ec2_sec = EXCLUDED.ec2_sec,
               total_sec = EXCLUDED.total_sec,
               warnings = EXCLUDED.warnings,
               graded_at = NOW()
           RETURNING *""",
        payload["serial_id"],
        analysis_key,
        payload.get("grade"),
        payload.get("live_weight"),
        payload.get("all_views_ok"),
        measurements_json,
        grade_details_json,
        manual_override_history_json,
        payload.get("side_raw_s3_key"),
        payload.get("top_raw_s3_key"),
        payload.get("front_raw_s3_key"),
        payload.get("side_debug_s3_key"),
        payload.get("top_debug_s3_key"),
        payload.get("front_debug_s3_key"),
        payload.get("capture_sec"),
        payload.get("ec2_sec"),
        payload.get("total_sec"),
        payload.get("warnings"),
    )

    if animal["species"] in GRADEABLE_SPECIES:
        table = animal["species"] + "s"
        await conn.execute(
            f"UPDATE {table} SET grade = $1 WHERE serial_id = $2",
            payload.get("grade"),
            payload["serial_id"],
        )

    return _normalize_grade_row(row)


def _table_for_species(species: str) -> str:
    if species not in GRADEABLE_SPECIES:
        raise HTTPException(400, f"Grading not supported for {species}")
    return species + "s"


@router.post("/sessions")
async def create_or_resume_grading_session(request: Request, body: GradingSessionCreateIn):
    if body.species not in GRADEABLE_SPECIES:
        raise HTTPException(400, "species must be goat or lamb")

    pool = await get_conn(request)
    username = request.headers.get("x-auth-username", "unknown")
    async with pool.acquire() as conn:
        async with conn.transaction():
            await lock_new_animal_creation_gate(conn)
            existing = await get_active_new_animal_session(conn)
            if existing:
                payload = await _serialize_session(conn, existing)
                payload["created"] = False
                payload["resumed"] = True
                return payload

            session_id = uuid4().hex
            analysis_key = f"draft-{session_id}"
            row = await conn.fetchrow(
                """
                INSERT INTO grading_sessions
                    (id, analysis_key, status, species, description, live_weight, created_by, updated_by)
                VALUES
                    ($1, $2, 'capturing', $3, $4, $5, $6, $6)
                RETURNING *
                """,
                session_id,
                analysis_key,
                body.species,
                body.description,
                body.live_weight,
                username,
            )
            payload = await _serialize_session(conn, row)
            payload["created"] = True
            payload["resumed"] = False
            return payload


@router.get("/sessions/pending")
async def get_pending_grading_session(request: Request):
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        session = await get_active_new_animal_session(conn)
        if not session:
            raise HTTPException(404, "No pending new-animal grading session")
        return await _serialize_session(conn, session)


@router.put("/sessions/{session_id}")
async def update_grading_session(request: Request, session_id: str, body: GradingSessionUpdateIn):
    if body.status not in SESSION_UPDATE_STATUSES:
        raise HTTPException(400, "Invalid grading session status")
    if body.status == "review_pending" and body.result_payload is None:
        raise HTTPException(400, "review_pending sessions require a result payload")

    pool = await get_conn(request)
    username = request.headers.get("x-auth-username", "unknown")
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT * FROM grading_sessions WHERE id = $1 FOR UPDATE",
                session_id,
            )
            if not row:
                raise HTTPException(404, "Grading session not found")
            if row["status"] in {"saved", "discarded"}:
                raise HTTPException(409, "Grading session is already closed")

            result_payload = body.result_payload
            if result_payload is not None:
                result_payload = dict(result_payload)
                result_payload["analysis_key"] = row["analysis_key"]
                result_payload["species"] = result_payload.get("species") or row["species"]
                result_payload["description"] = result_payload.get("description") if result_payload.get("description") is not None else row["description"]
                result_payload["live_weight"] = result_payload.get("live_weight") if result_payload.get("live_weight") is not None else row["live_weight"]

            updated = await conn.fetchrow(
                """
                UPDATE grading_sessions
                SET status = $2,
                    result_payload = COALESCE($3::jsonb, result_payload),
                    last_error = COALESCE($4, last_error),
                    description = COALESCE($5, description),
                    live_weight = COALESCE($6, live_weight),
                    updated_by = $7,
                    updated_at = NOW()
                WHERE id = $1
                RETURNING *
                """,
                session_id,
                body.status,
                json.dumps(result_payload) if result_payload is not None else None,
                body.last_error,
                body.description,
                body.live_weight,
                username,
            )
            return await _serialize_session(conn, updated)


@router.post("/sessions/{session_id}/discard")
async def discard_grading_session(request: Request, session_id: str):
    pool = await get_conn(request)
    username = request.headers.get("x-auth-username", "unknown")
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT * FROM grading_sessions WHERE id = $1 FOR UPDATE",
                session_id,
            )
            if not row:
                raise HTTPException(404, "Grading session not found")
            if row["status"] == "saved":
                raise HTTPException(409, "Saved grading sessions cannot be discarded")
            if row["status"] == "finalizing":
                raise HTTPException(409, "Grading session is currently finalizing")
            if row["status"] == "discarded":
                return await _serialize_session(conn, row)

            updated = await conn.fetchrow(
                """
                UPDATE grading_sessions
                SET status = 'discarded',
                    updated_by = $2,
                    updated_at = NOW()
                WHERE id = $1
                RETURNING *
                """,
                session_id,
                username,
            )
            return await _serialize_session(conn, updated)


@router.post("/sessions/{session_id}/finalize", status_code=201)
async def finalize_grading_session(request: Request, session_id: str, body: GradingSessionFinalizeIn):
    pool = await get_conn(request)
    username = request.headers.get("x-auth-username", "unknown")

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT * FROM grading_sessions WHERE id = $1 FOR UPDATE",
                session_id,
            )
            if not row:
                raise HTTPException(404, "Grading session not found")
            if row["status"] == "saved":
                serial_id = await conn.fetchval(
                    "SELECT serial_id FROM grade_results WHERE analysis_key = $1",
                    row["analysis_key"],
                )
                return {"session_id": session_id, "serial_id": serial_id}
            if row["status"] != "review_pending":
                raise HTTPException(409, "Grading session is not ready to save")

            payload = _parse_json_field(row["result_payload"]) or {}
            if not isinstance(payload, dict):
                raise HTTPException(409, "Grading session has no review payload")

            await conn.execute(
                """
                UPDATE grading_sessions
                SET status = 'finalizing',
                    updated_by = $2,
                    updated_at = NOW()
                WHERE id = $1
                """,
                session_id,
                username,
            )

            serial_id = await allocate_next_committed_serial(conn)
            species = row["species"]
            table = _table_for_species(species)
            description = payload.get("description")
            if description is None:
                description = row["description"]
            live_weight = payload.get("live_weight")
            if live_weight is None:
                live_weight = row["live_weight"]
            grade = payload.get("grade")
            animal_data = {
                "description": description,
                "live_weight": live_weight,
                "grade": grade,
                "prov_id": body.prov_id,
                "kill_date": body.kill_date,
            }
            animal_row, _ = await _create_species_animal(
                conn,
                table,
                animal_data,
                serial_id=serial_id,
            )
            grade_row = await _upsert_grade_result(
                conn,
                username=username,
                animal={"serial_id": serial_id, "species": species},
                payload={
                    **payload,
                    "serial_id": serial_id,
                    "analysis_key": row["analysis_key"],
                    "description": description,
                    "live_weight": live_weight,
                    "grade": grade,
                },
            )
            await conn.execute(
                """
                UPDATE grading_sessions
                SET status = 'saved',
                    updated_by = $2,
                    updated_at = NOW()
                WHERE id = $1
                """,
                session_id,
                username,
            )
            return {
                "session_id": session_id,
                "serial_id": serial_id,
                "animal": animal_row,
                "grade_result": grade_row,
            }


@router.get("/{serial_id}")
async def get_grades_for_animal(request: Request, serial_id: int):
    """Get the grading result for an animal."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        animal = await conn.fetchrow(
            "SELECT serial_id, species FROM animals WHERE serial_id = $1",
            serial_id,
        )
        if not animal:
            raise HTTPException(404, "Animal not found")
        if animal["species"] not in GRADEABLE_SPECIES:
            raise HTTPException(400, f"Grading not supported for {animal['species']}")

        rows = await conn.fetch(
            """SELECT *
               FROM grade_results
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
            "DELETE FROM grade_results WHERE id = $1 RETURNING id",
            result_id,
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
            f"UPDATE grade_results SET {sets} WHERE id = $1 RETURNING *",
            *vals,
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
                    row["grade"],
                    serial_id,
                )
        return _normalize_grade_row(row)


@router.post("", status_code=201)
async def create_grade_result(request: Request, body: GradeResultIn):
    """
    Record a grading result for an existing committed animal.
    """
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        username = request.headers.get("x-auth-username", "unknown")
        animal = await conn.fetchrow(
            "SELECT serial_id, species FROM animals WHERE serial_id = $1",
            body.serial_id,
        )
        if not animal:
            raise HTTPException(404, f"Animal with serial_id {body.serial_id} not found")
        if animal["species"] not in GRADEABLE_SPECIES:
            raise HTTPException(400, f"Grading not supported for {animal['species']}")

        return await _upsert_grade_result(
            conn,
            username=username,
            animal=animal,
            payload=body.model_dump(),
        )


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

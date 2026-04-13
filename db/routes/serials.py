from __future__ import annotations

from fastapi import HTTPException


ANIMAL_COUNTER_NAME = "animals"
ACTIVE_NEW_ANIMAL_SESSION_STATUSES = (
    "capturing",
    "review_pending",
    "finalizing",
)
STALE_CAPTURING_SESSION_INTERVAL = "30 minutes"
NEW_ANIMAL_GATE_LOCK_ID = 4319001


async def lock_new_animal_creation_gate(conn):
    await conn.execute(
        "SELECT pg_advisory_xact_lock($1)",
        NEW_ANIMAL_GATE_LOCK_ID,
    )


async def sync_animals_legacy_sequence(conn):
    current_value = await conn.fetchval(
        """
        SELECT current_value
        FROM serial_counters
        WHERE name = $1
        """,
        ANIMAL_COUNTER_NAME,
    )
    if current_value is None:
        raise HTTPException(500, "serial counter is not initialized")
    await conn.execute(
        """
        SELECT setval(
            pg_get_serial_sequence('animals', 'serial_id'),
            $1,
            $2
        )
        """,
        max(int(current_value), 1),
        int(current_value) > 0,
    )


async def get_next_committed_serial(conn) -> int:
    row = await conn.fetchrow(
        """
        SELECT current_value + 1 AS next_serial_id
        FROM serial_counters
        WHERE name = $1
        """,
        ANIMAL_COUNTER_NAME,
    )
    if not row:
        raise HTTPException(500, "serial counter is not initialized")
    return row["next_serial_id"]


async def allocate_next_committed_serial(conn) -> int:
    row = await conn.fetchrow(
        """
        UPDATE serial_counters
        SET current_value = current_value + 1,
            updated_at = NOW()
        WHERE name = $1
        RETURNING current_value AS serial_id
        """,
        ANIMAL_COUNTER_NAME,
    )
    if not row:
        raise HTTPException(500, "serial counter is not initialized")
    await sync_animals_legacy_sequence(conn)
    return row["serial_id"]


async def bump_serial_counter_to_at_least(conn, value: int):
    await conn.execute(
        """
        UPDATE serial_counters
        SET current_value = GREATEST(current_value, $2),
            updated_at = NOW()
        WHERE name = $1
        """,
        ANIMAL_COUNTER_NAME,
        value,
    )
    await sync_animals_legacy_sequence(conn)


async def _expire_stale_capturing_sessions(conn):
    await conn.execute(
        f"""
        UPDATE grading_sessions
        SET status = 'failed',
            last_error = COALESCE(last_error, 'Capture session expired before review'),
            updated_at = NOW()
        WHERE status = 'capturing'
          AND updated_at < NOW() - INTERVAL '{STALE_CAPTURING_SESSION_INTERVAL}'
        """
    )


async def get_active_new_animal_session(conn):
    await _expire_stale_capturing_sessions(conn)
    return await conn.fetchrow(
        """
        SELECT *
        FROM grading_sessions
        WHERE status = ANY($1::text[])
        ORDER BY created_at ASC
        LIMIT 1
        """,
        list(ACTIVE_NEW_ANIMAL_SESSION_STATUSES),
    )


async def get_blocking_new_animal_session(conn, *, exclude_session_id: str | None = None):
    session = await get_active_new_animal_session(conn)
    if not session:
        return None
    if exclude_session_id and session["id"] == exclude_session_id:
        return None
    return session

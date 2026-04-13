"""
Generic CRUD factory for animal tables (chickens, goats, lambs).
Each table has the same core fields (serial_id, weights, dates, prov_id)
with minor differences (goats have hook_id, lambs/goats have description + grade).
This factory generates a full CRUD router for any of them.

serial_id is always auto-assigned from the unified `animals` table.
No manual ID assignment — the system owns the sequence.
"""

import boto3
import os
import threading

from decimal import Decimal
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, create_model
from typing import Optional
from datetime import date
from routes import get_conn
from routes.serials import (
    allocate_next_committed_serial,
    bump_serial_counter_to_at_least,
    get_blocking_new_animal_session,
    lock_new_animal_creation_gate,
)

S3_CAPTURES_BUCKET = os.environ.get("S3_CAPTURES_BUCKET", "")
S3_PROCESSED_BUCKET = os.environ.get("S3_PROCESSED_BUCKET", "")

# All possible animal fields and their types
ANIMAL_FIELDS = {
    "live_weight": (Optional[float], None),
    "hang_weight": (Optional[float], None),
    "hang_portion": (Optional[float], None),
    "whole_hw": (Optional[float], None),
    "grade": (Optional[str], None),
    "hook_id": (Optional[str], None),
    "description": (Optional[str], None),
    "prov_id": (Optional[int], None),
    "kill_date": (Optional[date], None),
    "process_date": (Optional[date], None),
    "purchase_date": (Optional[date], None),
    "delivery_date": (Optional[date], None),
}

# Which fields each table uses
TABLE_FIELDS = {
    "chickens": [
        "live_weight", "hang_weight", "hang_portion", "whole_hw",
        "prov_id", "kill_date", "process_date", "purchase_date", "delivery_date",
    ],
    "goats": [
        "hook_id", "description", "live_weight", "hang_weight", "hang_portion",
        "whole_hw", "grade", "prov_id", "kill_date", "process_date",
        "purchase_date", "delivery_date",
    ],
    "lambs": [
        "description", "live_weight", "hang_weight", "hang_portion",
        "whole_hw", "grade", "prov_id", "kill_date", "process_date",
        "purchase_date", "delivery_date",
    ],
}

# Map table name to species value in animals table
TABLE_TO_SPECIES = {
    "chickens": "chicken",
    "goats": "goat",
    "lambs": "lamb",
}

def _s3_cleanup(prefixes: list[str]):
    """Delete all S3 objects for each prefix. Runs in background thread."""
    try:
        s3 = boto3.client("s3")
        for bucket in (S3_CAPTURES_BUCKET, S3_PROCESSED_BUCKET):
            if not bucket:
                continue
            for raw_prefix in prefixes:
                prefix = f"{raw_prefix}/"
                resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
                keys = [obj["Key"] for obj in resp.get("Contents", [])]
                if keys:
                    s3.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": [{"Key": k} for k in keys]},
                    )
                    print(f"[s3] Deleted {len(keys)} objects from {bucket}/{prefix}")
    except Exception as e:
        print(f"[s3] Cleanup failed for {prefixes}: {e}")


def _normalize_compare_value(value):
    if isinstance(value, Decimal):
        return value.normalize()
    if isinstance(value, float):
        return Decimal(str(value)).normalize()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _matches_non_null_fields(existing_row, incoming: dict) -> bool:
    for key, value in incoming.items():
        if value is None:
            continue
        if _normalize_compare_value(existing_row.get(key)) != _normalize_compare_value(value):
            return False
    return True


async def _create_species_animal(
    conn,
    table: str,
    data: dict,
    *,
    serial_id: Optional[int] = None,
):
    species = TABLE_TO_SPECIES[table]

    if serial_id is None:
        serial_id = await allocate_next_committed_serial(conn)
        await conn.execute(
            """
            INSERT INTO animals (serial_id, species)
            VALUES ($1, $2)
            """,
            serial_id,
            species,
        )
    else:
        if serial_id <= 0:
            raise HTTPException(400, "serial_id must be a positive integer")
        animal_row = await conn.fetchrow(
            """
            INSERT INTO animals (serial_id, species)
            VALUES ($1, $2)
            ON CONFLICT (serial_id) DO NOTHING
            RETURNING serial_id, species
            """,
            serial_id,
            species,
        )
        if animal_row is None:
            animal_row = await conn.fetchrow(
                "SELECT serial_id, species FROM animals WHERE serial_id = $1",
                serial_id,
            )
            if not animal_row:
                raise HTTPException(500, "Failed to reserve requested serial_id")
            if animal_row["species"] != species:
                raise HTTPException(
                    409,
                    f"serial_id {serial_id} already belongs to {animal_row['species']}",
                )
        await bump_serial_counter_to_at_least(conn, serial_id)

    insert_data = {"serial_id": serial_id, **data}
    cols = [k for k, v in insert_data.items() if v is not None]
    vals = [insert_data[k] for k in cols]
    placeholders = ", ".join(f"${i+1}" for i in range(len(cols)))
    col_names = ", ".join(cols)

    row = await conn.fetchrow(
        """
        INSERT INTO {table} ({col_names})
        VALUES ({placeholders})
        ON CONFLICT (serial_id) DO NOTHING
        RETURNING *
        """.format(table=table, col_names=col_names, placeholders=placeholders),
        *vals,
    )
    if row:
        return dict(row), True

    existing = await conn.fetchrow(
        f"SELECT * FROM {table} WHERE serial_id = $1",
        serial_id,
    )
    if existing and _matches_non_null_fields(dict(existing), data):
        return dict(existing), False

    singular = table.rstrip("s")
    raise HTTPException(
        409,
        f"{singular.title()} #{serial_id} already exists with conflicting data",
    )


def build_animal_router(table: str, *, allow_explicit_serial: bool = False) -> APIRouter:
    """Build a full CRUD router for an animal table."""
    router = APIRouter()
    fields = TABLE_FIELDS[table]
    singular = table.rstrip("s")  # "chickens" -> "chicken"
    species = TABLE_TO_SPECIES[table]

    # Build pydantic models dynamically based on table fields
    field_defs = {f: ANIMAL_FIELDS[f] for f in fields}

    create_defs = dict(field_defs)
    if allow_explicit_serial:
        create_defs = {"serial_id": (Optional[int], None), **create_defs}

    CreateModel = create_model(f"{singular.title()}Create", **create_defs)

    UpdateModel = create_model(
        f"{singular.title()}Update",
        **field_defs,
    )

    @router.get("")
    async def list_animals(request: Request, prov_id: Optional[int] = None):
        pool = await get_conn(request)
        async with pool.acquire() as conn:
            if prov_id:
                rows = await conn.fetch(
                    f"SELECT * FROM {table} WHERE prov_id = $1 ORDER BY serial_id", prov_id
                )
            else:
                rows = await conn.fetch(f"SELECT * FROM {table} ORDER BY serial_id")
            return [dict(r) for r in rows]

    @router.get("/{serial_id}")
    async def get_animal(request: Request, serial_id: int):
        pool = await get_conn(request)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {table} WHERE serial_id = $1", serial_id
            )
            if not row:
                raise HTTPException(404, f"{singular.title()} not found")
            return dict(row)

    @router.post("", status_code=201)
    async def create_animal(request: Request, body: CreateModel, response: Response):
        pool = await get_conn(request)
        data = body.model_dump()
        explicit_serial_id = data.pop("serial_id", None) if allow_explicit_serial else None

        async with pool.acquire() as conn:
            async with conn.transaction():
                await lock_new_animal_creation_gate(conn)
                blocking_session = await get_blocking_new_animal_session(conn)
                if blocking_session:
                    raise HTTPException(
                        409,
                        "Finish or discard the pending new-animal grade before creating another animal",
                    )

                row, created = await _create_species_animal(
                    conn,
                    table,
                    data,
                    serial_id=explicit_serial_id,
                )
                if not created:
                    response.status_code = 200
                return row

    @router.put("/{serial_id}")
    async def update_animal(request: Request, serial_id: int, body: UpdateModel):
        pool = await get_conn(request)
        fields_to_update = body.model_dump(exclude_unset=True)

        if not fields_to_update:
            raise HTTPException(400, "No fields to update")

        sets = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(fields_to_update))
        vals = [serial_id] + list(fields_to_update.values())

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE {table} SET {sets} WHERE serial_id = $1 RETURNING *", *vals
            )
            if not row:
                raise HTTPException(404, f"{singular.title()} not found")
            return dict(row)

    @router.delete("/{serial_id}")
    async def delete_animal(request: Request, serial_id: int):
        pool = await get_conn(request)
        cleanup_prefixes = [str(serial_id)]
        async with pool.acquire() as conn:
            async with conn.transaction():
                grade_row = await conn.fetchrow(
                    "SELECT analysis_key FROM grade_results WHERE serial_id = $1",
                    serial_id,
                )
                if grade_row and grade_row.get("analysis_key"):
                    cleanup_prefixes.append(grade_row["analysis_key"])

                await conn.execute(
                    "DELETE FROM grade_results WHERE serial_id = $1",
                    serial_id,
                )
                result = await conn.execute(
                    f"DELETE FROM {table} WHERE serial_id = $1", serial_id
                )
                if result == "DELETE 0":
                    raise HTTPException(404, f"{singular.title()} not found")

                await conn.execute(
                    "DELETE FROM animals WHERE serial_id = $1", serial_id
                )

        # S3 cleanup in background — don't block the response
        if S3_CAPTURES_BUCKET or S3_PROCESSED_BUCKET:
            threading.Thread(
                target=_s3_cleanup, args=(cleanup_prefixes,), daemon=True
            ).start()

        return {"deleted": serial_id}

    return router

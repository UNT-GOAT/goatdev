"""
Generic CRUD factory for animal tables (chickens, goats, lambs).

Each table has the same core fields (serial_id, weights, dates, prov_id)
with minor differences (goats have hook_id, lambs/goats have description + grade).
This factory generates a full CRUD router for any of them.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, create_model
from typing import Optional
from datetime import date
from routes import get_conn


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


def build_animal_router(table: str) -> APIRouter:
    """Build a full CRUD router for an animal table."""
    router = APIRouter()
    fields = TABLE_FIELDS[table]
    singular = table.rstrip("s")  # "chickens" -> "chicken"

    # Build pydantic models dynamically based on table fields
    field_defs = {f: ANIMAL_FIELDS[f] for f in fields}

    CreateModel = create_model(
        f"{singular.title()}Create",
        serial_id=(int, ...),  # required
        **field_defs,
    )

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
    async def create_animal(request: Request, body: CreateModel):
        pool = await get_conn(request)
        data = body.model_dump()
        cols = [k for k, v in data.items() if v is not None]
        vals = [data[k] for k in cols]
        placeholders = ", ".join(f"${i+1}" for i in range(len(cols)))
        col_names = ", ".join(cols)

        async with pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) RETURNING *",
                    *vals,
                )
                return dict(row)
            except asyncpg.UniqueViolationError:
                raise HTTPException(409, f"{singular.title()} with serial_id {data['serial_id']} already exists")

    @router.put("/{serial_id}")
    async def update_animal(request: Request, serial_id: int, body: UpdateModel):
        pool = await get_conn(request)
        fields_to_update = {k: v for k, v in body.model_dump().items() if v is not None}
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
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {table} WHERE serial_id = $1", serial_id
            )
            if result == "DELETE 0":
                raise HTTPException(404, f"{singular.title()} not found")
            return {"deleted": serial_id}

    return router


# Needed for the dynamic route closures to capture `table` correctly
import asyncpg

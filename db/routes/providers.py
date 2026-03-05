from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from datetime import date
from routes import get_conn

router = APIRouter()


class ProviderIn(BaseModel):
    name: str
    company: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    active_since: Optional[date] = None
    status: Optional[str] = "active"


class ProviderUpdate(BaseModel):
    name: Optional[str] = None
    company: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    active_since: Optional[date] = None
    status: Optional[str] = None


@router.get("")
async def list_providers(request: Request, status: Optional[str] = None):
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        if status:
            rows = await conn.fetch(
                "SELECT * FROM providers WHERE status = $1 ORDER BY name", status
            )
        else:
            rows = await conn.fetch("SELECT * FROM providers ORDER BY name")
        return [dict(r) for r in rows]


@router.get("/{provider_id}")
async def get_provider(request: Request, provider_id: int):
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM providers WHERE id = $1", provider_id)
        if not row:
            raise HTTPException(404, "Provider not found")
        return dict(row)


@router.post("", status_code=201)
async def create_provider(request: Request, body: ProviderIn):
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO providers (name, company, phone, email, address, active_since, status)
               VALUES ($1, $2, $3, $4, $5, $6, $7)
               RETURNING *""",
            body.name, body.company, body.phone, body.email,
            body.address, body.active_since, body.status,
        )
        return dict(row)


@router.put("/{provider_id}")
async def update_provider(request: Request, provider_id: int, body: ProviderUpdate):
    pool = await get_conn(request)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "No fields to update")

    sets = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(fields))
    vals = [provider_id] + list(fields.values())

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"UPDATE providers SET {sets} WHERE id = $1 RETURNING *", *vals
        )
        if not row:
            raise HTTPException(404, "Provider not found")
        return dict(row)


@router.delete("/{provider_id}")
async def delete_provider(request: Request, provider_id: int):
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM providers WHERE id = $1", provider_id)
        if result == "DELETE 0":
            raise HTTPException(404, "Provider not found")
        return {"deleted": provider_id}

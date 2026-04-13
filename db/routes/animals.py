"""
Animals route — unified serial_id assignment across all species.

POST /animals with {"species": "goat"} creates an animals row using the
committed serial counter.

GET /animals/next returns the authoritative next committed serial_id for UI
display. Serial IDs are assigned only when a new animal record is committed.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from routes import get_conn
from routes.serials import (
    allocate_next_committed_serial,
    get_blocking_new_animal_session,
    get_next_committed_serial,
    lock_new_animal_creation_gate,
)

router = APIRouter()

VALID_SPECIES = {"chicken", "goat", "lamb"}


class AnimalCreate(BaseModel):
    species: str


@router.post("", status_code=201)
async def create_animal(request: Request, body: AnimalCreate):
    """Create a new animal row using the database-owned sequence."""
    if body.species not in VALID_SPECIES:
        raise HTTPException(400, f"Invalid species. Must be one of: {', '.join(VALID_SPECIES)}")

    pool = await get_conn(request)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await lock_new_animal_creation_gate(conn)
            blocking_session = await get_blocking_new_animal_session(conn)
            if blocking_session:
                raise HTTPException(
                    409,
                    "Finish or discard the pending new-animal grade before creating another animal",
                )
            serial_id = await allocate_next_committed_serial(conn)
            row = await conn.fetchrow(
                """INSERT INTO animals (serial_id, species)
                   VALUES ($1, $2)
                   RETURNING serial_id, species, created_at""",
                serial_id,
                body.species,
            )
            return dict(row)


@router.post("/allocate", status_code=201)
async def allocate_serial_id(request: Request):
    """Deprecated — serial IDs are assigned only when a record is committed."""
    raise HTTPException(
        410,
        "serial reservation has been removed; create the animal or grading session instead",
    )


@router.get("/next")
async def peek_next_id(request: Request):
    """Preview the next committed serial_id (display-only)."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        next_id = await get_next_committed_serial(conn)
        return {"next_serial_id": next_id}


@router.get("")
async def list_animals(request: Request, species: Optional[str] = None):
    """List all animals, optionally filtered by species."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        if species:
            if species not in VALID_SPECIES:
                raise HTTPException(400, f"Invalid species. Must be one of: {', '.join(VALID_SPECIES)}")
            rows = await conn.fetch(
                "SELECT * FROM animals WHERE species = $1 ORDER BY serial_id", species
            )
        else:
            rows = await conn.fetch("SELECT * FROM animals ORDER BY serial_id")
        return [dict(r) for r in rows]


@router.get("/{serial_id}")
async def get_animal(request: Request, serial_id: int):
    """Get an animal by serial_id, including its species-specific data."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        animal = await conn.fetchrow(
            "SELECT * FROM animals WHERE serial_id = $1", serial_id
        )
        if not animal:
            raise HTTPException(404, "Animal not found")

        species = animal["species"]
        table = species + "s"  # chicken -> chickens

        detail = await conn.fetchrow(
            f"SELECT * FROM {table} WHERE serial_id = $1", serial_id
        )

        result = dict(animal)
        if detail:
            result["details"] = dict(detail)
        else:
            result["details"] = None

        return result


@router.delete("/{serial_id}")
async def delete_animal(request: Request, serial_id: int):
    """Delete an animal row by serial_id."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "DELETE FROM animals WHERE serial_id = $1 RETURNING serial_id", serial_id
        )
        if not row:
            raise HTTPException(404, "Animal not found")
        return {"deleted": serial_id}

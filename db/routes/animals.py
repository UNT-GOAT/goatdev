"""
Animals route — unified serial_id assignment across all species.

POST /animals with {"species": "goat"} assigns the next serial_id.
The frontend calls this first, then creates the species-specific record.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from routes import get_conn

router = APIRouter()

VALID_SPECIES = {"chicken", "goat", "lamb"}


class AnimalCreate(BaseModel):
    species: str


@router.post("", status_code=201)
async def create_animal(request: Request, body: AnimalCreate):
    """Assign the next serial_id for a new animal."""
    if body.species not in VALID_SPECIES:
        raise HTTPException(400, f"Invalid species. Must be one of: {', '.join(VALID_SPECIES)}")

    pool = await get_conn(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO animals (serial_id, species)
               VALUES ((SELECT COALESCE(MAX(serial_id), 0) + 1 FROM animals), $1)
               RETURNING serial_id, species, created_at""",
            body.species,
        )
        return dict(row)


@router.get("/next")
async def peek_next_id(request: Request):
    """Preview what the next serial_id will be (without assigning it)."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COALESCE(MAX(serial_id), 0) + 1 AS next_id FROM animals")
        return {"next_serial_id": row["next_id"]}


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
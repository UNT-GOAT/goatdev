"""
Animals route — unified serial_id assignment across all species.

POST /animals with {"species": "goat"} creates an animals row using the
database-owned sequence.

POST /animals/allocate reserves the next sequence value without creating a row.
The grading flow uses that value as a durable serial_id for the subsequent
species-specific create.
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
    """Create a new animal row using the database-owned sequence."""
    if body.species not in VALID_SPECIES:
        raise HTTPException(400, f"Invalid species. Must be one of: {', '.join(VALID_SPECIES)}")

    pool = await get_conn(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO animals (species)
               VALUES ($1)
               RETURNING serial_id, species, created_at""",
            body.species,
        )
        return dict(row)


@router.post("/allocate", status_code=201)
async def allocate_serial_id(request: Request):
    """Reserve the next serial_id from the animals sequence without inserting."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT nextval(pg_get_serial_sequence('animals', 'serial_id')) AS serial_id"""
        )
        return {"serial_id": row["serial_id"]}


@router.get("/next")
async def peek_next_id(request: Request):
    """Preview the next serial_id in the animals sequence (display-only)."""
    pool = await get_conn(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                CASE
                    WHEN is_called THEN last_value + 1
                    ELSE last_value
                END AS next_id
            FROM animals_serial_id_seq
            """
        )
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

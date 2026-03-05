"""
herdsync-db — Animal & Provider Data Service

Single gateway to the HerdSync Postgres database.
All reads/writes to animal, provider, and grading data go through this service.

Port: 8002
"""

from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
import asyncpg
import os
import json

from routes import providers, chickens, goats, lambs, grading


# ============================================================
# DATABASE
# ============================================================

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var is required")

pool: asyncpg.Pool = None


async def get_pool() -> asyncpg.Pool:
    return pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=2,
        max_size=10,
        ssl="require",
    )

    # Apply schema on startup (idempotent)
    async with pool.acquire() as conn:
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                await conn.execute(f.read())

    print(f"[herdsync-db] Connected to database, pool ready")
    yield
    await pool.close()
    print(f"[herdsync-db] Database pool closed")


# ============================================================
# APP
# ============================================================

app = FastAPI(title="herdsync-db", version="1.0.0", lifespan=lifespan)

# Dependency: inject pool into route modules
app.state.get_pool = get_pool

# Mount routes
app.include_router(providers.router, prefix="/providers", tags=["providers"])
app.include_router(chickens.router, prefix="/chickens", tags=["chickens"])
app.include_router(goats.router, prefix="/goats", tags=["goats"])
app.include_router(lambs.router, prefix="/lambs", tags=["lambs"])
app.include_router(grading.router, prefix="/grading", tags=["grading"])


@app.get("/health")
async def health():
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": str(e)}

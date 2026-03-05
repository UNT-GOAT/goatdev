# Shared helper to get db pool from app state
from fastapi import Request
import asyncpg


async def get_conn(request: Request) -> asyncpg.Pool:
    pool = await request.app.state.get_pool()
    return pool

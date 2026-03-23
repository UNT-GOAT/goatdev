"""
Audit log routes.

GET  /audit-logs         — List audit entries (filterable by user, action, resource, date range)
POST /audit-logs         — Create a new audit entry (called internally by db-proxy)
GET  /audit-logs/stats   — Summary counts by user and action type
"""

from fastapi import APIRouter, Request, Query
from typing import Optional
import json

router = APIRouter()


@router.get("")
async def list_audit_logs(
    request: Request,
    username: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    after: Optional[str] = Query(None, description="ISO datetime, e.g. 2026-03-01T00:00:00"),
    before: Optional[str] = Query(None, description="ISO datetime"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List audit log entries with optional filters."""
    pool = await request.app.state.get_pool()

    conditions = []
    params = []
    idx = 1

    if username:
        conditions.append(f"username = ${idx}")
        params.append(username)
        idx += 1

    if action:
        conditions.append(f"action = ${idx}")
        params.append(action)
        idx += 1

    if resource_type:
        conditions.append(f"resource_type = ${idx}")
        params.append(resource_type)
        idx += 1

    if after:
        conditions.append(f"timestamp >= ${idx}::timestamptz")
        params.append(after)
        idx += 1

    if before:
        conditions.append(f"timestamp <= ${idx}::timestamptz")
        params.append(before)
        idx += 1

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    # Get total count
    count_sql = f"SELECT COUNT(*) FROM audit_logs {where}"
    query_sql = f"""
        SELECT id, timestamp, username, role, action, resource_type,
               resource_id, detail, ip_address
        FROM audit_logs
        {where}
        ORDER BY timestamp DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        total = await conn.fetchval(count_sql, *params[:-2])
        rows = await conn.fetch(query_sql, *params)

    return {
        "logs": [
            {
                "id": r["id"],
                "timestamp": r["timestamp"].isoformat() if r["timestamp"] else None,
                "username": r["username"],
                "role": r["role"],
                "action": r["action"],
                "resource_type": r["resource_type"],
                "resource_id": r["resource_id"],
                "detail": json.loads(r["detail"]) if r["detail"] else None,
                "ip_address": r["ip_address"],
            }
            for r in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.post("", status_code=201)
async def create_audit_log(request: Request):
    """
    Create an audit log entry.

    Called internally by db-proxy after successful mutations.
    Not exposed publicly — db-proxy is the only caller.

    Expected body:
    {
        "username": "admin",
        "role": "admin",
        "action": "create",
        "resource_type": "goat",
        "resource_id": "42",
        "detail": {"live_weight": 85.5, "description": "meat"},
        "ip_address": "1.2.3.4"
    }
    """
    pool = await request.app.state.get_pool()
    body = await request.json()

    username = body.get("username", "unknown")
    role = body.get("role")
    action = body.get("action", "unknown")
    resource_type = body.get("resource_type", "unknown")
    resource_id = body.get("resource_id")
    detail = body.get("detail")
    ip_address = body.get("ip_address")

    detail_json = json.dumps(detail) if detail else None

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO audit_logs (username, role, action, resource_type, resource_id, detail, ip_address)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
            RETURNING id, timestamp
            """,
            username, role, action, resource_type, resource_id, detail_json, ip_address,
        )

    return {"id": row["id"], "timestamp": row["timestamp"].isoformat()}


@router.get("/stats")
async def audit_stats(request: Request):
    """Summary counts grouped by username and action."""
    pool = await request.app.state.get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT username, action, COUNT(*) as count
            FROM audit_logs
            GROUP BY username, action
            ORDER BY username, action
            """
        )

    return {
        "stats": [
            {"username": r["username"], "action": r["action"], "count": r["count"]}
            for r in rows
        ]
    }
"""
herdsync-db-proxy — DB Proxy Service (Auth-gated)

Purpose:
- Expose ONLY a /db/* surface publicly (behind ALB) while keeping the real DB
  service internal/private.
- Enforce auth on every /db/* request using herdsync-auth (JWT verify).
- Handle browser CORS correctly (OPTIONS preflight has no Authorization).

Port: 8003
"""

import os
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware


# =============================================================================
# ENV
# =============================================================================

# Internal URL of the db service (reachable from this container / VPC)
# Examples:
#   http://127.0.0.1:8002
#   http://db:8002         (if on same docker network)
DB_INTERNAL = os.environ.get("DB_INTERNAL", "http://127.0.0.1:8002").rstrip("/")

# Public auth base (or internal auth URL if you have one); must support /auth/verify
AUTH_BASE = os.environ.get("AUTH_BASE", "http://127.0.0.1:8001").rstrip("/")

# Comma-separated allowed origins for browser access (CloudFront site domain)
# Example: "https://herd-sync.com"
CORS_ALLOW_ORIGINS = os.environ.get("CORS_ALLOW_ORIGINS", "https://herd-sync.com")


# =============================================================================
# APP
# =============================================================================

app = FastAPI(title="herdsync-db-proxy", version="1.0.0")

# CORS MUST be middleware so it can handle preflight cleanly.
# OPTIONS requests do not carry Authorization.
allowed_origins = [o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# AUTH
# =============================================================================

async def verify_token(token: str) -> None:
    """
    Verify JWT via herdsync-auth.
    This assumes your auth service exposes a verification endpoint.

    Expected behavior:
      - 200 => token valid
      - 401/403 => invalid/expired
    """
    # Try common patterns:
    #   GET  /auth/verify   with Authorization: Bearer <token>
    # If your endpoint differs, change ONLY this function.
    url = f"{AUTH_BASE}/auth/verify"

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})

    if resp.status_code != 200:
        # Preserve detail if possible
        detail = "Invalid token"
        try:
            j = resp.json()
            detail = j.get("detail") or j.get("error") or detail
        except Exception:
            pass
        raise HTTPException(status_code=401, detail=detail)


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/health")
async def health():
    # Report db connectivity based on upstream db /health
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{DB_INTERNAL}/health")
        if r.status_code == 200:
            return {"status": "ok", "database": "connected"}
        return {"status": "degraded", "database": f"upstream_status_{r.status_code}"}
    except Exception as e:
        return {"status": "error", "database": str(e)}


# =============================================================================
# PROXY
# =============================================================================

@app.api_route("/db/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    """
    Proxy /db/* -> DB_INTERNAL/*
    - Enforce Bearer JWT on non-OPTIONS requests
    - Pass through querystring, body, and content-type
    - Return upstream response as-is (json or not)
    """

    # Let CORS preflight succeed (no Authorization header on preflight).
    if request.method == "OPTIONS":
        return Response(status_code=204)

    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = auth.split(" ", 1)[1]
    await verify_token(token)

    # Preserve query string
    qs = request.url.query
    upstream_url = f"{DB_INTERNAL}/{path}" + (f"?{qs}" if qs else "")

    body = await request.body()

    # Forward content-type if present
    forward_headers = {}
    ct = request.headers.get("content-type")
    if ct:
        forward_headers["content-type"] = ct

    async with httpx.AsyncClient(timeout=30) as client:
        upstream_resp = await client.request(
            request.method,
            upstream_url,
            content=body,
            headers=forward_headers,
        )

    # Return upstream response raw (avoids resp.json() crashing on empty/non-json)
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type", "application/json"),
    )
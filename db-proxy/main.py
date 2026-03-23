"""
herdsync-db-proxy — DB Proxy Service (Auth-gated)

Purpose:
- Expose ONLY a /db/* surface publicly (behind ALB) while keeping the real DB
  service internal/private.
- Enforce auth on every /db/* request using herdsync-auth (JWT verify via /auth/verify).
- Handle browser CORS correctly (OPTIONS preflight has no Authorization).
- Audit log all successful mutations (POST/PUT/DELETE) with user attribution.

Port: 8003
"""

import os
import time
import json
import asyncio
from typing import Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware


# =============================================================================
# ENV
# =============================================================================

DB_INTERNAL = os.environ.get("DB_INTERNAL", "http://127.0.0.1:8002").rstrip("/")

# MUST be set correctly in prod (e.g. http://herdsync-auth:8001 or http://127.0.0.1:8001)
AUTH_BASE = (os.environ.get("AUTH_BASE") or "").strip().rstrip("/")

# If your auth verify endpoint differs, adjust here via env
AUTH_VERIFY_PATH = os.environ.get("AUTH_VERIFY_PATH", "/auth/verify")

# Comma-separated allowed origins for browser access (CloudFront/site domain)
# Example: "https://herd-sync.com"
CORS_ALLOW_ORIGINS = os.environ.get("CORS_ALLOW_ORIGINS", "https://herd-sync.com")

# Optional: allowlist upstream route prefixes (relative to db service, without leading slash)
# Example: "providers,goats,lambs,chickens,grading,animals,audit-logs"
# If empty => allow everything under /db/*
ALLOW_PREFIXES = os.environ.get("ALLOW_PREFIXES", "").strip()

# Cache token verification results (seconds). Keeps auth from being called on every request.
VERIFY_CACHE_TTL_SEC = int(os.environ.get("VERIFY_CACHE_TTL_SEC", "30"))


# =============================================================================
# APP
# =============================================================================

app = FastAPI(title="herdsync-db-proxy", version="1.1.0")

allowed_origins = [o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    allow_headers=["authorization", "content-type"],
    max_age=600,
)


# =============================================================================
# HTTP CLIENTS (pooled)
# =============================================================================

_limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
_auth_client: Optional[httpx.AsyncClient] = None
_db_client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def _startup():
    global _auth_client, _db_client

    # Fail fast on missing AUTH_BASE (this is the exact failure you hit earlier)
    if not AUTH_BASE:
        raise RuntimeError("AUTH_BASE env var is required (e.g. http://herdsync-auth:8001)")

    _auth_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=2.0, read=6.0, write=6.0, pool=6.0),
        limits=_limits,
        follow_redirects=False,
    )
    _db_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=2.0, read=20.0, write=20.0, pool=20.0),
        limits=_limits,
        follow_redirects=False,
    )


@app.on_event("shutdown")
async def _shutdown():
    global _auth_client, _db_client
    if _auth_client:
        await _auth_client.aclose()
        _auth_client = None
    if _db_client:
        await _db_client.aclose()
        _db_client = None


# =============================================================================
# TOKEN VERIFY (cached, returns user info)
# =============================================================================

# token -> (expires_at_epoch_sec, ok_bool, detail_if_bad, user_info_dict_or_none)
_verify_cache: Dict[str, Tuple[float, bool, Optional[str], Optional[dict]]] = {}


def _cache_get(token: str) -> Optional[Tuple[bool, Optional[str], Optional[dict]]]:
    entry = _verify_cache.get(token)
    if not entry:
        return None
    exp, ok, detail, user_info = entry
    if time.time() > exp:
        _verify_cache.pop(token, None)
        return None
    return ok, detail, user_info


def _cache_set(token: str, ok: bool, detail: Optional[str], user_info: Optional[dict]) -> None:
    _verify_cache[token] = (time.time() + VERIFY_CACHE_TTL_SEC, ok, detail, user_info)


async def verify_token(token: str) -> dict:
    """
    Verify JWT via herdsync-auth (/auth/verify).
    Uses a short TTL cache to avoid hitting auth on every DB request.

    Returns user info dict with 'username' and 'role' keys.
    """
    cached = _cache_get(token)
    if cached is not None:
        ok, detail, user_info = cached
        if ok:
            return user_info or {}
        raise HTTPException(status_code=401, detail=detail or "Invalid token")

    url = f"{AUTH_BASE}{AUTH_VERIFY_PATH}"

    assert _auth_client is not None
    try:
        resp = await _auth_client.get(url, headers={"Authorization": f"Bearer {token}"})
    except httpx.RequestError:
        # Fail closed: if we can't verify, we reject.
        _cache_set(token, False, "Auth verifier unreachable", None)
        raise HTTPException(status_code=401, detail="Auth verifier unreachable")

    if resp.status_code == 200:
        user_info = {}
        try:
            j = resp.json()
            user_info = {"username": j.get("user", "unknown"), "role": j.get("role", "unknown")}
        except Exception:
            pass
        _cache_set(token, True, None, user_info)
        return user_info

    detail = "Invalid token"
    try:
        j = resp.json()
        detail = j.get("detail") or j.get("error") or detail
    except Exception:
        pass

    _cache_set(token, False, detail, None)
    raise HTTPException(status_code=401, detail=detail)


# =============================================================================
# AUDIT LOGGING
# =============================================================================

# Map HTTP methods to action names
_METHOD_ACTION = {
    "POST": "create",
    "PUT": "update",
    "PATCH": "update",
    "DELETE": "delete",
}

# Map path prefixes to resource types
_RESOURCE_MAP = {
    "goats": "goat",
    "lambs": "lamb",
    "chickens": "chicken",
    "providers": "provider",
    "grading": "grade",
}


def _parse_resource(path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the upstream path into (resource_type, resource_id).

    Examples:
        "goats"         -> ("goat", None)
        "goats/42"      -> ("goat", "42")
        "providers/5"   -> ("provider", "5")
        "grading"       -> ("grade", None)
        "grading/result/3" -> ("grade", "3")
    """
    parts = path.strip("/").split("/")
    if not parts:
        return None, None

    resource_type = _RESOURCE_MAP.get(parts[0])
    if not resource_type:
        return None, None

    # Extract ID
    resource_id = None
    if len(parts) >= 2 and parts[1].isdigit():
        resource_id = parts[1]
    elif len(parts) >= 3 and parts[2].isdigit():
        # e.g. grading/result/3
        resource_id = parts[2]

    return resource_type, resource_id


def _extract_detail(method: str, body: bytes, upstream_body: bytes) -> Optional[dict]:
    """
    Extract a concise detail dict for the audit log.

    For creates: includes the response body (has the created record).
    For updates: includes the request body (what was changed).
    For deletes: minimal info.
    """
    try:
        if method == "POST" and upstream_body:
            data = json.loads(upstream_body)
            # Don't log huge blobs — keep it concise
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if k not in ("measurements", "warnings")}
        elif method in ("PUT", "PATCH") and body:
            data = json.loads(body)
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return None


async def _fire_audit_log(
    user_info: dict,
    method: str,
    path: str,
    request_body: bytes,
    upstream_body: bytes,
    ip_address: str,
):
    """
    Write an audit log entry to the db service. Fire-and-forget.
    Errors are logged but never block the response to the client.
    """
    action = _METHOD_ACTION.get(method)
    if not action:
        return

    # Skip audit logging for dev account
    if user_info.get("username") == "dev":
        return

    resource_type, resource_id = _parse_resource(path)
    if not resource_type:
        return

    # Don't audit reads of audit logs themselves
    if resource_type == "audit-logs":
        return

    detail = _extract_detail(method, request_body, upstream_body)

    payload = {
        "username": user_info.get("username", "unknown"),
        "role": user_info.get("role"),
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "detail": detail,
        "ip_address": ip_address,
    }

    try:
        assert _db_client is not None
        await _db_client.post(
            f"{DB_INTERNAL}/audit-logs",
            json=payload,
            timeout=httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=5.0),
        )
    except Exception as e:
        # Never let audit failure affect the user's request
        print(f"[audit] Failed to write audit log: {e}")


# =============================================================================
# HELPERS
# =============================================================================

_allowed_prefixes = [p.strip().lstrip("/").rstrip("/") for p in ALLOW_PREFIXES.split(",") if p.strip()]


def _is_allowed_path(path: str) -> bool:
    if not _allowed_prefixes:
        return True
    p = path.lstrip("/")
    return any(p == pref or p.startswith(pref + "/") for pref in _allowed_prefixes)


def _sanitize_path(path: str) -> str:
    # Block obvious traversal / weirdness
    if ".." in path or "\\" in path:
        raise HTTPException(status_code=400, detail="Invalid path")
    # Normalize accidental leading slashes
    return path.lstrip("/")


def _pick_forward_headers(req: Request) -> Dict[str, str]:
    h: Dict[str, str] = {}
    ct = req.headers.get("content-type")
    if ct:
        h["content-type"] = ct
    accept = req.headers.get("accept")
    if accept:
        h["accept"] = accept
    return h


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting common proxy headers."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip
    return request.client.host if request.client else "unknown"


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/health")
async def health():
    try:
        assert _db_client is not None
        r = await _db_client.get(f"{DB_INTERNAL}/health")
        if r.status_code == 200:
            return {"status": "ok", "database": "connected"}
        return {"status": "degraded", "database": f"upstream_status_{r.status_code}"}
    except Exception as e:
        return {"status": "error", "database": str(e)}


# =============================================================================
# PROXY
# =============================================================================

@app.api_route(
    "/db/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def proxy(path: str, request: Request):
    """
    Proxy /db/* -> DB_INTERNAL/*
    - Enforce Bearer JWT on non-OPTIONS requests
    - Preserve querystring, body
    - Do NOT forward Authorization to the db service
    - Audit log successful mutations (POST/PUT/PATCH/DELETE)
    - Return upstream response raw (works for empty/non-json)
    """

    # Let CORS preflight succeed (no Authorization header on preflight).
    if request.method == "OPTIONS":
        return Response(status_code=200, content=b"OK", media_type="text/plain")

    # Path checks
    path = _sanitize_path(path)
    if not _is_allowed_path(path):
        # 404 reduces endpoint discovery
        raise HTTPException(status_code=404, detail="Not found")

    # Auth
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = auth.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    user_info = await verify_token(token)

    # Preserve query string
    qs = request.url.query
    upstream_url = f"{DB_INTERNAL}/{path}" + (f"?{qs}" if qs else "")

    body = await request.body()
    forward_headers = _pick_forward_headers(request)

    assert _db_client is not None
    upstream_resp = await _db_client.request(
        request.method,
        upstream_url,
        content=body if body else None,
        headers=forward_headers,
    )

    # Audit log successful mutations (fire-and-forget, non-blocking)
    is_mutation = request.method in ("POST", "PUT", "PATCH", "DELETE")
    is_success = 200 <= upstream_resp.status_code < 300

    if is_mutation and is_success:
        ip = _get_client_ip(request)
        asyncio.ensure_future(
            _fire_audit_log(user_info, request.method, path, body, upstream_resp.content, ip)
        )

    # Minimal safe response headers
    resp_headers: Dict[str, str] = {}
    for k in ("content-type", "cache-control"):
        v = upstream_resp.headers.get(k)
        if v:
            resp_headers[k] = v

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=upstream_resp.headers.get("content-type", "application/json"),
    )
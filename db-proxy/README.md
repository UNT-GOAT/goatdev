# DB Proxy Service - `herdsync-db-proxy`

Auth-gated reverse proxy between the public ALB and the internal database service. Every request to `/db/*` passes through here - JWT is verified, the request is forwarded to `herdsync-db`, and successful mutations are audit-logged.

Runs as a Docker container on EC2 (port 8003). Exposed through the ALB at `/db/*`.

## What It Does

- Enforces JWT authentication on every database request (via `herdsync-auth`'s `/auth/verify`)
- Proxies `/db/*` → internal `herdsync-db` service (strips `/db` prefix)
- Caches token verification results (30s TTL) to avoid hitting auth on every request
- Audit logs all successful POST/PUT/PATCH/DELETE with user attribution, change tracking, and client IP
- Handles CORS preflight (OPTIONS passes through without auth)
- Pre-fetches records before PUT/PATCH to capture before→after diffs for audit

## Architecture Decisions

**Separate proxy vs middleware** - keeping auth enforcement in a dedicated proxy means the db service has zero auth code. The db service is a pure data layer. If auth logic changes, only one service is modified.

**Token verification caching** - a 30-second TTL cache prevents a round-trip to the auth service on every single database request. Trade-off: a revoked token could work for up to 30 seconds. Acceptable given the 15-minute access token lifetime.

**Audit logging** - fires asynchronously after the response is sent to the client. Never blocks the user's request. Skips the `dev` account. Extracts field-level from→to changes for updates by comparing the pre-mutation snapshot to the request body.

**Path allowlisting** - `ALLOW_PREFIXES` env var restricts which upstream routes are accessible. Requests to unlisted paths get 404 (not 403) to reduce endpoint discovery.

## Directory Structure

```
db-proxy/
├── Dockerfile          # python:3.11-slim, exposes port 8003
├── requirements.txt    # FastAPI, uvicorn, httpx, python-jose
└── main.py             # Everything: proxy logic, auth, audit, CORS
```

Single-file service - the entire proxy is ~350 lines in `main.py`. No routes directory needed.

## How It Works

```
Browser → ALB → db-proxy (port 8003)
                    ↓ verify JWT
                herdsync-auth (port 8001)
                    ↓ proxy request
                herdsync-db (port 8002)
                    ↓ if mutation succeeded
                audit log → herdsync-db /audit-logs
```

## Environment Variables

| Variable               | Required | Description                                                |
| ---------------------- | -------- | ---------------------------------------------------------- |
| `AUTH_BASE`            | Yes      | Auth service URL (e.g., `http://herdsync-auth:8001`)       |
| `DB_INTERNAL`          | No       | DB service URL (default: `http://127.0.0.1:8002`)          |
| `CORS_ALLOW_ORIGINS`   | No       | Comma-separated origins (default: `https://herd-sync.com`) |
| `ALLOW_PREFIXES`       | No       | Comma-separated allowed route prefixes (empty = allow all) |
| `VERIFY_CACHE_TTL_SEC` | No       | Token cache lifetime in seconds (default: `30`)            |

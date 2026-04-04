# Auth Service - `herdsync-auth`

Standalone authentication and user management service for the HerdSync platform. Runs as a Docker container on EC2 (port 8001), independent of the grading pipeline.

## What It Does

- Username/password authentication with RS256 JWT tokens
- User management - admins create and manage operator accounts
- JWKS endpoint for offline token validation by other services (Pi, db-proxy)
- Token refresh flow with rotating refresh tokens
- Rate-limited login with sliding window + lockout
- Self-service password change and account deletion

## Architecture Decisions

**RS256 asymmetric JWTs** - the auth service holds the private key and signs tokens. All other services (Pi's auth-verifier, db-proxy) fetch the public key via the JWKS endpoint and validate tokens locally. No shared secrets, no network call on every request.

**Refresh token rotation** - each refresh grants a new refresh token and revokes the old one. Tokens are stored as SHA-256 hashes in Postgres, so a database leak doesn't compromise active sessions.

**Rate limiting** - in-memory sliding window tracks by both IP and username independently. 5 attempts per 5-minute window, 15-minute lockout. Not persistent across restarts (intentional - avoids DB complexity at this scale).

## Directory Structure

```
auth/
├── Dockerfile                  # python:3.11-slim, exposes port 8001
├── requirements.txt            # Pinned dependencies
└── auth/
    ├── __init__.py
    ├── main.py                 # FastAPI app, lifespan (key gen, DB init, admin seed)
    ├── config.py               # Environment variable loading
    ├── database.py             # SQLAlchemy engine, session factory, table creation
    ├── db_models.py            # User and RefreshToken ORM models
    ├── security.py             # RSA key management, JWT create/decode, bcrypt, deactivation tracking
    ├── rate_limiter.py         # Sliding window rate limiter (in-memory, per-IP + per-username)
    ├── models.py               # Pydantic request/response schemas
    └── routes/
        ├── __init__.py
        ├── login.py            # POST /auth/login, /auth/refresh, /auth/logout
        ├── verify.py           # GET /auth/verify (Caddy forward_auth), /auth/me, /auth/.well-known/jwks.json
        ├── users.py            # Admin CRUD: GET/POST/PUT/DELETE /auth/users
        └── account.py          # Self-service: POST /auth/change-password, DELETE /auth/me
```

## API Endpoints

| Method | Path                          | Auth   | Description                                                  |
| ------ | ----------------------------- | ------ | ------------------------------------------------------------ |
| POST   | `/auth/login`                 | None   | Authenticate, receive access + refresh tokens                |
| POST   | `/auth/refresh`               | None   | Exchange refresh token for new token pair                    |
| POST   | `/auth/logout`                | None   | Revoke a refresh token                                       |
| GET    | `/auth/verify`                | Bearer | Token validation for Caddy forward_auth (returns 200 or 401) |
| GET    | `/auth/me`                    | Bearer | Current user info                                            |
| GET    | `/auth/.well-known/jwks.json` | None   | RSA public key in JWKS format                                |
| POST   | `/auth/change-password`       | Bearer | Change own password (requires current password)              |
| DELETE | `/auth/me`                    | Bearer | Delete own account                                           |
| GET    | `/auth/users`                 | Admin  | List all users                                               |
| POST   | `/auth/users`                 | Admin  | Create user account                                          |
| GET    | `/auth/users/{id}`            | Admin  | Get user by ID                                               |
| PUT    | `/auth/users/{id}`            | Admin  | Update user (password, role, active status)                  |
| DELETE | `/auth/users/{id}`            | Admin  | Delete user and all their tokens                             |
| GET    | `/auth/health`                | None   | Health check with DB connectivity test                       |

## Environment Variables

| Variable                      | Required   | Description                                                            |
| ----------------------------- | ---------- | ---------------------------------------------------------------------- |
| `DATABASE_URL`                | Yes        | PostgreSQL connection string (`postgres://user:pass@host:5432/dbname`) |
| `ADMIN_USERNAME`              | First boot | Initial admin username (default: `admin`)                              |
| `ADMIN_PASSWORD`              | First boot | Initial admin password                                                 |
| `ALLOWED_ORIGINS`             | No         | Comma-separated CORS origins (default: CloudFront domain)              |
| `JWT_PRIVATE_KEY_PATH`        | No         | Path to RSA private key (default: `/app/keys/private.pem`)             |
| `JWT_PUBLIC_KEY_PATH`         | No         | Path to RSA public key (default: `/app/keys/public.pem`)               |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No         | Access token lifetime (default: `15`)                                  |
| `REFRESH_TOKEN_EXPIRE_DAYS`   | No         | Refresh token lifetime (default: `30`)                                 |

The `auth-keys` volume is shared with `db-proxy` and `goat-api` (read-only) so they can access the public key for local JWT validation. Auth must be deployed first on a fresh setup so the RSA key pair exists before other services start.

## Token Flow

```
Login: username + password → access_token (15min) + refresh_token (30 days)
API call: Authorization: Bearer <access_token> → 200 or 401
Refresh: refresh_token → new access_token + new refresh_token (old revoked)
Logout: refresh_token → revoked (access token expires naturally in ≤15 min)
```

## Password Policy

All passwords (admin seed, user creation, self-service change) must be 8+ characters with at least one uppercase letter, one lowercase letter, one number, and one special character.

## Deactivation

When an admin deactivates a user, their refresh tokens are revoked immediately and their user ID is added to an in-memory set. The `/auth/verify` endpoint checks this set on every request, so deactivated users are locked out within the same request - no 15-minute window. The set clears on restart, but that's acceptable since the user's `active=false` flag persists in the database.

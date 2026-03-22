"""
HerdSync Auth Service

Standalone authentication service for the HerdSync facility platform.
Runs on EC2 port 8001, independent of the goat grading pipeline.

Provides:
  - Username/password authentication with JWT tokens
  - User management (admin creates operators)
  - JWKS endpoint for offline token validation by other services
  - Token refresh flow for seamless session management

Architecture:
  - Postgres (RDS) for user storage
  - RS256 JWTs — asymmetric keys, no shared secrets
  - Access tokens: 15 min, validated locally by consumers
  - Refresh tokens: 30 days, stored hashed in DB, revocable
"""

import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ADMIN_USERNAME, ADMIN_PASSWORD, ALLOWED_ORIGINS
from .database import create_tables, SessionLocal
from .db_models import User
from .security import hash_password, _ensure_keys
from .routes import login, verify, users, account

logger = logging.getLogger("auth")


def _seed_admin():
    """Create initial admin account if no users exist."""
    db = SessionLocal()
    try:
        user_count = db.query(User).count()
        if user_count == 0:
            if not ADMIN_PASSWORD:
                logger.error(
                    "No users in database and ADMIN_PASSWORD not set. "
                    "Set ADMIN_PASSWORD env var to create initial admin account."
                )
                return

            admin = User(
                username=ADMIN_USERNAME,
                password_hash=hash_password(ADMIN_PASSWORD),
                role="admin",
                created_by="system",
            )
            db.add(admin)
            db.commit()
            logger.info(f"Created initial admin account: {ADMIN_USERNAME}")
        else:
            logger.info(f"Database has {user_count} user(s), skipping seed")
    finally:
        db.close()


def _cleanup_expired_tokens():
    """Remove expired refresh tokens on startup."""
    db = SessionLocal()
    try:
        from .db_models import RefreshToken
        deleted = db.query(RefreshToken).filter(
            RefreshToken.expires_at < datetime.utcnow()
        ).delete()
        db.commit()
        if deleted:
            logger.info(f"Cleaned up {deleted} expired refresh token(s)")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("=" * 50)
    logger.info("HERDSYNC AUTH SERVICE STARTING")

    # Generate/load RSA key pair
    _ensure_keys()
    logger.info("RSA keys ready")

    # Create tables
    create_tables()
    logger.info("Database tables ready")

    # Seed admin
    _seed_admin()

    # Cleanup
    _cleanup_expired_tokens()

    logger.info("Auth service ready")
    logger.info("=" * 50)

    yield

    logger.info("Auth service shutting down")


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="HerdSync Auth",
    description="Authentication service for the HerdSync facility platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(login.router)
app.include_router(verify.router)
app.include_router(users.router)
app.include_router(account.router)


@app.get("/auth/health")
def health():
    """Health check — verifies DB connectivity."""
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "ok" if db_ok else "degraded",
        "service": "auth",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_ok,
    }

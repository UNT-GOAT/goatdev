"""
Authentication routes — login, refresh, logout.

These are the only unauthenticated endpoints (login and refresh).
Everything else requires a valid access token.
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..database import get_db
from ..db_models import User, RefreshToken
from ..security import (
    verify_password, create_access_token, create_refresh_token,
    hash_refresh_token
)
from ..models import LoginRequest, LoginResponse, RefreshRequest, RefreshResponse
from ..config import ACCESS_TOKEN_EXPIRE_MINUTES
from ..rate_limiter import login_limiter

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest, request: Request, db: Session = Depends(get_db)):
    """
    Authenticate with username + password.
    Returns access token (short-lived) and refresh token (long-lived).
    """
    # Rate limit by IP and username independently.
    # Attacker spraying passwords across users from one IP → IP key triggers.
    # Attacker hitting one user from a botnet → username key triggers.
    client_ip = request.client.host if request.client else "unknown"
    ip_key = f"ip:{client_ip}"
    user_key = f"user:{req.username}"

    allowed, retry_after = login_limiter.check(ip_key, user_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many login attempts. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    user = db.query(User).filter(
        User.username == req.username,
        User.active == True
    ).first()

    if not user or not verify_password(req.password, user.password_hash):
        login_limiter.record_failure(ip_key, user_key)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create tokens
    access_token = create_access_token(user.id, user.username, user.role)
    raw_refresh, refresh_expires = create_refresh_token()

    # Store refresh token hash in DB
    db_token = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(raw_refresh),
        expires_at=refresh_expires,
    )
    db.add(db_token)

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    # Clear rate limit counters on success
    login_limiter.record_success(ip_key, user_key)

    return LoginResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user.to_dict(),
    )


@router.post("/refresh", response_model=RefreshResponse)
def refresh(req: RefreshRequest, db: Session = Depends(get_db)):
    """
    Exchange a valid refresh token for a new access token.
    The refresh token itself is NOT rotated — it stays valid until expiry.
    """
    token_hash = hash_refresh_token(req.refresh_token)

    db_token = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash
    ).first()

    if not db_token:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if db_token.expires_at < datetime.utcnow():
        # Clean up expired token
        db.delete(db_token)
        db.commit()
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user = db.query(User).filter(
        User.id == db_token.user_id,
        User.active == True
    ).first()

    if not user:
        raise HTTPException(status_code=401, detail="User deactivated")

    access_token = create_access_token(user.id, user.username, user.role)

    return RefreshResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout")
def logout(req: RefreshRequest, db: Session = Depends(get_db)):
    """
    Revoke a refresh token. Access tokens will expire naturally (15 min).
    For immediate revocation of an associate, deactivate their account
    via /auth/users — that prevents refresh and new access tokens.
    """
    token_hash = hash_refresh_token(req.refresh_token)
    db_token = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash
    ).first()

    if db_token:
        db.delete(db_token)
        db.commit()

    return {"status": "ok"}

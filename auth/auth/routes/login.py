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
    Authenticate with username + password, receive access + refresh tokens.
    """
    client_ip = request.client.host if request.client else "unknown"

    # Rate limit check
    allowed, retry_after = login_limiter.check(client_ip, req.username)
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
        login_limiter.record_failure(client_ip, req.username)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    login_limiter.record_success(client_ip, req.username)

    # Create tokens
    access_token = create_access_token(user.id, user.username, user.role)
    raw_refresh, refresh_expires = create_refresh_token()

    # Store hashed refresh token
    db_token = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(raw_refresh),
        expires_at=refresh_expires,
    )
    db.add(db_token)
    db.commit()

    return LoginResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={"id": user.id, "username": user.username, "role": user.role},
    )


@router.post("/refresh", response_model=RefreshResponse)
def refresh(req: RefreshRequest, db: Session = Depends(get_db)):
    """
    Exchange a valid refresh token for a new access token.
    The old refresh token is revoked and a new one issued (rotation).
    """
    token_hash = hash_refresh_token(req.refresh_token)

    db_token = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash
    ).first()

    if not db_token:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if db_token.expires_at < datetime.utcnow():
        db.delete(db_token)
        db.commit()
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user = db.query(User).filter(
        User.id == db_token.user_id,
        User.active == True
    ).first()

    if not user:
        raise HTTPException(status_code=401, detail="User deactivated")

    # Revoke old refresh token
    db.delete(db_token)

    # Issue new tokens
    access_token = create_access_token(user.id, user.username, user.role)
    new_raw_refresh, new_refresh_expires = create_refresh_token()

    new_db_token = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(new_raw_refresh),
        expires_at=new_refresh_expires,
    )
    db.add(new_db_token)
    db.commit()

    return RefreshResponse(
        access_token=access_token,
        refresh_token=new_raw_refresh,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout")
def logout(req: RefreshRequest, db: Session = Depends(get_db)):
    """
    Revoke a refresh token. Access tokens will expire naturally (15 min).
    """
    token_hash = hash_refresh_token(req.refresh_token)
    db_token = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash
    ).first()
    if db_token:
        db.delete(db_token)
        db.commit()
    return {"status": "ok"}
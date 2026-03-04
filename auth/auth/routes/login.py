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


@router.post("/refresh", response_model=RefreshResponse)
def refresh(req: RefreshRequest, db: Session = Depends(get_db)):
    """
    Exchange a valid refresh token for a new access token.
    The old refresh token is revoked and a new one issued (rotation).
    If someone steals a refresh token and uses it, the legitimate user's
    next refresh will fail — signaling compromise.
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

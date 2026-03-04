"""
User management routes — admin only.

Admins can create, list, update, and deactivate user accounts.
Deactivating a user immediately prevents token refresh. Existing
access tokens expire naturally within 15 minutes.
"""

import re
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..db_models import User, RefreshToken
from ..security import hash_password, mark_user_deactivated, mark_user_reactivated
from ..models import (
    CreateUserRequest, UpdateUserRequest,
    UserResponse, UserListResponse
)
from .verify import require_admin

router = APIRouter(prefix="/auth/users", tags=["users"])


@router.get("", response_model=UserListResponse)
def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """List all users."""
    users = db.query(User).order_by(User.created_at.desc()).all()
    return UserListResponse(
        users=[UserResponse(**u.to_dict()) for u in users],
        total=len(users),
    )


@router.post("", response_model=UserResponse, status_code=201)
def create_user(
    req: CreateUserRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """Create a new user account."""
    # Enforce password policy
    if (len(req.password) < 8 or
            not re.search(r'[A-Z]', req.password) or
            not re.search(r'[a-z]', req.password) or
            not re.search(r'[0-9]', req.password) or
            not re.search(r'[^A-Za-z0-9]', req.password)):
        raise HTTPException(status_code=400, detail="Password must be 8+ chars with uppercase, lowercase, number, and special character")

    # Check username not taken
    existing = db.query(User).filter(User.username == req.username).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")

    user = User(
        username=req.username,
        password_hash=hash_password(req.password),
        role=req.role,
        created_by=admin.username,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserResponse(**user.to_dict())


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """Get a single user by ID."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**user.to_dict())


@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    req: UpdateUserRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Update a user's password, role, or active status.

    Deactivating a user (active=false) revokes all their refresh tokens
    immediately. Their access tokens will expire within 15 minutes.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent admin from deactivating themselves
    if req.active is False and user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

    if req.password is not None:
        if (len(req.password) < 8 or
                not re.search(r'[A-Z]', req.password) or
                not re.search(r'[a-z]', req.password) or
                not re.search(r'[0-9]', req.password) or
                not re.search(r'[^A-Za-z0-9]', req.password)):
            raise HTTPException(status_code=400, detail="Password must be 8+ chars with uppercase, lowercase, number, and special character")
        user.password_hash = hash_password(req.password)

    if req.role is not None:
        # Prevent admin from demoting themselves
        if user.id == admin.id and req.role != "admin":
            raise HTTPException(status_code=400, detail="Cannot change your own role")
        user.role = req.role

    if req.active is not None:
        user.active = req.active
        if not req.active:
            # Revoke all refresh tokens
            db.query(RefreshToken).filter(
                RefreshToken.user_id == user.id
            ).delete()
            # Immediately reject any existing access tokens
            mark_user_deactivated(user.id)
        else:
            mark_user_reactivated(user.id)

    db.commit()
    db.refresh(user)

    return UserResponse(**user.to_dict())


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Permanently delete a user and all their tokens.
    Prefer deactivation (PUT with active=false) for audit trail.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    mark_user_deactivated(user.id)
    db.delete(user)
    db.commit()

    return {"status": "deleted", "username": user.username}

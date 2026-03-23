"""
Self-service account route.

Any authenticated user can change their own password.
Requires current password verification — prevents someone
with a stolen access token from locking out the real user.
"""

import re
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..db_models import User
from ..security import verify_password, hash_password
from .verify import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/change-password")
def change_password(
    req: ChangePasswordRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Change the authenticated user's own password.

    Requires current password for verification.
    New password must meet the same policy as account creation:
    8+ chars, uppercase, lowercase, number, special character.
    """
    # Verify current password
    if not verify_password(req.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    # Enforce password policy
    if (len(req.new_password) < 8 or
            not re.search(r'[A-Z]', req.new_password) or
            not re.search(r'[a-z]', req.new_password) or
            not re.search(r'[0-9]', req.new_password) or
            not re.search(r'[^A-Za-z0-9]', req.new_password)):
        raise HTTPException(
            status_code=400,
            detail="Password must be 8+ chars with uppercase, lowercase, number, and special character"
        )

    # Don't allow same password
    if verify_password(req.new_password, user.password_hash):
        raise HTTPException(status_code=400, detail="New password must be different from current password")

    user.password_hash = hash_password(req.new_password)
    db.commit()

    return {"status": "ok", "message": "Password updated"}


@router.delete("/me")
def delete_own_account(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Delete the authenticated user's own account.

    Admins can only delete themselves if at least one other admin remains.
    Operators can always delete themselves.
    """
    if user.role == "admin":
        admin_count = db.query(User).filter(User.role == "admin").count()
        if admin_count <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete the last admin account"
            )

    db.delete(user)
    db.commit()

    return {"status": "ok", "message": f"Account '{user.username}' deleted"}
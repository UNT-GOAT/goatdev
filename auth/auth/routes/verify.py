"""
Token verification routes.

/auth/verify — Called by Caddy's forward_auth on every request to the Pi.
  Returns 200 if valid, 401 if not. Caddy only checks the status code.

/auth/me — Returns current user info. Used by frontends to check session.

/auth/.well-known/jwks.json — Public key for offline JWT validation.
  Pi fetches this once at startup and caches it, so it can validate
  tokens without calling EC2 on every request.
"""

import base64
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from ..database import get_db
from ..db_models import User
from ..security import decode_access_token, get_public_key_pem, get_key_id, is_user_deactivated

router = APIRouter(prefix="/auth", tags=["auth"])


def _extract_token(request: Request) -> str:
    """Extract Bearer token from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return auth[7:]


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """
    FastAPI dependency — extracts and validates JWT, returns User object.
    Used by protected endpoints.
    """
    token = _extract_token(request)
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = db.query(User).filter(
        User.id == int(payload["sub"]),
        User.active == True
    ).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found or deactivated")

    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency — requires admin role."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


@router.get("/verify")
def verify(request: Request):
    """
    Token verification for Caddy forward_auth.

    Caddy sends the original request headers here. We check the
    Authorization header and return 200 (allow) or 401 (deny).
    No body needed — Caddy only looks at the status code.

    Also checks the in-memory deactivated users set for immediate
    revocation (no DB query needed on the hot path).
    """
    token = _extract_token(request)
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if is_user_deactivated(payload.get("sub", "")):
        raise HTTPException(status_code=401, detail="User deactivated")
    return {"status": "ok", "user": payload.get("username"), "role": payload.get("role")}


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    """Return current authenticated user info."""
    return user.to_dict()


@router.get("/.well-known/jwks.json")
def jwks():
    """
    Public key in JWKS format.

    Services (Pi Caddy, EC2 middleware, future services) fetch this once
    at startup to validate JWTs locally. No shared secrets needed.
    """
    pub_pem = get_public_key_pem()
    pub_key = load_pem_public_key(pub_pem)
    pub_numbers = pub_key.public_numbers()

    # Encode RSA public key components as base64url
    def _b64url(num, length):
        data = num.to_bytes(length, byteorder="big")
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    n_bytes = (pub_numbers.n.bit_length() + 7) // 8

    return {
        "keys": [{
            "kty": "RSA",
            "alg": "RS256",
            "use": "sig",
            "kid": get_key_id(),
            "n": _b64url(pub_numbers.n, n_bytes),
            "e": _b64url(pub_numbers.e, 3),
        }]
    }

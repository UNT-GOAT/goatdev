"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# =============================================================================
# AUTH
# =============================================================================

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until access token expires
    user: dict


class RefreshRequest(BaseModel):
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# =============================================================================
# USERS
# =============================================================================

class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(..., min_length=8)
    role: str = Field(default="operator", pattern=r"^(admin|operator)$")


class UpdateUserRequest(BaseModel):
    password: Optional[str] = Field(default=None, min_length=8)
    role: Optional[str] = Field(default=None, pattern=r"^(admin|operator)$")
    active: Optional[bool] = None


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    active: bool
    created_by: Optional[str]
    created_at: Optional[str]
    last_login: Optional[str]


class UserListResponse(BaseModel):
    users: List[UserResponse]
    total: int

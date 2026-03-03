"""
Database ORM models.

Users:
  - admin: can create/delete associates, full access
  - associate: can use grading, capture, view features

RefreshTokens:
  - Stored hashed (SHA-256) so a DB leak doesn't compromise active sessions
  - Linked to user, expires after REFRESH_TOKEN_EXPIRE_DAYS
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Boolean, DateTime, ForeignKey, Text, Integer
)
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="associate")  # admin | associate
    active = Column(Boolean, nullable=False, default=True)
    created_by = Column(String(50), nullable=True)  # username of creator
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "active": self.active,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA-256
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    user = relationship("User", back_populates="refresh_tokens")

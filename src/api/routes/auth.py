"""
Authentication API Routes
=========================
Login, logout, token refresh, and API key management.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..config import JWTConfig, APIKeyConfig
from ..models import (
    APIKeyCreate,
    APIKeyResponse,
    ErrorResponse,
    RefreshTokenRequest,
    SuccessResponse,
    TokenRequest,
    TokenResponse,
)
from ..middleware.auth import (
    APIKeyHandler,
    AuthenticatedUser,
    JWTHandler,
    get_authenticated_user,
    require_permission,
)


router = APIRouter(prefix="/auth", tags=["Authentication"])


# =============================================================================
# Storage (Demo)
# =============================================================================

# User storage (would be replaced by Security layer integration)
_users: dict[str, dict[str, Any]] = {
    "admin": {
        "id": "user-admin-001",
        "username": "admin",
        "email": "admin@example.com",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "roles": ["admin", "user"],
        "permissions": ["*"],
        "organization_id": "org-001",
        "is_active": True,
        "created_at": datetime.utcnow(),
    },
    "user": {
        "id": "user-standard-001",
        "username": "user",
        "email": "user@example.com",
        "password_hash": hashlib.sha256("user123".encode()).hexdigest(),
        "roles": ["user"],
        "permissions": ["documents:read", "documents:create", "search:execute"],
        "organization_id": "org-001",
        "is_active": True,
        "created_at": datetime.utcnow(),
    },
}

# Refresh token storage
_refresh_tokens: dict[str, dict[str, Any]] = {}

# Session storage
_sessions: dict[str, dict[str, Any]] = {}

# Initialize handlers
_jwt_handler = JWTHandler(JWTConfig())
_api_key_handler = APIKeyHandler(APIKeyConfig())


# =============================================================================
# Authentication Endpoints
# =============================================================================

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User login",
    description="Authenticate user and get access tokens.",
    responses={401: {"model": ErrorResponse}},
)
async def login(
    request: Request,
    credentials: TokenRequest,
) -> TokenResponse:
    """Authenticate user and return tokens."""
    # Find user
    user = _users.get(credentials.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Verify password
    password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()
    if password_hash != user["password_hash"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Check if active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is disabled",
        )

    # Create session
    session_id = str(uuid4())
    _sessions[session_id] = {
        "user_id": user["id"],
        "created_at": datetime.utcnow(),
        "last_activity": datetime.utcnow(),
        "ip_address": request.client.host if request.client else None,
    }

    # Generate tokens
    access_token = _jwt_handler.create_access_token(
        user_id=user["id"],
        username=user["username"],
        email=user.get("email"),
        roles=user.get("roles", []),
        permissions=user.get("permissions", []),
        organization_id=user.get("organization_id"),
        session_id=session_id,
    )

    refresh_token = _jwt_handler.create_refresh_token(
        user_id=user["id"],
        session_id=session_id,
    )

    # Store refresh token
    _refresh_tokens[refresh_token] = {
        "user_id": user["id"],
        "session_id": session_id,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(days=7),
    }

    # Update last login
    user["last_login_at"] = datetime.utcnow()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=_jwt_handler.config.access_token_expire_minutes * 60,
        scope="read write",
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="Get new access token using refresh token.",
    responses={401: {"model": ErrorResponse}},
)
async def refresh_token(
    request: Request,
    refresh_request: RefreshTokenRequest,
) -> TokenResponse:
    """Refresh access token."""
    # Validate refresh token
    token_data = _refresh_tokens.get(refresh_request.refresh_token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    # Check expiration
    if datetime.utcnow() > token_data["expires_at"]:
        del _refresh_tokens[refresh_request.refresh_token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired",
        )

    # Get user
    user = None
    for u in _users.values():
        if u["id"] == token_data["user_id"]:
            user = u
            break

    if not user or not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )

    # Update session
    session_id = token_data["session_id"]
    if session_id in _sessions:
        _sessions[session_id]["last_activity"] = datetime.utcnow()

    # Generate new access token (keep same session)
    access_token = _jwt_handler.create_access_token(
        user_id=user["id"],
        username=user["username"],
        email=user.get("email"),
        roles=user.get("roles", []),
        permissions=user.get("permissions", []),
        organization_id=user.get("organization_id"),
        session_id=session_id,
    )

    # Optionally rotate refresh token
    new_refresh_token = _jwt_handler.create_refresh_token(
        user_id=user["id"],
        session_id=session_id,
    )

    # Remove old, add new
    del _refresh_tokens[refresh_request.refresh_token]
    _refresh_tokens[new_refresh_token] = {
        "user_id": user["id"],
        "session_id": session_id,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(days=7),
    }

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=_jwt_handler.config.access_token_expire_minutes * 60,
        scope="read write",
    )


@router.post(
    "/logout",
    response_model=SuccessResponse,
    summary="User logout",
    description="Invalidate current session and tokens.",
)
async def logout(
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> SuccessResponse:
    """Logout user and invalidate session."""
    # Remove session
    if user.session_id and user.session_id in _sessions:
        del _sessions[user.session_id]

    # Remove associated refresh tokens
    tokens_to_remove = [
        token for token, data in _refresh_tokens.items()
        if data.get("session_id") == user.session_id
    ]
    for token in tokens_to_remove:
        del _refresh_tokens[token]

    return SuccessResponse(
        success=True,
        message="Successfully logged out",
    )


@router.get(
    "/me",
    summary="Get current user",
    description="Get information about the authenticated user.",
)
async def get_me(
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> dict[str, Any]:
    """Get current user information."""
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "roles": user.roles,
        "permissions": user.permissions,
        "organization_id": user.organization_id,
        "auth_method": user.auth_method,
    }


@router.get(
    "/sessions",
    summary="List active sessions",
    description="List all active sessions for current user.",
)
async def list_sessions(
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> dict[str, Any]:
    """List active sessions."""
    user_sessions = [
        {
            "id": session_id,
            "created_at": data["created_at"],
            "last_activity": data["last_activity"],
            "ip_address": data.get("ip_address"),
            "current": session_id == user.session_id,
        }
        for session_id, data in _sessions.items()
        if data["user_id"] == user.id
    ]

    return {"sessions": user_sessions}


@router.delete(
    "/sessions/{session_id}",
    response_model=SuccessResponse,
    summary="Revoke session",
    description="Revoke a specific session.",
)
async def revoke_session(
    session_id: str,
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> SuccessResponse:
    """Revoke a specific session."""
    session = _sessions.get(session_id)
    if not session or session["user_id"] != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Remove session
    del _sessions[session_id]

    # Remove associated refresh tokens
    tokens_to_remove = [
        token for token, data in _refresh_tokens.items()
        if data.get("session_id") == session_id
    ]
    for token in tokens_to_remove:
        del _refresh_tokens[token]

    return SuccessResponse(
        success=True,
        message="Session revoked",
    )


# =============================================================================
# API Key Management
# =============================================================================

@router.get(
    "/api-keys",
    summary="List API keys",
    description="List all API keys for current user.",
)
async def list_api_keys(
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> dict[str, Any]:
    """List user's API keys."""
    keys = _api_key_handler.list_keys(user.id)

    return {
        "keys": [
            APIKeyResponse(
                id=k.id,
                name=k.name,
                key_prefix=f"ldk_{k.key_hash[:8]}...",
                created_at=k.created_at,
                expires_at=k.expires_at,
                last_used_at=k.last_used_at,
                scopes=k.scopes,
                is_active=k.is_active,
            ).model_dump()
            for k in keys
        ]
    }


@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
    description="Create a new API key.",
)
async def create_api_key(
    key_request: APIKeyCreate,
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> APIKeyResponse:
    """Create new API key."""
    # Limit number of keys per user
    existing_keys = _api_key_handler.list_keys(user.id)
    if len(existing_keys) >= 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum number of API keys reached (10)",
        )

    # Create key
    raw_key, api_key = _api_key_handler.create_key(
        name=key_request.name,
        user_id=user.id,
        organization_id=user.organization_id,
        scopes=key_request.scopes,
        expires_at=key_request.expires_at,
    )

    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key=raw_key,  # Only shown on creation
        key_prefix=raw_key[:12] + "...",
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        scopes=api_key.scopes,
        is_active=api_key.is_active,
    )


@router.delete(
    "/api-keys/{key_id}",
    response_model=SuccessResponse,
    summary="Revoke API key",
    description="Revoke an API key by ID.",
)
async def revoke_api_key(
    key_id: str,
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> SuccessResponse:
    """Revoke API key."""
    # Verify ownership
    api_key = _api_key_handler.get_key_by_id(key_id)
    if not api_key or api_key.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    # Revoke
    _api_key_handler.revoke_key(key_id)

    return SuccessResponse(
        success=True,
        message="API key revoked",
    )


# =============================================================================
# Password Management
# =============================================================================

@router.post(
    "/change-password",
    response_model=SuccessResponse,
    summary="Change password",
    description="Change current user's password.",
)
async def change_password(
    current_password: str,
    new_password: str,
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> SuccessResponse:
    """Change user password."""
    # Find user record
    user_record = None
    for u in _users.values():
        if u["id"] == user.id:
            user_record = u
            break

    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Verify current password
    current_hash = hashlib.sha256(current_password.encode()).hexdigest()
    if current_hash != user_record["password_hash"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Validate new password
    if len(new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 8 characters",
        )

    # Update password
    user_record["password_hash"] = hashlib.sha256(new_password.encode()).hexdigest()

    # Invalidate all other sessions
    sessions_to_remove = [
        sid for sid, data in _sessions.items()
        if data["user_id"] == user.id and sid != user.session_id
    ]
    for sid in sessions_to_remove:
        del _sessions[sid]

    return SuccessResponse(
        success=True,
        message="Password changed successfully. Other sessions have been logged out.",
    )

"""
User API Routes
===============
User management endpoints for administrators.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..models import (
    ErrorResponse,
    PaginatedResponse,
    PasswordChange,
    SuccessResponse,
    UserCreate,
    UserResponse,
    UserUpdate,
)
from ..middleware.auth import (
    AuthenticatedUser,
    get_authenticated_user,
    require_permission,
    require_role,
)


router = APIRouter(prefix="/users", tags=["Users"])


# =============================================================================
# Storage (Demo)
# =============================================================================

# Shared with auth module (in production, use database)
_users: dict[str, dict[str, Any]] = {
    "admin": {
        "id": "user-admin-001",
        "username": "admin",
        "email": "admin@example.com",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "first_name": "Admin",
        "last_name": "User",
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
        "first_name": "Standard",
        "last_name": "User",
        "roles": ["user"],
        "permissions": ["documents:read", "documents:create", "search:execute"],
        "organization_id": "org-001",
        "is_active": True,
        "created_at": datetime.utcnow(),
    },
}


def _user_to_response(user: dict[str, Any]) -> UserResponse:
    """Convert user dict to response model."""
    return UserResponse(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        first_name=user.get("first_name"),
        last_name=user.get("last_name"),
        organization_id=user.get("organization_id"),
        roles=user.get("roles", []),
        is_active=user.get("is_active", True),
        created_at=user["created_at"],
        last_login_at=user.get("last_login_at"),
    )


# =============================================================================
# User CRUD (Admin)
# =============================================================================

@router.get(
    "",
    response_model=PaginatedResponse[UserResponse],
    summary="List users",
    description="List all users (admin only).",
)
async def list_users(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    role: Optional[str] = Query(None, description="Filter by role"),
    search: Optional[str] = Query(None, description="Search by username or email"),
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> PaginatedResponse[UserResponse]:
    """List users with pagination and filtering."""
    users = list(_users.values())

    # Filter by organization (multi-tenancy)
    if admin.organization_id:
        users = [u for u in users if u.get("organization_id") == admin.organization_id]

    # Apply filters
    if is_active is not None:
        users = [u for u in users if u.get("is_active") == is_active]
    if role:
        users = [u for u in users if role in u.get("roles", [])]
    if search:
        search_lower = search.lower()
        users = [
            u for u in users
            if search_lower in u["username"].lower() or
               search_lower in u.get("email", "").lower()
        ]

    # Sort by created_at descending
    users.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)

    # Paginate
    total = len(users)
    start = (page - 1) * size
    end = start + size
    page_users = users[start:end]

    items = [_user_to_response(u) for u in page_users]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user",
    description="Get user by ID (admin only).",
    responses={404: {"model": ErrorResponse}},
)
async def get_user(
    user_id: str,
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> UserResponse:
    """Get user by ID."""
    # Find user by ID
    user = None
    for u in _users.values():
        if u["id"] == user_id:
            user = u
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Check organization access
    if admin.organization_id and user.get("organization_id") != admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    return _user_to_response(user)


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create user",
    description="Create a new user (admin only).",
    responses={400: {"model": ErrorResponse}},
)
async def create_user(
    user_data: UserCreate,
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> UserResponse:
    """Create new user."""
    # Check if username exists
    if user_data.username in _users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{user_data.username}' already exists",
        )

    # Check if email exists
    for u in _users.values():
        if u.get("email") == user_data.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Email '{user_data.email}' already registered",
            )

    # Create user
    user_id = str(uuid4())
    now = datetime.utcnow()

    user = {
        "id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "password_hash": hashlib.sha256(user_data.password.encode()).hexdigest(),
        "first_name": user_data.first_name,
        "last_name": user_data.last_name,
        "roles": ["user"],
        "permissions": ["documents:read", "documents:create", "search:execute"],
        "organization_id": user_data.organization_id or admin.organization_id,
        "is_active": True,
        "created_at": now,
    }

    _users[user_data.username] = user

    return _user_to_response(user)


@router.patch(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user",
    description="Update user information (admin only).",
    responses={404: {"model": ErrorResponse}},
)
async def update_user(
    user_id: str,
    update: UserUpdate,
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> UserResponse:
    """Update user."""
    # Find user
    user = None
    username_key = None
    for uname, u in _users.items():
        if u["id"] == user_id:
            user = u
            username_key = uname
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Check organization access
    if admin.organization_id and user.get("organization_id") != admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Update fields
    if update.email is not None:
        # Check email uniqueness
        for u in _users.values():
            if u.get("email") == update.email and u["id"] != user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Email '{update.email}' already registered",
                )
        user["email"] = update.email

    if update.first_name is not None:
        user["first_name"] = update.first_name
    if update.last_name is not None:
        user["last_name"] = update.last_name
    if update.is_active is not None:
        user["is_active"] = update.is_active

    return _user_to_response(user)


@router.delete(
    "/{user_id}",
    response_model=SuccessResponse,
    summary="Delete user",
    description="Delete a user (admin only).",
    responses={404: {"model": ErrorResponse}},
)
async def delete_user(
    user_id: str,
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> SuccessResponse:
    """Delete user."""
    # Find user
    user = None
    username_key = None
    for uname, u in _users.items():
        if u["id"] == user_id:
            user = u
            username_key = uname
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Check organization access
    if admin.organization_id and user.get("organization_id") != admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Prevent self-deletion
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    del _users[username_key]

    return SuccessResponse(
        success=True,
        message=f"User {user_id} deleted",
    )


# =============================================================================
# Role Management
# =============================================================================

@router.post(
    "/{user_id}/roles",
    response_model=UserResponse,
    summary="Add role to user",
    description="Add a role to a user (admin only).",
)
async def add_user_role(
    user_id: str,
    role: str,
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> UserResponse:
    """Add role to user."""
    # Find user
    user = None
    for u in _users.values():
        if u["id"] == user_id:
            user = u
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Check organization access
    if admin.organization_id and user.get("organization_id") != admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Add role
    if role not in user.get("roles", []):
        user["roles"] = user.get("roles", []) + [role]

    return _user_to_response(user)


@router.delete(
    "/{user_id}/roles/{role}",
    response_model=UserResponse,
    summary="Remove role from user",
    description="Remove a role from a user (admin only).",
)
async def remove_user_role(
    user_id: str,
    role: str,
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> UserResponse:
    """Remove role from user."""
    # Find user
    user = None
    for u in _users.values():
        if u["id"] == user_id:
            user = u
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Check organization access
    if admin.organization_id and user.get("organization_id") != admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Remove role
    if role in user.get("roles", []):
        user["roles"] = [r for r in user["roles"] if r != role]

    return _user_to_response(user)


# =============================================================================
# Permission Management
# =============================================================================

@router.get(
    "/{user_id}/permissions",
    summary="Get user permissions",
    description="Get all permissions for a user (admin only).",
)
async def get_user_permissions(
    user_id: str,
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> dict[str, Any]:
    """Get user permissions."""
    # Find user
    user = None
    for u in _users.values():
        if u["id"] == user_id:
            user = u
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    return {
        "user_id": user_id,
        "permissions": user.get("permissions", []),
        "roles": user.get("roles", []),
    }


@router.put(
    "/{user_id}/permissions",
    response_model=SuccessResponse,
    summary="Set user permissions",
    description="Set permissions for a user (admin only).",
)
async def set_user_permissions(
    user_id: str,
    permissions: list[str],
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> SuccessResponse:
    """Set user permissions."""
    # Find user
    user = None
    for u in _users.values():
        if u["id"] == user_id:
            user = u
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Check organization access
    if admin.organization_id and user.get("organization_id") != admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Update permissions
    user["permissions"] = permissions

    return SuccessResponse(
        success=True,
        message=f"Permissions updated for user {user_id}",
    )


# =============================================================================
# User Self-Service
# =============================================================================

@router.get(
    "/me/profile",
    response_model=UserResponse,
    summary="Get my profile",
    description="Get current user's profile.",
)
async def get_my_profile(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
) -> UserResponse:
    """Get current user's profile."""
    # Find user
    user = None
    for u in _users.values():
        if u["id"] == current_user.id:
            user = u
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return _user_to_response(user)


@router.patch(
    "/me/profile",
    response_model=UserResponse,
    summary="Update my profile",
    description="Update current user's profile.",
)
async def update_my_profile(
    update: UserUpdate,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
) -> UserResponse:
    """Update current user's profile."""
    # Find user
    user = None
    for u in _users.values():
        if u["id"] == current_user.id:
            user = u
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Update allowed fields (not roles or is_active)
    if update.email is not None:
        # Check email uniqueness
        for u in _users.values():
            if u.get("email") == update.email and u["id"] != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Email '{update.email}' already registered",
                )
        user["email"] = update.email

    if update.first_name is not None:
        user["first_name"] = update.first_name
    if update.last_name is not None:
        user["last_name"] = update.last_name

    return _user_to_response(user)

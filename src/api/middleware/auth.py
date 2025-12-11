"""
Authentication Middleware
=========================
JWT and API key authentication with Security layer integration.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional, TYPE_CHECKING
from uuid import uuid4
import hashlib
import hmac
import base64
import json

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from starlette.types import ASGIApp

from ..config import JWTConfig, APIKeyConfig


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TokenPayload:
    """JWT token payload."""
    sub: str  # Subject (user ID)
    username: str
    email: Optional[str] = None
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    organization_id: Optional[str] = None
    session_id: Optional[str] = None
    iat: Optional[datetime] = None
    exp: Optional[datetime] = None
    iss: Optional[str] = None
    aud: Optional[str] = None


@dataclass
class AuthenticatedUser:
    """Authenticated user context."""
    id: str
    username: str
    email: Optional[str] = None
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    organization_id: Optional[str] = None
    session_id: Optional[str] = None
    auth_method: str = "jwt"  # jwt, api_key
    api_key_id: Optional[str] = None
    token_exp: Optional[datetime] = None


@dataclass
class APIKey:
    """API key record."""
    id: str
    key_hash: str
    name: str
    user_id: str
    organization_id: Optional[str] = None
    scopes: list[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    rate_limit_tier: str = "professional"


# =============================================================================
# Request Context
# =============================================================================

# Thread-local-like storage for request context
_request_context: dict[str, Any] = {}


def set_current_user(user: Optional[AuthenticatedUser]) -> None:
    """Set current user in request context."""
    _request_context["current_user"] = user


def get_current_user() -> Optional[AuthenticatedUser]:
    """Get current authenticated user from request context."""
    return _request_context.get("current_user")


def clear_request_context() -> None:
    """Clear request context."""
    _request_context.clear()


# =============================================================================
# JWT Utilities
# =============================================================================

class JWTHandler:
    """JWT token handler."""

    def __init__(self, config: JWTConfig):
        self.config = config
        self._secret = config.secret_key.encode()

    def _base64url_encode(self, data: bytes) -> str:
        """Base64url encode without padding."""
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    def _base64url_decode(self, data: str) -> bytes:
        """Base64url decode with padding."""
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    def _sign(self, message: str) -> str:
        """Create HMAC signature."""
        signature = hmac.new(
            self._secret,
            message.encode(),
            hashlib.sha256
        ).digest()
        return self._base64url_encode(signature)

    def create_access_token(
        self,
        user_id: str,
        username: str,
        email: Optional[str] = None,
        roles: Optional[list[str]] = None,
        permissions: Optional[list[str]] = None,
        organization_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Create JWT access token."""
        now = datetime.utcnow()
        exp = now + timedelta(minutes=self.config.access_token_expire_minutes)

        payload = {
            "sub": user_id,
            "username": username,
            "email": email,
            "roles": roles or [],
            "permissions": permissions or [],
            "organization_id": organization_id,
            "session_id": session_id or str(uuid4()),
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "type": "access",
        }

        return self._create_token(payload)

    def create_refresh_token(
        self,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Create JWT refresh token."""
        now = datetime.utcnow()
        exp = now + timedelta(days=self.config.refresh_token_expire_days)

        payload = {
            "sub": user_id,
            "session_id": session_id or str(uuid4()),
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "type": "refresh",
        }

        return self._create_token(payload)

    def _create_token(self, payload: dict[str, Any]) -> str:
        """Create JWT token from payload."""
        header = {"alg": "HS256", "typ": "JWT"}

        header_b64 = self._base64url_encode(json.dumps(header).encode())
        payload_b64 = self._base64url_encode(json.dumps(payload).encode())

        message = f"{header_b64}.{payload_b64}"
        signature = self._sign(message)

        return f"{message}.{signature}"

    def decode_token(self, token: str) -> Optional[TokenPayload]:
        """Decode and validate JWT token."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature = parts

            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            expected_signature = self._sign(message)
            if not hmac.compare_digest(signature, expected_signature):
                return None

            # Decode payload
            payload_json = self._base64url_decode(payload_b64)
            payload = json.loads(payload_json)

            # Check expiration
            if "exp" in payload:
                exp = datetime.fromtimestamp(payload["exp"])
                if datetime.utcnow() > exp:
                    return None

            # Check issuer and audience
            if payload.get("iss") != self.config.issuer:
                return None
            if payload.get("aud") != self.config.audience:
                return None

            return TokenPayload(
                sub=payload.get("sub", ""),
                username=payload.get("username", ""),
                email=payload.get("email"),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                organization_id=payload.get("organization_id"),
                session_id=payload.get("session_id"),
                iat=datetime.fromtimestamp(payload["iat"]) if "iat" in payload else None,
                exp=datetime.fromtimestamp(payload["exp"]) if "exp" in payload else None,
                iss=payload.get("iss"),
                aud=payload.get("aud"),
            )

        except Exception:
            return None

    def is_refresh_token(self, token: str) -> bool:
        """Check if token is a refresh token."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return False
            payload_json = self._base64url_decode(parts[1])
            payload = json.loads(payload_json)
            return payload.get("type") == "refresh"
        except Exception:
            return False


# =============================================================================
# API Key Utilities
# =============================================================================

class APIKeyHandler:
    """API key handler."""

    def __init__(self, config: APIKeyConfig):
        self.config = config
        self._keys: dict[str, APIKey] = {}  # In-memory store for demo

    def generate_key(self) -> tuple[str, str]:
        """Generate new API key. Returns (key, hash)."""
        import secrets
        raw_key = secrets.token_urlsafe(self.config.key_length)
        full_key = f"{self.config.prefix}{raw_key}"
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        return full_key, key_hash

    def hash_key(self, key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def create_key(
        self,
        name: str,
        user_id: str,
        organization_id: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        expires_at: Optional[datetime] = None,
        rate_limit_tier: str = "professional",
    ) -> tuple[str, APIKey]:
        """Create new API key. Returns (raw_key, APIKey)."""
        raw_key, key_hash = self.generate_key()

        api_key = APIKey(
            id=str(uuid4()),
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            organization_id=organization_id,
            scopes=scopes or ["read"],
            expires_at=expires_at,
            rate_limit_tier=rate_limit_tier,
        )

        self._keys[key_hash] = api_key
        return raw_key, api_key

    def validate_key(self, key: str) -> Optional[APIKey]:
        """Validate API key and return record if valid."""
        if not key.startswith(self.config.prefix):
            return None

        key_hash = self.hash_key(key)
        api_key = self._keys.get(key_hash)

        if not api_key:
            return None

        if not api_key.is_active:
            return None

        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            return None

        # Update last used
        api_key.last_used_at = datetime.utcnow()

        return api_key

    def revoke_key(self, key_id: str) -> bool:
        """Revoke API key by ID."""
        for api_key in self._keys.values():
            if api_key.id == key_id:
                api_key.is_active = False
                return True
        return False

    def get_key_by_id(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        for api_key in self._keys.values():
            if api_key.id == key_id:
                return api_key
        return None

    def list_keys(self, user_id: str) -> list[APIKey]:
        """List API keys for user."""
        return [k for k in self._keys.values() if k.user_id == user_id]


# =============================================================================
# Authentication Middleware
# =============================================================================

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware supporting JWT and API key auth.

    Integrates with the Security layer for user validation and
    permission checking.
    """

    # Paths that don't require authentication
    PUBLIC_PATHS = {
        "/",
        "/health",
        "/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/refresh",
        "/api/v2/auth/login",
        "/api/v2/auth/register",
        "/api/v2/auth/refresh",
    }

    def __init__(
        self,
        app: "ASGIApp",
        jwt_config: Optional[JWTConfig] = None,
        api_key_config: Optional[APIKeyConfig] = None,
        security_service: Optional[Any] = None,
    ):
        super().__init__(app)
        self.jwt_handler = JWTHandler(jwt_config or JWTConfig())
        self.api_key_handler = APIKeyHandler(api_key_config or APIKeyConfig())
        self.security_service = security_service

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        """Process authentication for each request."""
        # Clear previous context
        clear_request_context()

        # Check if path requires auth
        path = request.url.path
        if self._is_public_path(path):
            return await call_next(request)

        # Try JWT authentication first
        user = await self._authenticate_jwt(request)

        # Fall back to API key
        if not user:
            user = await self._authenticate_api_key(request)

        # No valid auth
        if not user:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Invalid or missing authentication credentials",
                    "request_id": str(uuid4()),
                },
            )

        # Set user in context
        set_current_user(user)

        # Add user info to request state
        request.state.user = user

        # Call next middleware/handler
        response = await call_next(request)

        return response

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no auth required)."""
        # Exact match
        if path in self.PUBLIC_PATHS:
            return True

        # Prefix match for docs
        if path.startswith("/docs") or path.startswith("/redoc"):
            return True

        return False

    async def _authenticate_jwt(self, request: Request) -> Optional[AuthenticatedUser]:
        """Authenticate via JWT token."""
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None

        token = parts[1]
        payload = self.jwt_handler.decode_token(token)

        if not payload:
            return None

        # Optionally validate with security service
        if self.security_service:
            try:
                # Integration with security layer
                is_valid = await self._validate_with_security_service(
                    payload.sub, payload.session_id
                )
                if not is_valid:
                    return None
            except Exception:
                pass  # Fall through if security service unavailable

        return AuthenticatedUser(
            id=payload.sub,
            username=payload.username,
            email=payload.email,
            roles=payload.roles,
            permissions=payload.permissions,
            organization_id=payload.organization_id,
            session_id=payload.session_id,
            auth_method="jwt",
            token_exp=payload.exp,
        )

    async def _authenticate_api_key(self, request: Request) -> Optional[AuthenticatedUser]:
        """Authenticate via API key."""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            # Also check query param for certain endpoints
            api_key = request.query_params.get("api_key")

        if not api_key:
            return None

        key_record = self.api_key_handler.validate_key(api_key)
        if not key_record:
            return None

        # Convert scopes to permissions
        permissions = self._scopes_to_permissions(key_record.scopes)

        return AuthenticatedUser(
            id=key_record.user_id,
            username=f"api_key:{key_record.name}",
            organization_id=key_record.organization_id,
            permissions=permissions,
            auth_method="api_key",
            api_key_id=key_record.id,
        )

    def _scopes_to_permissions(self, scopes: list[str]) -> list[str]:
        """Convert API key scopes to permission strings."""
        permission_map = {
            "read": ["documents:read", "search:execute"],
            "write": ["documents:create", "documents:update"],
            "delete": ["documents:delete"],
            "admin": ["admin:*"],
            "analytics": ["analytics:read"],
            "export": ["export:create"],
        }

        permissions = []
        for scope in scopes:
            if scope in permission_map:
                permissions.extend(permission_map[scope])
            else:
                permissions.append(scope)

        return list(set(permissions))

    async def _validate_with_security_service(
        self, user_id: str, session_id: Optional[str]
    ) -> bool:
        """Validate token with security service."""
        if not self.security_service:
            return True

        # Check if session is valid (not revoked)
        try:
            if hasattr(self.security_service, "validate_session"):
                return await self.security_service.validate_session(
                    user_id, session_id
                )
        except Exception:
            pass

        return True


# =============================================================================
# Permission Decorators
# =============================================================================

def require_permissions(*required_permissions: str):
    """
    Decorator to require specific permissions for an endpoint.

    Usage:
        @require_permissions("documents:read", "documents:write")
        async def my_endpoint(request: Request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from context
            user = get_current_user()

            if not user:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "unauthorized",
                        "message": "Authentication required",
                    },
                )

            # Check permissions
            if not _has_permissions(user, required_permissions):
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "forbidden",
                        "message": f"Missing required permissions: {', '.join(required_permissions)}",
                    },
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def require_roles(*required_roles: str):
    """
    Decorator to require specific roles for an endpoint.

    Usage:
        @require_roles("admin", "manager")
        async def admin_endpoint(request: Request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = get_current_user()

            if not user:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "unauthorized",
                        "message": "Authentication required",
                    },
                )

            # Check if user has any of the required roles
            if not any(role in user.roles for role in required_roles):
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "forbidden",
                        "message": f"Required roles: {', '.join(required_roles)}",
                    },
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def _has_permissions(user: AuthenticatedUser, required: tuple[str, ...]) -> bool:
    """Check if user has all required permissions."""
    user_perms = set(user.permissions)

    # Check for admin wildcard
    if "admin:*" in user_perms or "*" in user_perms:
        return True

    for perm in required:
        if perm not in user_perms:
            # Check for wildcard match (e.g., "documents:*" matches "documents:read")
            resource = perm.split(":")[0] if ":" in perm else perm
            if f"{resource}:*" not in user_perms:
                return False

    return True


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def get_authenticated_user(request: Request) -> AuthenticatedUser:
    """
    FastAPI dependency to get authenticated user.

    Usage:
        @app.get("/protected")
        async def protected(user: AuthenticatedUser = Depends(get_authenticated_user)):
            return {"user": user.username}
    """
    user = getattr(request.state, "user", None) or get_current_user()

    if not user:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_optional_user(request: Request) -> Optional[AuthenticatedUser]:
    """
    FastAPI dependency to optionally get authenticated user.

    Returns None if not authenticated (doesn't raise).
    """
    return getattr(request.state, "user", None) or get_current_user()


def require_permission(permission: str):
    """
    FastAPI dependency factory for permission checking.

    Usage:
        @app.delete("/documents/{id}")
        async def delete_document(
            id: str,
            user: AuthenticatedUser = Depends(require_permission("documents:delete"))
        ):
            ...
    """
    async def dependency(request: Request) -> AuthenticatedUser:
        user = await get_authenticated_user(request)

        if not _has_permissions(user, (permission,)):
            from fastapi import HTTPException
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {permission} required",
            )

        return user

    return dependency


def require_role(role: str):
    """
    FastAPI dependency factory for role checking.

    Usage:
        @app.get("/admin/users")
        async def list_users(
            user: AuthenticatedUser = Depends(require_role("admin"))
        ):
            ...
    """
    async def dependency(request: Request) -> AuthenticatedUser:
        user = await get_authenticated_user(request)

        if role not in user.roles:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=403,
                detail=f"Role required: {role}",
            )

        return user

    return dependency

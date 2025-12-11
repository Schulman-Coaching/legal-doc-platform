"""
Legal Document Platform - Integration & API Layer
=================================================
Unified API gateway providing REST, GraphQL, and webhook capabilities
with comprehensive authentication, rate limiting, and API management.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional
from uuid import uuid4

import jwt
from fastapi import FastAPI, HTTPException, Request, Response, Depends, Header, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration & Models
# ============================================================================

class APIVersion(str, Enum):
    """Supported API versions."""
    V1 = "v1"
    V2 = "v2"


class AuthType(str, Enum):
    """Authentication types."""
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    MTLS = "mtls"


class RateLimitTier(str, Enum):
    """Rate limit tiers."""
    FREE = "free"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"


class WebhookEventType(str, Enum):
    """Webhook event types."""
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_ANALYZED = "document.analyzed"
    DOCUMENT_DELETED = "document.deleted"
    ANALYSIS_COMPLETED = "analysis.completed"
    ALERT_TRIGGERED = "alert.triggered"
    EXPORT_READY = "export.ready"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_size: int
    documents_per_day: int
    storage_gb: int


RATE_LIMITS = {
    RateLimitTier.FREE: RateLimitConfig(60, 1000, 5000, 10, 100, 1),
    RateLimitTier.PROFESSIONAL: RateLimitConfig(300, 10000, 50000, 50, 1000, 50),
    RateLimitTier.ENTERPRISE: RateLimitConfig(1000, 50000, 200000, 100, 10000, 500),
    RateLimitTier.UNLIMITED: RateLimitConfig(10000, 500000, 2000000, 500, 100000, 10000),
}


class User(BaseModel):
    """Authenticated user model."""
    id: str
    email: str
    organization_id: str
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    tier: RateLimitTier = RateLimitTier.FREE
    metadata: dict[str, Any] = Field(default_factory=dict)


class APIKey(BaseModel):
    """API key model."""
    id: str
    key_hash: str
    name: str
    user_id: str
    organization_id: str
    permissions: list[str] = Field(default_factory=list)
    rate_limit_tier: RateLimitTier = RateLimitTier.FREE
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    is_active: bool = True


class WebhookSubscription(BaseModel):
    """Webhook subscription model."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    organization_id: str
    url: str
    events: list[WebhookEventType]
    secret: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebhookDelivery(BaseModel):
    """Webhook delivery record."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    subscription_id: str
    event_type: WebhookEventType
    payload: dict[str, Any]
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 5
    next_retry_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Authentication & Authorization
# ============================================================================

class JWTConfig(BaseModel):
    """JWT configuration."""
    secret_key: str = "your-secret-key-here"  # Use env var in production
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    issuer: str = "legal-doc-platform"
    audience: str = "legal-doc-api"


class AuthService:
    """Authentication and authorization service."""

    def __init__(self, jwt_config: JWTConfig):
        self.jwt_config = jwt_config
        self._api_keys: dict[str, APIKey] = {}  # In production, use database
        self._users: dict[str, User] = {}  # In production, integrate with Keycloak

    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create JWT access token."""
        expire = datetime.utcnow() + (
            expires_delta or
            timedelta(minutes=self.jwt_config.access_token_expire_minutes)
        )

        payload = {
            "sub": user.id,
            "email": user.email,
            "org": user.organization_id,
            "roles": user.roles,
            "permissions": user.permissions,
            "tier": user.tier.value,
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": self.jwt_config.issuer,
            "aud": self.jwt_config.audience,
        }

        return jwt.encode(
            payload,
            self.jwt_config.secret_key,
            algorithm=self.jwt_config.algorithm
        )

    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        expire = datetime.utcnow() + timedelta(
            days=self.jwt_config.refresh_token_expire_days
        )

        payload = {
            "sub": user.id,
            "type": "refresh",
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": self.jwt_config.issuer,
        }

        return jwt.encode(
            payload,
            self.jwt_config.secret_key,
            algorithm=self.jwt_config.algorithm
        )

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.jwt_config.secret_key,
                algorithms=[self.jwt_config.algorithm],
                audience=self.jwt_config.audience,
                issuer=self.jwt_config.issuer,
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    def create_api_key(
        self,
        user_id: str,
        organization_id: str,
        name: str,
        permissions: list[str],
        tier: RateLimitTier = RateLimitTier.FREE,
        expires_in_days: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """Create a new API key."""
        # Generate secure random key
        raw_key = f"ldk_{uuid4().hex}{uuid4().hex}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        api_key = APIKey(
            id=str(uuid4()),
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            organization_id=organization_id,
            permissions=permissions,
            rate_limit_tier=tier,
            expires_at=(
                datetime.utcnow() + timedelta(days=expires_in_days)
                if expires_in_days else None
            ),
        )

        self._api_keys[key_hash] = api_key

        # Return raw key only once - it won't be stored
        return raw_key, api_key

    def verify_api_key(self, raw_key: str) -> Optional[APIKey]:
        """Verify an API key."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = self._api_keys.get(key_hash)

        if not api_key:
            return None

        if not api_key.is_active:
            return None

        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None

        # Update last used
        api_key.last_used_at = datetime.utcnow()

        return api_key

    def check_permission(
        self,
        user: User,
        required_permission: str,
    ) -> bool:
        """Check if user has required permission."""
        # Admin has all permissions
        if "admin" in user.roles:
            return True

        # Check direct permission
        if required_permission in user.permissions:
            return True

        # Check role-based permissions
        role_permissions = {
            "viewer": ["documents:read", "search:read"],
            "editor": ["documents:read", "documents:write", "search:read"],
            "analyst": ["documents:read", "analysis:read", "analysis:write", "search:read"],
            "manager": ["documents:*", "analysis:*", "search:*", "users:read"],
        }

        for role in user.roles:
            if role in role_permissions:
                perms = role_permissions[role]
                for perm in perms:
                    if perm == required_permission or perm.endswith(":*"):
                        resource = perm.split(":")[0]
                        if required_permission.startswith(resource):
                            return True

        return False


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self):
        self._buckets: dict[str, dict] = {}

    async def check_rate_limit(
        self,
        key: str,
        tier: RateLimitTier,
    ) -> tuple[bool, dict[str, int]]:
        """
        Check if request is within rate limits.

        Returns:
            Tuple of (is_allowed, headers_dict)
        """
        config = RATE_LIMITS[tier]
        now = time.time()

        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": config.burst_size,
                "last_update": now,
                "minute_count": 0,
                "minute_reset": now + 60,
            }

        bucket = self._buckets[key]

        # Refill tokens
        elapsed = now - bucket["last_update"]
        tokens_to_add = elapsed * (config.requests_per_minute / 60)
        bucket["tokens"] = min(config.burst_size, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now

        # Reset minute counter if needed
        if now > bucket["minute_reset"]:
            bucket["minute_count"] = 0
            bucket["minute_reset"] = now + 60

        # Check limits
        if bucket["tokens"] < 1 or bucket["minute_count"] >= config.requests_per_minute:
            return False, {
                "X-RateLimit-Limit": str(config.requests_per_minute),
                "X-RateLimit-Remaining": str(max(0, config.requests_per_minute - bucket["minute_count"])),
                "X-RateLimit-Reset": str(int(bucket["minute_reset"])),
                "Retry-After": str(int(bucket["minute_reset"] - now)),
            }

        # Consume token
        bucket["tokens"] -= 1
        bucket["minute_count"] += 1

        return True, {
            "X-RateLimit-Limit": str(config.requests_per_minute),
            "X-RateLimit-Remaining": str(config.requests_per_minute - bucket["minute_count"]),
            "X-RateLimit-Reset": str(int(bucket["minute_reset"])),
        }


# ============================================================================
# Webhook Service
# ============================================================================

class WebhookService:
    """Service for managing webhooks."""

    def __init__(self):
        self._subscriptions: dict[str, WebhookSubscription] = {}
        self._deliveries: list[WebhookDelivery] = []

    def create_subscription(
        self,
        user_id: str,
        organization_id: str,
        url: str,
        events: list[WebhookEventType],
    ) -> WebhookSubscription:
        """Create a new webhook subscription."""
        # Generate secret for signing
        secret = f"whsec_{uuid4().hex}"

        subscription = WebhookSubscription(
            user_id=user_id,
            organization_id=organization_id,
            url=url,
            events=events,
            secret=secret,
        )

        self._subscriptions[subscription.id] = subscription
        return subscription

    def get_subscriptions(
        self,
        organization_id: str,
    ) -> list[WebhookSubscription]:
        """Get all subscriptions for an organization."""
        return [
            sub for sub in self._subscriptions.values()
            if sub.organization_id == organization_id
        ]

    def delete_subscription(self, subscription_id: str) -> bool:
        """Delete a webhook subscription."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False

    async def trigger_event(
        self,
        event_type: WebhookEventType,
        organization_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Trigger a webhook event to all relevant subscribers."""
        # Find matching subscriptions
        matching_subs = [
            sub for sub in self._subscriptions.values()
            if sub.organization_id == organization_id
            and event_type in sub.events
            and sub.is_active
        ]

        # Queue deliveries
        for sub in matching_subs:
            delivery = WebhookDelivery(
                subscription_id=sub.id,
                event_type=event_type,
                payload=payload,
            )
            self._deliveries.append(delivery)
            # In production, queue for async delivery
            asyncio.create_task(self._deliver_webhook(delivery, sub))

    async def _deliver_webhook(
        self,
        delivery: WebhookDelivery,
        subscription: WebhookSubscription,
    ) -> None:
        """Deliver a webhook with retry logic."""
        import httpx

        # Build payload
        webhook_payload = {
            "id": delivery.id,
            "event": delivery.event_type.value,
            "created_at": delivery.created_at.isoformat(),
            "data": delivery.payload,
        }

        # Sign payload
        signature = self._sign_payload(
            json.dumps(webhook_payload),
            subscription.secret
        )

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-ID": delivery.id,
            "X-Webhook-Signature": signature,
            "X-Webhook-Timestamp": str(int(time.time())),
        }

        # Attempt delivery with retries
        retry_delays = [0, 60, 300, 1800, 3600]  # Exponential backoff

        for attempt in range(delivery.max_attempts):
            if attempt > 0:
                await asyncio.sleep(retry_delays[min(attempt, len(retry_delays) - 1)])

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        subscription.url,
                        json=webhook_payload,
                        headers=headers,
                    )

                    delivery.status_code = response.status_code
                    delivery.response_body = response.text[:1000]
                    delivery.attempts = attempt + 1

                    if 200 <= response.status_code < 300:
                        delivery.delivered_at = datetime.utcnow()
                        logger.info(f"Webhook {delivery.id} delivered successfully")
                        return

            except Exception as e:
                logger.warning(f"Webhook delivery attempt {attempt + 1} failed: {e}")
                delivery.attempts = attempt + 1

        logger.error(f"Webhook {delivery.id} delivery failed after {delivery.max_attempts} attempts")

    def _sign_payload(self, payload: str, secret: str) -> str:
        """Sign webhook payload."""
        timestamp = str(int(time.time()))
        message = f"{timestamp}.{payload}"
        signature = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"t={timestamp},v1={signature}"

    def verify_signature(
        self,
        payload: str,
        signature: str,
        secret: str,
        tolerance_seconds: int = 300,
    ) -> bool:
        """Verify webhook signature."""
        try:
            parts = dict(p.split("=") for p in signature.split(","))
            timestamp = int(parts["t"])
            expected_sig = parts["v1"]

            # Check timestamp
            if abs(time.time() - timestamp) > tolerance_seconds:
                return False

            # Verify signature
            message = f"{timestamp}.{payload}"
            computed_sig = hmac.new(
                secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(expected_sig, computed_sig)

        except Exception:
            return False


# ============================================================================
# API Request/Response Models
# ============================================================================

class DocumentUploadRequest(BaseModel):
    """Document upload request."""
    filename: str
    content_type: str
    client_id: Optional[str] = None
    matter_id: Optional[str] = None
    classification: str = "internal"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """Document upload response."""
    document_id: str
    status: str
    upload_url: Optional[str] = None
    message: str


class SearchRequest(BaseModel):
    """Search request."""
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    page: int = 1
    page_size: int = 20
    sort_by: str = "relevance"
    sort_order: str = "desc"


class SearchResponse(BaseModel):
    """Search response."""
    total: int
    page: int
    page_size: int
    results: list[dict[str, Any]]
    facets: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    took_ms: int


class AnalysisRequest(BaseModel):
    """Analysis request."""
    document_id: str
    analysis_types: list[str]
    options: dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """Analysis response."""
    document_id: str
    results: dict[str, Any]
    status: str
    processing_time_ms: int


class WebhookCreateRequest(BaseModel):
    """Webhook creation request."""
    url: str
    events: list[str]


class WebhookResponse(BaseModel):
    """Webhook response."""
    id: str
    url: str
    events: list[str]
    secret: str
    created_at: datetime


class APIErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[dict[str, Any]] = None
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Middleware
# ============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all requests."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id

        start_time = time.time()

        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        # Log request
        duration = time.time() - start_time
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"- {response.status_code} - {duration:.3f}s"
        )

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting."""

    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next):
        # Get user identifier for rate limiting
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            # Use IP for unauthenticated requests
            user_id = request.client.host if request.client else "unknown"

        tier = getattr(request.state, "tier", RateLimitTier.FREE)

        # Check rate limit
        allowed, headers = await self.rate_limiter.check_rate_limit(
            f"{user_id}:{request.url.path}",
            tier
        )

        if not allowed:
            return Response(
                content=json.dumps({
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests",
                }),
                status_code=429,
                headers=headers,
                media_type="application/json",
            )

        response = await call_next(request)

        # Add rate limit headers
        for key, value in headers.items():
            response.headers[key] = value

        return response


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Legal Document Platform API",
    description="Comprehensive API for legal document processing and analysis",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
auth_service = AuthService(JWTConfig())
rate_limiter = RateLimiter()
webhook_service = WebhookService()

# Add middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# Security
security = HTTPBearer(auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


# ============================================================================
# Dependencies
# ============================================================================

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None),
) -> User:
    """Get current authenticated user."""
    # Try API key first
    if x_api_key:
        api_key = auth_service.verify_api_key(x_api_key)
        if api_key:
            # Create user from API key
            return User(
                id=api_key.user_id,
                email="api-key@system",
                organization_id=api_key.organization_id,
                roles=[],
                permissions=api_key.permissions,
                tier=api_key.rate_limit_tier,
            )

    # Try JWT token
    if credentials:
        payload = auth_service.verify_token(credentials.credentials)
        return User(
            id=payload["sub"],
            email=payload["email"],
            organization_id=payload["org"],
            roles=payload.get("roles", []),
            permissions=payload.get("permissions", []),
            tier=RateLimitTier(payload.get("tier", "free")),
        )

    raise HTTPException(
        status_code=401,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_permission(permission: str):
    """Dependency for checking permissions."""
    async def check(user: User = Depends(get_current_user)):
        if not auth_service.check_permission(user, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {permission} required"
            )
        return user
    return check


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/api/v1/auth/token", tags=["Authentication"])
async def login(username: str, password: str):
    """
    Authenticate user and return access token.

    In production, validate against Keycloak or identity provider.
    """
    # Placeholder authentication
    # In production, validate credentials against IdP
    user = User(
        id="user-123",
        email=username,
        organization_id="org-123",
        roles=["editor"],
        permissions=["documents:read", "documents:write"],
        tier=RateLimitTier.PROFESSIONAL,
    )

    access_token = auth_service.create_access_token(user)
    refresh_token = auth_service.create_refresh_token(user)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": auth_service.jwt_config.access_token_expire_minutes * 60,
    }


@app.post("/api/v1/auth/refresh", tags=["Authentication"])
async def refresh_token(refresh_token: str):
    """Refresh access token."""
    payload = auth_service.verify_token(refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=400, detail="Invalid refresh token")

    # Get user and generate new tokens
    user = User(
        id=payload["sub"],
        email="user@example.com",  # Fetch from DB
        organization_id="org-123",
        roles=["editor"],
        permissions=["documents:read", "documents:write"],
        tier=RateLimitTier.PROFESSIONAL,
    )

    new_access_token = auth_service.create_access_token(user)

    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "expires_in": auth_service.jwt_config.access_token_expire_minutes * 60,
    }


@app.post("/api/v1/auth/api-keys", tags=["Authentication"])
async def create_api_key(
    name: str,
    permissions: list[str],
    expires_in_days: Optional[int] = None,
    user: User = Depends(require_permission("api_keys:create")),
):
    """Create a new API key."""
    raw_key, api_key = auth_service.create_api_key(
        user_id=user.id,
        organization_id=user.organization_id,
        name=name,
        permissions=permissions,
        tier=user.tier,
        expires_in_days=expires_in_days,
    )

    return {
        "api_key": raw_key,  # Only shown once
        "id": api_key.id,
        "name": api_key.name,
        "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
        "warning": "Save this API key - it will not be shown again",
    }


# ============================================================================
# Document Endpoints
# ============================================================================

@app.post(
    "/api/v1/documents",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
)
async def upload_document(
    request: DocumentUploadRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_permission("documents:write")),
):
    """
    Initiate document upload.

    Returns a presigned URL for uploading the document file.
    """
    document_id = str(uuid4())

    # Generate presigned upload URL
    # In production, generate MinIO presigned URL
    upload_url = f"https://storage.example.com/upload/{document_id}?signed=true"

    # Trigger webhook event
    background_tasks.add_task(
        webhook_service.trigger_event,
        WebhookEventType.DOCUMENT_UPLOADED,
        user.organization_id,
        {"document_id": document_id, "filename": request.filename},
    )

    return DocumentUploadResponse(
        document_id=document_id,
        status="pending_upload",
        upload_url=upload_url,
        message="Upload document to the provided URL",
    )


@app.get("/api/v1/documents/{document_id}", tags=["Documents"])
async def get_document(
    document_id: str,
    user: User = Depends(require_permission("documents:read")),
):
    """Get document metadata."""
    # In production, fetch from storage layer
    return {
        "id": document_id,
        "filename": "example.pdf",
        "status": "processed",
        "document_type": "contract",
        "practice_areas": ["corporate"],
        "created_at": datetime.utcnow().isoformat(),
    }


@app.delete("/api/v1/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_permission("documents:delete")),
):
    """Delete a document."""
    # In production, delete from storage layer

    background_tasks.add_task(
        webhook_service.trigger_event,
        WebhookEventType.DOCUMENT_DELETED,
        user.organization_id,
        {"document_id": document_id},
    )

    return {"message": "Document deleted successfully"}


@app.get("/api/v1/documents/{document_id}/download", tags=["Documents"])
async def download_document(
    document_id: str,
    user: User = Depends(require_permission("documents:read")),
):
    """Get presigned download URL."""
    # In production, generate presigned URL
    download_url = f"https://storage.example.com/download/{document_id}?signed=true"

    return {
        "download_url": download_url,
        "expires_in_seconds": 3600,
    }


# ============================================================================
# Search Endpoints
# ============================================================================

@app.post("/api/v1/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(
    request: SearchRequest,
    user: User = Depends(require_permission("search:read")),
):
    """
    Search documents with full-text and filters.

    Supports boolean queries, phrase matching, and faceted search.
    """
    # In production, route to Elasticsearch
    return SearchResponse(
        total=0,
        page=request.page,
        page_size=request.page_size,
        results=[],
        facets={
            "document_type": [
                {"value": "contract", "count": 100},
                {"value": "brief", "count": 50},
            ],
            "practice_area": [
                {"value": "corporate", "count": 80},
                {"value": "litigation", "count": 70},
            ],
        },
        took_ms=10,
    )


@app.get("/api/v1/search/suggest", tags=["Search"])
async def search_suggestions(
    q: str = Query(..., min_length=2),
    user: User = Depends(require_permission("search:read")),
):
    """Get search suggestions/autocomplete."""
    # In production, use Elasticsearch completion suggester
    return {
        "suggestions": [
            {"text": q + " contract", "score": 0.9},
            {"text": q + " agreement", "score": 0.8},
        ]
    }


# ============================================================================
# Analysis Endpoints
# ============================================================================

@app.post("/api/v1/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_document(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_permission("analysis:write")),
):
    """
    Request document analysis.

    Available analysis types:
    - summarization
    - risk_analysis
    - contract_review
    - entity_extraction
    - similarity
    """
    # Queue analysis job
    # In production, send to processing queue

    background_tasks.add_task(
        webhook_service.trigger_event,
        WebhookEventType.ANALYSIS_COMPLETED,
        user.organization_id,
        {
            "document_id": request.document_id,
            "analysis_types": request.analysis_types,
        },
    )

    return AnalysisResponse(
        document_id=request.document_id,
        results={},
        status="processing",
        processing_time_ms=0,
    )


@app.get("/api/v1/analyze/{document_id}/results", tags=["Analysis"])
async def get_analysis_results(
    document_id: str,
    analysis_type: Optional[str] = None,
    user: User = Depends(require_permission("analysis:read")),
):
    """Get analysis results for a document."""
    # In production, fetch from storage
    return {
        "document_id": document_id,
        "results": {
            "summarization": {
                "summary": "This is a sample summary...",
                "key_points": ["Point 1", "Point 2"],
            },
            "risk_analysis": {
                "overall_risk": "medium",
                "risk_factors": [],
            },
        },
    }


# ============================================================================
# Webhook Endpoints
# ============================================================================

@app.post(
    "/api/v1/webhooks",
    response_model=WebhookResponse,
    tags=["Webhooks"],
)
async def create_webhook(
    request: WebhookCreateRequest,
    user: User = Depends(require_permission("webhooks:write")),
):
    """Create a webhook subscription."""
    events = [WebhookEventType(e) for e in request.events]

    subscription = webhook_service.create_subscription(
        user_id=user.id,
        organization_id=user.organization_id,
        url=request.url,
        events=events,
    )

    return WebhookResponse(
        id=subscription.id,
        url=subscription.url,
        events=[e.value for e in subscription.events],
        secret=subscription.secret,
        created_at=subscription.created_at,
    )


@app.get("/api/v1/webhooks", tags=["Webhooks"])
async def list_webhooks(
    user: User = Depends(require_permission("webhooks:read")),
):
    """List webhook subscriptions."""
    subscriptions = webhook_service.get_subscriptions(user.organization_id)

    return {
        "webhooks": [
            {
                "id": sub.id,
                "url": sub.url,
                "events": [e.value for e in sub.events],
                "is_active": sub.is_active,
                "created_at": sub.created_at.isoformat(),
            }
            for sub in subscriptions
        ]
    }


@app.delete("/api/v1/webhooks/{webhook_id}", tags=["Webhooks"])
async def delete_webhook(
    webhook_id: str,
    user: User = Depends(require_permission("webhooks:write")),
):
    """Delete a webhook subscription."""
    if not webhook_service.delete_subscription(webhook_id):
        raise HTTPException(status_code=404, detail="Webhook not found")

    return {"message": "Webhook deleted successfully"}


# ============================================================================
# Export Endpoints
# ============================================================================

@app.post("/api/v1/export", tags=["Export"])
async def create_export(
    document_ids: list[str],
    format: str = "zip",
    include_metadata: bool = True,
    include_analysis: bool = False,
    background_tasks: BackgroundTasks = None,
    user: User = Depends(require_permission("documents:export")),
):
    """
    Create an export job for multiple documents.

    Supported formats: zip, pdf, csv (metadata only)
    """
    export_id = str(uuid4())

    # Queue export job
    # In production, send to processing queue

    if background_tasks:
        background_tasks.add_task(
            webhook_service.trigger_event,
            WebhookEventType.EXPORT_READY,
            user.organization_id,
            {"export_id": export_id, "document_count": len(document_ids)},
        )

    return {
        "export_id": export_id,
        "status": "processing",
        "document_count": len(document_ids),
        "format": format,
    }


@app.get("/api/v1/export/{export_id}", tags=["Export"])
async def get_export_status(
    export_id: str,
    user: User = Depends(require_permission("documents:export")),
):
    """Get export job status."""
    return {
        "export_id": export_id,
        "status": "completed",
        "download_url": f"https://storage.example.com/exports/{export_id}.zip",
        "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
    }


# ============================================================================
# Admin Endpoints
# ============================================================================

@app.get("/api/v1/admin/stats", tags=["Admin"])
async def get_platform_stats(
    user: User = Depends(require_permission("admin:stats")),
):
    """Get platform statistics."""
    return {
        "total_documents": 10000,
        "total_users": 100,
        "storage_used_gb": 50.5,
        "api_calls_today": 5000,
        "processing_queue_size": 10,
    }


@app.get("/api/v1/admin/usage", tags=["Admin"])
async def get_usage_metrics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user: User = Depends(require_permission("admin:usage")),
):
    """Get API usage metrics."""
    return {
        "period": {
            "start": (start_date or datetime.utcnow() - timedelta(days=30)).isoformat(),
            "end": (end_date or datetime.utcnow()).isoformat(),
        },
        "metrics": {
            "total_requests": 100000,
            "total_documents": 5000,
            "total_analyses": 2000,
            "avg_response_time_ms": 150,
        },
    }


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/api/v1/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/status", tags=["System"])
async def system_status():
    """Detailed system status."""
    return {
        "api": "healthy",
        "database": "healthy",
        "search": "healthy",
        "storage": "healthy",
        "queue": "healthy",
        "cache": "healthy",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

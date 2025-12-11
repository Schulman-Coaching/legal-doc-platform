"""
API Gateway Models
==================
Request/response models for API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Generic, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field, EmailStr


# =============================================================================
# Enums
# =============================================================================

class APIVersion(str, Enum):
    """Supported API versions."""
    V1 = "v1"
    V2 = "v2"


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
    DOCUMENT_CLASSIFIED = "document.classified"
    ANALYSIS_COMPLETED = "analysis.completed"
    ALERT_TRIGGERED = "alert.triggered"
    EXPORT_READY = "export.ready"
    LEGAL_HOLD_APPLIED = "legal_hold.applied"
    LEGAL_HOLD_RELEASED = "legal_hold.released"
    COMPLIANCE_VIOLATION = "compliance.violation"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    FAILED = "failed"
    ARCHIVED = "archived"


class SortOrder(str, Enum):
    """Sort order."""
    ASC = "asc"
    DESC = "desc"


# =============================================================================
# Generic Response Models
# =============================================================================

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""
    items: list[T]
    total: int
    page: int
    size: int
    pages: int

    @classmethod
    def create(cls, items: list[T], total: int, page: int, size: int):
        """Create paginated response."""
        pages = (total + size - 1) // size if size > 0 else 0
        return cls(items=items, total=total, page=page, size=size, pages=pages)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    details: Optional[dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SuccessResponse(BaseModel):
    """Success response model."""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: dict[str, dict[str, Any]] = Field(default_factory=dict)


# =============================================================================
# Authentication Models
# =============================================================================

class TokenRequest(BaseModel):
    """Token request (login)."""
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8)


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str = "read write"


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class APIKeyCreate(BaseModel):
    """API key creation request."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    expires_at: Optional[datetime] = None
    scopes: list[str] = Field(default_factory=lambda: ["read"])


class APIKeyResponse(BaseModel):
    """API key response."""
    id: str
    name: str
    key: Optional[str] = None  # Only shown on creation
    key_prefix: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    scopes: list[str]
    is_active: bool = True


# =============================================================================
# User Models
# =============================================================================

class UserCreate(BaseModel):
    """User creation request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    organization_id: Optional[str] = None


class UserUpdate(BaseModel):
    """User update request."""
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """User response."""
    id: str
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    organization_id: Optional[str] = None
    roles: list[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime
    last_login_at: Optional[datetime] = None


class PasswordChange(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=8)


# =============================================================================
# Document Models
# =============================================================================

class DocumentUploadRequest(BaseModel):
    """Document upload metadata."""
    filename: str
    content_type: str
    client_id: Optional[str] = None
    matter_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """Document response."""
    id: str
    filename: str
    original_filename: str
    content_type: str
    file_size: int
    checksum: str
    status: DocumentStatus
    client_id: Optional[str] = None
    matter_id: Optional[str] = None
    classification: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    created_by: str
    download_url: Optional[str] = None


class DocumentUpdate(BaseModel):
    """Document update request."""
    client_id: Optional[str] = None
    matter_id: Optional[str] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    classification: Optional[str] = None


class DocumentSearchRequest(BaseModel):
    """Document search request."""
    query: str = Field(..., min_length=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    client_id: Optional[str] = None
    matter_id: Optional[str] = None
    document_types: list[str] = Field(default_factory=list)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sort_by: str = "relevance"
    sort_order: SortOrder = SortOrder.DESC


class DocumentSearchResult(BaseModel):
    """Document search result."""
    id: str
    filename: str
    content_type: str
    score: float
    highlights: list[str] = Field(default_factory=list)
    classification: Optional[str] = None
    created_at: datetime


class DocumentAnalysis(BaseModel):
    """Document analysis result."""
    document_id: str
    document_type: Optional[str] = None
    classification: str
    summary: Optional[str] = None
    key_entities: list[dict[str, Any]] = Field(default_factory=list)
    key_dates: list[dict[str, Any]] = Field(default_factory=list)
    key_clauses: list[dict[str, Any]] = Field(default_factory=list)
    risk_factors: list[dict[str, Any]] = Field(default_factory=list)
    pii_detected: list[dict[str, Any]] = Field(default_factory=list)
    compliance_status: dict[str, Any] = Field(default_factory=dict)
    analyzed_at: datetime


# =============================================================================
# Client & Matter Models
# =============================================================================

class ClientCreate(BaseModel):
    """Client creation request."""
    name: str = Field(..., min_length=1, max_length=200)
    code: Optional[str] = None
    industry: Optional[str] = None
    contact_email: Optional[EmailStr] = None
    contact_phone: Optional[str] = None
    address: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClientResponse(BaseModel):
    """Client response."""
    id: str
    name: str
    code: Optional[str] = None
    industry: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    address: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_count: int = 0
    matter_count: int = 0
    created_at: datetime
    updated_at: datetime


class MatterCreate(BaseModel):
    """Matter creation request."""
    client_id: str
    name: str = Field(..., min_length=1, max_length=200)
    code: Optional[str] = None
    description: Optional[str] = None
    practice_area: Optional[str] = None
    status: str = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatterResponse(BaseModel):
    """Matter response."""
    id: str
    client_id: str
    name: str
    code: Optional[str] = None
    description: Optional[str] = None
    practice_area: Optional[str] = None
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_count: int = 0
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Webhook Models
# =============================================================================

class WebhookCreate(BaseModel):
    """Webhook creation request."""
    url: str = Field(..., pattern=r'^https?://')
    events: list[WebhookEventType]
    secret: Optional[str] = None  # Auto-generated if not provided
    description: Optional[str] = None
    is_active: bool = True


class WebhookResponse(BaseModel):
    """Webhook response."""
    id: str
    url: str
    events: list[WebhookEventType]
    secret_preview: str  # First 8 chars + ***
    description: Optional[str] = None
    is_active: bool
    created_at: datetime
    last_triggered_at: Optional[datetime] = None
    failure_count: int = 0


class WebhookDelivery(BaseModel):
    """Webhook delivery record."""
    id: str
    webhook_id: str
    event_type: WebhookEventType
    payload: dict[str, Any]
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    delivered_at: Optional[datetime] = None
    attempts: int = 0
    next_retry_at: Optional[datetime] = None


# =============================================================================
# Legal Hold Models
# =============================================================================

class LegalHoldCreate(BaseModel):
    """Legal hold creation request."""
    matter_id: str
    matter_name: str
    document_ids: list[str]
    custodians: list[str] = Field(default_factory=list)
    reason: str


class LegalHoldResponse(BaseModel):
    """Legal hold response."""
    id: str
    matter_id: str
    matter_name: Optional[str] = None
    document_ids: list[str]
    custodians: list[str]
    reason: str
    status: str
    applied_by: str
    applied_at: datetime
    released_by: Optional[str] = None
    released_at: Optional[datetime] = None


# =============================================================================
# Export Models
# =============================================================================

class ExportRequest(BaseModel):
    """Export request."""
    document_ids: list[str]
    format: str = "zip"  # zip, pdf, json
    include_metadata: bool = True
    include_analysis: bool = False


class ExportResponse(BaseModel):
    """Export response."""
    id: str
    status: str
    document_count: int
    format: str
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: datetime


# =============================================================================
# Analytics Models
# =============================================================================

class AnalyticsQuery(BaseModel):
    """Analytics query request."""
    metric: str
    dimensions: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    granularity: str = "day"  # hour, day, week, month


class AnalyticsResponse(BaseModel):
    """Analytics response."""
    metric: str
    data: list[dict[str, Any]]
    total: Optional[float] = None
    average: Optional[float] = None
    period: dict[str, datetime] = Field(default_factory=dict)

"""
API Gateway Configuration
=========================
Configuration classes for API gateway settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CORSConfig:
    """CORS configuration."""
    allow_origins: list[str] = field(default_factory=lambda: ["*"])
    allow_methods: list[str] = field(default_factory=lambda: ["*"])
    allow_headers: list[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    max_age: int = 600


@dataclass
class RateLimitConfig:
    """Rate limit configuration per tier."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 20
    documents_per_day: int = 100
    storage_gb: int = 10


@dataclass
class JWTConfig:
    """JWT configuration."""
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    issuer: str = "legal-doc-platform"
    audience: str = "legal-doc-api"


@dataclass
class WebhookConfig:
    """Webhook configuration."""
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 30
    max_payload_size_kb: int = 256
    signature_header: str = "X-Webhook-Signature"
    timestamp_header: str = "X-Webhook-Timestamp"


@dataclass
class APIKeyConfig:
    """API key configuration."""
    header_name: str = "X-API-Key"
    prefix: str = "ldk_"  # legal-doc-key
    key_length: int = 32


@dataclass
class PaginationConfig:
    """Pagination defaults."""
    default_page_size: int = 20
    max_page_size: int = 100
    page_param: str = "page"
    size_param: str = "size"


@dataclass
class UploadConfig:
    """File upload configuration."""
    max_file_size_mb: int = 100
    allowed_extensions: list[str] = field(default_factory=lambda: [
        ".pdf", ".doc", ".docx", ".txt", ".rtf",
        ".xls", ".xlsx", ".ppt", ".pptx",
        ".jpg", ".jpeg", ".png", ".tiff",
        ".zip", ".tar", ".gz",
    ])
    chunk_size_kb: int = 1024
    temp_directory: str = "/tmp/uploads"


@dataclass
class APIConfig:
    """Combined API configuration."""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 4

    # API settings
    title: str = "Legal Document Platform API"
    description: str = "Comprehensive API for legal document management and analysis"
    version: str = "1.0.0"
    api_prefix: str = "/api"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    # Sub-configs
    cors: CORSConfig = field(default_factory=CORSConfig)
    jwt: JWTConfig = field(default_factory=JWTConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    api_key: APIKeyConfig = field(default_factory=APIKeyConfig)
    pagination: PaginationConfig = field(default_factory=PaginationConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)

    # Rate limits by tier
    rate_limits: dict[str, RateLimitConfig] = field(default_factory=lambda: {
        "free": RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000,
            burst_size=10,
            documents_per_day=50,
            storage_gb=5,
        ),
        "professional": RateLimitConfig(
            requests_per_minute=120,
            requests_per_hour=5000,
            requests_per_day=50000,
            burst_size=50,
            documents_per_day=500,
            storage_gb=50,
        ),
        "enterprise": RateLimitConfig(
            requests_per_minute=600,
            requests_per_hour=30000,
            requests_per_day=500000,
            burst_size=200,
            documents_per_day=5000,
            storage_gb=500,
        ),
        "unlimited": RateLimitConfig(
            requests_per_minute=10000,
            requests_per_hour=1000000,
            requests_per_day=10000000,
            burst_size=1000,
            documents_per_day=100000,
            storage_gb=10000,
        ),
    })

    # Redis for rate limiting (optional)
    redis_url: Optional[str] = None

    # Environment
    environment: str = "development"

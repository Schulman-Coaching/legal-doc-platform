"""
Tests for API Configuration
============================
Tests for configuration classes and defaults.
"""

import pytest

from ..config import (
    APIConfig,
    APIKeyConfig,
    CORSConfig,
    JWTConfig,
    PaginationConfig,
    RateLimitConfig,
    UploadConfig,
    WebhookConfig,
)


class TestCORSConfig:
    """Tests for CORS configuration."""

    def test_defaults(self):
        """Test default CORS config."""
        config = CORSConfig()

        assert config.allow_origins == ["*"]
        assert config.allow_methods == ["*"]
        assert config.allow_headers == ["*"]
        assert config.allow_credentials is True
        assert config.max_age == 600

    def test_custom_origins(self):
        """Test custom CORS origins."""
        config = CORSConfig(
            allow_origins=["https://example.com", "https://app.example.com"],
            allow_credentials=False,
        )

        assert len(config.allow_origins) == 2
        assert "https://example.com" in config.allow_origins
        assert config.allow_credentials is False


class TestRateLimitConfig:
    """Tests for rate limit configuration."""

    def test_defaults(self):
        """Test default rate limits."""
        config = RateLimitConfig()

        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.requests_per_day == 10000
        assert config.burst_size == 20
        assert config.documents_per_day == 100
        assert config.storage_gb == 10

    def test_custom_limits(self):
        """Test custom rate limits."""
        config = RateLimitConfig(
            requests_per_minute=120,
            burst_size=50,
        )

        assert config.requests_per_minute == 120
        assert config.burst_size == 50


class TestJWTConfig:
    """Tests for JWT configuration."""

    def test_defaults(self):
        """Test default JWT config."""
        config = JWTConfig()

        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 30
        assert config.refresh_token_expire_days == 7
        assert config.issuer == "legal-doc-platform"
        assert config.audience == "legal-doc-api"

    def test_custom_expiry(self):
        """Test custom token expiry."""
        config = JWTConfig(
            access_token_expire_minutes=60,
            refresh_token_expire_days=30,
        )

        assert config.access_token_expire_minutes == 60
        assert config.refresh_token_expire_days == 30


class TestWebhookConfig:
    """Tests for webhook configuration."""

    def test_defaults(self):
        """Test default webhook config."""
        config = WebhookConfig()

        assert config.max_retries == 3
        assert config.retry_delay_seconds == 60
        assert config.timeout_seconds == 30
        assert config.max_payload_size_kb == 256
        assert config.signature_header == "X-Webhook-Signature"


class TestAPIKeyConfig:
    """Tests for API key configuration."""

    def test_defaults(self):
        """Test default API key config."""
        config = APIKeyConfig()

        assert config.header_name == "X-API-Key"
        assert config.prefix == "ldk_"
        assert config.key_length == 32


class TestPaginationConfig:
    """Tests for pagination configuration."""

    def test_defaults(self):
        """Test default pagination config."""
        config = PaginationConfig()

        assert config.default_page_size == 20
        assert config.max_page_size == 100
        assert config.page_param == "page"
        assert config.size_param == "size"


class TestUploadConfig:
    """Tests for upload configuration."""

    def test_defaults(self):
        """Test default upload config."""
        config = UploadConfig()

        assert config.max_file_size_mb == 100
        assert ".pdf" in config.allowed_extensions
        assert ".docx" in config.allowed_extensions
        assert config.chunk_size_kb == 1024

    def test_allowed_extensions(self):
        """Test allowed file extensions."""
        config = UploadConfig()

        # Common document types
        assert ".pdf" in config.allowed_extensions
        assert ".doc" in config.allowed_extensions
        assert ".docx" in config.allowed_extensions
        assert ".txt" in config.allowed_extensions

        # Spreadsheets
        assert ".xls" in config.allowed_extensions
        assert ".xlsx" in config.allowed_extensions

        # Archives
        assert ".zip" in config.allowed_extensions


class TestAPIConfig:
    """Tests for main API configuration."""

    def test_defaults(self):
        """Test default API config."""
        config = APIConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.debug is False
        assert config.workers == 4
        assert config.environment == "development"

    def test_sub_configs_initialized(self):
        """Test sub-configs are initialized."""
        config = APIConfig()

        assert isinstance(config.cors, CORSConfig)
        assert isinstance(config.jwt, JWTConfig)
        assert isinstance(config.webhook, WebhookConfig)
        assert isinstance(config.api_key, APIKeyConfig)
        assert isinstance(config.pagination, PaginationConfig)
        assert isinstance(config.upload, UploadConfig)

    def test_rate_limits_by_tier(self):
        """Test rate limits by tier."""
        config = APIConfig()

        assert "free" in config.rate_limits
        assert "professional" in config.rate_limits
        assert "enterprise" in config.rate_limits
        assert "unlimited" in config.rate_limits

        # Verify tiers have increasing limits
        free = config.rate_limits["free"]
        pro = config.rate_limits["professional"]
        enterprise = config.rate_limits["enterprise"]

        assert pro.requests_per_minute > free.requests_per_minute
        assert enterprise.requests_per_minute > pro.requests_per_minute

    def test_api_metadata(self):
        """Test API metadata."""
        config = APIConfig()

        assert config.title == "Legal Document Platform API"
        assert config.version == "1.0.0"
        assert config.api_prefix == "/api"
        assert config.docs_url == "/docs"

    def test_custom_config(self):
        """Test custom API config."""
        config = APIConfig(
            host="127.0.0.1",
            port=9000,
            debug=True,
            environment="production",
        )

        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.debug is True
        assert config.environment == "production"

    def test_redis_url_optional(self):
        """Test Redis URL is optional."""
        config = APIConfig()
        assert config.redis_url is None

        config_with_redis = APIConfig(redis_url="redis://localhost:6379")
        assert config_with_redis.redis_url == "redis://localhost:6379"

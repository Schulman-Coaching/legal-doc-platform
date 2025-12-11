"""
Tests for API Middleware
========================
Tests for authentication, rate limiting, and request context middleware.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from ..config import APIKeyConfig, JWTConfig, RateLimitConfig
from ..middleware.auth import (
    APIKey,
    APIKeyHandler,
    AuthenticatedUser,
    AuthMiddleware,
    JWTHandler,
    TokenPayload,
    get_current_user,
    set_current_user,
    clear_request_context,
    require_permissions,
    require_roles,
    _has_permissions,
)
from ..middleware.rate_limit import (
    InMemoryRateLimitStore,
    RateLimitChecker,
    RateLimitInfo,
    TokenBucket,
)
from ..middleware.request_context import (
    RequestContext,
    RequestContextManager,
    get_request_id,
    get_correlation_id,
    get_request_context,
    set_request_context,
    clear_request_context as clear_ctx,
    TraceContext,
)


# =============================================================================
# JWT Handler Tests
# =============================================================================

class TestJWTHandler:
    """Tests for JWT token handling."""

    @pytest.fixture
    def jwt_handler(self):
        """Create JWT handler with test config."""
        config = JWTConfig(
            secret_key="test-secret-key-for-testing",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7,
        )
        return JWTHandler(config)

    def test_create_access_token(self, jwt_handler):
        """Test creating access token."""
        token = jwt_handler.create_access_token(
            user_id="user-123",
            username="testuser",
            email="test@example.com",
            roles=["user", "admin"],
            permissions=["documents:read", "documents:write"],
        )

        assert token is not None
        assert len(token.split(".")) == 3  # JWT format

    def test_decode_valid_token(self, jwt_handler):
        """Test decoding valid token."""
        token = jwt_handler.create_access_token(
            user_id="user-123",
            username="testuser",
            email="test@example.com",
            roles=["user"],
            permissions=["documents:read"],
        )

        payload = jwt_handler.decode_token(token)

        assert payload is not None
        assert payload.sub == "user-123"
        assert payload.username == "testuser"
        assert payload.email == "test@example.com"
        assert "user" in payload.roles

    def test_decode_invalid_token(self, jwt_handler):
        """Test decoding invalid token."""
        payload = jwt_handler.decode_token("invalid.token.here")
        assert payload is None

    def test_decode_tampered_token(self, jwt_handler):
        """Test decoding tampered token."""
        token = jwt_handler.create_access_token(
            user_id="user-123",
            username="testuser",
        )

        # Tamper with token
        parts = token.split(".")
        parts[1] = parts[1] + "tampered"
        tampered = ".".join(parts)

        payload = jwt_handler.decode_token(tampered)
        assert payload is None

    def test_decode_expired_token(self, jwt_handler):
        """Test decoding expired token."""
        # Create handler with very short expiry
        config = JWTConfig(
            secret_key="test-secret",
            access_token_expire_minutes=-1,  # Already expired
        )
        handler = JWTHandler(config)

        token = handler.create_access_token(
            user_id="user-123",
            username="testuser",
        )

        payload = handler.decode_token(token)
        assert payload is None

    def test_create_refresh_token(self, jwt_handler):
        """Test creating refresh token."""
        token = jwt_handler.create_refresh_token(
            user_id="user-123",
            session_id="session-456",
        )

        assert token is not None
        assert jwt_handler.is_refresh_token(token)

    def test_is_refresh_token(self, jwt_handler):
        """Test refresh token identification."""
        access = jwt_handler.create_access_token(
            user_id="user-123",
            username="testuser",
        )
        refresh = jwt_handler.create_refresh_token(user_id="user-123")

        assert jwt_handler.is_refresh_token(access) is False
        assert jwt_handler.is_refresh_token(refresh) is True


# =============================================================================
# API Key Handler Tests
# =============================================================================

class TestAPIKeyHandler:
    """Tests for API key handling."""

    @pytest.fixture
    def api_key_handler(self):
        """Create API key handler."""
        config = APIKeyConfig(
            header_name="X-API-Key",
            prefix="ldk_",
            key_length=32,
        )
        return APIKeyHandler(config)

    def test_generate_key(self, api_key_handler):
        """Test API key generation."""
        key, key_hash = api_key_handler.generate_key()

        assert key.startswith("ldk_")
        assert len(key_hash) == 64  # SHA256 hex

    def test_create_key(self, api_key_handler):
        """Test creating API key."""
        raw_key, api_key = api_key_handler.create_key(
            name="Test Key",
            user_id="user-123",
            organization_id="org-456",
            scopes=["read", "write"],
        )

        assert raw_key.startswith("ldk_")
        assert api_key.name == "Test Key"
        assert api_key.user_id == "user-123"
        assert "read" in api_key.scopes

    def test_validate_key(self, api_key_handler):
        """Test validating API key."""
        raw_key, created_key = api_key_handler.create_key(
            name="Test Key",
            user_id="user-123",
        )

        validated = api_key_handler.validate_key(raw_key)

        assert validated is not None
        assert validated.id == created_key.id
        assert validated.last_used_at is not None

    def test_validate_invalid_key(self, api_key_handler):
        """Test validating invalid key."""
        result = api_key_handler.validate_key("invalid_key")
        assert result is None

    def test_validate_wrong_prefix(self, api_key_handler):
        """Test validating key with wrong prefix."""
        result = api_key_handler.validate_key("wrong_prefix_key")
        assert result is None

    def test_revoke_key(self, api_key_handler):
        """Test revoking API key."""
        raw_key, api_key = api_key_handler.create_key(
            name="Test Key",
            user_id="user-123",
        )

        result = api_key_handler.revoke_key(api_key.id)
        assert result is True

        # Key should no longer validate
        validated = api_key_handler.validate_key(raw_key)
        assert validated is None

    def test_list_keys(self, api_key_handler):
        """Test listing user's API keys."""
        api_key_handler.create_key(name="Key 1", user_id="user-123")
        api_key_handler.create_key(name="Key 2", user_id="user-123")
        api_key_handler.create_key(name="Key 3", user_id="user-456")

        user_keys = api_key_handler.list_keys("user-123")

        assert len(user_keys) == 2
        assert all(k.user_id == "user-123" for k in user_keys)


# =============================================================================
# Permission Tests
# =============================================================================

class TestPermissions:
    """Tests for permission checking."""

    def test_has_permissions_exact_match(self):
        """Test exact permission matching."""
        user = AuthenticatedUser(
            id="user-123",
            username="testuser",
            permissions=["documents:read", "documents:write"],
        )

        assert _has_permissions(user, ("documents:read",))
        assert _has_permissions(user, ("documents:read", "documents:write"))

    def test_has_permissions_missing(self):
        """Test missing permission."""
        user = AuthenticatedUser(
            id="user-123",
            username="testuser",
            permissions=["documents:read"],
        )

        assert not _has_permissions(user, ("documents:delete",))

    def test_has_permissions_admin_wildcard(self):
        """Test admin wildcard permission."""
        user = AuthenticatedUser(
            id="user-123",
            username="testuser",
            permissions=["admin:*"],
        )

        assert _has_permissions(user, ("documents:read",))
        assert _has_permissions(user, ("users:delete",))

    def test_has_permissions_resource_wildcard(self):
        """Test resource wildcard permission."""
        user = AuthenticatedUser(
            id="user-123",
            username="testuser",
            permissions=["documents:*"],
        )

        assert _has_permissions(user, ("documents:read",))
        assert _has_permissions(user, ("documents:delete",))
        assert not _has_permissions(user, ("users:read",))


# =============================================================================
# Rate Limit Tests
# =============================================================================

class TestTokenBucket:
    """Tests for token bucket algorithm."""

    def test_bucket_consume_success(self):
        """Test successful token consumption."""
        bucket = TokenBucket(
            tokens=10.0,
            last_refill=0,
            rate=1.0,
            capacity=10,
        )
        bucket.last_refill = asyncio.get_event_loop().time()

        assert bucket.consume(1) is True
        assert bucket.remaining == 9

    def test_bucket_consume_failure(self):
        """Test failed token consumption when requesting more than available."""
        import time
        bucket = TokenBucket(
            tokens=0.5,  # Less than 1 token
            last_refill=time.time(),  # Current time
            rate=0.001,  # Very slow refill (1 token per 1000 seconds)
            capacity=10,
        )

        # Trying to consume 1 token when only 0.5 available should fail
        assert bucket.consume(1) is False

    def test_bucket_refill(self):
        """Test token refill."""
        import time

        bucket = TokenBucket(
            tokens=5.0,
            last_refill=time.time() - 2,  # 2 seconds ago
            rate=1.0,  # 1 token per second
            capacity=10,
        )

        # Should have ~7 tokens now (5 + 2)
        bucket.consume(0)  # Trigger refill
        assert bucket.remaining >= 6


class TestInMemoryRateLimitStore:
    """Tests for in-memory rate limit store."""

    @pytest.fixture
    def store(self):
        """Create rate limit store."""
        return InMemoryRateLimitStore()

    @pytest.mark.asyncio
    async def test_get_bucket_creates_new(self, store):
        """Test creating new bucket."""
        bucket = await store.get_bucket("test-key", rate=1.0, capacity=10)

        assert bucket is not None
        assert bucket.capacity == 10

    @pytest.mark.asyncio
    async def test_get_bucket_returns_existing(self, store):
        """Test returning existing bucket."""
        bucket1 = await store.get_bucket("test-key", rate=1.0, capacity=10)
        bucket1.tokens = 5.0

        bucket2 = await store.get_bucket("test-key", rate=1.0, capacity=10)

        assert bucket2.tokens == 5.0

    @pytest.mark.asyncio
    async def test_increment_counter(self, store):
        """Test counter incrementing."""
        count1 = await store.increment_counter("test-key", "minute", 60)
        count2 = await store.increment_counter("test-key", "minute", 60)

        assert count1 == 1
        assert count2 == 2


class TestRateLimitChecker:
    """Tests for rate limit checker."""

    @pytest.fixture
    def checker(self):
        """Create rate limit checker."""
        store = InMemoryRateLimitStore()
        return RateLimitChecker(store)

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, checker):
        """Test allowed request."""
        allowed, info = await checker.check_rate_limit("user-123", "professional")

        assert allowed is True
        assert info.remaining > 0
        assert info.tier == "professional"

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, checker):
        """Test rate limit exceeded."""
        # Use very restrictive limits
        checker.limits["test"] = RateLimitConfig(
            requests_per_minute=1,
            requests_per_hour=1,
            requests_per_day=1,
            burst_size=1,
        )

        # First request allowed
        allowed1, _ = await checker.check_rate_limit("user-123", "test")
        assert allowed1 is True

        # Second request denied
        allowed2, info = await checker.check_rate_limit("user-123", "test")
        assert allowed2 is False
        assert info.retry_after is not None

    @pytest.mark.asyncio
    async def test_different_tiers_have_different_limits(self, checker):
        """Test different tier limits."""
        free_limits = checker.get_limits_for_tier("free")
        enterprise_limits = checker.get_limits_for_tier("enterprise")

        assert enterprise_limits.requests_per_minute > free_limits.requests_per_minute


# =============================================================================
# Request Context Tests
# =============================================================================

class TestRequestContext:
    """Tests for request context management."""

    def test_context_creation(self):
        """Test creating request context."""
        import time

        context = RequestContext(
            request_id="req-123",
            correlation_id="corr-456",
            method="GET",
            path="/api/v1/documents",
            query_string="page=1",
            client_ip="192.168.1.1",
            user_agent="TestClient/1.0",
            start_time=time.time(),
        )

        assert context.request_id == "req-123"
        assert context.correlation_id == "corr-456"
        assert context.elapsed_ms >= 0

    def test_context_to_dict(self):
        """Test converting context to dict."""
        import time

        context = RequestContext(
            request_id="req-123",
            correlation_id="corr-456",
            method="GET",
            path="/api/v1/documents",
            query_string="",
            client_ip="127.0.0.1",
            user_agent="TestClient",
            start_time=time.time(),
            user_id="user-789",
        )

        d = context.to_dict()

        assert d["request_id"] == "req-123"
        assert d["user_id"] == "user-789"
        assert "elapsed_ms" in d

    def test_set_and_get_context(self):
        """Test setting and getting context."""
        import time

        context = RequestContext(
            request_id="test-123",
            correlation_id="test-123",
            method="POST",
            path="/test",
            query_string="",
            client_ip="127.0.0.1",
            user_agent="Test",
            start_time=time.time(),
        )

        set_request_context(context)

        assert get_request_id() == "test-123"
        assert get_correlation_id() == "test-123"
        assert get_request_context() == context

        clear_ctx()

        assert get_request_id() is None


class TestTraceContext:
    """Tests for distributed tracing context."""

    def test_trace_context_from_headers(self):
        """Test parsing trace context from headers."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "tracestate": "congo=lZWRzIHRoNhcm5telefo",
        }

        context = TraceContext.from_headers(headers)

        assert context is not None
        assert context.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert context.span_id == "b7ad6b7169203331"
        assert context.sampled is True
        assert "congo" in context.trace_state

    def test_trace_context_to_headers(self):
        """Test converting trace context to headers."""
        context = TraceContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            sampled=True,
        )

        headers = context.to_headers()

        assert "traceparent" in headers
        assert "0af7651916cd43dd8448eb211c80319c" in headers["traceparent"]

    def test_create_child_span(self):
        """Test creating child span."""
        parent = TraceContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            sampled=True,
        )

        child = parent.create_child_span()

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        assert child.parent_span_id == parent.span_id


class TestRequestContextManager:
    """Tests for request context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager."""
        async with RequestContextManager(
            request_id="async-123",
            user_id="user-456",
        ) as ctx:
            assert ctx.request_id == "async-123"
            assert ctx.user_id == "user-456"
            assert get_request_id() == "async-123"

        # Context should be cleared
        assert get_request_id() is None

    def test_sync_context_manager(self):
        """Test sync context manager."""
        with RequestContextManager(request_id="sync-123") as ctx:
            assert ctx.request_id == "sync-123"
            assert get_request_id() == "sync-123"

        assert get_request_id() is None


# =============================================================================
# Auth Context Tests
# =============================================================================

class TestAuthContext:
    """Tests for auth request context."""

    def test_set_and_get_current_user(self):
        """Test setting and getting current user."""
        user = AuthenticatedUser(
            id="user-123",
            username="testuser",
            roles=["user"],
        )

        set_current_user(user)
        retrieved = get_current_user()

        assert retrieved is not None
        assert retrieved.id == "user-123"

        clear_request_context()
        assert get_current_user() is None

    def test_authenticated_user_defaults(self):
        """Test authenticated user defaults."""
        user = AuthenticatedUser(
            id="user-123",
            username="testuser",
        )

        assert user.roles == []
        assert user.permissions == []
        assert user.auth_method == "jwt"
        assert user.api_key_id is None

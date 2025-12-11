"""
Rate Limiting Middleware
========================
Token bucket rate limiting with Redis support and tier-based limits.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, TYPE_CHECKING
from uuid import uuid4
import hashlib

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from starlette.types import ASGIApp

from ..config import RateLimitConfig


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RateLimitInfo:
    """Rate limit status information."""
    limit: int
    remaining: int
    reset_at: datetime
    retry_after: Optional[int] = None
    tier: str = "free"


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    tokens: float
    last_refill: float
    rate: float  # tokens per second
    capacity: int

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        now = time.time()
        elapsed = now - self.last_refill

        # Refill tokens
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_refill = now

        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    @property
    def remaining(self) -> int:
        """Get remaining tokens."""
        return int(self.tokens)

    @property
    def time_until_refill(self) -> float:
        """Time until next token is available."""
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) / self.rate


# =============================================================================
# Rate Limit Store
# =============================================================================

class InMemoryRateLimitStore:
    """In-memory rate limit storage."""

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._counters: dict[str, dict[str, int]] = {}
        self._lock = asyncio.Lock()

    async def get_bucket(
        self,
        key: str,
        rate: float,
        capacity: int,
    ) -> TokenBucket:
        """Get or create token bucket for key."""
        async with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(
                    tokens=float(capacity),
                    last_refill=time.time(),
                    rate=rate,
                    capacity=capacity,
                )
            return self._buckets[key]

    async def increment_counter(
        self,
        key: str,
        window: str,  # "minute", "hour", "day"
        ttl: int,
    ) -> int:
        """Increment counter for sliding window."""
        async with self._lock:
            now = time.time()
            window_key = f"{key}:{window}:{int(now // ttl)}"

            if window_key not in self._counters:
                self._counters[window_key] = {"count": 0, "expires": now + ttl}

            self._counters[window_key]["count"] += 1

            # Clean expired
            self._cleanup_expired()

            return self._counters[window_key]["count"]

    async def get_counter(self, key: str, window: str, ttl: int) -> int:
        """Get current counter value."""
        async with self._lock:
            now = time.time()
            window_key = f"{key}:{window}:{int(now // ttl)}"
            counter = self._counters.get(window_key, {})
            return counter.get("count", 0)

    def _cleanup_expired(self) -> None:
        """Remove expired counters."""
        now = time.time()
        expired = [
            k for k, v in self._counters.items()
            if v.get("expires", 0) < now
        ]
        for k in expired:
            del self._counters[k]


class RedisRateLimitStore:
    """Redis-backed rate limit storage."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: Any = None
        self._connected = False

    async def _get_redis(self) -> Any:
        """Get Redis connection."""
        if not self._connected:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self.redis_url)
                await self._redis.ping()
                self._connected = True
            except Exception:
                self._redis = None
                self._connected = False
        return self._redis

    async def get_bucket(
        self,
        key: str,
        rate: float,
        capacity: int,
    ) -> TokenBucket:
        """Get or create token bucket using Redis."""
        redis = await self._get_redis()
        if not redis:
            # Fall back to simple counter
            return TokenBucket(
                tokens=float(capacity),
                last_refill=time.time(),
                rate=rate,
                capacity=capacity,
            )

        bucket_key = f"ratelimit:bucket:{key}"

        try:
            # Get existing bucket data
            data = await redis.hgetall(bucket_key)

            if data:
                bucket = TokenBucket(
                    tokens=float(data.get(b"tokens", capacity)),
                    last_refill=float(data.get(b"last_refill", time.time())),
                    rate=rate,
                    capacity=capacity,
                )
            else:
                bucket = TokenBucket(
                    tokens=float(capacity),
                    last_refill=time.time(),
                    rate=rate,
                    capacity=capacity,
                )

            return bucket

        except Exception:
            return TokenBucket(
                tokens=float(capacity),
                last_refill=time.time(),
                rate=rate,
                capacity=capacity,
            )

    async def save_bucket(self, key: str, bucket: TokenBucket) -> None:
        """Save bucket state to Redis."""
        redis = await self._get_redis()
        if not redis:
            return

        bucket_key = f"ratelimit:bucket:{key}"

        try:
            await redis.hset(bucket_key, mapping={
                "tokens": str(bucket.tokens),
                "last_refill": str(bucket.last_refill),
            })
            await redis.expire(bucket_key, 3600)  # 1 hour TTL
        except Exception:
            pass

    async def increment_counter(
        self,
        key: str,
        window: str,
        ttl: int,
    ) -> int:
        """Increment counter using Redis."""
        redis = await self._get_redis()
        if not redis:
            return 0

        counter_key = f"ratelimit:counter:{key}:{window}"

        try:
            count = await redis.incr(counter_key)
            if count == 1:
                await redis.expire(counter_key, ttl)
            return count
        except Exception:
            return 0

    async def get_counter(self, key: str, window: str, ttl: int) -> int:
        """Get counter from Redis."""
        redis = await self._get_redis()
        if not redis:
            return 0

        counter_key = f"ratelimit:counter:{key}:{window}"

        try:
            count = await redis.get(counter_key)
            return int(count) if count else 0
        except Exception:
            return 0


# =============================================================================
# Rate Limit Checker
# =============================================================================

class RateLimitChecker:
    """Rate limit checker with tier support."""

    # Default rate limits by tier
    DEFAULT_LIMITS: dict[str, RateLimitConfig] = {
        "free": RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000,
            burst_size=10,
        ),
        "professional": RateLimitConfig(
            requests_per_minute=120,
            requests_per_hour=5000,
            requests_per_day=50000,
            burst_size=50,
        ),
        "enterprise": RateLimitConfig(
            requests_per_minute=600,
            requests_per_hour=30000,
            requests_per_day=500000,
            burst_size=200,
        ),
        "unlimited": RateLimitConfig(
            requests_per_minute=10000,
            requests_per_hour=1000000,
            requests_per_day=10000000,
            burst_size=1000,
        ),
    }

    def __init__(
        self,
        store: InMemoryRateLimitStore | RedisRateLimitStore,
        limits: Optional[dict[str, RateLimitConfig]] = None,
    ):
        self.store = store
        self.limits = limits or self.DEFAULT_LIMITS

    async def check_rate_limit(
        self,
        identifier: str,
        tier: str = "free",
        cost: int = 1,
    ) -> tuple[bool, RateLimitInfo]:
        """
        Check if request is within rate limits.

        Returns (allowed, info).
        """
        config = self.limits.get(tier, self.limits["free"])

        # Get rate (tokens per second based on per-minute limit)
        rate = config.requests_per_minute / 60.0
        capacity = config.burst_size

        # Get or create bucket
        bucket = await self.store.get_bucket(
            f"{identifier}:{tier}",
            rate=rate,
            capacity=capacity,
        )

        # Try to consume tokens
        allowed = bucket.consume(cost)

        # Calculate reset time
        reset_at = datetime.utcnow()

        # Build info
        info = RateLimitInfo(
            limit=config.requests_per_minute,
            remaining=max(0, bucket.remaining),
            reset_at=reset_at,
            tier=tier,
        )

        if not allowed:
            info.retry_after = int(bucket.time_until_refill) + 1

        # Also check sliding windows
        if allowed:
            # Check hourly limit
            hourly_count = await self.store.increment_counter(
                identifier, "hour", 3600
            )
            if hourly_count > config.requests_per_hour:
                allowed = False
                info.retry_after = 3600 - (int(time.time()) % 3600)

            # Check daily limit
            if allowed:
                daily_count = await self.store.increment_counter(
                    identifier, "day", 86400
                )
                if daily_count > config.requests_per_day:
                    allowed = False
                    info.retry_after = 86400 - (int(time.time()) % 86400)

        # Save bucket state (for Redis)
        if hasattr(self.store, "save_bucket"):
            await self.store.save_bucket(f"{identifier}:{tier}", bucket)

        return allowed, info

    def get_limits_for_tier(self, tier: str) -> RateLimitConfig:
        """Get rate limit config for tier."""
        return self.limits.get(tier, self.limits["free"])


# =============================================================================
# Rate Limit Middleware
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with tier-based limits.

    Features:
    - Token bucket algorithm for burst handling
    - Sliding window counters for minute/hour/day limits
    - Per-user and per-IP limiting
    - Redis support for distributed rate limiting
    - Tier-based limits (free, pro, enterprise)
    """

    # Paths excluded from rate limiting
    EXCLUDED_PATHS = {
        "/health",
        "/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    def __init__(
        self,
        app: "ASGIApp",
        redis_url: Optional[str] = None,
        limits: Optional[dict[str, RateLimitConfig]] = None,
        get_identifier: Optional[Callable[[Request], str]] = None,
        get_tier: Optional[Callable[[Request], str]] = None,
    ):
        super().__init__(app)

        # Create store
        if redis_url:
            store = RedisRateLimitStore(redis_url)
        else:
            store = InMemoryRateLimitStore()

        self.checker = RateLimitChecker(store, limits)
        self._get_identifier = get_identifier or self._default_get_identifier
        self._get_tier = get_tier or self._default_get_tier

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process rate limiting for each request."""
        # Skip excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        # Get identifier (user ID or IP)
        identifier = self._get_identifier(request)

        # Get tier
        tier = self._get_tier(request)

        # Check rate limit
        allowed, info = await self.checker.check_rate_limit(
            identifier, tier
        )

        # Build response headers
        headers = {
            "X-RateLimit-Limit": str(info.limit),
            "X-RateLimit-Remaining": str(info.remaining),
            "X-RateLimit-Reset": info.reset_at.isoformat(),
            "X-RateLimit-Tier": info.tier,
        }

        if not allowed:
            headers["Retry-After"] = str(info.retry_after or 60)

            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please slow down.",
                    "retry_after": info.retry_after,
                    "limit": info.limit,
                    "tier": info.tier,
                    "request_id": str(uuid4()),
                },
                headers=headers,
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response

    def _default_get_identifier(self, request: Request) -> str:
        """Get rate limit identifier from request."""
        # Try to get user ID from state (set by auth middleware)
        user = getattr(request.state, "user", None)
        if user and hasattr(user, "id"):
            return f"user:{user.id}"

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"

    def _default_get_tier(self, request: Request) -> str:
        """Get rate limit tier from request."""
        # Try to get tier from user
        user = getattr(request.state, "user", None)
        if user:
            # Check for tier in user attributes
            if hasattr(user, "rate_limit_tier"):
                return user.rate_limit_tier
            # Check roles for tier hints
            roles = getattr(user, "roles", [])
            if "enterprise" in roles:
                return "enterprise"
            if "professional" in roles or "pro" in roles:
                return "professional"

        # Check API key tier
        api_key_tier = getattr(request.state, "api_key_tier", None)
        if api_key_tier:
            return api_key_tier

        return "free"


# =============================================================================
# Utility Functions
# =============================================================================

async def check_rate_limit(
    identifier: str,
    tier: str = "free",
    redis_url: Optional[str] = None,
) -> tuple[bool, RateLimitInfo]:
    """
    Standalone rate limit check.

    Usage:
        allowed, info = await check_rate_limit("user:123", "professional")
    """
    if redis_url:
        store = RedisRateLimitStore(redis_url)
    else:
        store = InMemoryRateLimitStore()

    checker = RateLimitChecker(store)
    return await checker.check_rate_limit(identifier, tier)


def rate_limit(
    requests_per_minute: int = 60,
    burst: int = 10,
):
    """
    Decorator for per-endpoint rate limiting.

    Usage:
        @rate_limit(requests_per_minute=10, burst=5)
        async def expensive_endpoint():
            ...
    """
    # Per-endpoint stores
    store = InMemoryRateLimitStore()

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Get request from args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                # Can't rate limit without request
                return await func(*args, **kwargs)

            # Get identifier
            user = getattr(request.state, "user", None)
            if user and hasattr(user, "id"):
                identifier = f"endpoint:{func.__name__}:user:{user.id}"
            else:
                ip = request.client.host if request.client else "unknown"
                identifier = f"endpoint:{func.__name__}:ip:{ip}"

            # Check rate limit
            rate = requests_per_minute / 60.0
            bucket = await store.get_bucket(identifier, rate, burst)

            if not bucket.consume():
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": f"Rate limit exceeded for this endpoint",
                        "retry_after": int(bucket.time_until_refill) + 1,
                    },
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator

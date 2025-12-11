"""
Cache Service
==============
Caching layer for AI/ML operations using Redis or in-memory fallback.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BaseCacheService(ABC):
    """Abstract base class for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass


class InMemoryCache(BaseCacheService):
    """In-memory cache with LRU eviction and TTL support."""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        """
        Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                return None

            value, expires_at = self._cache[key]

            # Check TTL
            if expires_at > 0 and time.time() > expires_at:
                del self._cache[key]
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        async with self._lock:
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl if ttl > 0 else -1

            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (value, expires_at)
            self._cache.move_to_end(key)
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        async with self._lock:
            if key not in self._cache:
                return False

            _, expires_at = self._cache[key]
            if expires_at > 0 and time.time() > expires_at:
                del self._cache[key]
                return False

            return True

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()

    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        async with self._lock:
            now = time.time()
            expired_keys = [
                key for key, (_, expires_at) in self._cache.items()
                if expires_at > 0 and now > expires_at
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "type": "in_memory",
        }


class RedisCache(BaseCacheService):
    """Redis-based cache implementation."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "aiml:",
        default_ttl: int = 3600,
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            prefix: Key prefix for all cache entries
            default_ttl: Default TTL in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self.default_ttl = default_ttl
        self._redis = None

    async def _get_client(self):
        """Get or create Redis client."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=False,
                )
                # Test connection
                await self._redis.ping()
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
            except ImportError:
                logger.error("redis package not installed")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._redis

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        try:
            client = await self._get_client()
            data = await client.get(self._make_key(key))
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            client = await self._get_client()
            ttl = ttl or self.default_ttl
            data = json.dumps(value)
            if ttl > 0:
                await client.setex(self._make_key(key), ttl, data)
            else:
                await client.set(self._make_key(key), data)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        try:
            client = await self._get_client()
            result = await client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        try:
            client = await self._get_client()
            return await client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    async def clear(self) -> None:
        try:
            client = await self._get_client()
            # Delete all keys with our prefix
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=f"{self.prefix}*", count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Redis clear error: {e}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "host": self.host,
            "port": self.port,
            "prefix": self.prefix,
            "type": "redis",
        }


class CacheService:
    """
    High-level cache service with automatic backend selection.
    Uses Redis if available, falls back to in-memory cache.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        max_memory_size: int = 10000,
        default_ttl: int = 3600,
    ):
        """
        Initialize cache service.

        Args:
            redis_url: Redis URL (redis://host:port/db) or None for auto-detect
            max_memory_size: Max size for in-memory fallback
            default_ttl: Default TTL in seconds
        """
        self.default_ttl = default_ttl
        self._backend: Optional[BaseCacheService] = None
        self._redis_url = redis_url or os.getenv("REDIS_URL")
        self._max_memory_size = max_memory_size

    async def _get_backend(self) -> BaseCacheService:
        """Get or initialize cache backend."""
        if self._backend is not None:
            return self._backend

        # Try Redis first
        if self._redis_url:
            try:
                # Parse URL: redis://[:password@]host:port/db
                from urllib.parse import urlparse
                parsed = urlparse(self._redis_url)

                host = parsed.hostname or "localhost"
                port = parsed.port or 6379
                password = parsed.password
                db = int(parsed.path.lstrip("/") or 0)

                self._backend = RedisCache(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    default_ttl=self.default_ttl,
                )
                # Test connection
                await self._backend._get_client()
                logger.info("Using Redis cache backend")
                return self._backend
            except Exception as e:
                logger.warning(f"Redis not available, falling back to in-memory: {e}")

        # Fall back to in-memory
        self._backend = InMemoryCache(
            max_size=self._max_memory_size,
            default_ttl=self.default_ttl,
        )
        logger.info("Using in-memory cache backend")
        return self._backend

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        backend = await self._get_backend()
        return await backend.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        backend = await self._get_backend()
        return await backend.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        backend = await self._get_backend()
        return await backend.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        backend = await self._get_backend()
        return await backend.exists(key)

    async def clear(self) -> None:
        """Clear cache."""
        backend = await self._get_backend()
        await backend.clear()

    async def get_or_set(
        self,
        key: str,
        factory,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get value from cache or compute and store it.

        Args:
            key: Cache key
            factory: Async callable to compute value if not cached
            ttl: TTL for cached value

        Returns:
            Cached or computed value
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl)
        return value

    def make_key(self, *parts: str) -> str:
        """Create cache key from parts."""
        key_str = ":".join(str(p) for p in parts)
        if len(key_str) > 200:
            # Hash long keys
            return hashlib.sha256(key_str.encode()).hexdigest()
        return key_str

"""Tests for Cache Service."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai_ml.cache_service import InMemoryCache, CacheService


class TestInMemoryCache:
    """Test InMemoryCache class."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test initialization."""
        cache = InMemoryCache(max_size=100, default_ttl=60)
        assert cache.max_size == 100
        assert cache.default_ttl == 60

    @pytest.mark.asyncio
    async def test_set_get(self):
        """Test basic set and get."""
        cache = InMemoryCache()

        await cache.set("key1", {"data": "value"})
        result = await cache.get("key1")

        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test getting non-existent key."""
        cache = InMemoryCache()

        result = await cache.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting key."""
        cache = InMemoryCache()

        await cache.set("key1", "value")
        deleted = await cache.delete("key1")

        assert deleted is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Test deleting non-existent key."""
        cache = InMemoryCache()

        deleted = await cache.delete("nonexistent")

        assert deleted is False

    @pytest.mark.asyncio
    async def test_exists(self):
        """Test checking key existence."""
        cache = InMemoryCache()

        await cache.set("key1", "value")

        assert await cache.exists("key1") is True
        assert await cache.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing cache."""
        cache = InMemoryCache()

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration."""
        import time

        cache = InMemoryCache()

        # Set with very short TTL
        await cache.set("key1", "value", ttl=1)

        # Should exist immediately
        assert await cache.get("key1") == "value"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = InMemoryCache(max_size=3)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add another key, should evict key2 (least recently used)
        await cache.set("key4", "value4")

        assert await cache.get("key1") is not None  # Recently used
        assert await cache.get("key2") is None  # Evicted
        assert await cache.get("key3") is not None
        assert await cache.get("key4") is not None

    @pytest.mark.asyncio
    async def test_stats(self):
        """Test cache statistics."""
        cache = InMemoryCache(max_size=100)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        stats = cache.stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 100
        assert stats["type"] == "in_memory"

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        import time

        cache = InMemoryCache()

        # Add entries with short TTL
        await cache.set("key1", "value1", ttl=0)  # Immediately expired
        await cache.set("key2", "value2", ttl=3600)  # Long TTL

        # Force expiration
        async with cache._lock:
            cache._cache["key1"] = (cache._cache["key1"][0], time.time() - 1)

        removed = await cache.cleanup_expired()

        assert removed == 1
        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"


class TestCacheService:
    """Test high-level CacheService class."""

    @pytest.mark.asyncio
    async def test_default_backend(self):
        """Test that default backend is in-memory."""
        service = CacheService()

        await service.set("key", "value")
        result = await service.get("key")

        assert result == "value"

    @pytest.mark.asyncio
    async def test_get_or_set_cached(self):
        """Test get_or_set with cached value."""
        service = CacheService()

        await service.set("key", "cached_value")

        factory = AsyncMock(return_value="new_value")
        result = await service.get_or_set("key", factory)

        assert result == "cached_value"
        factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_set_not_cached(self):
        """Test get_or_set without cached value."""
        service = CacheService()

        factory = AsyncMock(return_value="computed_value")
        result = await service.get_or_set("key", factory)

        assert result == "computed_value"
        factory.assert_called_once()

        # Should be cached now
        cached = await service.get("key")
        assert cached == "computed_value"

    @pytest.mark.asyncio
    async def test_get_or_set_sync_factory(self):
        """Test get_or_set with synchronous factory."""
        service = CacheService()

        def sync_factory():
            return "sync_value"

        result = await service.get_or_set("key", sync_factory)

        assert result == "sync_value"

    def test_make_key(self):
        """Test key generation."""
        service = CacheService()

        # Short key
        key1 = service.make_key("part1", "part2", "part3")
        assert key1 == "part1:part2:part3"

        # Long key should be hashed
        long_parts = ["part"] * 100
        key2 = service.make_key(*long_parts)
        assert len(key2) == 64  # SHA-256 hex length

    @pytest.mark.asyncio
    async def test_redis_fallback(self):
        """Test fallback to in-memory when Redis unavailable."""
        # Non-existent Redis URL
        service = CacheService(redis_url="redis://nonexistent:6379/0")

        # Should fall back to in-memory
        await service.set("key", "value")
        result = await service.get("key")

        assert result == "value"


class TestCacheServiceWithRedis:
    """Test CacheService with mocked Redis."""

    @pytest.mark.asyncio
    async def test_redis_get(self):
        """Test Redis get operation."""
        import json

        mock_redis = AsyncMock()
        mock_redis.get.return_value = json.dumps({"data": "value"}).encode()

        with patch('redis.asyncio.Redis', return_value=mock_redis):
            from ..cache_service import RedisCache
            cache = RedisCache()
            cache._redis = mock_redis

            result = await cache.get("key")

            assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_redis_set(self):
        """Test Redis set operation."""
        mock_redis = AsyncMock()

        with patch('redis.asyncio.Redis', return_value=mock_redis):
            from ..cache_service import RedisCache
            cache = RedisCache()
            cache._redis = mock_redis

            result = await cache.set("key", {"data": "value"}, ttl=3600)

            assert result is True
            mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_delete(self):
        """Test Redis delete operation."""
        mock_redis = AsyncMock()
        mock_redis.delete.return_value = 1

        with patch('redis.asyncio.Redis', return_value=mock_redis):
            from ..cache_service import RedisCache
            cache = RedisCache()
            cache._redis = mock_redis

            result = await cache.delete("key")

            assert result is True

    @pytest.mark.asyncio
    async def test_redis_exists(self):
        """Test Redis exists operation."""
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 1

        with patch('redis.asyncio.Redis', return_value=mock_redis):
            from ..cache_service import RedisCache
            cache = RedisCache()
            cache._redis = mock_redis

            result = await cache.exists("key")

            assert result is True


class TestConcurrency:
    """Test concurrent cache operations."""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self):
        """Test concurrent read operations."""
        cache = InMemoryCache()
        await cache.set("key", "value")

        async def read():
            return await cache.get("key")

        results = await asyncio.gather(*[read() for _ in range(100)])

        assert all(r == "value" for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_writes(self):
        """Test concurrent write operations."""
        cache = InMemoryCache()

        async def write(i):
            await cache.set(f"key{i}", f"value{i}")

        await asyncio.gather(*[write(i) for i in range(100)])

        # All writes should succeed
        for i in range(100):
            assert await cache.get(f"key{i}") == f"value{i}"

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self):
        """Test concurrent read and write operations."""
        cache = InMemoryCache()

        async def operation(i):
            if i % 2 == 0:
                await cache.set(f"key{i}", f"value{i}")
            else:
                await cache.get(f"key{i-1}")

        await asyncio.gather(*[operation(i) for i in range(100)])

        # Even keys should be set
        for i in range(0, 100, 2):
            assert await cache.get(f"key{i}") == f"value{i}"

"""
Redis Repository
================
Async Redis repository for caching, sessions, and pub/sub.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Optional, Union

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.asyncio.cluster import RedisCluster

from ..config import RedisConfig

logger = logging.getLogger(__name__)


class RedisRepository:
    """
    Redis repository for caching and sessions.

    Supports standalone and cluster modes, pub/sub, and rate limiting.
    """

    # Key prefixes for organization
    PREFIX_CACHE = "cache"
    PREFIX_SESSION = "session"
    PREFIX_SEARCH = "search"
    PREFIX_RATE_LIMIT = "ratelimit"
    PREFIX_LOCK = "lock"
    PREFIX_DOCUMENT = "doc"

    def __init__(self, config: RedisConfig):
        self.config = config
        self._client: Optional[Union[Redis, RedisCluster]] = None
        self._pubsub = None

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            if self.config.cluster_mode:
                self._client = RedisCluster(
                    host=self.config.host,
                    port=self.config.port,
                    password=self.config.password,
                    decode_responses=self.config.decode_responses,
                )
            else:
                self._client = Redis(
                    host=self.config.host,
                    port=self.config.port,
                    password=self.config.password,
                    db=self.config.db,
                    ssl=self.config.ssl,
                    decode_responses=self.config.decode_responses,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    max_connections=self.config.max_connections,
                )

            # Test connection
            await self._client.ping()
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None

        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Redis")

    def _key(self, prefix: str, key: str) -> str:
        """Generate namespaced key."""
        return f"{prefix}:{key}"

    # Basic Operations

    async def get(self, key: str, prefix: str = PREFIX_CACHE) -> Optional[str]:
        """Get value from cache."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        return await self._client.get(self._key(prefix, key))

    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
        prefix: str = PREFIX_CACHE,
    ) -> bool:
        """Set value in cache."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        ttl = ttl or self.config.default_ttl
        full_key = self._key(prefix, key)

        await self._client.set(full_key, value, ex=ttl)
        logger.debug(f"Cache set: {full_key} (TTL: {ttl}s)")
        return True

    async def delete(self, key: str, prefix: str = PREFIX_CACHE) -> bool:
        """Delete key from cache."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        result = await self._client.delete(self._key(prefix, key))
        return result > 0

    async def exists(self, key: str, prefix: str = PREFIX_CACHE) -> bool:
        """Check if key exists."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        return await self._client.exists(self._key(prefix, key)) > 0

    async def expire(self, key: str, ttl: int, prefix: str = PREFIX_CACHE) -> bool:
        """Set TTL on existing key."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        return await self._client.expire(self._key(prefix, key), ttl)

    async def ttl(self, key: str, prefix: str = PREFIX_CACHE) -> int:
        """Get remaining TTL for key."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        return await self._client.ttl(self._key(prefix, key))

    # JSON Operations

    async def get_json(self, key: str, prefix: str = PREFIX_CACHE) -> Optional[dict]:
        """Get JSON value from cache."""
        value = await self.get(key, prefix)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None

    async def set_json(
        self,
        key: str,
        value: dict,
        ttl: Optional[int] = None,
        prefix: str = PREFIX_CACHE,
    ) -> bool:
        """Set JSON value in cache."""
        return await self.set(key, json.dumps(value), ttl, prefix)

    # Batch Operations

    async def mget(self, keys: list[str], prefix: str = PREFIX_CACHE) -> dict[str, Optional[str]]:
        """Get multiple values."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        full_keys = [self._key(prefix, k) for k in keys]
        values = await self._client.mget(full_keys)

        return {k: v for k, v in zip(keys, values)}

    async def mset(
        self,
        mapping: dict[str, str],
        ttl: Optional[int] = None,
        prefix: str = PREFIX_CACHE,
    ) -> bool:
        """Set multiple values."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        # Use pipeline for atomic operation
        pipe = self._client.pipeline()

        for key, value in mapping.items():
            full_key = self._key(prefix, key)
            pipe.set(full_key, value, ex=ttl or self.config.default_ttl)

        await pipe.execute()
        return True

    async def delete_pattern(self, pattern: str, prefix: str = PREFIX_CACHE) -> int:
        """Delete all keys matching pattern."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        full_pattern = self._key(prefix, pattern)
        deleted = 0

        async for key in self._client.scan_iter(match=full_pattern):
            await self._client.delete(key)
            deleted += 1

        logger.debug(f"Deleted {deleted} keys matching {full_pattern}")
        return deleted

    # Session Management

    async def get_session(self, session_id: str) -> Optional[dict]:
        """Get user session."""
        return await self.get_json(session_id, self.PREFIX_SESSION)

    async def set_session(
        self,
        session_id: str,
        data: dict,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set user session."""
        return await self.set_json(
            session_id,
            data,
            ttl or self.config.session_ttl,
            self.PREFIX_SESSION,
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete user session."""
        return await self.delete(session_id, self.PREFIX_SESSION)

    async def extend_session(self, session_id: str, ttl: Optional[int] = None) -> bool:
        """Extend session TTL."""
        return await self.expire(
            session_id,
            ttl or self.config.session_ttl,
            self.PREFIX_SESSION,
        )

    # Document Caching

    async def cache_document(
        self,
        document_id: str,
        data: dict,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache document metadata."""
        return await self.set_json(
            document_id,
            data,
            ttl or self.config.default_ttl,
            self.PREFIX_DOCUMENT,
        )

    async def get_cached_document(self, document_id: str) -> Optional[dict]:
        """Get cached document metadata."""
        return await self.get_json(document_id, self.PREFIX_DOCUMENT)

    async def invalidate_document(self, document_id: str) -> bool:
        """Invalidate document cache."""
        return await self.delete(document_id, self.PREFIX_DOCUMENT)

    # Search Result Caching

    async def cache_search_results(
        self,
        query_hash: str,
        results: dict,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache search results."""
        return await self.set_json(
            query_hash,
            results,
            ttl or self.config.search_cache_ttl,
            self.PREFIX_SEARCH,
        )

    async def get_cached_search(self, query_hash: str) -> Optional[dict]:
        """Get cached search results."""
        return await self.get_json(query_hash, self.PREFIX_SEARCH)

    async def invalidate_search_cache(self) -> int:
        """Invalidate all search caches."""
        return await self.delete_pattern("*", self.PREFIX_SEARCH)

    # Rate Limiting

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int, int]:
        """
        Check and update rate limit using sliding window.

        Args:
            key: Rate limit key (e.g., user_id, ip)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds

        Returns:
            Tuple of (allowed, remaining, reset_in_seconds)
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")

        full_key = self._key(self.PREFIX_RATE_LIMIT, key)
        now = datetime.utcnow().timestamp()
        window_start = now - window_seconds

        # Use pipeline for atomic operations
        pipe = self._client.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(full_key, 0, window_start)

        # Count current entries
        pipe.zcard(full_key)

        # Add current request
        pipe.zadd(full_key, {str(now): now})

        # Set expiry
        pipe.expire(full_key, window_seconds)

        results = await pipe.execute()
        current_count = results[1]

        allowed = current_count < limit
        remaining = max(0, limit - current_count - 1)

        # Calculate reset time
        oldest = await self._client.zrange(full_key, 0, 0, withscores=True)
        if oldest:
            reset_in = int(oldest[0][1] + window_seconds - now)
        else:
            reset_in = window_seconds

        return allowed, remaining, max(0, reset_in)

    async def get_rate_limit_info(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> dict[str, Any]:
        """Get rate limit information without incrementing."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        full_key = self._key(self.PREFIX_RATE_LIMIT, key)
        now = datetime.utcnow().timestamp()
        window_start = now - window_seconds

        # Clean and count
        await self._client.zremrangebyscore(full_key, 0, window_start)
        current_count = await self._client.zcard(full_key)

        return {
            "limit": limit,
            "remaining": max(0, limit - current_count),
            "used": current_count,
            "window_seconds": window_seconds,
        }

    # Distributed Locking

    async def acquire_lock(
        self,
        lock_name: str,
        owner: str,
        ttl: int = 30,
    ) -> bool:
        """
        Acquire a distributed lock.

        Args:
            lock_name: Name of the lock
            owner: Unique identifier for lock owner
            ttl: Lock expiry in seconds

        Returns:
            True if lock acquired, False otherwise
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")

        full_key = self._key(self.PREFIX_LOCK, lock_name)

        # Use SET NX to atomically acquire lock
        acquired = await self._client.set(
            full_key,
            owner,
            nx=True,
            ex=ttl,
        )

        if acquired:
            logger.debug(f"Lock acquired: {lock_name} by {owner}")

        return acquired is not None

    async def release_lock(self, lock_name: str, owner: str) -> bool:
        """
        Release a distributed lock.

        Only releases if owner matches.
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")

        full_key = self._key(self.PREFIX_LOCK, lock_name)

        # Lua script for atomic check-and-delete
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        result = await self._client.eval(script, 1, full_key, owner)

        if result:
            logger.debug(f"Lock released: {lock_name} by {owner}")

        return result == 1

    async def extend_lock(self, lock_name: str, owner: str, ttl: int = 30) -> bool:
        """Extend lock TTL if owner matches."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        full_key = self._key(self.PREFIX_LOCK, lock_name)

        # Lua script for atomic check-and-extend
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """

        result = await self._client.eval(script, 1, full_key, owner, ttl)
        return result == 1

    # Pub/Sub

    async def publish(self, channel: str, message: dict) -> int:
        """Publish message to channel."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        return await self._client.publish(channel, json.dumps(message))

    async def subscribe(self, channel: str) -> AsyncIterator[dict]:
        """Subscribe to channel and yield messages."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        self._pubsub = self._client.pubsub()
        await self._pubsub.subscribe(channel)

        logger.info(f"Subscribed to channel: {channel}")

        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    try:
                        yield json.loads(message["data"])
                    except json.JSONDecodeError:
                        yield {"raw": message["data"]}
        finally:
            await self._pubsub.unsubscribe(channel)

    # Counter Operations

    async def incr(self, key: str, prefix: str = PREFIX_CACHE) -> int:
        """Increment counter."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        return await self._client.incr(self._key(prefix, key))

    async def decr(self, key: str, prefix: str = PREFIX_CACHE) -> int:
        """Decrement counter."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        return await self._client.decr(self._key(prefix, key))

    async def incrby(self, key: str, amount: int, prefix: str = PREFIX_CACHE) -> int:
        """Increment counter by amount."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        return await self._client.incrby(self._key(prefix, key), amount)

    # List Operations (for queues)

    async def lpush(self, key: str, *values: str, prefix: str = PREFIX_CACHE) -> int:
        """Push values to list head."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        return await self._client.lpush(self._key(prefix, key), *values)

    async def rpop(self, key: str, prefix: str = PREFIX_CACHE) -> Optional[str]:
        """Pop value from list tail."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        return await self._client.rpop(self._key(prefix, key))

    async def llen(self, key: str, prefix: str = PREFIX_CACHE) -> int:
        """Get list length."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        return await self._client.llen(self._key(prefix, key))

    # Health Check

    async def health_check(self) -> dict[str, Any]:
        """Check Redis health."""
        if not self._client:
            return {"status": "disconnected"}

        try:
            info = await self._client.info()

            return {
                "status": "healthy",
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory_human"),
                "uptime_seconds": info.get("uptime_in_seconds"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    # Statistics

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self._client:
            raise RuntimeError("Redis client not connected")

        info = await self._client.info()

        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses

        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / total if total > 0 else 0,
            "keys": await self._client.dbsize(),
            "memory_used": info.get("used_memory"),
            "memory_peak": info.get("used_memory_peak"),
        }

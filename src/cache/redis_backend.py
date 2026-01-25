"""
Redis-based cache backend implementation.

This module provides a production-ready Redis cache backend that implements
the CacheBackend interface with proper error handling, connection pooling,
and optimized batch operations.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import json
import logging
from threading import Lock

try:
    import redis
    from redis import Redis, ConnectionPool
    from redis.exceptions import (
        ConnectionError as RedisConnectionError,
        TimeoutError as RedisTimeoutError,
        RedisError
    )
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    # Define dummy classes for type hints when redis is not installed
    Redis = None
    ConnectionPool = None
    RedisConnectionError = Exception
    RedisTimeoutError = Exception
    RedisError = Exception

from .backend import CacheBackend, CacheStats

# Set up logging
logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base exception for cache operations."""
    pass


class CacheConnectionError(CacheError):
    """Exception for connection-related errors."""
    pass


class CacheOperationError(CacheError):
    """Exception for operation failures."""
    pass


@dataclass
class RedisConfig:
    """Configuration for Redis connection."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_memory: int = 1024 * 1024 * 100  # 100MB default
    eviction_policy: str = "allkeys-lru"
    key_prefix: str = "markov_cache:"
    socket_timeout: float = 5.0
    max_connections: int = 50
    decode_responses: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'max_memory': self.max_memory,
            'eviction_policy': self.eviction_policy,
            'key_prefix': self.key_prefix,
            'socket_timeout': self.socket_timeout,
            'max_connections': self.max_connections
        }


class RedisBackend(CacheBackend):
    """
    Redis-based cache backend implementation.

    Features:
    - Connection pooling for thread safety
    - Automatic reconnection on transient failures
    - Optimized batch operations using pipelines
    - Metadata storage in separate keys
    - Comprehensive error handling
    - Context manager support
    """

    def __init__(self, config: Optional[RedisConfig] = None):
        """
        Initialize Redis backend.

        Args:
            config: Redis configuration, defaults to RedisConfig()
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package is required for RedisBackend. "
                "Install it with: pip install redis"
            )

        self.config = config or RedisConfig()
        self._client: Optional[Redis] = None
        self._pool: Optional[ConnectionPool] = None
        self._connected = False

        # Thread-safe statistics tracking
        self._stats = CacheStats(max_size_bytes=self.config.max_memory)
        self._stats_lock = Lock()

        logger.info(f"RedisBackend initialized with config: {self.config.to_dict()}")

    def _prefixed_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.config.key_prefix}{key}"

    def _unprefixed_key(self, key: str) -> str:
        """Remove prefix from key."""
        prefix = self.config.key_prefix
        if key.startswith(prefix):
            return key[len(prefix):]
        return key

    def _metadata_key(self, key: str) -> str:
        """Get metadata key for a cache key."""
        return f"{self._prefixed_key(key)}:meta"

    def connect(self) -> bool:
        """
        Establish connection to Redis.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create connection pool
            self._pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                max_connections=self.config.max_connections,
                decode_responses=self.config.decode_responses
            )

            # Create Redis client
            self._client = Redis(connection_pool=self._pool)

            # Test connection
            self._client.ping()
            self._connected = True

            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            return True

        except (RedisConnectionError, RedisTimeoutError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close Redis connection cleanly."""
        if self._client:
            try:
                self._client.close()
                logger.info("Disconnected from Redis")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._connected = False
                self._client = None

        if self._pool:
            try:
                self._pool.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting pool: {e}")
            finally:
                self._pool = None

    def ping(self) -> bool:
        """
        Health check for Redis connection.

        Returns:
            True if Redis responds, False otherwise
        """
        if not self._client:
            return False

        try:
            return self._client.ping()
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        """Check if connection is alive."""
        return self._connected and self.ping()

    def _ensure_connected(self) -> None:
        """Ensure connection is established, raise if not."""
        if not self._client or not self._connected:
            raise CacheConnectionError("Not connected to Redis. Call connect() first.")

    def _handle_error(self, operation: str, error: Exception) -> None:
        """
        Handle Redis errors with logging and reconnection attempts.

        Args:
            operation: Name of the operation that failed
            error: The exception that occurred
        """
        logger.error(f"Redis {operation} failed: {error}")

        # Mark as disconnected on connection errors
        if isinstance(error, (RedisConnectionError, RedisTimeoutError)):
            self._connected = False
            raise CacheConnectionError(f"Connection error during {operation}: {error}")

        raise CacheOperationError(f"Operation {operation} failed: {error}")

    def get(self, key: str) -> Optional[bytes]:
        """
        Retrieve value from Redis.

        Args:
            key: The cache key

        Returns:
            The cached value as bytes, or None if not found
        """
        try:
            self._ensure_connected()

            prefixed_key = self._prefixed_key(key)
            value = self._client.get(prefixed_key)

            with self._stats_lock:
                if value is not None:
                    self._stats.hits += 1
                else:
                    self._stats.misses += 1

            return value

        except CacheError:
            raise
        except Exception as e:
            self._handle_error("get", e)

    def set(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store value in Redis.

        Args:
            key: The cache key
            value: The value to cache (as bytes)
            ttl: Time to live in seconds, None for no expiration
            metadata: Optional dictionary for extra info

        Returns:
            True if successfully stored, False otherwise
        """
        try:
            self._ensure_connected()

            prefixed_key = self._prefixed_key(key)

            # Use pipeline for atomic operations
            pipe = self._client.pipeline()

            # Set the value with optional TTL
            if ttl:
                pipe.setex(prefixed_key, ttl, value)
            else:
                pipe.set(prefixed_key, value)

            # Store metadata if provided
            if metadata:
                meta_key = self._metadata_key(key)
                meta_json = json.dumps(metadata).encode()
                if ttl:
                    pipe.setex(meta_key, ttl, meta_json)
                else:
                    pipe.set(meta_key, meta_json)

            pipe.execute()

            with self._stats_lock:
                self._stats.sets += 1

            return True

        except CacheError:
            return False
        except Exception as e:
            logger.warning(f"Set operation failed: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Remove a key from Redis.

        Args:
            key: The cache key to remove

        Returns:
            True if the key existed and was deleted, False otherwise
        """
        try:
            self._ensure_connected()

            prefixed_key = self._prefixed_key(key)
            meta_key = self._metadata_key(key)

            # Delete both value and metadata
            pipe = self._client.pipeline()
            pipe.delete(prefixed_key)
            pipe.delete(meta_key)
            results = pipe.execute()

            deleted = results[0] > 0

            if deleted:
                with self._stats_lock:
                    self._stats.deletes += 1

            return deleted

        except CacheError:
            return False
        except Exception as e:
            logger.warning(f"Delete operation failed: {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis.

        Args:
            key: The cache key to check

        Returns:
            True if key exists, False otherwise
        """
        try:
            self._ensure_connected()

            prefixed_key = self._prefixed_key(key)
            return self._client.exists(prefixed_key) > 0

        except Exception as e:
            logger.warning(f"Exists check failed: {e}")
            return False

    def clear(self) -> int:
        """
        Remove all entries from Redis database.

        WARNING: This flushes the entire Redis database!

        Returns:
            Number of entries deleted (approximate)
        """
        try:
            self._ensure_connected()

            # Get count before clearing
            count = self._client.dbsize()

            # Flush database
            self._client.flushdb()

            logger.warning(f"Cleared {count} entries from Redis database {self.config.db}")

            return count

        except Exception as e:
            logger.error(f"Clear operation failed: {e}")
            return 0

    def get_stats(self) -> CacheStats:
        """
        Get cache statistics combining our tracking with Redis INFO.

        Returns:
            CacheStats object with current metrics
        """
        try:
            self._ensure_connected()

            # Get Redis server info
            info = self._client.info('stats')
            keyspace_info = self._client.info('keyspace')

            with self._stats_lock:
                # Update stats with Redis info
                db_key = f'db{self.config.db}'
                if db_key in keyspace_info:
                    self._stats.current_entries = keyspace_info[db_key].get('keys', 0)

                # Estimate size (Redis doesn't track this directly)
                # We use our max_memory config as the limit
                self._stats.current_size_bytes = self._client.info('memory').get('used_memory', 0)

                return self._stats

        except Exception as e:
            logger.warning(f"get_stats failed: {e}")
            return self._stats

    def get_many(self, keys: List[str]) -> Dict[str, bytes]:
        """
        Get multiple keys using Redis MGET for efficiency.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to values (only includes found keys)
        """
        try:
            self._ensure_connected()

            if not keys:
                return {}

            # Prefix all keys
            prefixed_keys = [self._prefixed_key(k) for k in keys]

            # Use MGET for single round trip
            values = self._client.mget(prefixed_keys)

            # Build result dict, skipping None values
            result = {}
            hits = 0
            misses = 0

            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = value
                    hits += 1
                else:
                    misses += 1

            with self._stats_lock:
                self._stats.hits += hits
                self._stats.misses += misses

            return result

        except CacheError:
            raise
        except Exception as e:
            self._handle_error("get_many", e)

    def set_many(
        self,
        items: Dict[str, bytes],
        ttl: Optional[int] = None
    ) -> int:
        """
        Set multiple key-value pairs using Redis pipeline.

        Args:
            items: Dictionary mapping keys to values
            ttl: Time to live in seconds for all items

        Returns:
            Number of items successfully set
        """
        try:
            self._ensure_connected()

            if not items:
                return 0

            # Use pipeline for efficiency
            pipe = self._client.pipeline()

            for key, value in items.items():
                prefixed_key = self._prefixed_key(key)
                if ttl:
                    pipe.setex(prefixed_key, ttl, value)
                else:
                    pipe.set(prefixed_key, value)

            results = pipe.execute()

            # Count successful sets
            count = sum(1 for r in results if r)

            with self._stats_lock:
                self._stats.sets += count

            return count

        except CacheError:
            return 0
        except Exception as e:
            logger.warning(f"set_many failed: {e}")
            return 0

    def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple keys using Redis pipeline.

        Args:
            keys: List of cache keys to delete

        Returns:
            Number of keys successfully deleted
        """
        try:
            self._ensure_connected()

            if not keys:
                return 0

            # Use pipeline for efficiency
            pipe = self._client.pipeline()

            for key in keys:
                prefixed_key = self._prefixed_key(key)
                meta_key = self._metadata_key(key)
                pipe.delete(prefixed_key)
                pipe.delete(meta_key)

            results = pipe.execute()

            # Count deleted keys (every 2 results = 1 key + metadata)
            count = sum(1 for i in range(0, len(results), 2) if results[i] > 0)

            with self._stats_lock:
                self._stats.deletes += count

            return count

        except CacheError:
            return 0
        except Exception as e:
            logger.warning(f"delete_many failed: {e}")
            return 0

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        List keys matching pattern using SCAN (non-blocking).

        Args:
            pattern: Optional glob-style pattern

        Returns:
            List of matching keys (without prefix)
        """
        try:
            self._ensure_connected()

            # Build match pattern with prefix
            if pattern:
                match_pattern = f"{self.config.key_prefix}{pattern}"
            else:
                match_pattern = f"{self.config.key_prefix}*"

            # Use SCAN instead of KEYS to avoid blocking
            keys = []
            cursor = 0

            while True:
                cursor, batch = self._client.scan(
                    cursor=cursor,
                    match=match_pattern,
                    count=100
                )

                # Remove prefix and filter out metadata keys
                for key in batch:
                    if isinstance(key, bytes):
                        key = key.decode()

                    # Skip metadata keys
                    if key.endswith(':meta'):
                        continue

                    keys.append(self._unprefixed_key(key))

                if cursor == 0:
                    break

            return keys

        except Exception as e:
            logger.warning(f"keys operation failed: {e}")
            return []

    def __enter__(self):
        """Context manager entry - connect to Redis."""
        if not self.connect():
            raise CacheConnectionError("Failed to connect to Redis")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disconnect from Redis."""
        self.disconnect()
        return False

    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.disconnect()
        except Exception:
            pass


"""
High-level cache manager with serialization, compression, and prefetch coordination.

This module provides application-level caching that sits on top of the raw cache backend,
adding intelligence like Python object serialization, compression, and prefetch coordination.
"""

import pickle
import json
import zlib
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Tuple, Callable
from collections import OrderedDict
from threading import Lock

from .backend import CacheBackend, InMemoryBackend
from .redis_backend import RedisBackend, RedisConfig

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class CacheManagerConfig:
    """Configuration for the cache manager."""

    backend_type: str = 'memory'  # 'redis' or 'memory'
    backend_config: Optional[Any] = None  # Config for the chosen backend
    default_ttl: int = 300  # Default TTL in seconds
    max_entry_size: int = 1024 * 1024  # 1MB max per entry
    compression_enabled: bool = True
    compression_threshold: int = 1024  # Compress if over 1KB
    compression_level: int = 6  # zlib compression level (1-9)
    serialization_format: str = 'pickle'  # 'pickle' or 'json'

    def __post_init__(self):
        """Validate configuration."""
        if self.backend_type not in ['redis', 'memory']:
            raise ValueError(f"Invalid backend_type: {self.backend_type}")

        if self.serialization_format not in ['pickle', 'json']:
            raise ValueError(f"Invalid serialization_format: {self.serialization_format}")

        if not (1 <= self.compression_level <= 9):
            raise ValueError(f"compression_level must be 1-9, got {self.compression_level}")

        if self.compression_threshold < 0:
            raise ValueError("compression_threshold must be non-negative")

        if self.max_entry_size <= 0:
            raise ValueError("max_entry_size must be positive")


class CacheManager:
    """
    High-level cache manager with serialization, compression, and prefetch coordination.

    Features:
    - Automatic serialization/deserialization of Python objects
    - Optional compression for large values
    - Prefetch queue coordination
    - Comprehensive metrics tracking
    - RL-driven eviction support
    """

    def __init__(self, config: CacheManagerConfig):
        """
        Initialize cache manager.

        Args:
            config: Cache manager configuration
        """
        self._config = config
        self._backend: Optional[CacheBackend] = None
        self._running = False
        self._lock = Lock()

        # Prefetch queue: OrderedDict[(endpoint, params): (priority, timestamp)]
        self._prefetch_queue: OrderedDict = OrderedDict()
        self._prefetch_lock = Lock()

        # Metrics tracking
        self._metrics = {
            'serialization_time_ms': 0.0,
            'deserialization_time_ms': 0.0,
            'compression_time_ms': 0.0,
            'decompression_time_ms': 0.0,
            'compression_ratio': 0.0,
            'compression_count': 0,
            'prefetch_requests': 0,
            'prefetch_hits': 0,
            'cache_operations': 0,
            'serialization_errors': 0,
            'compression_errors': 0,
        }
        self._metrics_lock = Lock()

        logger.info(f"CacheManager initialized with backend_type={config.backend_type}")

    def start(self) -> bool:
        """
        Start the cache manager and connect backend.

        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self._running:
                logger.warning("CacheManager already running")
                return True

            try:
                # Create and initialize backend
                self._backend = self._create_backend()

                # For Redis, connect explicitly
                if self._config.backend_type == 'redis':
                    if not self._backend.connect():
                        logger.error("Failed to connect to Redis backend")
                        return False

                self._running = True
                logger.info("CacheManager started successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to start CacheManager: {e}")
                return False

    def stop(self):
        """Gracefully shut down the cache manager."""
        with self._lock:
            if not self._running:
                logger.warning("CacheManager not running")
                return

            try:
                # Disconnect backend
                if self._backend and hasattr(self._backend, 'disconnect'):
                    self._backend.disconnect()

                # Clear prefetch queue
                with self._prefetch_lock:
                    self._prefetch_queue.clear()

                self._running = False
                logger.info("CacheManager stopped successfully")

            except Exception as e:
                logger.error(f"Error stopping CacheManager: {e}")

    @property
    def is_running(self) -> bool:
        """Check if cache manager is running."""
        return self._running

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve and deserialize a value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Deserialized value or default if not found
        """
        if not self._running:
            logger.warning("CacheManager not running")
            return default

        try:
            with self._metrics_lock:
                self._metrics['cache_operations'] += 1

            # Get from backend
            data = self._backend.get(key)
            if data is None:
                logger.debug(f"Cache miss: {key}")
                return default

            # Decompress if needed
            data = self._decompress(data)

            # Deserialize
            value = self._deserialize(data)

            logger.debug(f"Cache hit: {key}")
            return value

        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Serialize and store a value in cache.

        Args:
            key: Cache key
            value: Python object to cache
            ttl: Time to live in seconds (None = use default)
            metadata: Optional metadata for the entry

        Returns:
            True if successfully cached, False otherwise
        """
        if not self._running:
            logger.warning("CacheManager not running")
            return False

        try:
            with self._metrics_lock:
                self._metrics['cache_operations'] += 1

            # Serialize
            data = self._serialize(value)

            # Check size limit
            if len(data) > self._config.max_entry_size:
                logger.warning(
                    f"Entry too large: {len(data)} bytes exceeds max {self._config.max_entry_size}"
                )
                return False

            # Compress if needed
            data = self._compress(data)

            # Use default TTL if not specified
            if ttl is None:
                ttl = self._config.default_ttl

            # Store in backend
            success = self._backend.set(key, data, ttl=ttl, metadata=metadata)

            if success:
                logger.debug(f"Cached key: {key} (ttl={ttl})")
            else:
                logger.warning(f"Failed to cache key: {key}")

            return success

        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Remove a key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False if not found or error
        """
        if not self._running:
            logger.warning("CacheManager not running")
            return False

        try:
            with self._metrics_lock:
                self._metrics['cache_operations'] += 1

            success = self._backend.delete(key)

            if success:
                logger.debug(f"Deleted key: {key}")
            else:
                logger.debug(f"Key not found: {key}")

            return success

        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get value from cache, or call factory and cache result if not found.

        Args:
            key: Cache key
            factory: Callable that generates the value if not cached
            ttl: Time to live in seconds

        Returns:
            Cached or newly generated value
        """
        # Try to get from cache
        value = self.get(key)

        if value is not None:
            return value

        # Cache miss - generate value
        try:
            value = factory()

            # Cache the result
            self.set(key, value, ttl=ttl)

            return value

        except Exception as e:
            logger.error(f"Error in factory for key {key}: {e}")
            raise

    def prefetch(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        priority: float = 0.5
    ):
        """
        Queue an endpoint for prefetching.

        Args:
            endpoint: API endpoint to prefetch
            params: Optional parameters for the endpoint
            priority: Prefetch priority (0.0 to 1.0)
        """
        if not self._running:
            logger.warning("CacheManager not running")
            return

        try:
            # Generate cache key
            cache_key = self._generate_key(endpoint, params)

            with self._prefetch_lock:
                # Add to prefetch queue
                self._prefetch_queue[cache_key] = {
                    'endpoint': endpoint,
                    'params': params,
                    'priority': priority,
                    'timestamp': time.time()
                }

                # Sort by priority (highest first)
                self._prefetch_queue = OrderedDict(
                    sorted(
                        self._prefetch_queue.items(),
                        key=lambda x: x[1]['priority'],
                        reverse=True
                    )
                )

            with self._metrics_lock:
                self._metrics['prefetch_requests'] += 1

            logger.debug(f"Queued for prefetch: {endpoint} (priority={priority})")

        except Exception as e:
            logger.error(f"Error queuing prefetch for {endpoint}: {e}")

    def prefetch_many(self, items: List[Tuple[str, Optional[Dict], float]]):
        """
        Queue multiple endpoints for prefetching.

        Args:
            items: List of (endpoint, params, priority) tuples
        """
        for endpoint, params, priority in items:
            self.prefetch(endpoint, params, priority)

    def get_prefetch_queue(self) -> List[Dict[str, Any]]:
        """
        Get current prefetch queue for debugging.

        Returns:
            List of prefetch queue items
        """
        with self._prefetch_lock:
            return [
                {
                    'key': key,
                    'endpoint': item['endpoint'],
                    'params': item['params'],
                    'priority': item['priority'],
                    'timestamp': item['timestamp']
                }
                for key, item in self._prefetch_queue.items()
            ]

    def evict_lru(self, count: int = 10) -> int:
        """
        Manually evict least-recently-used entries.

        Args:
            count: Number of entries to evict

        Returns:
            Number of entries actually evicted
        """
        if not self._running:
            logger.warning("CacheManager not running")
            return 0

        try:
            # Get all keys
            keys = self._backend.keys()

            if len(keys) <= count:
                # Evict all
                evicted = len(keys)
                for key in keys:
                    self._backend.delete(key)
            else:
                # Evict oldest entries
                # Note: This is a simple implementation. For production,
                # you'd want to track actual access times.
                evicted = 0
                for key in keys[:count]:
                    if self._backend.delete(key):
                        evicted += 1

            logger.info(f"Evicted {evicted} LRU entries")
            return evicted

        except Exception as e:
            logger.error(f"Error evicting LRU entries: {e}")
            return 0

    def evict_low_probability(
        self,
        predictions: Dict[str, float],
        count: int = 10
    ) -> int:
        """
        Evict entries unlikely to be accessed based on Markov predictions.

        Args:
            predictions: Dict mapping cache keys to access probabilities
            count: Number of entries to evict

        Returns:
            Number of entries evicted
        """
        if not self._running:
            logger.warning("CacheManager not running")
            return 0

        try:
            # Sort keys by probability (lowest first)
            sorted_keys = sorted(predictions.items(), key=lambda x: x[1])

            # Evict lowest probability entries
            evicted = 0
            for key, prob in sorted_keys[:count]:
                if self._backend.delete(key):
                    evicted += 1
                    logger.debug(f"Evicted low-probability key: {key} (prob={prob:.4f})")

            logger.info(f"Evicted {evicted} low-probability entries")
            return evicted

        except Exception as e:
            logger.error(f"Error evicting low-probability entries: {e}")
            return 0

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache metrics.

        Returns:
            Dictionary of metrics including backend stats and manager stats
        """
        if not self._running:
            return {}

        try:
            # Get backend stats
            backend_stats = self._backend.get_stats()

            # Combine with manager metrics
            with self._metrics_lock:
                metrics = {
                    # Backend stats
                    'hits': backend_stats.hits,
                    'misses': backend_stats.misses,
                    'sets': backend_stats.sets,
                    'deletes': backend_stats.deletes,
                    'hit_rate': backend_stats.hit_rate,
                    'current_entries': backend_stats.current_entries,
                    'current_size_bytes': backend_stats.current_size_bytes,

                    # Manager stats
                    'serialization_time_ms': self._metrics['serialization_time_ms'],
                    'deserialization_time_ms': self._metrics['deserialization_time_ms'],
                    'compression_time_ms': self._metrics['compression_time_ms'],
                    'decompression_time_ms': self._metrics['decompression_time_ms'],
                    'compression_ratio': (
                        self._metrics['compression_ratio'] / self._metrics['compression_count']
                        if self._metrics['compression_count'] > 0 else 0.0
                    ),
                    'compression_count': self._metrics['compression_count'],
                    'prefetch_requests': self._metrics['prefetch_requests'],
                    'prefetch_hits': self._metrics['prefetch_hits'],
                    'prefetch_hit_rate': (
                        self._metrics['prefetch_hits'] / self._metrics['prefetch_requests']
                        if self._metrics['prefetch_requests'] > 0 else 0.0
                    ),
                    'cache_operations': self._metrics['cache_operations'],
                    'serialization_errors': self._metrics['serialization_errors'],
                    'compression_errors': self._metrics['compression_errors'],

                    # Queue size
                    'prefetch_queue_size': len(self._prefetch_queue),
                }

            return metrics

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}

    # Internal helper methods

    def _create_backend(self) -> CacheBackend:
        """Create the appropriate cache backend."""
        if self._config.backend_type == 'memory':
            return InMemoryBackend(max_size_bytes=100 * 1024 * 1024)  # 100MB

        elif self._config.backend_type == 'redis':
            # Use provided config or create default
            redis_config = self._config.backend_config
            if redis_config is None:
                redis_config = RedisConfig()

            return RedisBackend(redis_config)

        else:
            raise ValueError(f"Unknown backend_type: {self._config.backend_type}")

    def _serialize(self, value: Any) -> bytes:
        """
        Convert Python object to bytes.

        Args:
            value: Python object to serialize

        Returns:
            Serialized bytes
        """
        start = time.perf_counter()

        try:
            if self._config.serialization_format == 'pickle':
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            elif self._config.serialization_format == 'json':
                # JSON requires encoding to bytes
                json_str = json.dumps(value)
                data = json_str.encode('utf-8')
            else:
                raise ValueError(f"Unknown serialization format: {self._config.serialization_format}")

            elapsed_ms = (time.perf_counter() - start) * 1000
            with self._metrics_lock:
                self._metrics['serialization_time_ms'] += elapsed_ms

            return data

        except Exception as e:
            with self._metrics_lock:
                self._metrics['serialization_errors'] += 1
            logger.error(f"Serialization error: {e}")
            raise

    def _deserialize(self, data: bytes) -> Any:
        """
        Convert bytes back to Python object.

        Args:
            data: Serialized bytes

        Returns:
            Deserialized Python object
        """
        start = time.perf_counter()

        try:
            if self._config.serialization_format == 'pickle':
                value = pickle.loads(data)
            elif self._config.serialization_format == 'json':
                json_str = data.decode('utf-8')
                value = json.loads(json_str)
            else:
                raise ValueError(f"Unknown serialization format: {self._config.serialization_format}")

            elapsed_ms = (time.perf_counter() - start) * 1000
            with self._metrics_lock:
                self._metrics['deserialization_time_ms'] += elapsed_ms

            return value

        except Exception as e:
            with self._metrics_lock:
                self._metrics['serialization_errors'] += 1
            logger.error(f"Deserialization error: {e}")
            raise

    def _compress(self, data: bytes) -> bytes:
        """
        Compress data if over threshold.

        Args:
            data: Data to compress

        Returns:
            Compressed or original data with marker
        """
        if not self._config.compression_enabled:
            # No compression - prepend marker
            return b'\x00' + data

        if len(data) < self._config.compression_threshold:
            # Too small to compress - prepend marker
            return b'\x00' + data

        start = time.perf_counter()

        try:
            compressed = zlib.compress(data, level=self._config.compression_level)

            elapsed_ms = (time.perf_counter() - start) * 1000

            # Only use compression if it actually saves space
            if len(compressed) < len(data):
                compression_ratio = len(compressed) / len(data)

                with self._metrics_lock:
                    self._metrics['compression_time_ms'] += elapsed_ms
                    self._metrics['compression_ratio'] += compression_ratio
                    self._metrics['compression_count'] += 1

                # Prepend marker indicating compression
                return b'\x01' + compressed
            else:
                # Compression didn't help
                return b'\x00' + data

        except Exception as e:
            with self._metrics_lock:
                self._metrics['compression_errors'] += 1
            logger.error(f"Compression error: {e}")
            # Return uncompressed on error
            return b'\x00' + data

    def _decompress(self, data: bytes) -> bytes:
        """
        Decompress data if needed.

        Args:
            data: Potentially compressed data with marker

        Returns:
            Decompressed data
        """
        if len(data) == 0:
            return data

        # Check compression marker
        marker = data[0:1]
        payload = data[1:]

        if marker == b'\x00':
            # Not compressed
            return payload
        elif marker == b'\x01':
            # Compressed - decompress it
            start = time.perf_counter()

            try:
                decompressed = zlib.decompress(payload)

                elapsed_ms = (time.perf_counter() - start) * 1000
                with self._metrics_lock:
                    self._metrics['decompression_time_ms'] += elapsed_ms

                return decompressed

            except Exception as e:
                with self._metrics_lock:
                    self._metrics['compression_errors'] += 1
                logger.error(f"Decompression error: {e}")
                raise
        else:
            # Unknown marker
            logger.warning(f"Unknown compression marker: {marker}")
            return payload

    def _generate_key(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate consistent cache key from endpoint and parameters.

        Args:
            endpoint: API endpoint
            params: Optional parameters

        Returns:
            Cache key string
        """
        return generate_cache_key(endpoint, params)


def generate_cache_key(endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a deterministic cache key from endpoint and parameters.

    The key is deterministic: same endpoint+params always gives same key.
    Parameter ordering is handled by sorting keys.
    Long keys are hashed to keep them reasonable length.

    Args:
        endpoint: API endpoint (e.g., '/api/users/123')
        params: Optional parameters dict

    Returns:
        Cache key string
    """
    # Start with endpoint
    key_parts = [endpoint]

    # Add sorted parameters if present
    if params:
        # Sort parameters by key for deterministic ordering
        sorted_params = sorted(params.items())

        # Convert to string representation
        param_str = '&'.join(f"{k}={v}" for k, v in sorted_params)
        key_parts.append(param_str)

    # Join parts
    key = '|'.join(key_parts)

    # If key is too long, hash it
    if len(key) > 200:
        # Use SHA256 hash for long keys
        hash_obj = hashlib.sha256(key.encode('utf-8'))
        key_hash = hash_obj.hexdigest()[:32]  # Use first 32 chars

        # Keep a prefix for debugging
        prefix = endpoint.split('?')[0][:50]
        key = f"{prefix}|hash:{key_hash}"

    return key


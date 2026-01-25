"""Cache management utilities and interfaces (Redis, local caches)."""

from .backend import CacheBackend, CacheEntry, CacheStats, InMemoryBackend

# Redis backend (optional, requires redis package)
try:
    from .redis_backend import RedisBackend, RedisConfig, CacheError, CacheConnectionError, CacheOperationError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisBackend = None
    RedisConfig = None
    CacheError = None
    CacheConnectionError = None
    CacheOperationError = None

__all__ = [
    "CacheBackend",
    "CacheEntry",
    "CacheStats",
    "InMemoryBackend",
    "RedisBackend",
    "RedisConfig",
    "CacheError",
    "CacheConnectionError",
    "CacheOperationError",
    "REDIS_AVAILABLE",
]


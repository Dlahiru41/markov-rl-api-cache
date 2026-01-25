"""Cache management utilities and interfaces (Redis, local caches)."""

from .backend import CacheBackend, CacheEntry, CacheStats, InMemoryBackend

__all__ = [
    "CacheBackend",
    "CacheEntry",
    "CacheStats",
    "InMemoryBackend",
]


"""
Abstract interface for cache backends.

This module defines the abstraction layer that allows swapping cache implementations
(Redis, in-memory, etc.) without changing the rest of the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
import time


@dataclass
class CacheEntry:
    """Represents a cached item with metadata and expiration info."""

    key: str
    value: bytes
    created_at: float  # Unix timestamp
    expires_at: Optional[float] = None  # Unix timestamp, None if no expiration
    size_bytes: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Calculate size if not provided."""
        if self.size_bytes == 0:
            self.size_bytes = len(self.value)
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_expired(self) -> bool:
        """Check if current time is past expires_at."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def ttl_remaining(self) -> Optional[float]:
        """
        Seconds until expiration.

        Returns:
            0 if expired, None if no expiry, otherwise seconds remaining
        """
        if self.expires_at is None:
            return None

        remaining = self.expires_at - time.time()
        return max(0.0, remaining)


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    current_entries: int = 0
    current_size_bytes: int = 0
    max_size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate as hits / (hits + misses), or 0 if no requests."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def utilization(self) -> float:
        """Calculate utilization as current_size_bytes / max_size_bytes."""
        if self.max_size_bytes == 0:
            return 0.0
        return self.current_size_bytes / self.max_size_bytes

    def reset(self) -> None:
        """Zero out the counters."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        # Don't reset current_entries and current_size_bytes as they represent state

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'evictions': self.evictions,
            'current_entries': self.current_entries,
            'current_size_bytes': self.current_size_bytes,
            'max_size_bytes': self.max_size_bytes,
            'hit_rate': self.hit_rate,
            'utilization': self.utilization
        }


class CacheBackend(ABC):
    """
    Abstract base class for cache backends.

    Defines the contract all cache implementations must follow.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """
        Retrieve value from cache.

        Args:
            key: The cache key

        Returns:
            The cached value as bytes, or None if not found or expired
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store value in cache.

        Args:
            key: The cache key
            value: The value to cache (as bytes)
            ttl: Time to live in seconds, None for no expiration
            metadata: Optional dictionary for extra info (endpoint, user_type, etc.)

        Returns:
            True if successfully stored, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Remove a key from cache.

        Args:
            key: The cache key to remove

        Returns:
            True if the key existed and was deleted, False otherwise
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists and is not expired.

        Args:
            key: The cache key to check

        Returns:
            True if key exists and is not expired, False otherwise
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """
        Remove all entries from cache.

        Returns:
            Number of entries deleted
        """
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """
        Get current cache statistics.

        Returns:
            CacheStats object with current performance metrics
        """
        pass

    # Optional methods with default implementations

    def get_many(self, keys: List[str]) -> Dict[str, bytes]:
        """
        Get multiple keys at once.

        Default implementation loops calling get() for each key.
        Subclasses can override for optimized batch operations.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to values (only includes found keys)
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    def set_many(
        self,
        items: Dict[str, bytes],
        ttl: Optional[int] = None
    ) -> int:
        """
        Set multiple key-value pairs.

        Default implementation loops calling set() for each pair.
        Subclasses can override for optimized batch operations.

        Args:
            items: Dictionary mapping keys to values
            ttl: Time to live in seconds for all items

        Returns:
            Number of items successfully set
        """
        count = 0
        for key, value in items.items():
            if self.set(key, value, ttl=ttl):
                count += 1
        return count

    def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple keys.

        Default implementation loops calling delete() for each key.
        Subclasses can override for optimized batch operations.

        Args:
            keys: List of cache keys to delete

        Returns:
            Number of keys successfully deleted
        """
        count = 0
        for key in keys:
            if self.delete(key):
                count += 1
        return count

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        List keys matching pattern.

        Default implementation raises NotImplementedError.
        Subclasses should implement this based on their capabilities.

        Args:
            pattern: Optional pattern to match (implementation-specific)

        Returns:
            List of matching keys

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("keys() must be implemented by subclass")


class InMemoryBackend(CacheBackend):
    """
    Simple in-memory cache implementation for testing.

    Uses a dictionary internally with TTL support and LRU eviction.
    """

    def __init__(self, max_size_bytes: int = 1024 * 1024 * 100):  # 100MB default
        """
        Initialize in-memory cache.

        Args:
            max_size_bytes: Maximum cache size in bytes
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size_bytes = max_size_bytes
        self._stats = CacheStats(max_size_bytes=max_size_bytes)
        self._access_order: List[str] = []  # For LRU tracking

    def _evict_if_needed(self, required_bytes: int) -> None:
        """
        Evict least recently used entries if needed to make space.

        Args:
            required_bytes: Bytes needed for new entry
        """
        while (self._stats.current_size_bytes + required_bytes > self._max_size_bytes
               and self._access_order):
            # Remove least recently used
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                entry = self._cache[lru_key]
                self._stats.current_size_bytes -= entry.size_bytes
                self._stats.current_entries -= 1
                self._stats.evictions += 1
                del self._cache[lru_key]

    def _update_access(self, key: str) -> None:
        """
        Update access order for LRU tracking.

        Args:
            key: The key that was accessed
        """
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _remove_expired(self) -> None:
        """Remove all expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        for key in expired_keys:
            self.delete(key)

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve value from cache."""
        entry = self._cache.get(key)

        if entry is None:
            self._stats.misses += 1
            return None

        if entry.is_expired:
            self.delete(key)
            self._stats.misses += 1
            return None

        self._update_access(key)
        self._stats.hits += 1
        return entry.value

    def set(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store value in cache."""
        try:
            old_size = 0
            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                old_size = old_entry.size_bytes
                self._stats.current_size_bytes -= old_size
                self._stats.current_entries -= 1
                # Remove from access order - will be re-added
                if key in self._access_order:
                    self._access_order.remove(key)

            # Create new entry
            created_at = time.time()
            expires_at = created_at + ttl if ttl is not None else None
            size_bytes = len(value)

            # Check if entry is too large for cache
            if size_bytes > self._max_size_bytes:
                # Restore old entry if it existed
                if key in self._cache:
                    self._stats.current_size_bytes += old_size
                    self._stats.current_entries += 1
                return False

            # Evict if needed (current size already reduced by old entry)
            self._evict_if_needed(size_bytes)


            entry = CacheEntry(
                key=key,
                value=value,
                created_at=created_at,
                expires_at=expires_at,
                size_bytes=size_bytes,
                metadata=metadata or {}
            )

            self._cache[key] = entry
            self._update_access(key)
            self._stats.current_size_bytes += size_bytes
            self._stats.current_entries += 1
            self._stats.sets += 1

            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Remove a key from cache."""
        if key not in self._cache:
            return False

        entry = self._cache[key]
        self._stats.current_size_bytes -= entry.size_bytes
        self._stats.current_entries -= 1
        self._stats.deletes += 1

        del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)

        return True

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        entry = self._cache.get(key)
        if entry is None:
            return False

        if entry.is_expired:
            self.delete(key)
            return False

        return True

    def clear(self) -> int:
        """Remove all entries from cache."""
        count = len(self._cache)
        self._cache.clear()
        self._access_order.clear()
        self._stats.current_entries = 0
        self._stats.current_size_bytes = 0
        return count

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        # Clean up expired entries first for accurate stats
        self._remove_expired()
        return self._stats

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        List keys matching pattern.

        Args:
            pattern: Optional glob-style pattern (supports * and ?)

        Returns:
            List of matching keys
        """
        import fnmatch

        all_keys = list(self._cache.keys())

        if pattern is None:
            return all_keys

        return [key for key in all_keys if fnmatch.fnmatch(key, pattern)]

    def get_many(self, keys: List[str]) -> Dict[str, bytes]:
        """Get multiple keys at once (optimized for in-memory)."""
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    def set_many(
        self,
        items: Dict[str, bytes],
        ttl: Optional[int] = None
    ) -> int:
        """Set multiple key-value pairs (optimized for in-memory)."""
        count = 0
        for key, value in items.items():
            if self.set(key, value, ttl=ttl):
                count += 1
        return count

    def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys (optimized for in-memory)."""
        count = 0
        for key in keys:
            if self.delete(key):
                count += 1
        return count


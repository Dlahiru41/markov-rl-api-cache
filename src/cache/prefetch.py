"""
Prefetch queue system for proactive caching of predicted API responses.

This module manages prefetching of API responses based on Markov chain predictions.
It includes a priority queue for pending requests, background workers to execute them,
and a scheduler to decide what to prefetch.
"""

import time
import heapq
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Any, Set
from queue import PriorityQueue, Empty, Full
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class PrefetchRequest:
    """
    Represents a prefetch job for proactive caching.

    Attributes:
        endpoint: API endpoint to prefetch (e.g., "/api/users/123")
        params: Optional request parameters dictionary
        priority: Float 0-1, higher = more urgent (typically Markov probability)
        created_at: When the request was created (Unix timestamp)
        source_prediction: Markov probability that led to this request
        max_age: Maximum seconds to wait in queue before discarding
    """

    endpoint: str
    params: Optional[Dict[str, Any]]
    priority: float
    created_at: float
    source_prediction: float
    max_age: float = 30.0

    def __post_init__(self):
        """Validate the request."""
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError(f"Priority must be 0-1, got {self.priority}")

        if not 0.0 <= self.source_prediction <= 1.0:
            raise ValueError(f"source_prediction must be 0-1, got {self.source_prediction}")

        if self.max_age <= 0:
            raise ValueError(f"max_age must be positive, got {self.max_age}")

        if self.params is None:
            self.params = {}

    def __lt__(self, other: 'PrefetchRequest') -> bool:
        """
        Compare for priority queue ordering.

        Higher priority comes first (so we negate for min-heap).
        If priorities are equal, earlier created_at comes first.
        """
        if not isinstance(other, PrefetchRequest):
            return NotImplemented

        # Negate priority for max-heap behavior in min-heap
        if self.priority != other.priority:
            return self.priority > other.priority

        # Earlier timestamp has higher priority
        return self.created_at < other.created_at

    def __eq__(self, other: 'PrefetchRequest') -> bool:
        """Check equality based on endpoint and params."""
        if not isinstance(other, PrefetchRequest):
            return NotImplemented

        return (self.endpoint == other.endpoint and
                self.params == other.params)

    def __hash__(self) -> int:
        """Hash based on endpoint and params for set operations."""
        # Convert params dict to sorted tuple for hashing
        params_tuple = tuple(sorted(self.params.items())) if self.params else ()
        return hash((self.endpoint, params_tuple))

    def is_expired(self) -> bool:
        """Check if request has exceeded max_age."""
        return (time.time() - self.created_at) > self.max_age

    def age_seconds(self) -> float:
        """Get age of request in seconds."""
        return time.time() - self.created_at

    def get_cache_key(self) -> str:
        """Generate cache key for this request."""
        from ..cache.cache_manager import generate_cache_key
        return generate_cache_key(self.endpoint, self.params)


class PrefetchQueue:
    """
    Thread-safe priority queue for managing pending prefetch requests.

    Uses a priority queue with lock protection. Tracks pending requests
    by endpoint to avoid duplicates.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize prefetch queue.

        Args:
            max_size: Maximum number of pending requests
        """
        self._max_size = max_size
        self._queue = PriorityQueue(maxsize=max_size)
        self._pending: Set[str] = set()  # Track endpoints to avoid duplicates
        self._lock = threading.Lock()
        self._stats = {
            'total_added': 0,
            'total_removed': 0,
            'duplicates_rejected': 0,
            'full_rejections': 0,
        }

        logger.info(f"PrefetchQueue initialized with max_size={max_size}")

    def put(self, request: PrefetchRequest, block: bool = False, timeout: Optional[float] = None) -> bool:
        """
        Add request to queue.

        Args:
            request: PrefetchRequest to add
            block: If True, block if queue is full
            timeout: Timeout for blocking

        Returns:
            True if added, False if duplicate or full
        """
        with self._lock:
            # Check for duplicate
            cache_key = request.get_cache_key()
            if cache_key in self._pending:
                logger.debug(f"Rejecting duplicate request: {request.endpoint}")
                self._stats['duplicates_rejected'] += 1
                return False

        try:
            # Add to queue
            self._queue.put(request, block=block, timeout=timeout)

            with self._lock:
                self._pending.add(cache_key)
                self._stats['total_added'] += 1

            logger.debug(f"Added to queue: {request.endpoint} (priority={request.priority:.3f})")
            return True

        except Full:
            logger.warning(f"Queue full, rejecting: {request.endpoint}")
            with self._lock:
                self._stats['full_rejections'] += 1
            return False

    def get(self, timeout: Optional[float] = 1.0) -> Optional[PrefetchRequest]:
        """
        Get highest priority request from queue.

        Args:
            timeout: Timeout in seconds

        Returns:
            PrefetchRequest or None if timeout/empty
        """
        try:
            request = self._queue.get(block=True, timeout=timeout)

            with self._lock:
                # Remove from pending set
                cache_key = request.get_cache_key()
                self._pending.discard(cache_key)
                self._stats['total_removed'] += 1

            logger.debug(f"Retrieved from queue: {request.endpoint}")
            return request

        except Empty:
            return None

    def remove(self, endpoint: str) -> bool:
        """
        Cancel a pending request for this endpoint.

        Note: This is expensive as it requires rebuilding the queue.

        Args:
            endpoint: Endpoint to cancel

        Returns:
            True if request was found and removed
        """
        with self._lock:
            # Get all items from queue
            items = []
            removed = False

            while not self._queue.empty():
                try:
                    request = self._queue.get_nowait()
                    if request.endpoint != endpoint:
                        items.append(request)
                    else:
                        removed = True
                        cache_key = request.get_cache_key()
                        self._pending.discard(cache_key)
                except Empty:
                    break

            # Rebuild queue
            for item in items:
                try:
                    self._queue.put_nowait(item)
                except Full:
                    # Should not happen since we removed items
                    logger.error("Failed to rebuild queue after remove")
                    break

            if removed:
                logger.debug(f"Removed from queue: {endpoint}")

            return removed

    def clear(self) -> int:
        """
        Empty the queue.

        Returns:
            Number of requests removed
        """
        with self._lock:
            count = 0
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    count += 1
                except Empty:
                    break

            self._pending.clear()
            logger.info(f"Cleared queue: {count} requests removed")
            return count

    def contains(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if endpoint is already queued.

        Args:
            endpoint: Endpoint to check
            params: Optional parameters

        Returns:
            True if endpoint is in queue
        """
        from ..cache.cache_manager import generate_cache_key
        cache_key = generate_cache_key(endpoint, params)

        with self._lock:
            return cache_key in self._pending

    @property
    def size(self) -> int:
        """Current queue length."""
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        """True if no pending requests."""
        return self._queue.empty()

    @property
    def is_full(self) -> bool:
        """True if at max_size."""
        return self._queue.full()

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                'size': self.size,
                'max_size': self._max_size,
                'is_empty': self.is_empty,
                'is_full': self.is_full,
                'total_added': self._stats['total_added'],
                'total_removed': self._stats['total_removed'],
                'duplicates_rejected': self._stats['duplicates_rejected'],
                'full_rejections': self._stats['full_rejections'],
            }


class PrefetchWorker:
    """
    Background worker that processes prefetch queue.

    Pulls requests from queue, fetches API responses, and stores in cache.
    Multiple workers can run in parallel for throughput.
    """

    def __init__(
        self,
        queue: PrefetchQueue,
        fetcher: Callable[[str, Optional[Dict]], Any],
        cache_manager: Any,
        num_workers: int = 2,
        max_requests_per_second: float = 10.0
    ):
        """
        Initialize prefetch worker.

        Args:
            queue: PrefetchQueue to pull from
            fetcher: Callable that fetches API responses (endpoint, params) -> data
            cache_manager: CacheManager to store results
            num_workers: Number of worker threads
            max_requests_per_second: Rate limit for prefetch requests
        """
        self._queue = queue
        self._fetcher = fetcher
        self._cache_manager = cache_manager
        self._num_workers = num_workers
        self._max_requests_per_second = max_requests_per_second

        # Worker state
        self._running = False
        self._threads: List[threading.Thread] = []
        self._stop_event = threading.Event()

        # Rate limiting
        self._rate_limiter_lock = threading.Lock()
        self._last_request_time = 0.0
        self._min_interval = 1.0 / max_requests_per_second if max_requests_per_second > 0 else 0.0

        # Metrics
        self._metrics_lock = threading.Lock()
        self._metrics = {
            'total_processed': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'expired_requests': 0,
            'total_wait_time': 0.0,
            'total_fetch_time': 0.0,
            'consecutive_errors': 0,
        }

        logger.info(
            f"PrefetchWorker initialized with {num_workers} workers, "
            f"rate limit={max_requests_per_second} req/s"
        )

    def start(self):
        """Start background worker threads."""
        if self._running:
            logger.warning("PrefetchWorker already running")
            return

        self._running = True
        self._stop_event.clear()

        # Start worker threads
        for i in range(self._num_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"PrefetchWorker-{i}",
                daemon=True
            )
            thread.start()
            self._threads.append(thread)

        logger.info(f"Started {self._num_workers} prefetch worker threads")

    def stop(self, timeout: float = 10.0):
        """
        Gracefully shut down workers.

        Args:
            timeout: Maximum seconds to wait for workers to finish
        """
        if not self._running:
            logger.warning("PrefetchWorker not running")
            return

        logger.info("Stopping prefetch workers...")
        self._running = False
        self._stop_event.set()

        # Wait for threads to finish
        for thread in self._threads:
            thread.join(timeout=timeout / len(self._threads))
            if thread.is_alive():
                logger.warning(f"Worker thread {thread.name} did not stop gracefully")

        self._threads.clear()
        logger.info("Prefetch workers stopped")

    @property
    def is_running(self) -> bool:
        """Check if workers are running."""
        return self._running

    def _worker_loop(self):
        """Main worker loop (runs in background thread)."""
        thread_name = threading.current_thread().name
        logger.debug(f"{thread_name} started")

        while self._running and not self._stop_event.is_set():
            try:
                # Get next request from queue
                request = self._queue.get(timeout=1.0)

                if request is None:
                    continue

                # Process the request
                self._process_request(request)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)

                # Back off on consecutive errors
                with self._metrics_lock:
                    self._metrics['consecutive_errors'] += 1
                    if self._metrics['consecutive_errors'] >= 5:
                        logger.warning("Multiple consecutive errors, backing off...")
                        time.sleep(5.0)

        logger.debug(f"{thread_name} stopped")

    def _process_request(self, request: PrefetchRequest):
        """
        Process a single prefetch request.

        Args:
            request: PrefetchRequest to process
        """
        wait_time = request.age_seconds()

        with self._metrics_lock:
            self._metrics['total_processed'] += 1
            self._metrics['total_wait_time'] += wait_time

        # Check if request is too old
        if request.is_expired():
            logger.debug(f"Discarding expired request: {request.endpoint} (age={wait_time:.1f}s)")
            with self._metrics_lock:
                self._metrics['expired_requests'] += 1
            return

        # Rate limiting
        self._apply_rate_limit()

        # Fetch the API response
        fetch_start = time.time()
        try:
            logger.debug(f"Fetching: {request.endpoint}")
            data = self._fetcher(request.endpoint, request.params)
            fetch_time = time.time() - fetch_start

            # Store in cache
            cache_key = request.get_cache_key()
            # Use prediction probability to determine TTL (higher prob = longer TTL)
            ttl = int(300 * request.source_prediction)  # 0-300 seconds
            ttl = max(60, min(ttl, 600))  # Clamp to 60-600 seconds

            success = self._cache_manager.set(
                cache_key,
                data,
                ttl=ttl,
                metadata={
                    'prefetched': True,
                    'prediction': request.source_prediction,
                    'wait_time': wait_time,
                    'fetch_time': fetch_time,
                }
            )

            if success:
                logger.info(
                    f"Prefetched: {request.endpoint} "
                    f"(prob={request.source_prediction:.3f}, ttl={ttl}s, fetch={fetch_time:.3f}s)"
                )

                with self._metrics_lock:
                    self._metrics['successful_fetches'] += 1
                    self._metrics['total_fetch_time'] += fetch_time
                    self._metrics['consecutive_errors'] = 0
            else:
                logger.warning(f"Failed to cache: {request.endpoint}")
                with self._metrics_lock:
                    self._metrics['failed_fetches'] += 1

        except Exception as e:
            fetch_time = time.time() - fetch_start
            logger.error(f"Failed to fetch {request.endpoint}: {e}")

            with self._metrics_lock:
                self._metrics['failed_fetches'] += 1
                self._metrics['total_fetch_time'] += fetch_time
                self._metrics['consecutive_errors'] += 1

    def _apply_rate_limit(self):
        """Apply rate limiting before making request."""
        if self._min_interval <= 0:
            return

        with self._rate_limiter_lock:
            now = time.time()
            elapsed = now - self._last_request_time

            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                time.sleep(sleep_time)

            self._last_request_time = time.time()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get prefetch statistics.

        Returns:
            Dictionary of metrics
        """
        with self._metrics_lock:
            metrics = self._metrics.copy()

        # Calculate derived metrics
        if metrics['total_processed'] > 0:
            metrics['success_rate'] = (
                metrics['successful_fetches'] / metrics['total_processed']
            )
            metrics['avg_wait_time'] = (
                metrics['total_wait_time'] / metrics['total_processed']
            )
        else:
            metrics['success_rate'] = 0.0
            metrics['avg_wait_time'] = 0.0

        if metrics['successful_fetches'] > 0:
            metrics['avg_fetch_time'] = (
                metrics['total_fetch_time'] /
                (metrics['successful_fetches'] + metrics['failed_fetches'])
            )
        else:
            metrics['avg_fetch_time'] = 0.0

        metrics['is_running'] = self._running
        metrics['num_workers'] = self._num_workers

        return metrics


class PrefetchScheduler:
    """
    Decides what to prefetch based on Markov predictions.

    Connects Markov chain predictions to the prefetch queue,
    filtering and prioritizing based on various criteria.
    """

    def __init__(
        self,
        queue: PrefetchQueue,
        cache_manager: Any,
        min_probability: float = 0.3,
        max_prefetch_per_schedule: int = 10,
        cache_space_threshold: float = 0.8
    ):
        """
        Initialize prefetch scheduler.

        Args:
            queue: PrefetchQueue to add requests to
            cache_manager: CacheManager to check what's already cached
            min_probability: Minimum prediction probability to prefetch
            max_prefetch_per_schedule: Maximum requests to schedule at once
            cache_space_threshold: Don't prefetch if cache utilization > this
        """
        self._queue = queue
        self._cache_manager = cache_manager
        self._min_probability = min_probability
        self._max_prefetch_per_schedule = max_prefetch_per_schedule
        self._cache_space_threshold = cache_space_threshold

        self._stats = {
            'total_predictions': 0,
            'scheduled': 0,
            'filtered_low_prob': 0,
            'filtered_cached': 0,
            'filtered_space': 0,
        }
        self._stats_lock = threading.Lock()

        logger.info(
            f"PrefetchScheduler initialized (min_prob={min_probability}, "
            f"max_per_schedule={max_prefetch_per_schedule})"
        )

    def schedule_from_predictions(
        self,
        predictions: List[Tuple[str, float]],
        current_cache_state: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Schedule prefetch requests from Markov predictions.

        Args:
            predictions: List of (endpoint, probability) from Markov predictor
            current_cache_state: Optional cache state (if None, will query cache_manager)

        Returns:
            List of endpoints that were scheduled for prefetch
        """
        with self._stats_lock:
            self._stats['total_predictions'] += len(predictions)

        # Get current cache state
        if current_cache_state is None:
            metrics = self._cache_manager.get_metrics()
            cache_utilization = metrics.get('hit_rate', 0)  # Use hit rate as proxy
        else:
            cache_utilization = current_cache_state.get('utilization', 0)

        # Check cache space
        if cache_utilization > self._cache_space_threshold:
            logger.debug(
                f"Cache utilization {cache_utilization:.1%} > threshold "
                f"{self._cache_space_threshold:.1%}, skipping prefetch"
            )
            with self._stats_lock:
                self._stats['filtered_space'] += len(predictions)
            return []

        # Get currently cached keys
        cached_keys = set()
        try:
            # Try to get keys from cache backend
            if hasattr(self._cache_manager, '_backend') and hasattr(self._cache_manager._backend, 'keys'):
                cached_keys = set(self._cache_manager._backend.keys())
        except Exception as e:
            logger.warning(f"Could not get cached keys: {e}")

        # Get candidates
        candidates = self.get_prefetch_candidates(
            predictions,
            cached_keys,
            min_probability=self._min_probability
        )

        # Limit number of prefetches
        candidates = candidates[:self._max_prefetch_per_schedule]

        # Create and queue requests
        scheduled = []
        now = time.time()

        for endpoint, probability in candidates:
            request = PrefetchRequest(
                endpoint=endpoint,
                params=None,  # Could be enhanced to include params
                priority=probability,
                created_at=now,
                source_prediction=probability,
                max_age=30.0
            )

            if self._queue.put(request):
                scheduled.append(endpoint)
                with self._stats_lock:
                    self._stats['scheduled'] += 1

        if scheduled:
            logger.info(f"Scheduled {len(scheduled)} prefetch requests")

        return scheduled

    def get_prefetch_candidates(
        self,
        predictions: List[Tuple[str, float]],
        cache_keys: Set[str],
        min_probability: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Filter predictions to get prefetch candidates.

        This is a pure function for testing: determines which predictions
        should become prefetch requests.

        Args:
            predictions: List of (endpoint, probability)
            cache_keys: Set of keys already in cache
            min_probability: Minimum probability threshold

        Returns:
            List of (endpoint, probability) that should be prefetched,
            sorted by probability (highest first)
        """
        candidates = []

        for endpoint, probability in predictions:
            # Filter low probability
            if probability < min_probability:
                with self._stats_lock:
                    self._stats['filtered_low_prob'] += 1
                continue

            # Filter already cached
            # Note: endpoint might not be the cache key, could need transformation
            if endpoint in cache_keys:
                with self._stats_lock:
                    self._stats['filtered_cached'] += 1
                continue

            # This is a good candidate
            candidates.append((endpoint, probability))

        # Sort by probability (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self._stats_lock:
            stats = self._stats.copy()

        # Calculate derived metrics
        if stats['total_predictions'] > 0:
            stats['schedule_rate'] = stats['scheduled'] / stats['total_predictions']
        else:
            stats['schedule_rate'] = 0.0

        return stats


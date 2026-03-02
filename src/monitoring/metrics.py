"""
Prometheus metrics collector for the Markov-RL API Cache system.

This module defines and exposes all Prometheus metrics covering:

  CACHE PERFORMANCE
  -  cache_hits_total          — Counter of cache hits (labelled by service/endpoint)
  -  cache_misses_total        — Counter of cache misses
  -  cache_hit_rate            — Gauge: rolling hit-rate fraction [0,1]
  -  cache_entries             — Gauge: current number of cached entries
  -  cache_utilization         — Gauge: fraction of capacity used [0,1]
  -  cache_evictions_total     — Counter: eviction events (labelled by strategy)
  -  cache_sets_total          — Counter: set operations
  -  cache_deletes_total       — Counter: delete operations
  -  cache_entry_size_bytes    — Histogram: individual entry sizes
  -  cache_operation_latency_seconds — Histogram: backend get/set latency

  PREFETCH ENGINE
  -  prefetch_requests_total   — Counter: prefetch attempts (labelled by strategy)
  -  prefetch_hits_total       — Counter: times a prefetched item was used
  -  prefetch_wasted_total     — Counter: prefetches that expired unused
  -  prefetch_efficiency       — Gauge: prefetch_hits / prefetch_requests [0,1]
  -  prefetch_bandwidth_bytes_total — Counter: bytes transferred for prefetching

  MARKOV CHAIN PREDICTOR
  -  markov_predictions_total  — Counter: predictions made
  -  markov_correct_top1_total — Counter: top-1 prediction was correct
  -  markov_correct_topk_total — Counter: labelled by k={1,3,5}
  -  markov_accuracy_top1      — Gauge: rolling top-1 accuracy [0,1]
  -  markov_accuracy_topk      — Gauge: labelled by k
  -  markov_confidence         — Histogram: prediction confidence scores
  -  markov_vocab_size         — Gauge: number of known API endpoints
  -  markov_transition_entropy — Gauge: Shannon entropy of transition distribution

  REINFORCEMENT LEARNING AGENT
  -  rl_episodes_total         — Counter: completed training episodes
  -  rl_steps_total            — Counter: total environment steps
  -  rl_episode_reward         — Histogram: reward per episode
  -  rl_episode_reward_mean    — Gauge: rolling mean reward (last 100)
  -  rl_episode_length         — Histogram: steps per episode
  -  rl_epsilon                — Gauge: current exploration rate
  -  rl_loss                   — Gauge: latest training loss
  -  rl_q_value_mean           — Gauge: mean Q-value (diagnostics)
  -  rl_replay_buffer_size     — Gauge: transitions in replay buffer
  -  rl_target_updates_total   — Counter: target network hard-updates
  -  rl_action_counts_total    — Counter: labelled by action name
  -  rl_training_steps_total   — Counter: gradient update steps

  CASCADE PREVENTION
  -  cascade_risk_score        — Gauge: current cascade risk in [0,1]
  -  cascade_events_total      — Counter: cascade failures detected
  -  cascade_prevented_total   — Counter: cascades the agent prevented
  -  cascade_prevention_rate   — Gauge: prevented/(prevented+occurred) [0,1]

  REWARD DECOMPOSITION
  -  reward_component          — Gauge: labelled by component
                                 {cache, cascade, prefetch, latency, bandwidth, shaping}

  SYSTEM / INFRASTRUCTURE
  -  request_latency_seconds   — Histogram: end-to-end API latency
                                 labelled by service + endpoint
  -  request_count_total       — Counter: requests served (service, status)
  -  backend_call_latency_seconds — Histogram: downstream backend call latency
  -  system_cpu_usage          — Gauge: CPU fraction [0,1]
  -  system_memory_usage       — Gauge: memory fraction [0,1]

  SESSION / TRAFFIC
  -  active_sessions           — Gauge: concurrent user sessions
  -  session_length            — Histogram: API calls per session
  -  requests_per_second       — Gauge: rolling RPS

Usage
-----
    from src.monitoring import MetricsCollector, start_metrics_server

    collector = MetricsCollector()
    start_metrics_server(port=9200)     # /metrics on :9200

    # Record a cache hit
    collector.record_cache_hit(service="api-gateway", endpoint="/products")

    # Record a training step
    collector.record_training_step(loss=0.034, epsilon=0.12, q_mean=4.5)

    # Record a full episode
    collector.record_episode(reward=342.1, length=85,
                             hit_rate=0.78, cascade_occurred=False)
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    # Define stub names so the rest of the module can be imported safely
    CollectorRegistry = None  # type: ignore[assignment,misc]
    Counter = None            # type: ignore[assignment,misc]
    Gauge = None              # type: ignore[assignment,misc]
    Histogram = None          # type: ignore[assignment,misc]
    start_http_server = None  # type: ignore[assignment,misc]
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed – metrics will be no-ops. "
        "Install with: pip install prometheus-client"
    )


# ---------------------------------------------------------------------------
# Histogram bucket presets
# ---------------------------------------------------------------------------
_LATENCY_BUCKETS = (
    0.001, 0.005, 0.010, 0.025, 0.050,
    0.075, 0.100, 0.150, 0.200, 0.300,
    0.500, 0.750, 1.0, 2.5, 5.0, 10.0,
)

_REWARD_BUCKETS = (
    -100, -50, -20, -10, -5, 0,
    5, 10, 20, 50, 100, 200, 400, 600,
)

_SIZE_BUCKETS = (
    64, 256, 1_024, 4_096, 16_384,
    65_536, 262_144, 1_048_576,
)

_CONFIDENCE_BUCKETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_LENGTH_BUCKETS = (10, 25, 50, 100, 200, 500, 1_000)


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def start_metrics_server(port: int = 9200, registry=None) -> None:
    """
    Start the Prometheus HTTP metrics server in a daemon thread.

    Args:
        port:     TCP port to listen on (default 9200).
        registry: CollectorRegistry to serve (default: global REGISTRY).
    """
    if not PROMETHEUS_AVAILABLE:
        logger.error("Cannot start metrics server: prometheus_client not installed.")
        return
    try:
        if registry is None:
            start_http_server(port)
        else:
            start_http_server(port, registry=registry)
        logger.info(f"Prometheus metrics server started on :{port}/metrics")
    except OSError as exc:
        logger.error(f"Failed to start metrics server on port {port}: {exc}")


# ---------------------------------------------------------------------------
# Main collector class
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Central Prometheus metrics collector for the Markov-RL caching system.

    All metrics are registered on a *private* CollectorRegistry so the
    collector can be safely instantiated multiple times (e.g. in tests)
    without duplicate-metric errors on the global registry.

    Call ``start_metrics_server(registry=collector.registry)`` if you want
    to expose this collector's metrics over HTTP; or integrate it with the
    FastAPI ``/metrics`` endpoint via ``generate_latest(collector.registry)``.
    """

    def __init__(self, namespace: str = "markov_rl", service: str = "api-cache"):
        """
        Args:
            namespace: Prometheus metric name prefix (default ``markov_rl``).
            service:   Default ``service`` label value.
        """
        self.namespace = namespace
        self.service = service

        if not PROMETHEUS_AVAILABLE:
            self._noop = True
            return
        self._noop = False

        self.registry = CollectorRegistry()
        ns = namespace

        # ── Cache Performance ────────────────────────────────────────────────
        self.cache_hits = Counter(
            f"{ns}_cache_hits_total",
            "Total cache hits",
            ["service", "endpoint"],
            registry=self.registry,
        )
        self.cache_misses = Counter(
            f"{ns}_cache_misses_total",
            "Total cache misses",
            ["service", "endpoint"],
            registry=self.registry,
        )
        self.cache_hit_rate = Gauge(
            f"{ns}_cache_hit_rate",
            "Rolling cache hit rate [0,1]",
            ["service"],
            registry=self.registry,
        )
        self.cache_entries = Gauge(
            f"{ns}_cache_entries",
            "Current number of entries in cache",
            ["service"],
            registry=self.registry,
        )
        self.cache_utilization = Gauge(
            f"{ns}_cache_utilization",
            "Cache capacity utilisation fraction [0,1]",
            ["service"],
            registry=self.registry,
        )
        self.cache_evictions = Counter(
            f"{ns}_cache_evictions_total",
            "Cache eviction events",
            ["service", "strategy"],   # strategy: lru | low_prob | ttl
            registry=self.registry,
        )
        self.cache_sets = Counter(
            f"{ns}_cache_sets_total",
            "Cache set (write) operations",
            ["service"],
            registry=self.registry,
        )
        self.cache_deletes = Counter(
            f"{ns}_cache_deletes_total",
            "Cache delete operations",
            ["service"],
            registry=self.registry,
        )
        self.cache_entry_size_bytes = Histogram(
            f"{ns}_cache_entry_size_bytes",
            "Size of individual cache entries in bytes",
            ["service"],
            buckets=_SIZE_BUCKETS,
            registry=self.registry,
        )
        self.cache_op_latency = Histogram(
            f"{ns}_cache_operation_latency_seconds",
            "Latency of cache backend get/set operations",
            ["service", "operation"],  # operation: get | set | delete
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )

        # ── Prefetch Engine ──────────────────────────────────────────────────
        self.prefetch_requests = Counter(
            f"{ns}_prefetch_requests_total",
            "Total prefetch attempts",
            ["service", "strategy"],  # strategy: conservative | moderate | aggressive
            registry=self.registry,
        )
        self.prefetch_hits = Counter(
            f"{ns}_prefetch_hits_total",
            "Prefetched entries that were actually used",
            ["service", "strategy"],
            registry=self.registry,
        )
        self.prefetch_wasted = Counter(
            f"{ns}_prefetch_wasted_total",
            "Prefetched entries that expired unused",
            ["service", "strategy"],
            registry=self.registry,
        )
        self.prefetch_efficiency = Gauge(
            f"{ns}_prefetch_efficiency",
            "Fraction of prefetches that were used (hits/requests) [0,1]",
            ["service"],
            registry=self.registry,
        )
        self.prefetch_bandwidth_bytes = Counter(
            f"{ns}_prefetch_bandwidth_bytes_total",
            "Bytes transferred to populate prefetch entries",
            ["service"],
            registry=self.registry,
        )

        # ── Markov Chain Predictor ───────────────────────────────────────────
        self.markov_predictions = Counter(
            f"{ns}_markov_predictions_total",
            "Total Markov chain predictions made",
            ["service", "order"],  # order: 1 | 2
            registry=self.registry,
        )
        self.markov_correct = Counter(
            f"{ns}_markov_correct_total",
            "Correct Markov predictions (top-k)",
            ["service", "k"],
            registry=self.registry,
        )
        self.markov_accuracy_topk = Gauge(
            f"{ns}_markov_accuracy_topk",
            "Rolling top-k accuracy of the Markov predictor",
            ["service", "k"],
            registry=self.registry,
        )
        self.markov_confidence = Histogram(
            f"{ns}_markov_confidence",
            "Confidence score (probability) of the top-1 prediction",
            ["service"],
            buckets=_CONFIDENCE_BUCKETS,
            registry=self.registry,
        )
        self.markov_vocab_size = Gauge(
            f"{ns}_markov_vocab_size",
            "Number of distinct API endpoints known to the Markov chain",
            ["service"],
            registry=self.registry,
        )
        self.markov_transition_entropy = Gauge(
            f"{ns}_markov_transition_entropy",
            "Shannon entropy of the outgoing transition distribution from current state",
            ["service"],
            registry=self.registry,
        )

        # ── RL Agent ─────────────────────────────────────────────────────────
        self.rl_episodes = Counter(
            f"{ns}_rl_episodes_total",
            "Total completed training episodes",
            ["service"],
            registry=self.registry,
        )
        self.rl_steps = Counter(
            f"{ns}_rl_steps_total",
            "Total environment interaction steps",
            ["service"],
            registry=self.registry,
        )
        self.rl_episode_reward = Histogram(
            f"{ns}_rl_episode_reward",
            "Cumulative reward per training episode",
            ["service"],
            buckets=_REWARD_BUCKETS,
            registry=self.registry,
        )
        self.rl_episode_reward_mean = Gauge(
            f"{ns}_rl_episode_reward_mean",
            "Rolling mean episode reward (last 100 episodes)",
            ["service"],
            registry=self.registry,
        )
        self.rl_episode_length = Histogram(
            f"{ns}_rl_episode_length",
            "Number of steps per training episode",
            ["service"],
            buckets=_LENGTH_BUCKETS,
            registry=self.registry,
        )
        self.rl_epsilon = Gauge(
            f"{ns}_rl_epsilon",
            "Current epsilon for ε-greedy exploration",
            ["service"],
            registry=self.registry,
        )
        self.rl_loss = Gauge(
            f"{ns}_rl_loss",
            "Latest TD-error / training loss value",
            ["service"],
            registry=self.registry,
        )
        self.rl_q_value_mean = Gauge(
            f"{ns}_rl_q_value_mean",
            "Mean Q-value across sampled states (diagnostic)",
            ["service"],
            registry=self.registry,
        )
        self.rl_replay_buffer_size = Gauge(
            f"{ns}_rl_replay_buffer_size",
            "Number of transitions stored in the replay buffer",
            ["service"],
            registry=self.registry,
        )
        self.rl_target_updates = Counter(
            f"{ns}_rl_target_updates_total",
            "Number of hard target-network synchronisations",
            ["service"],
            registry=self.registry,
        )
        self.rl_action_counts = Counter(
            f"{ns}_rl_action_counts_total",
            "Actions selected by the agent",
            ["service", "action"],  # action name from CacheAction enum
            registry=self.registry,
        )
        self.rl_training_steps = Counter(
            f"{ns}_rl_training_steps_total",
            "Gradient update (mini-batch) steps performed",
            ["service"],
            registry=self.registry,
        )

        # ── Cascade Prevention ───────────────────────────────────────────────
        self.cascade_risk = Gauge(
            f"{ns}_cascade_risk_score",
            "Current cascade failure risk score [0,1]",
            ["service"],
            registry=self.registry,
        )
        self.cascade_events = Counter(
            f"{ns}_cascade_events_total",
            "Total cascade failure events detected",
            ["service"],
            registry=self.registry,
        )
        self.cascade_prevented = Counter(
            f"{ns}_cascade_prevented_total",
            "Total cascade failures prevented by the RL agent",
            ["service"],
            registry=self.registry,
        )
        self.cascade_prevention_rate = Gauge(
            f"{ns}_cascade_prevention_rate",
            "Fraction of cascade threats that were prevented [0,1]",
            ["service"],
            registry=self.registry,
        )

        # ── Reward Decomposition ─────────────────────────────────────────────
        self.reward_component = Gauge(
            f"{ns}_reward_component",
            "Mean reward contribution from each component (last episode)",
            ["service", "component"],  # cache|cascade|prefetch|latency|bandwidth|shaping
            registry=self.registry,
        )

        # ── System / Infrastructure ──────────────────────────────────────────
        self.request_latency = Histogram(
            f"{ns}_request_latency_seconds",
            "End-to-end API request latency",
            ["service", "endpoint", "status"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.request_count = Counter(
            f"{ns}_request_count_total",
            "Total API requests handled",
            ["service", "endpoint", "status"],
            registry=self.registry,
        )
        self.backend_call_latency = Histogram(
            f"{ns}_backend_call_latency_seconds",
            "Downstream (cache-miss) backend call latency",
            ["service", "endpoint"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.system_cpu = Gauge(
            f"{ns}_system_cpu_usage",
            "System CPU utilisation fraction [0,1]",
            ["service"],
            registry=self.registry,
        )
        self.system_memory = Gauge(
            f"{ns}_system_memory_usage",
            "System memory utilisation fraction [0,1]",
            ["service"],
            registry=self.registry,
        )

        # ── Session / Traffic ────────────────────────────────────────────────
        self.active_sessions = Gauge(
            f"{ns}_active_sessions",
            "Number of concurrent user sessions being served",
            ["service"],
            registry=self.registry,
        )
        self.session_length = Histogram(
            f"{ns}_session_length",
            "Number of API calls per user session",
            ["service"],
            buckets=_LENGTH_BUCKETS,
            registry=self.registry,
        )
        self.requests_per_second = Gauge(
            f"{ns}_requests_per_second",
            "Current rolling requests-per-second rate",
            ["service"],
            registry=self.registry,
        )

        # ── Internal state for rolling calculations ──────────────────────────
        self._reward_window: deque = deque(maxlen=100)
        self._hit_window: deque = deque(maxlen=500)   # True/False per request
        self._rps_window: deque = deque(maxlen=100)   # timestamps
        self._cascade_prevented_count = 0
        self._cascade_occurred_count = 0
        self._prefetch_req_count = 0
        self._prefetch_hit_count = 0
        self._markov_pred_count = 0
        self._markov_correct_k: dict = {1: 0, 3: 0, 5: 0}
        self._lock = threading.Lock()

        logger.info(
            f"MetricsCollector initialised (namespace={namespace}, service={service})"
        )

    # ------------------------------------------------------------------ #
    #  Cache helpers                                                       #
    # ------------------------------------------------------------------ #

    def record_cache_hit(self, endpoint: str = "unknown") -> None:
        if self._noop:
            return
        with self._lock:
            self.cache_hits.labels(service=self.service, endpoint=endpoint).inc()
            self._hit_window.append(True)
            self._update_hit_rate()
            self._record_rps()

    def record_cache_miss(self, endpoint: str = "unknown") -> None:
        if self._noop:
            return
        with self._lock:
            self.cache_misses.labels(service=self.service, endpoint=endpoint).inc()
            self._hit_window.append(False)
            self._update_hit_rate()
            self._record_rps()

    def record_cache_set(
        self,
        entry_size_bytes: int = 0,
        latency_seconds: float = 0.0,
    ) -> None:
        if self._noop:
            return
        self.cache_sets.labels(service=self.service).inc()
        if entry_size_bytes > 0:
            self.cache_entry_size_bytes.labels(service=self.service).observe(
                entry_size_bytes
            )
        if latency_seconds > 0:
            self.cache_op_latency.labels(
                service=self.service, operation="set"
            ).observe(latency_seconds)

    def record_cache_eviction(self, strategy: str = "lru") -> None:
        if self._noop:
            return
        self.cache_evictions.labels(service=self.service, strategy=strategy).inc()

    def update_cache_state(
        self,
        entries: int,
        utilization: float,
    ) -> None:
        if self._noop:
            return
        self.cache_entries.labels(service=self.service).set(entries)
        self.cache_utilization.labels(service=self.service).set(utilization)

    def record_cache_get_latency(self, latency_seconds: float) -> None:
        if self._noop:
            return
        self.cache_op_latency.labels(
            service=self.service, operation="get"
        ).observe(latency_seconds)

    # ------------------------------------------------------------------ #
    #  Prefetch helpers                                                    #
    # ------------------------------------------------------------------ #

    def record_prefetch(
        self,
        strategy: str,
        used: int,
        wasted: int,
        bandwidth_bytes: int = 0,
    ) -> None:
        if self._noop:
            return
        n = used + wasted
        if n == 0:
            return
        with self._lock:
            self.prefetch_requests.labels(
                service=self.service, strategy=strategy
            ).inc(n)
            self.prefetch_hits.labels(
                service=self.service, strategy=strategy
            ).inc(used)
            self.prefetch_wasted.labels(
                service=self.service, strategy=strategy
            ).inc(wasted)
            self._prefetch_req_count += n
            self._prefetch_hit_count += used
            if self._prefetch_req_count > 0:
                eff = self._prefetch_hit_count / self._prefetch_req_count
                self.prefetch_efficiency.labels(service=self.service).set(eff)
        if bandwidth_bytes > 0:
            self.prefetch_bandwidth_bytes.labels(service=self.service).inc(
                bandwidth_bytes
            )

    # ------------------------------------------------------------------ #
    #  Markov helpers                                                      #
    # ------------------------------------------------------------------ #

    def record_markov_prediction(
        self,
        correct_at_k: dict,        # e.g. {1: True, 3: True, 5: False}
        confidence: float,
        order: int = 1,
        vocab_size: Optional[int] = None,
        entropy: Optional[float] = None,
    ) -> None:
        """
        Record one Markov prediction event.

        Args:
            correct_at_k: Dict mapping k → bool (was top-k prediction correct?)
            confidence:   Probability of the top-1 prediction.
            order:        Markov order (1 or 2).
            vocab_size:   Current vocabulary size (optional).
            entropy:      Transition entropy at current state (optional).
        """
        if self._noop:
            return
        with self._lock:
            self.markov_predictions.labels(
                service=self.service, order=str(order)
            ).inc()
            self._markov_pred_count += 1
            for k, correct in correct_at_k.items():
                if correct:
                    self.markov_correct.labels(
                        service=self.service, k=str(k)
                    ).inc()
                    self._markov_correct_k[k] = (
                        self._markov_correct_k.get(k, 0) + 1
                    )
                if self._markov_pred_count > 0:
                    acc = self._markov_correct_k.get(k, 0) / self._markov_pred_count
                    self.markov_accuracy_topk.labels(
                        service=self.service, k=str(k)
                    ).set(acc)
        self.markov_confidence.labels(service=self.service).observe(confidence)
        if vocab_size is not None:
            self.markov_vocab_size.labels(service=self.service).set(vocab_size)
        if entropy is not None:
            self.markov_transition_entropy.labels(service=self.service).set(entropy)

    # ------------------------------------------------------------------ #
    #  RL agent helpers                                                    #
    # ------------------------------------------------------------------ #

    def record_episode(
        self,
        reward: float,
        length: int,
        hit_rate: float,
        cascade_occurred: bool,
        cascade_prevented: bool = False,
        reward_breakdown: Optional[dict] = None,
    ) -> None:
        """Record metrics at the end of a training episode."""
        if self._noop:
            return
        with self._lock:
            self.rl_episodes.labels(service=self.service).inc()
            self._reward_window.append(reward)
            mean_r = sum(self._reward_window) / len(self._reward_window)
            self.rl_episode_reward_mean.labels(service=self.service).set(mean_r)
            if cascade_occurred:
                self._cascade_occurred_count += 1
                self.cascade_events.labels(service=self.service).inc()
            if cascade_prevented:
                self._cascade_prevented_count += 1
                self.cascade_prevented.labels(service=self.service).inc()
            total_threats = (
                self._cascade_prevented_count + self._cascade_occurred_count
            )
            if total_threats > 0:
                rate = self._cascade_prevented_count / total_threats
                self.cascade_prevention_rate.labels(service=self.service).set(rate)
        self.rl_episode_reward.labels(service=self.service).observe(reward)
        self.rl_episode_length.labels(service=self.service).observe(length)
        self.cache_hit_rate.labels(service=self.service).set(hit_rate)
        if reward_breakdown:
            for component, value in reward_breakdown.items():
                if component != "total":
                    self.reward_component.labels(
                        service=self.service, component=component
                    ).set(value)

    def record_training_step(
        self,
        loss: float,
        epsilon: float,
        q_mean: Optional[float] = None,
        buffer_size: Optional[int] = None,
    ) -> None:
        """Record one gradient update step."""
        if self._noop:
            return
        self.rl_training_steps.labels(service=self.service).inc()
        self.rl_loss.labels(service=self.service).set(loss)
        self.rl_epsilon.labels(service=self.service).set(epsilon)
        if q_mean is not None:
            self.rl_q_value_mean.labels(service=self.service).set(q_mean)
        if buffer_size is not None:
            self.rl_replay_buffer_size.labels(service=self.service).set(buffer_size)

    def record_env_step(
        self,
        action_name: str,
        steps: int = 1,
    ) -> None:
        """Record an environment interaction step."""
        if self._noop:
            return
        self.rl_steps.labels(service=self.service).inc(steps)
        self.rl_action_counts.labels(
            service=self.service, action=action_name
        ).inc()

    def record_target_update(self) -> None:
        """Record a target network hard synchronisation."""
        if self._noop:
            return
        self.rl_target_updates.labels(service=self.service).inc()

    def update_epsilon(self, epsilon: float) -> None:
        if self._noop:
            return
        self.rl_epsilon.labels(service=self.service).set(epsilon)

    # ------------------------------------------------------------------ #
    #  Cascade helpers                                                     #
    # ------------------------------------------------------------------ #

    def update_cascade_risk(self, risk: float) -> None:
        """Update the live cascade risk score."""
        if self._noop:
            return
        self.cascade_risk.labels(service=self.service).set(risk)

    # ------------------------------------------------------------------ #
    #  System helpers                                                      #
    # ------------------------------------------------------------------ #

    def update_system_metrics(self, cpu: float, memory: float) -> None:
        if self._noop:
            return
        self.system_cpu.labels(service=self.service).set(cpu)
        self.system_memory.labels(service=self.service).set(memory)

    def record_request(
        self,
        endpoint: str,
        latency_seconds: float,
        status: str = "200",
    ) -> None:
        if self._noop:
            return
        self.request_count.labels(
            service=self.service, endpoint=endpoint, status=status
        ).inc()
        self.request_latency.labels(
            service=self.service, endpoint=endpoint, status=status
        ).observe(latency_seconds)
        with self._lock:
            self._record_rps()

    def record_backend_call(self, endpoint: str, latency_seconds: float) -> None:
        if self._noop:
            return
        self.backend_call_latency.labels(
            service=self.service, endpoint=endpoint
        ).observe(latency_seconds)

    def update_sessions(self, active: int) -> None:
        if self._noop:
            return
        self.active_sessions.labels(service=self.service).set(active)

    def record_session_end(self, length: int) -> None:
        if self._noop:
            return
        self.session_length.labels(service=self.service).observe(length)

    # ------------------------------------------------------------------ #
    #  Bulk update from controller.get_metrics() dict                     #
    # ------------------------------------------------------------------ #

    def update_from_metrics_dict(self, metrics: dict) -> None:
        """
        Convenience method: ingest the dict returned by
        ``IntegrationController.get_metrics()`` and push all values
        to the relevant gauges/counters.
        """
        if self._noop:
            return
        # Markov
        markov = metrics.get("markov", {})
        if "vocab_size" in markov:
            self.markov_vocab_size.labels(service=self.service).set(
                markov["vocab_size"]
            )
        if "accuracy" in markov:
            self.markov_accuracy_topk.labels(service=self.service, k="1").set(
                markov["accuracy"]
            )

        # Cache
        cache = metrics.get("cache", {})
        if "hit_rate" in cache:
            self.cache_hit_rate.labels(service=self.service).set(cache["hit_rate"])
        if "utilization" in cache:
            self.cache_utilization.labels(service=self.service).set(
                cache["utilization"]
            )
        if "current_entries" in cache:
            self.cache_entries.labels(service=self.service).set(
                cache["current_entries"]
            )

        # Agent
        agent = metrics.get("agent", {})
        if "epsilon" in agent:
            self.rl_epsilon.labels(service=self.service).set(agent["epsilon"])
        if "replay_buffer_size" in agent:
            self.rl_replay_buffer_size.labels(service=self.service).set(
                agent["replay_buffer_size"]
            )

        # Environment
        env = metrics.get("environment", {})
        if env.get("cascade_detected"):
            self.cascade_risk.labels(service=self.service).set(1.0)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _update_hit_rate(self) -> None:
        """Recompute rolling hit rate from window (call inside lock)."""
        if not self._hit_window:
            return
        rate = sum(self._hit_window) / len(self._hit_window)
        self.cache_hit_rate.labels(service=self.service).set(rate)

    def _record_rps(self) -> None:
        """Track timestamps to compute RPS (call inside lock)."""
        now = time.monotonic()
        self._rps_window.append(now)
        # only keep last 10 s worth
        cutoff = now - 10.0
        while self._rps_window and self._rps_window[0] < cutoff:
            self._rps_window.popleft()
        span = self._rps_window[-1] - self._rps_window[0] if len(self._rps_window) > 1 else 1.0
        rps = len(self._rps_window) / max(span, 1.0)
        self.requests_per_second.labels(service=self.service).set(rps)

    def get_snapshot(self) -> dict:
        """
        Return a plain-dict snapshot of key gauge values.
        Useful for logging without touching Prometheus.
        """
        if self._noop:
            return {}
        with self._lock:
            hit_rate = (
                sum(self._hit_window) / len(self._hit_window)
                if self._hit_window else 0.0
            )
            reward_mean = (
                sum(self._reward_window) / len(self._reward_window)
                if self._reward_window else 0.0
            )
        return {
            "cache_hit_rate": round(hit_rate, 4),
            "rl_episode_reward_mean": round(reward_mean, 3),
            "cascade_prevented": self._cascade_prevented_count,
            "cascade_occurred": self._cascade_occurred_count,
            "prefetch_requests": self._prefetch_req_count,
            "prefetch_hits": self._prefetch_hit_count,
            "markov_predictions": self._markov_pred_count,
        }


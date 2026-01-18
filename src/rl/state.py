import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class StateConfig:
    """Configuration for the RL state representation."""
    markov_top_k: int = 5
    include_probabilities: bool = True
    include_confidence: bool = True
    include_cache_metrics: bool = True
    include_system_metrics: bool = True
    include_user_context: bool = True
    include_temporal_context: bool = True
    include_session_context: bool = True
    vocab_size: int = 1000

    @property
    def state_dim(self) -> int:
        """Calculate total dimension based on configuration."""
        dim = 0
        
        # Markov predictions: top-k indices
        dim += self.markov_top_k
        
        # Probabilities
        if self.include_probabilities:
            dim += self.markov_top_k
            
        # Confidence
        if self.include_confidence:
            dim += 1
            
        # Cache metrics: utilization, hit rate, entries, eviction rate
        if self.include_cache_metrics:
            dim += 4
            
        # System metrics: cpu, memory, request_rate, p50, p95, p99, errors, connections, queue
        if self.include_system_metrics:
            dim += 9
            
        # User context: is_premium, is_free, is_guest
        if self.include_user_context:
            dim += 3
            
        # Temporal context: hour_sin, hour_cos, day_sin, day_cos, is_weekend, is_peak_hour
        if self.include_temporal_context:
            dim += 6
            
        # Session context: position, duration, call count
        if self.include_session_context:
            dim += 3
            
        return dim

class StateBuilder:
    """Builds fixed-size state vectors for the RL agent."""
    
    def __init__(self, config: StateConfig):
        self.config = config
        self.api_to_idx = {}
        self.is_fitted = False
        
    def fit(self, vocabulary: List[str]):
        """Fit the builder to the API vocabulary to map APIs to indices."""
        self.api_to_idx = {api: i for i, api in enumerate(vocabulary)}
        self.is_fitted = True
        return self
        
    def _normalize_api(self, api_name: str) -> float:
        """Normalize API index to 0-1."""
        if api_name in self.api_to_idx:
            idx = self.api_to_idx[api_name]
            return idx / self.config.vocab_size
        return 0.0

    def build_state(
        self,
        markov_predictions: List[Tuple[str, float]] = None,
        cache_metrics: Dict[str, float] = None,
        system_metrics: Dict[str, float] = None,
        context: Dict[str, Any] = None
    ) -> np.ndarray:
        """Constructs a fixed-size numpy array state vector."""
        if not self.is_fitted:
            raise ValueError("StateBuilder must be fit() on vocabulary before building states.")
            
        markov_predictions = markov_predictions or []
        cache_metrics = cache_metrics or {}
        system_metrics = system_metrics or {}
        context = context or {}
        
        state = []
        
        # 1. Markov predictions
        # Pad predictions with (None, 0.0)
        padded_predictions = (markov_predictions + [(None, 0.0)] * self.config.markov_top_k)[:self.config.markov_top_k]
        
        # API Indices
        for api, prob in padded_predictions:
            state.append(self._normalize_api(api) if api else 0.0)
            
        # Probabilities
        if self.config.include_probabilities:
            for api, prob in padded_predictions:
                state.append(float(prob))
                
        # Confidence
        if self.config.include_confidence:
            if markov_predictions:
                # Max probability as confidence
                confidence = max(p for _, p in markov_predictions) if markov_predictions else 0.0
            else:
                confidence = 0.0
            state.append(float(confidence))
            
        # 2. Cache metrics
        if self.config.include_cache_metrics:
            state.append(float(cache_metrics.get('utilization', 0.0)))
            state.append(float(cache_metrics.get('hit_rate', 0.0)))
            # Normalized entries (assume max 10000)
            state.append(float(cache_metrics.get('entries', 0.0)) / 10000.0)
            # Normalized eviction rate (assume max 1000)
            state.append(float(cache_metrics.get('eviction_rate', 0.0)) / 1000.0)
            
        # 3. System metrics
        if self.config.include_system_metrics:
            state.append(float(system_metrics.get('cpu', 0.0)))
            state.append(float(system_metrics.get('memory', 0.0)))
            # Normalized request rate (assume max 5000)
            state.append(float(system_metrics.get('request_rate', 0.0)) / 5000.0)
            # Latency percentiles normalized by 1000ms
            state.append(float(system_metrics.get('p50_latency', 0.0)) / 1000.0)
            state.append(float(system_metrics.get('p95_latency', 0.0)) / 1000.0)
            state.append(float(system_metrics.get('p99_latency', 0.0)) / 1000.0)
            state.append(float(system_metrics.get('error_rate', 0.0)))
            # Connection count (assume max 1000)
            state.append(float(system_metrics.get('connections', 0.0)) / 1000.0)
            # Queue depth (assume max 500)
            state.append(float(system_metrics.get('queue_depth', 0.0)) / 500.0)
            
        # 4. Context encoding
        if self.config.include_user_context:
            user_type = context.get('user_type', 'guest')
            state.append(1.0 if user_type == 'premium' else 0.0)
            state.append(1.0 if user_type == 'free' else 0.0)
            state.append(1.0 if user_type == 'guest' else 0.0)
            
        if self.config.include_temporal_context:
            hour = context.get('hour', 0)
            day = context.get('day', 0) # 0-6
            
            # Cyclical encoding
            state.append(np.sin(2 * np.pi * hour / 24))
            state.append(np.cos(2 * np.pi * hour / 24))
            state.append(np.sin(2 * np.pi * day / 7))
            state.append(np.cos(2 * np.pi * day / 7))
            
            # Binary flags
            state.append(1.0 if day >= 5 else 0.0) # weekend
            state.append(1.0 if 9 <= hour <= 17 else 0.0) # peak hour (business hours)
            
        if self.config.include_session_context:
            # Normalized position (assume max 100)
            state.append(float(context.get('session_position', 0)) / 100.0)
            # Normalized duration (assume max 3600s)
            state.append(float(context.get('session_duration', 0)) / 3600.0)
            # Call count (assume max 500)
            state.append(float(context.get('call_count', 0)) / 500.0)
            
        return np.array(state, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names."""
        names = []
        
        # Markov
        for i in range(self.config.markov_top_k):
            names.append(f"markov_api_idx_{i}")
        if self.config.include_probabilities:
            for i in range(self.config.markov_top_k):
                names.append(f"markov_prob_{i}")
        if self.config.include_confidence:
            names.append("markov_confidence")
            
        # Cache
        if self.config.include_cache_metrics:
            names.extend(["cache_utilization", "cache_hit_rate", "cache_entries", "cache_eviction_rate"])
            
        # System
        if self.config.include_system_metrics:
            names.extend([
                "sys_cpu", "sys_memory", "sys_request_rate", 
                "sys_p50_latency", "sys_p95_latency", "sys_p99_latency",
                "sys_error_rate", "sys_connections", "sys_queue_depth"
            ])
            
        # Context
        if self.config.include_user_context:
            names.extend(["user_is_premium", "user_is_free", "user_is_guest"])
            
        if self.config.include_temporal_context:
            names.extend([
                "time_hour_sin", "time_hour_cos", "time_day_sin", "time_day_cos",
                "time_is_weekend", "time_is_peak"
            ])
            
        if self.config.include_session_context:
            names.extend(["session_position", "session_duration", "session_call_count"])
            
        return names

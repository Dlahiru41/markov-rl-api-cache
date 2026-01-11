"""
Data Models Architecture and Flow
==================================

ASCII Diagram of Data Model Relationships
"""

ARCHITECTURE = """
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW API LOGS                                │
│  {"id": "1", "path": "/api/users/123", "method": "GET", ...}       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ Parse & Validate
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         APICall Objects                             │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ APICall(                                                 │       │
│  │   call_id="1",                                           │       │
│  │   endpoint="/api/users/123",                             │       │
│  │   method="GET",                                          │       │
│  │   timestamp=...,                                         │       │
│  │   response_time_ms=100,                                  │       │
│  │   status_code=200,                                       │       │
│  │   ...                                                    │       │
│  │ )                                                        │       │
│  │                                                          │       │
│  │ Methods:                                                 │       │
│  │   • get_service_name() → "users"                         │       │
│  │   • is_successful() → True                               │       │
│  │   • to_dict() / from_dict()                              │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ Group by session_id
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Session Objects                             │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ Session(                                                 │       │
│  │   session_id="sess1",                                    │       │
│  │   user_id="user1",                                       │       │
│  │   calls=[call1, call2, call3, ...]                       │       │
│  │ )                                                        │       │
│  │                                                          │       │
│  │ Properties:                                              │       │
│  │   • endpoint_sequence → ['/api/home', '/api/products']   │       │
│  │   • num_calls → 5                                        │       │
│  │   • duration_seconds → 25.3                              │       │
│  │                                                          │       │
│  │ Methods:                                                 │       │
│  │   • get_endpoint_transitions() → [                       │       │
│  │       ('/api/home', '/api/products'),          ◄────────┼──┐    │
│  │       ('/api/products', '/api/cart'),                    │  │    │
│  │       ('/api/cart', '/api/checkout')                     │  │    │
│  │     ]                                                    │  │    │
│  │   • append_call()                                        │  │    │
│  └─────────────────────────────────────────────────────────┘  │    │
└─────────────────────────────────────────────────────────────────────┘
                                  │                              │    │
                                  │ Collect sessions             │    │
                                  ▼                              │    │
┌─────────────────────────────────────────────────────────────────────┐
│                         Dataset Object                              │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ Dataset(                                                 │       │
│  │   name="training_data",                                  │       │
│  │   sessions=[session1, session2, ...]                     │       │
│  │ )                                                        │       │
│  │                                                          │       │
│  │ Properties:                                              │       │
│  │   • total_calls → 120                                    │       │
│  │   • num_unique_users → 15                                │       │
│  │   • unique_endpoints → {'/api/home', '/api/products'...} │       │
│  │                                                          │       │
│  │ Methods:                                                 │       │
│  │   • get_all_sequences() → [[seq1], [seq2], ...]          │       │
│  │   • count_endpoint_occurrences() → {'/api/home': 50}     │       │
│  │   • split(0.8) → (train_dataset, test_dataset)           │       │
│  │   • save_to_parquet() / load_from_parquet()              │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                                  │                              │    │
                    ┌─────────────┴─────────────┐                │    │
                    ▼                           ▼                │    │
         ┌────────────────────┐      ┌────────────────────┐     │    │
         │  Train Dataset     │      │  Test Dataset      │     │    │
         │  (80% of data)     │      │  (20% of data)     │     │    │
         └────────────────────┘      └────────────────────┘     │    │
                    │                                            │    │
                    │ Extract sequences & transitions            │    │
                    ▼                                            │    │
         ┌────────────────────────────────────────────┐         │    │
         │  Markov Chain Training Data                │ ◄───────┘    │
         │                                            │              │
         │  Transitions:                              │              │
         │    ('/api/home', '/api/products'): 25      │ ◄────────────┘
         │    ('/api/products', '/api/cart'): 18      │  Used for
         │    ('/api/cart', '/api/checkout'): 12      │  Markov Chain
         │    ...                                     │  Training!
         │                                            │
         │  Sequences:                                │
         │    ['/api/home', '/api/products', ...]     │
         │    ['/api/home', '/api/search', ...]       │
         │    ...                                     │
         └────────────────────────────────────────────┘
"""

DATA_FLOW = """
Data Flow for Markov Chain Training
====================================

1. Raw Logs → APICall
   ────────────────────
   Raw JSON/dict → APICall object
   - Validate fields
   - Convert timestamps
   - Extract metadata

2. APICall → Session
   ──────────────────
   Group by session_id
   - Sort by timestamp
   - Calculate durations
   - Track user behavior

3. Session → Transitions
   ──────────────────────
   Extract consecutive pairs
   - State: current endpoint
   - Next state: next endpoint
   - Build transition matrix

4. Sessions → Dataset
   ───────────────────
   Collect all sessions
   - Aggregate statistics
   - Count frequencies
   - Analyze patterns

5. Dataset → Training Data
   ────────────────────────
   Split and prepare
   - Train/test split
   - Extract sequences
   - Build transition counts
   - Feed to Markov model

Example Pipeline:
─────────────────

Raw Log:
  {"id": "1", "path": "/api/users/123", ...}
           │
           ▼
APICall:
  APICall(endpoint="/api/users/123", ...)
  .get_service_name() → "users"
           │
           ▼
Session:
  Session(calls=[call1, call2, call3])
  .get_endpoint_transitions() → [
    ("/api/home", "/api/users/123"),
    ("/api/users/123", "/api/products")
  ]
           │
           ▼
Dataset:
  Dataset(sessions=[sess1, sess2, ...])
  .get_all_sequences() → [[seq1], [seq2]]
  .split(0.8) → train, test
           │
           ▼
Markov Model Input:
  Transitions: {
    ("/api/home", "/api/users/123"): 0.35,
    ("/api/users/123", "/api/products"): 0.62,
    ...
  }
"""

INTEGRATION_EXAMPLE = """
Integration with Markov Chain Module
=====================================

from preprocessing.models import Dataset
from markov.model import MarkovChain  # Your future implementation

# 1. Load dataset
dataset = Dataset.load_from_parquet("data/training.parquet")

# 2. Extract training data
all_transitions = []
for session in dataset.sessions:
    transitions = session.get_endpoint_transitions()
    all_transitions.extend(transitions)

# 3. Build transition counts
from collections import Counter
transition_counts = Counter(all_transitions)

# transition_counts = {
#   ('/api/home', '/api/products'): 50,
#   ('/api/products', '/api/cart'): 30,
#   ...
# }

# 4. Train Markov model
markov_model = MarkovChain()
markov_model.train(transition_counts)

# 5. Predict next endpoint
current_state = "/api/products"
next_endpoint = markov_model.predict(current_state)
# → "/api/cart" (with 0.6 probability)

# 6. Preemptive caching decision
if markov_model.get_probability(current_state, next_endpoint) > 0.5:
    cache.prefetch(next_endpoint)
"""

if __name__ == "__main__":
    print(ARCHITECTURE)
    print("\n\n")
    print(DATA_FLOW)
    print("\n\n")
    print(INTEGRATION_EXAMPLE)


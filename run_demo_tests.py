import argparse
import logging
import math
import random
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# ANSI + output formatting
# ----------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

PASS_ICON = "✅"
FAIL_ICON = "❌"
SKIP_ICON = "⚠️"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ----------------------------
# Minimal standalone components
# ----------------------------
class InMemoryBackend:
    def __init__(self):
        self.store: Dict[str, object] = {}

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value: object):
        self.store[key] = value

    def delete(self, key: str):
        if key in self.store:
            del self.store[key]


class RedisBackendMock:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.store: Dict[str, object] = {}
        if self.should_fail:
            raise ConnectionError("Redis unavailable (simulated)")

    def get(self, key: str):
        if self.should_fail:
            raise ConnectionError("Redis read failed")
        return self.store.get(key)

    def set(self, key: str, value: object):
        if self.should_fail:
            raise ConnectionError("Redis write failed")
        self.store[key] = value

    def delete(self, key: str):
        if self.should_fail:
            raise ConnectionError("Redis delete failed")
        self.store.pop(key, None)


class CacheManager:
    def __init__(self, capacity: int = 5, redis_fail: bool = False):
        self.logger = logging.getLogger("cache_manager")
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.lru = OrderedDict()
        self.backend = self._init_backend(redis_fail)

    def _init_backend(self, redis_fail: bool):
        try:
            return RedisBackendMock(should_fail=redis_fail)
        except Exception as e:
            self.logger.warning(
                "Redis unavailable, falling back to InMemoryBackend: %s", e
            )
            return InMemoryBackend()

    def _safe_get(self, key: str):
        try:
            return self.backend.get(key)
        except Exception as e:
            self.logger.warning("Backend get failed, switching to InMemoryBackend: %s", e)
            self.backend = InMemoryBackend()
            return self.backend.get(key)

    def _safe_set(self, key: str, value: object):
        try:
            self.backend.set(key, value)
        except Exception as e:
            self.logger.warning("Backend set failed, switching to InMemoryBackend: %s", e)
            self.backend = InMemoryBackend()
            self.backend.set(key, value)

    def process_request(self, key: str) -> bool:
        val = self._safe_get(key)
        if val is not None:
            self.hits += 1
            self._touch_lru(key)
            return True
        self.misses += 1
        self.cache_key(key, f"value:{key}")
        return False

    def cache_key(self, key: str, value: object):
        if key in self.lru:
            self.lru.pop(key)
        elif len(self.lru) >= self.capacity:
            old_key, _ = self.lru.popitem(last=False)
            self.backend.delete(old_key)
        self._safe_set(key, value)
        self.lru[key] = True

    def prefetch(self, keys: List[str]):
        for k in keys:
            self.cache_key(k, f"value:{k}")

    def evict_lru(self):
        if self.lru:
            k, _ = self.lru.popitem(last=False)
            self.backend.delete(k)

    def _touch_lru(self, key: str):
        if key in self.lru:
            self.lru.pop(key)
        self.lru[key] = True

    @property
    def occupancy(self) -> int:
        return len(self.lru)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class FirstOrderMarkovPredictor:
    def __init__(self, vocabulary: List[str]):
        self.vocab = vocabulary
        self.n = len(vocabulary)
        self.idx = {v: i for i, v in enumerate(vocabulary)}
        self.counts = np.ones((self.n, self.n), dtype=float)  # Laplace smoothing
        self.matrix = self._normalize(self.counts)

    def _normalize(self, m: np.ndarray) -> np.ndarray:
        row_sums = m.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return m / row_sums

    def update_sequence(self, seq: List[str]):
        if len(seq) < 2:
            return
        for a, b in zip(seq[:-1], seq[1:]):
            if a in self.idx and b in self.idx:
                self.counts[self.idx[a], self.idx[b]] += 1.0
        self.matrix = self._normalize(self.counts)

    def predict(self, current: Optional[str], top_k: int = 3) -> List[Tuple[str, float]]:
        if current is None or current not in self.idx:
            probs = np.ones(self.n, dtype=float) / self.n
        else:
            probs = self.matrix[self.idx[current]]
        order = np.argsort(probs)[::-1][:top_k]
        return [(self.vocab[i], float(probs[i])) for i in order]

    def predict_from_sequence(self, seq: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        if not seq:
            probs = np.ones(self.n, dtype=float) / self.n
            order = np.argsort(probs)[::-1][:top_k]
            return [(self.vocab[i], float(probs[i])) for i in order]
        return self.predict(seq[-1], top_k=top_k)


class SecondOrderMarkovPredictor:
    def __init__(self, vocabulary: List[str], first_order: FirstOrderMarkovPredictor):
        self.vocab = vocabulary
        self.n = len(vocabulary)
        self.idx = {v: i for i, v in enumerate(vocabulary)}
        self.first_order = first_order
        self.pair_counts: Dict[Tuple[str, str], np.ndarray] = {}

    def update_sequence(self, seq: List[str]):
        if len(seq) < 3:
            return
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
            if a in self.idx and b in self.idx and c in self.idx:
                key = (a, b)
                if key not in self.pair_counts:
                    self.pair_counts[key] = np.ones(self.n, dtype=float)
                self.pair_counts[key][self.idx[c]] += 1.0
        self.first_order.update_sequence(seq)

    def predict(self, prev2: str, prev1: str, top_k: int = 3) -> List[Tuple[str, float]]:
        key = (prev2, prev1)
        if key not in self.pair_counts:
            return self.first_order.predict(prev1, top_k=top_k)
        probs = self.pair_counts[key]
        probs = probs / probs.sum()
        order = np.argsort(probs)[::-1][:top_k]
        return [(self.vocab[i], float(probs[i])) for i in order]


class ReplayBuffer:
    def __init__(self, capacity: int = 1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return None
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idxs]

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, in_dim: int = 60, out_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim=60, action_dim=7):
        self.action_dim = action_dim
        self.online = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.sync_target_network()
        self.optim = optim.Adam(self.online.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.05
        self.replay = ReplayBuffer(2000)

    def select_action(self, state: np.ndarray) -> int:
        clean = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            s = torch.tensor(clean).unsqueeze(0)
            q = self.online(s)
            return int(torch.argmax(q, dim=1).item())

    def sync_target_network(self):
        self.target.load_state_dict(self.online.state_dict())

    def decay_epsilon_step(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self, batch_size=32):
        batch = self.replay.sample(batch_size)
        if batch is None:
            return None

        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)

        q_pred = self.online(states).gather(1, actions)

        with torch.no_grad():
            next_online_actions = torch.argmax(self.online(next_states), dim=1, keepdim=True)
            next_q_target = self.target(next_states).gather(1, next_online_actions)
            target_q = rewards + self.gamma * (1.0 - dones) * next_q_target

        loss = nn.functional.mse_loss(q_pred, target_q)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optim.step()
        return float(loss.item())


class CachingEnv:
    DO_NOTHING = 0
    CACHE_CURRENT = 1
    PREFETCH_TOP1 = 2
    PREFETCH_TOP3 = 3
    EVICT_LRU = 4
    EVICT_LOW_PROB = 5
    HYBRID = 6

    def __init__(self, cache_manager: CacheManager, predictor: FirstOrderMarkovPredictor):
        self.cache = cache_manager
        self.predictor = predictor
        self.session: List[str] = []
        self.ptr = 0

    def reset(self, session: Optional[List[str]] = None):
        self.session = session if session is not None else []
        self.ptr = 0
        return np.zeros(60, dtype=np.float32)

    def step(self, action: int):
        if len(self.session) == 0:
            raise ValueError("Session is empty; cannot step()")

        endpoint = self.session[self.ptr]
        hit = self.cache.process_request(endpoint)

        if action == self.CACHE_CURRENT:
            self.cache.cache_key(endpoint, f"value:{endpoint}")
        elif action == self.PREFETCH_TOP1:
            preds = self.predictor.predict(endpoint, top_k=1)
            self.cache.prefetch([preds[0][0]])
        elif action == self.PREFETCH_TOP3:
            preds = self.predictor.predict(endpoint, top_k=3)
            self.cache.prefetch([p[0] for p in preds])
        elif action == self.EVICT_LRU:
            self.cache.evict_lru()
        elif action == self.HYBRID:
            preds = self.predictor.predict(endpoint, top_k=1)
            self.cache.prefetch([preds[0][0]])
            self.cache.evict_lru()

        reward = 1.0 if hit else -1.0
        self.ptr += 1
        done = self.ptr >= len(self.session)
        next_state = np.zeros(60, dtype=np.float32)
        return next_state, reward, done, {"hit": hit}


# ----------------------------
# Test harness
# ----------------------------
class SkipTest(Exception):
    pass


@dataclass
class TestCase:
    test_id: str
    name: str
    category: str  # "positive" or "negative"
    fn: Callable[[], Optional[str]]


def make_world():
    vocab = [
        "login",
        "profile",
        "orders",
        "checkout",
        "search",
        "cart",
        "home",
        "apiA",
        "apiB",
        "apiC",
    ]
    predictor = FirstOrderMarkovPredictor(vocab)
    seq = ["login", "profile", "orders", "checkout"] * 20 + ["home", "search", "cart"] * 10
    predictor.update_sequence(seq)
    second = SecondOrderMarkovPredictor(vocab, predictor)
    second.update_sequence(seq)
    cache = CacheManager(capacity=4, redis_fail=False)
    env = CachingEnv(cache, predictor)
    agent = DQNAgent()
    return vocab, predictor, second, cache, env, agent


# ----------------------------
# Positive tests P01-P15
# ----------------------------
def test_p01():
    _, predictor, _, _, _, _ = make_world()
    row_sums = predictor.matrix.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-6), "Rows do not sum to 1.0"
    return f"max row-sum error={float(np.max(np.abs(row_sums-1.0))):.2e}"


def test_p02():
    _, predictor, _, _, _, _ = make_world()
    preds = predictor.predict("login", top_k=3)
    assert len(preds) == 3, "Did not return exactly 3 predictions"
    probs = [p[1] for p in preds]
    assert probs[0] >= probs[1] >= probs[2], "Probabilities are not descending"
    return f"preds={preds}"


def test_p03():
    _, _, _, cache, _, _ = make_world()
    for _ in range(10):
        cache.process_request("login")
    assert cache.hit_rate > 0.8, f"Hit rate too low: {cache.hit_rate:.2f}"
    return f"hit_rate={cache.hit_rate:.2f}"


def test_p04():
    _, _, _, _, _, agent = make_world()
    state = np.random.randn(60).astype(np.float32)
    action = agent.select_action(state)
    assert 0 <= action <= 6, f"Action out of range: {action}"
    return f"action={action}"


def test_p05():
    q = QNetwork()
    x = torch.randn(8, 60)
    out = q(x)
    assert tuple(out.shape) == (8, 7), f"Unexpected shape: {tuple(out.shape)}"
    return f"shape={tuple(out.shape)}"


def test_p06():
    _, _, _, _, _, agent = make_world()
    with torch.no_grad():
        for p in agent.online.parameters():
            p.add_(torch.randn_like(p) * 0.001)
    agent.sync_target_network()
    for p1, p2 in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.allclose(p1, p2), "Target and online parameters differ after sync"
    return "target sync OK"


def test_p07():
    rb = ReplayBuffer(capacity=100)
    for i in range(20):
        rb.push((np.zeros(60), i % 7, 1.0, np.ones(60), 0.0))
    sample = rb.sample(8)
    assert sample is not None and len(sample) == 8, "Replay sample failed"
    return f"buffer={len(rb)}, sample={len(sample)}"


def test_p08():
    _, _, _, _, _, agent = make_world()
    start = agent.epsilon
    for _ in range(3):
        agent.decay_epsilon_step()
    assert agent.epsilon < start, "Epsilon did not decay"
    return f"epsilon {start:.3f}->{agent.epsilon:.3f}"


def test_p09():
    vocab, predictor, _, _, _, _ = make_world()
    a, b = "login", "profile"
    before = predictor.predict(a, top_k=len(vocab))
    pb = dict(before)[b]
    predictor.update_sequence(["login", "profile"] * 20)
    after = predictor.predict(a, top_k=len(vocab))
    pa = dict(after)[b]
    assert pa > pb, "Transition probability did not increase after online update"
    return f"P({b}|{a}) {pb:.4f}->{pa:.4f}"


def test_p10():
    _, predictor, second, _, _, _ = make_world()
    # unseen pair should fallback to first-order of prev1
    p2 = second.predict("apiB", "apiC", top_k=3)
    p1 = predictor.predict("apiC", top_k=3)
    assert [x[0] for x in p2] == [x[0] for x in p1], "Second-order did not fallback properly"
    return f"fallback_preds={p2}"


def test_p11():
    _, predictor, _, cache, env, _ = make_world()
    cache.cache_key("login", 1)
    cache.cache_key("profile", 2)
    cache.cache_key("orders", 3)
    cache.cache_key("checkout", 4)
    before = cache.occupancy
    env.reset(["login"])
    env.step(CachingEnv.EVICT_LRU)
    after = cache.occupancy
    assert after < before, "Occupancy did not drop after EVICT_LRU"
    return f"occupancy {before}->{after}"


def test_p12():
    _, predictor, _, cache, env, _ = make_world()
    workflow = ["login", "profile", "orders", "checkout"]
    # Warm-up
    for _ in range(3):
        env.reset(workflow)
        done = False
        while not done:
            _, _, done, _ = env.step(CachingEnv.CACHE_CURRENT)
    # Measure
    pre_hits = cache.hits
    pre_total = cache.hits + cache.misses
    for _ in range(5):
        env.reset(workflow)
        done = False
        while not done:
            _, _, done, _ = env.step(CachingEnv.CACHE_CURRENT)
    d_hits = cache.hits - pre_hits
    d_total = (cache.hits + cache.misses) - pre_total
    hit_rate = d_hits / d_total if d_total else 0.0
    assert hit_rate > 0.8, f"Workflow hit rate too low: {hit_rate:.2f}"
    return f"workflow_hit_rate={hit_rate:.2f}"


def test_p13():
    # A/B simulation using fixed timing model
    workflow = ["login", "profile", "orders", "checkout"] * 50

    def run(passive: bool):
        cache = CacheManager(capacity=4, redis_fail=False)
        total_ms = 0.0
        for ep in workflow:
            if passive:
                hit = False  # passive baseline: no proactive or reactive caching
            else:
                hit = cache.process_request(ep)
                cache.cache_key(ep, f"value:{ep}")  # heuristic
            total_ms += 20.0 if hit else 35.0
        return total_ms / len(workflow)

    baseline = run(passive=True)
    heuristic = run(passive=False)
    assert (
        heuristic < baseline
    ), f"Heuristic latency {heuristic:.2f} not lower than baseline {baseline:.2f}"
    return f"{baseline:.2f}ms->{heuristic:.2f}ms"


def test_p14():
    _, _, _, cache, env, _ = make_world()
    cache.cache_key("login", "v")
    env.reset(["login", "apiA"])
    _, r1, _, info1 = env.step(CachingEnv.DO_NOTHING)  # hit expected
    _, r2, _, info2 = env.step(CachingEnv.DO_NOTHING)  # miss expected
    assert info1["hit"] and r1 > 0, "Hit did not produce positive reward"
    assert (not info2["hit"]) and r2 < 0, "Miss did not produce negative reward"
    return f"rewards hit={r1}, miss={r2}"


def test_p15():
    cache = CacheManager(capacity=3, redis_fail=True)
    assert isinstance(cache.backend, InMemoryBackend), "Did not fallback to InMemoryBackend"
    ok1 = cache.process_request("login")
    ok2 = cache.process_request("login")
    assert (not ok1) and ok2, "Fallback cache did not continue serving requests"
    return f"hit_rate={cache.hit_rate:.2f}"


# ----------------------------
# Negative tests N01-N08
# ----------------------------
def test_n01():
    vocab, predictor, _, _, _, _ = make_world()
    preds = predictor.predict_from_sequence([], top_k=3)
    assert len(preds) == 3, "Uniform fallback did not return top-3"
    probs = [p for _, p in preds]
    assert all(abs(p - (1 / len(vocab))) < 1e-6 for p in probs), "Not uniform distribution"
    return f"uniform_prob={probs[0]:.4f}"


def test_n02():
    _, predictor, _, _, _, _ = make_world()
    preds = predictor.predict("unknown_endpoint_xyz", top_k=3)
    assert len(preds) == 3, "Unknown endpoint handling failed"
    return f"preds={preds}"


def test_n03():
    rb = ReplayBuffer(capacity=10)
    rb.push((np.zeros(60), 0, 0.0, np.zeros(60), 0.0))
    out = rb.sample(8)
    assert out is None, "Expected None when insufficient samples"
    return "safe underfilled sampling"


def test_n04():
    _, _, _, _, _, agent = make_world()
    state = np.random.randn(60).astype(np.float32)
    state[5] = np.nan
    action = agent.select_action(state)
    assert 0 <= action <= 6, "Agent crashed or produced invalid action with NaN state"
    q = agent.online(torch.tensor(np.nan_to_num(state), dtype=torch.float32).unsqueeze(0))
    assert torch.isfinite(q).all(), "Q-values not finite after NaN handling"
    return f"action={action}"


def test_n05():
    cache = CacheManager(capacity=2, redis_fail=False)
    cache.cache_key("k1", "first")
    cache.cache_key("k1", "second")
    val = cache.backend.get("k1")
    assert val == "second", "Key collision overwrite failed"
    return f"value={val}"


def test_n06():
    logger = logging.getLogger("cache_manager")
    logger.setLevel(logging.WARNING)

    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = ListHandler()
    logger.addHandler(handler)
    try:
        cache = CacheManager(capacity=2, redis_fail=True)
        cache.process_request("apiA")
        cache.process_request("apiA")
    finally:
        logger.removeHandler(handler)

    assert isinstance(cache.backend, InMemoryBackend), "No fallback backend after Redis failure"
    assert any("falling back" in m.lower() for m in records), "Warning log about fallback not found"
    return "fallback warning logged"


def test_n07():
    _, _, _, _, _, agent = make_world()
    # Fill replay with extreme rewards
    for i in range(64):
        s = np.random.randn(60).astype(np.float32)
        ns = np.random.randn(60).astype(np.float32)
        rew = 1000.0 if i % 2 == 0 else -1000.0
        agent.replay.push((s, i % 7, rew, ns, 0.0))
    loss = agent.train_step(batch_size=32)
    assert loss is not None and math.isfinite(loss), f"Loss is not finite: {loss}"
    with torch.no_grad():
        qvals = agent.online(torch.randn(4, 60))
    assert torch.isfinite(qvals).all(), "Q-values exploded to non-finite"
    return f"finite_loss={loss:.4f}"


def test_n08():
    _, predictor, _, cache, env, _ = make_world()
    _ = env.reset([])  # should succeed
    try:
        env.step(CachingEnv.DO_NOTHING)
    except ValueError as e:
        assert "empty" in str(e).lower(), "ValueError message not clear"
        return "clear ValueError on empty session"
    raise AssertionError("Expected ValueError on step() with zero-length session")


# ----------------------------
# Runner
# ----------------------------
def build_tests() -> List[TestCase]:
    return [
        TestCase("P01", "Markov normalisation", "positive", test_p01),
        TestCase("P02", "Markov top-3 prediction order", "positive", test_p02),
        TestCase("P03", "Cache hit rate improves on repeats", "positive", test_p03),
        TestCase("P04", "DQN action validity (0-6)", "positive", test_p04),
        TestCase("P05", "Q-network output shape", "positive", test_p05),
        TestCase("P06", "Target network sync", "positive", test_p06),
        TestCase("P07", "Replay buffer store/sample", "positive", test_p07),
        TestCase("P08", "Epsilon decay", "positive", test_p08),
        TestCase("P09", "Markov online update changes probs", "positive", test_p09),
        TestCase("P10", "Second-order fallback to first-order", "positive", test_p10),
        TestCase("P11", "Evict LRU reduces occupancy", "positive", test_p11),
        TestCase("P12", "Workflow >80% hit rate after warm-up", "positive", test_p12),
        TestCase("P13", "A/B heuristic latency improvement", "positive", test_p13),
        TestCase("P14", "Reward sign for hit/miss", "positive", test_p14),
        TestCase("P15", "Graceful degradation to in-memory backend", "positive", test_p15),
        TestCase("N01", "Empty sequence uniform fallback", "negative", test_n01),
        TestCase("N02", "Unknown endpoint graceful handling", "negative", test_n02),
        TestCase("N03", "Underfilled replay sampling safe", "negative", test_n03),
        TestCase("N04", "NaN state handling", "negative", test_n04),
        TestCase("N05", "Cache key collision overwrite", "negative", test_n05),
        TestCase("N06", "Redis failure fallback + warning", "negative", test_n06),
        TestCase("N07", "Extreme rewards remain finite", "negative", test_n07),
        TestCase("N08", "Zero-length session clear error", "negative", test_n08),
    ]


def print_banner(title: str):
    print(f"\n{CYAN}{'='*14} {title} {'='*14}{RESET}")


def run_tests(selected_category: Optional[str], verbose: bool):
    tests = build_tests()
    if selected_category in ("positive", "negative"):
        tests = [t for t in tests if t.category == selected_category]

    total = len(tests)
    passed = failed = skipped = 0
    start_all = time.perf_counter()

    if selected_category is None:
        print_banner("POSITIVE TESTS")
        print_banner("NEGATIVE TESTS")
    elif selected_category == "positive":
        print_banner("POSITIVE TESTS")
    elif selected_category == "negative":
        print_banner("NEGATIVE TESTS")

    for t in tests:
        t0 = time.perf_counter()
        try:
            detail = t.fn()
            dt = time.perf_counter() - t0
            passed += 1
            print(f"  {GREEN}{PASS_ICON} {t.test_id} | {t.name:<34} | {dt:.3f}s{RESET}")
            if verbose and detail:
                print(f"      {detail}")
        except SkipTest as e:
            dt = time.perf_counter() - t0
            skipped += 1
            print(f"  {YELLOW}{SKIP_ICON} {t.test_id} | {t.name:<34} | {dt:.3f}s{RESET}")
            if str(e):
                print(f"      {YELLOW}{str(e)}{RESET}")
        except Exception as e:
            dt = time.perf_counter() - t0
            failed += 1
            print(f"  {RED}{FAIL_ICON} {t.test_id} | {t.name:<34} | {dt:.3f}s{RESET}")
            print(f"      {RED}{type(e).__name__}: {e}{RESET}")

    total_time = time.perf_counter() - start_all
    passed_total_str = f"{passed}/{total}"

    print("\n╔══════════════════════════════╗")
    print(f"║  RESULTS: {passed_total_str:<17}║")
    print(f"║  Time: {total_time:<20.2f}s║")
    print("╚══════════════════════════════╝")

    return failed == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone demo test runner")
    parser.add_argument("--verbose", action="store_true", help="Print extra detail per passing test")
    parser.add_argument(
        "--filter",
        choices=["positive", "negative"],
        default=None,
        help="Run only one test group",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args()
    ok = run_tests(args.filter, args.verbose)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

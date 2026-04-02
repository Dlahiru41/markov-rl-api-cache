# 8.1 Chapter Overview

This chapter reports the testing and evaluation outcomes for the Markov-RL API cache system using only executed results collected in this session. Functional, non-functional, model, and benchmark-oriented evaluations were run using the repository’s existing test and script infrastructure.

# 8.2 Testing Criteria

The evaluation used four complementary test classes aligned to the architecture:

- HTTP gateway functional validation (FR-01 to FR-20) for proxy, cache, invalidation, observability, and health behavior.
- Non-functional validation (NFR-01 to NFR-08) for latency, concurrency, resilience, availability, and graceful degradation.
- Model validation for Markov prediction quality and DQN agent mechanics.
- Benchmark/performance runs for latency and throughput in component and end-to-end paths.

This selection was appropriate because the system combines gateway networking behavior, cache-state correctness, online prediction behavior, and RL-driven adaptation.

# 8.3 Model Testing

## 8.3.1 Markov Chain Evaluation

The Markov model evaluation suite was executed (`tests/model/test_model_evaluation.py`) and passed.

**Table 1: Markov Chain Evaluation Metrics (Executed)**

| Metric | Measured Value |
|---|---:|
| Top-1 accuracy | 0.0983 |
| Top-3 accuracy | 0.3027 |
| Top-5 accuracy | 0.5055 |
| Mean Reciprocal Rank (MRR) | 0.2921 |
| Coverage | 1.0000 |

**Table 2: First-Order vs Second-Order Markov Benchmark (Executed)**

| Model | Top-1 | Top-3 | Top-5 | MRR | Coverage |
|---|---:|---:|---:|---:|---:|
| First-order | 0.0938 | 0.2982 | 0.5276 | 0.2910 | 1.0000 |
| Second-order | 0.0921 | 0.2529 | 0.3920 | 0.2340 | 0.8325 |

The measured results indicated that the first-order variant produced better top-k accuracy and coverage than the second-order variant under the executed synthetic split.

## 8.3.2 DQN Agent Evaluation

DQN reward-convergence outputs requested in the problem statement (reward-per-episode trend, last-N Q-value variance, and training loss curve points from an end-to-end training run) were **not fully executable** in this session.

`scripts/train.py --episodes 60 --output results/ch8_train` terminated with:
- `ValueError: too many values to unpack (expected 2)` at environment reset in trainer episode execution.

**Table 3: DQN Evaluation Artifacts Status**

| Required Output | Execution Status | Evidence |
|---|---|---|
| Reward per episode (training run) | Not Executed | Training script failed at episode 0 |
| Q-value convergence stability (last N variance) | Not Executed | No completed episode trajectory |
| Training loss curve data points | Not Executed | `results/ch8_train/metrics.json` contains empty arrays |
| DQN model unit/mechanic checks | Executed | `tests/model/test_model_evaluation.py` passed (40 tests total) |

# 8.4 Benchmarking

The repository’s benchmark-style infrastructure was executed through:
- `tests/performance/test_latency.py`
- `tests/performance/test_throughput.py`
- `scripts/api_simulation_compare.py`
- `scripts/compare_baselines.py --episodes 30 --output results/ch8_baselines --save-json` (partial success; CSV produced, JSON export failed at final serialization step)

**Table 4: Component Performance Benchmarks (Executed)**

| Metric | p50 / Median | p95 | p99 | Notes |
|---|---:|---:|---:|---|
| Markov prediction latency | 3.1025 ms | 4.6176 ms | 5.2967 ms | From `tests/performance/test_latency.py` |
| Cache hit latency (in-memory backend) | 0.0999 ms | 0.1916 ms | 0.2156 ms | From `tests/performance/test_latency.py` |
| Full decision latency | 2.1153 ms | 2.5012 ms | 2.5527 ms | From `tests/performance/test_latency.py` |

**Table 5: Throughput Benchmarks (Executed)**

| Metric | Measured Throughput |
|---|---:|
| Markov prediction throughput (vocab=50) | 705 predictions/s |
| Markov prediction throughput (vocab=200) | 431 predictions/s |
| Markov prediction throughput (vocab=500) | 616 predictions/s |
| Markov prediction throughput (vocab=1000) | 839 predictions/s |
| Cache GET throughput | 4,327 ops/s |
| Cache SET throughput | 3,898 ops/s |
| Agent action selection throughput | 4,567 actions/s |
| Training-step throughput | 506 steps/s |
| Environment step throughput | 372 steps/s |

**Table 6: Scenario Comparison (a)–(d) Requested in Problem Statement**

| Scenario | Status | Measured Results |
|---|---|---|
| (a) No cache, no RL (pure proxy baseline) | Not Executed | No dedicated executed script in this session produced this isolated mode |
| (b) Cache only, static TTL (no RL) | Executed (approximation via `without_solution`) | Mean reward 758.458, cache hit rate 87.16%, success 98.06%, mean p95 latency 0.61 ms |
| (c) Cache + Markov prefetching | Partially Executed (static Markov baseline script family) | Static Markov (30 episodes): mean reward 424.86, mean hit rate 99.64%, cascade rate 0.00% |
| (d) Cache + RL-tuned TTL (full system) | Not Executed | Blocked by `scripts/train.py` runtime failure before training |

No direct comparison against published external API gateway benchmarks was executed in this session.

# 8.5 Further Evaluations

## 8.5.1 Prefetch Efficiency Deep-Dive

`scripts/api_simulation_compare.py` produced:
- without solution prefetch efficiency: 0.00%
- with solution prefetch efficiency: 0.00%

The executed heuristic policy used mostly `DO_NOTHING` with limited `CACHE_CURRENT`, and did not produce measurable prefetch gain in that run.

## 8.5.2 Cache Invalidation Correctness under Mutation

FR-05, FR-09, and FR-10 related tests passed in `tests/functional/test_functional_requirements.py`:
- mutation-triggered invalidation for POST/PUT/PATCH/DELETE,
- admin flush endpoint behavior,
- pattern invalidation behavior.

No failing assertion was recorded for these cases in the functional suite.

# 8.6 Results Discussion

The executed results showed:
- strong functional correctness in the implemented mocked gateway test harness (FR suite passed),
- mostly successful non-functional behavior with one critical latency miss (NFR-01),
- consistent Markov-model predictive behavior, with first-order outperforming second-order in this run,
- mixed performance outcomes: low-latency decision path but comparatively low throughput against aggressive in-test target thresholds.

The key bottleneck identified from executed NFR results was gateway-added p99 overhead in NFR-01.

# 8.7 Functional Testing

Functional suite executed: `tests/functional/test_functional_requirements.py`

- Passed: 76
- Failed: 0
- Functional pass rate: **100.00%**

All FR classes (FR-01 to FR-20) in the implemented functional suite passed in this execution.

# 8.8 Non-Functional Testing

Non-functional suite executed: `tests/nonfunctional/test_nfr.py`

- Passed: 22
- Failed: 1
- NFR suite pass rate: **95.65%**

## NFR-01 — Response Latency <50 ms Overhead

- **Objective:** Verify proxy overhead p95 target (<50 ms stated in requirement; test asserts p99 against 50 ms).
- **Method:** 100 proxied requests with mocked upstream.
- **Result:** p99 = 81.16 ms, mean = 36.56 ms (p50/p95 not emitted by this test).
- **Target:** <50 ms.
- **Pass/Fail:** **Fail**.
- **Discussion:** This was the only failing NFR test in the executed run.

## NFR-02 — Cache Hit Latency <10 ms

- **Objective:** Verify cache-hit response latency.
- **Method:** Cache warm-up then 200 hit requests.
- **Result:** p99 = 2.59 ms, mean = 2.29 ms (p50/p95 not emitted by this test).
- **Target:** p95 <10 ms (test enforced p99 <10 ms).
- **Pass/Fail:** **Pass**.
- **Discussion:** Measured hit latency remained well below the threshold.

## NFR-03 — ≥500 Concurrent Requests

- **Objective:** Validate concurrent request handling and error rate.
- **Method:** 500 concurrent requests through gateway test harness.
- **Result:** success=500, errors=0, success rate=100.00%.
- **Target:** error rate <1%.
- **Pass/Fail:** **Pass**.
- **Discussion:** No concurrency failures were observed in this test.

## NFR-04 — Redis Connection Pooling ≥50 Concurrent Ops

- **Objective:** Validate concurrent Redis operation stability.
- **Method:** 50 concurrent mock-Redis read/write/delete operations.
- **Result:** 0 operation exceptions in the executed assertion.
- **Target:** no connection errors/timeouts at 50 concurrent ops.
- **Pass/Fail:** **Pass**.
- **Discussion:** The test harness validated operation-level stability.

## NFR-05 — Background Thread Resilience

- **Objective:** Ensure background RL/scheduler/collector failures do not crash gateway.
- **Method:** Injected exceptions in RL hook, collector, session tracker, and Redis paths.
- **Result:** Requests still returned successful HTTP responses in assertions.
- **Target:** gateway remains operational.
- **Pass/Fail:** **Pass**.
- **Discussion:** Pytest emitted thread-exception warnings, but assertions passed.

## NFR-06 — Uptime ≥99.5%

- **Objective:** Validate availability under sustained normal requests.
- **Method:** 200 requests in normal operation test.
- **Result:** success=200/200 (100.00%).
- **Target:** ≥99.5%.
- **Pass/Fail:** **Pass**.
- **Discussion:** Availability exceeded target in this run.

## NFR-07 — Header Sanitization

- **Objective:** Verify hop-by-hop header stripping.
- **Method:** Forwarding tests with unsafe headers and safe-header controls.
- **Result:** Hop-by-hop headers stripped; safe headers preserved.
- **Target:** no hop-by-hop leakage upstream.
- **Pass/Fail:** **Pass**.
- **Discussion:** Sanitization behavior was validated by assertion.

## NFR-08 — Fault Tolerance / Graceful Degradation

- **Objective:** Verify graceful behavior when RL/Markov components fail.
- **Method:** Fault injection with RL crashes and degraded component scenarios.
- **Result:** Assertions for continued gateway operation and cache behavior passed.
- **Target:** no client-facing degradation to unstable behavior.
- **Pass/Fail:** **Pass**.
- **Discussion:** Test emitted expected thread-crash warnings while preserving response behavior.

# 8.10 Limitations of the Testing Process

1. Several requested outputs could not be fully executed due runtime/tooling constraints:
   - `scripts/train.py` failed before producing DQN training curves and convergence metrics.
   - Scenario (a) and scenario (d) in the requested benchmark matrix were not available as completed executed artifacts in this session.
2. Some tests used mocked Redis/upstream behavior rather than live service stacks.
3. Non-functional latency tests did not emit full p50/p95/p99 for all NFR cases.
4. Full repository pytest run showed unrelated pre-existing failures outside the targeted FR/NFR/model/performance suites.

# 8.11 Chapter Summary

The executed evidence demonstrated complete pass of the implemented functional suite (FR-01 to FR-20 test classes), strong non-functional stability with one latency regression (NFR-01), valid Markov-model behavior, and mixed benchmark outcomes. The RL training-evaluation path required for full DQN convergence reporting did not execute successfully in this session and remains an identified gap for final thesis-grade completion.

---

# APPENDIX A — Full Functional Test Case Table

**Table 7: Functional Test Cases (FR-01 to FR-20)**

| Test ID | FR ID | Description | Preconditions | Steps | Expected Result | Actual Result | Pass/Fail |
|---|---|---|---|---|---|---|---|
| TC-FR01 | FR-01 | HTTP method forwarding for GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS | Gateway test client + mocked upstream | Send each method through catch-all proxy | Method and response forwarded correctly | Class `TestFR01_ForwardHTTPMethods` assertions passed | Pass |
| TC-FR02 | FR-02 | Cache successful GET, reject non-2xx caching | Cache-enabled gateway fixture | Perform first and repeat GET; perform non-2xx GET | Hit/miss behavior correct; non-2xx not cached | `TestFR02_CacheGETResponses` passed | Pass |
| TC-FR03 | FR-03 | Cache-key generation rules | Cache key utilities available | Compare keys across query order/method/header variants | Deterministic and differentiated keys per rules | `TestFR03_CacheKeyGeneration` passed | Pass |
| TC-FR04 | FR-04 | Cache hit/miss statistics updates | Stats endpoint enabled | Trigger miss/hit patterns, read `/admin/stats` | Counters and rates update correctly | `TestFR04_CacheStatistics` passed | Pass |
| TC-FR05 | FR-05 | Mutation-triggered invalidation | Seeded cache entry | Perform POST/PUT/PATCH/DELETE then re-GET | Related cache invalidated and miss observed | `TestFR05_CacheInvalidationOnMutation` passed | Pass |
| TC-FR06 | FR-06 | Upstream timeout/connect error handling | Mocked upstream exceptions | Inject timeout/connect errors | 504/502 with error details; stats increment | `TestFR06_UpstreamErrorHandling` passed | Pass |
| TC-FR07 | FR-07 | Hop-by-hop header stripping | Header-capture upstream mock | Send host/transfer-encoding headers | Headers removed before forwarding | `TestFR07_HopByHopHeaderRemoval` passed | Pass |
| TC-FR08 | FR-08 | Health status endpoint behavior | Health/admin endpoints available | Query `/admin/health`; simulate degraded deps | Status payload reflects health/degraded state | `TestFR08_HealthStatus` passed | Pass |
| TC-FR09 | FR-09 | Cache flush endpoint | Seeded cache keys | Call `POST /admin/cache/flush` | Keys removed; idempotent response | `TestFR09_CacheFlush` passed | Pass |
| TC-FR10 | FR-10 | Pattern-based invalidation | Seeded multiple path keys | Call `POST /admin/cache/invalidate` with pattern | Matching keys invalidated, response fields present | `TestFR10_CacheInvalidateByPattern` passed | Pass |
| TC-FR11 | FR-11 | Markov/RL prefetch invocation smoke | Gateway + background thread hook | Trigger request and inspect async hook/stat field | Background invocation occurs; stats field exists | `TestFR11_MarkovPrefetch` passed | Pass |
| TC-FR12 | FR-12 | Prefetch statistics tracking | Stats endpoint + injected prefetched entry | Read stats and consume prefetched cache entry | Prefetch fields exist and usage updates | `TestFR12_PrefetchStatistics` passed | Pass |
| TC-FR13 | FR-13 | Async RL invocation non-blocking | Slow RL hook mock | Send request while RL hook sleeps | Client response not blocked | `TestFR13_AsyncRLHook` passed | Pass |
| TC-FR14 | FR-14 | API call collection/session recording behavior | Collector + stats available | Send requests and inspect counters/headers | Requests recorded and response succeeds | `TestFR14_APICallCollection` passed | Pass |
| TC-FR15 | FR-15 | Periodic training scheduler endpoints | Scheduler initialized in app state | Query `/scheduler/status`, trigger job endpoint | Scheduler endpoints respond correctly | `TestFR15_PeriodicTrainingJobs` passed | Pass |
| TC-FR16 | FR-16 | Session extraction from header/IP fallback | Session tracker active | Send with and without `x-session-id` | Session handling succeeds across cases | `TestFR16_SessionTracking` passed | Pass |
| TC-FR17 | FR-17 | Detailed health/metrics endpoint | Health monitor initialized | Call `/health/detailed` and `/health` | Structured health responses returned | `TestFR17_DetailedHealthMetrics` passed | Pass |
| TC-FR18 | FR-18 | Prefetched cache flag persistence | Direct cache payload injection + normal cache write | Verify prefetched flag and normal entry behavior | Prefetched flagged; normal entries unflagged | `TestFR18_PrefetchedFlagInCache` passed | Pass |
| TC-FR19 | FR-19 | x-request-id tracing handling | Request/response proxy flow | Call with and without `x-request-id` | Requests proceed and provided ID accepted | `TestFR19_RequestIDTracing` passed | Pass |
| TC-FR20 | FR-20 | Prometheus metrics endpoint availability | Metrics endpoint enabled | Call `GET /metrics` | 200 response with non-empty metrics payload | `TestFR20_PrometheusMetrics` passed | Pass |

# 🎯 Demo Scripts - Quick Reference Card

## Which Demo Should I Use?

```
┌─────────────────────────────────────────────────────────────┐
│  Need quick business overview? (5-10 min)                   │
│  → Use: ENTERPRISE_LIVE_DEMO.py                            │
│  → Best for: Executives, investors, quick pitches           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Need technical deep-dive? (10-15 min)                      │
│  → Use: ENTERPRISE_INTERACTIVE_DEMO.py                      │
│  → Best for: Engineers, architects, technical reviews       │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start Commands

```bash
# 1. Setup (one-time)
python setup_demo_dependencies.py

# 2. Verify everything works
python verify_demo.py

# 3. Run business demo
python ENTERPRISE_LIVE_DEMO.py

# 4. Run technical demo
python ENTERPRISE_INTERACTIVE_DEMO.py
```

---

## What Each Demo Shows

### ENTERPRISE_LIVE_DEMO.py
```
Section 1: Business Problem ($2.1M ROI)
Section 2: System Architecture
Section 3: Markov Prediction (Live)
Section 4: DQN Training (Live 20 episodes)
Section 5: Baseline Comparison (6 policies)
Section 6: Business Value & ROI
Section 7: Production Readiness
Section 8: Competitive Differentiation
Section 9: Strategic Vision
```
**Duration:** 5-10 minutes  
**Focus:** Business value, ROI, strategy

---

### ENTERPRISE_INTERACTIVE_DEMO.py
```
Section 1: Microservices Simulation (7 services)
Section 2: Comprehensive Baselines (13+ strategies)
Section 3: Live Benchmarking (20 episodes each)
Section 4: Policy Analysis (Why DQN wins)
Section 5: Production Deployment (5-step guide)
```
**Duration:** 10-15 minutes  
**Focus:** Technical depth, architecture, benchmarks

---

## Performance Comparison

| Metric | ENTERPRISE_LIVE_DEMO | ENTERPRISE_INTERACTIVE_DEMO |
|--------|---------------------|----------------------------|
| **Microservices** | Mentioned | Visualized architecture |
| **Baselines** | 6 policies | 13+ policies |
| **Training** | 20 episodes | 30 episodes |
| **Benchmarking** | Basic | Comprehensive |
| **Analysis** | Surface | Deep dive |
| **Deployment** | Brief | Detailed 5-step |
| **Best For** | Executives | Engineers |

---

## Caching Strategies Available

### Traditional (4)
- LRU (Least Recently Used)
- Adaptive LRU
- LFU (Least Frequently Used)
- Windowed LFU

### Predictive (3)
- Static Markov
- Inverse Static Markov
- Balanced Static Markov

### Hybrid (2)
- Epsilon-Random
- Biased Random

### Adaptive (2)
- Adaptive Heuristic
- Multi-Objective Adaptive

### Theoretical (2)
- Oracle (Upper Bound)
- Partial Oracle

### Our Solution (1)
- **DQN Agent** ✨

---

## Expected Performance

```
Hit Rate Rankings:
1. DQN Agent       80-90% ✨
2. Adaptive        75-85%
3. Balanced Markov 72-82%
4. Static Markov   70-80%
5. LFU             68-78%
6. LRU             65-75%
7. Random          20-30%
```

**DQN Advantage:** 25-40% better than LRU

---

## Documentation Map

```
QUICK_START_DEMO.md
├─ Basic setup instructions
└─ 3-step quick start

ADVANCED_DEMO_GUIDE.md
├─ Detailed walkthrough
├─ Customization options
└─ Troubleshooting

CACHING_STRATEGIES_COMPARISON.md
├─ All 13+ strategies explained
├─ Performance comparison tables
└─ Decision trees

ADVANCED_DEMO_SUMMARY.md
├─ Implementation overview
├─ What was built
└─ Key improvements

FIXES_APPLIED.md
├─ Original demo issues
└─ Solutions implemented

README_DEMO_FIXED.md
└─ Before/after comparison
```

---

## Troubleshooting Quick Fixes

### Error: ModuleNotFoundError
```bash
python setup_demo_dependencies.py
```

### Error: ImportError from baselines
```bash
# Already fixed in ENTERPRISE_INTERACTIVE_DEMO.py
# Uses simpler policies to avoid import issues
```

### Demo runs slowly
```python
# In demo file, reduce episodes:
for episode in range(10):  # Was 30
num_eval_episodes = 10     # Was 20
```

### Want to use real microservices
```bash
# Start orchestrator first:
cd simulator/services/ecommerce
python orchestrator.py start

# Then in demo, set:
use_real_services=True
```

---

## Key Files at a Glance

| File | Purpose | Size |
|------|---------|------|
| **ENTERPRISE_INTERACTIVE_DEMO.py** | Advanced technical demo | 740 lines |
| **ENTERPRISE_LIVE_DEMO.py** | Business-focused demo | 1,000+ lines |
| **setup_demo_dependencies.py** | Install dependencies | 150 lines |
| **verify_demo.py** | Test setup | 170 lines |
| **ADVANCED_DEMO_GUIDE.md** | Complete guide | 400+ lines |
| **CACHING_STRATEGIES_COMPARISON.md** | Strategy reference | 500+ lines |

---

## Command Cheat Sheet

```bash
# Setup
python setup_demo_dependencies.py   # Install dependencies
python verify_demo.py               # Verify everything works

# Run Demos
python ENTERPRISE_LIVE_DEMO.py           # Business demo
python ENTERPRISE_INTERACTIVE_DEMO.py    # Technical demo

# Check Syntax
python -m py_compile ENTERPRISE_*.py     # Validate syntax

# Test Baselines (Advanced)
python -m pytest tests/ -k baseline      # Test baseline policies
python compare_baselines.py              # Compare all baselines
python demo_baselines.py                 # Demo baseline usage
```

---

## Environment Variables (Optional)

```bash
# For advanced users
export CACHE_ENV_SEED=42              # Reproducible results
export CACHE_LOG_LEVEL=INFO           # Logging verbosity
export CACHE_BACKEND=redis            # Use Redis (vs memory)
export REDIS_HOST=localhost           # Redis connection
export REDIS_PORT=6379                # Redis port
```

---

## Decision Tree

```
What do you want to show?

Business Value & ROI
  └─> ENTERPRISE_LIVE_DEMO.py
      ├─ Section 1: Business Problem
      ├─ Section 6: ROI ($2.1M)
      └─ Section 8: Competitive Edge

Technical Architecture
  └─> ENTERPRISE_INTERACTIVE_DEMO.py
      ├─ Section 1: Microservices
      └─ Section 5: Deployment

Performance Benchmarks
  └─> ENTERPRISE_INTERACTIVE_DEMO.py
      ├─ Section 2: 13+ Baselines
      ├─ Section 3: Live Benchmarking
      └─ Section 4: Analysis

All of the Above
  └─> Run both demos:
      1. ENTERPRISE_LIVE_DEMO.py (business)
      2. ENTERPRISE_INTERACTIVE_DEMO.py (technical)
      Total: 20-25 minutes
```

---

## One-Liner Summaries

**ENTERPRISE_LIVE_DEMO.py:**  
"Business-focused demonstration showing $2.1M ROI and strategic value"

**ENTERPRISE_INTERACTIVE_DEMO.py:**  
"Technical deep-dive with microservices, 13+ baselines, and live benchmarking"

**Both Together:**  
"Complete presentation covering business value, technical excellence, and production deployment"

---

## Audience Guide

| Audience | Use This Demo | Focus On |
|----------|--------------|----------|
| **CEO/CFO** | LIVE_DEMO | Sections 1, 6 (ROI) |
| **CTO** | INTERACTIVE | Sections 1, 5 (Architecture, Deployment) |
| **VP Eng** | Both | All sections |
| **Principal Eng** | INTERACTIVE | Sections 2, 3, 4 (Baselines, Benchmarks) |
| **Investors** | LIVE_DEMO | Sections 1, 6, 9 (Problem, ROI, Vision) |
| **Architects** | INTERACTIVE | Sections 1, 5 (Microservices, Deployment) |

---

## Success Metrics

After running either demo, you should be able to explain:

✅ **Problem:** Why traditional caching fails  
✅ **Solution:** How RL + Markov works  
✅ **Performance:** 25-40% better than LRU  
✅ **Value:** $2.1M+ annual ROI  
✅ **Deployment:** 5-step production strategy  
✅ **Confidence:** Statistical proof via benchmarks  

**Ready to present?** Run the demo and impress your stakeholders! 🚀

---

## Need Help?

1. Check **ADVANCED_DEMO_GUIDE.md** for detailed instructions
2. Check **CACHING_STRATEGIES_COMPARISON.md** for strategy info
3. Check **FIXES_APPLIED.md** for troubleshooting
4. Run **verify_demo.py** to diagnose issues

Still stuck? Check the documentation files - they cover 95% of questions!

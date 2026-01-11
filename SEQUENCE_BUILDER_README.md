# 🎉 SequenceBuilder Module - Complete Implementation

## Quick Start

```python
from preprocessing.sequence_builder import SequenceBuilder
from preprocessing.models import Dataset

# Load your data
dataset = Dataset.load_from_parquet('sessions.parquet')

# Initialize builder with normalization
builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

# Extract sequences for Markov training
sequences = builder.build_sequences(dataset.sessions)

# Calculate transition probabilities
probs = builder.get_transition_probabilities(dataset.sessions)
```

## 📦 What's Included

### Core Module
- **preprocessing/sequence_builder.py** - Main implementation (450+ lines)

### Documentation (Choose Your Level)
1. **IMPLEMENTATION_COMPLETE.md** - Start here! Overview and status
2. **SEQUENCE_BUILDER_QUICK_REF.md** - Quick reference for developers
3. **preprocessing/SEQUENCE_BUILDER_GUIDE.md** - Comprehensive guide
4. **SEQUENCE_BUILDER_IMPLEMENTATION.md** - Technical details

### Examples & Tests
1. **demo_sequence_builder.py** - Full feature demonstration
2. **example_integration.py** - Integration with existing models
3. **test_sequence_builder.py** - Test suite
4. **test_user_validation.py** - User validation tests

## 🚀 Run Examples

```bash
# Comprehensive demonstration (recommended first run)
python demo_sequence_builder.py

# Integration examples
python example_integration.py

# Test suite
python test_sequence_builder.py
```

## 📚 Documentation Guide

| If you want to... | Read this |
|------------------|-----------|
| Get started quickly | SEQUENCE_BUILDER_QUICK_REF.md |
| Understand all features | preprocessing/SEQUENCE_BUILDER_GUIDE.md |
| See implementation details | SEQUENCE_BUILDER_IMPLEMENTATION.md |
| Check completion status | IMPLEMENTATION_COMPLETE.md |

## 🎯 Key Features

### 1. Endpoint Normalization ⭐
Converts endpoint variations into consistent patterns:
```python
"/API/Users/123/Profile/" → "/api/users/{id}/profile"
```

### 2. Sequence Extraction
Four different methods for different use cases:
- Basic sequences
- Labeled (history, next) pairs
- N-grams (bigrams, trigrams)
- Contextual sequences with metadata

### 3. Markov Chain Support
Direct support for training:
```python
# Train
train_probs = builder.get_transition_probabilities(train_sessions)

# Predict
next_endpoint = max(train_probs[current].items(), key=lambda x: x[1])[0]
```

### 4. Context-Aware Analysis
Includes user type, time of day, day type, session length:
```python
contextual = builder.build_contextual_sequences(sessions)
premium_seqs = [c.sequence for c in contextual if c.user_type == 'premium']
```

## ✅ Validation

All features tested and working:
- ✓ Endpoint normalization
- ✓ Sequence extraction (all 4 methods)
- ✓ Transition probability calculation
- ✓ Context metadata extraction
- ✓ Train/test splitting
- ✓ Statistics computation
- ✓ Integration with existing models

## 🔗 Integration

Works seamlessly with:
- **preprocessing.models** - APICall, Session, Dataset
- **src.markov** - Markov chain training
- **src.rl** - RL model development
- **evaluation** - Model evaluation

## 💡 Common Use Cases

### Use Case 1: Train Markov Chain
```python
builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)
train_sessions, test_sessions = builder.split_sequences(sessions, train_ratio=0.8)
train_probs = builder.get_transition_probabilities(train_sessions)
```

### Use Case 2: Evaluate Predictions
```python
test_labeled = builder.build_labeled_sequences(test_sessions)
for history, actual_next in test_labeled:
    predicted_next = model.predict(history)
    # Compare predicted_next with actual_next
```

### Use Case 3: Context-Aware Modeling
```python
contextual = builder.build_contextual_sequences(sessions)
premium_probs = builder.get_transition_probabilities(
    [s for s in sessions if s.user_type == 'premium']
)
```

## 📊 Performance

- Fast: O(n) complexity for most operations
- Memory efficient: Only stores unique endpoints and transitions
- Scalable: Tested with hundreds of sessions and thousands of calls

## 🛠️ Configuration

```python
SequenceBuilder(
    normalize_endpoints=True,    # Recommended: True
    min_sequence_length=2        # Recommended: 2 or higher
)
```

## 📖 Next Steps

1. **Read**: Start with `IMPLEMENTATION_COMPLETE.md`
2. **Run**: Execute `python demo_sequence_builder.py`
3. **Learn**: Review `preprocessing/SEQUENCE_BUILDER_GUIDE.md`
4. **Integrate**: Use `example_integration.py` as template
5. **Build**: Start building your Markov chain model!

## 🆘 Need Help?

- Quick syntax? → `SEQUENCE_BUILDER_QUICK_REF.md`
- Full guide? → `preprocessing/SEQUENCE_BUILDER_GUIDE.md`
- Examples? → Run the demo scripts
- Integration? → See `example_integration.py`

## 🎓 Learning Path

1. **Beginner**: Run `demo_sequence_builder.py` and read output
2. **Intermediate**: Review `SEQUENCE_BUILDER_QUICK_REF.md`
3. **Advanced**: Study `preprocessing/SEQUENCE_BUILDER_GUIDE.md`
4. **Expert**: Read `SEQUENCE_BUILDER_IMPLEMENTATION.md`

---

**Status**: ✅ Complete and Ready for Use

The SequenceBuilder module is fully implemented, tested, and documented. All features work correctly and are ready for production use in Markov chain training and RL model development.


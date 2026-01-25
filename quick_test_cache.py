"""Quick test for cache backend."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing imports...")
    from src.cache.backend import CacheBackend, CacheEntry, CacheStats, InMemoryBackend
    print("[OK] All imports successful")

    print("\nTesting InMemoryBackend...")
    cache = InMemoryBackend(max_size_bytes=1024*1024)
    print("[OK] InMemoryBackend created")

    print("\nTesting set/get...")
    cache.set("key1", b"hello world", ttl=60)
    value = cache.get("key1")
    assert value == b"hello world", f"Expected b'hello world', got {value}"
    print(f"[OK] Retrieved: {value}")

    print("\nTesting stats...")
    stats = cache.get_stats()
    print(f"[OK] Hit rate: {stats.hit_rate:.2f}")
    print(f"[OK] Utilization: {stats.utilization:.2%}")
    print(f"[OK] Stats: {stats.to_dict()}")

    print("\n" + "="*50)
    print("[SUCCESS] ALL QUICK TESTS PASSED!")
    print("="*50)

except Exception as e:
    print(f"\n[ERROR]: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


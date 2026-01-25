from src.cache.backend import InMemoryBackend

cache = InMemoryBackend(max_size_bytes=100)

print("Setting key1...")
cache.set('key1', b'A' * 30)
print(f"After key1: size={cache._stats.current_size_bytes}/{cache._max_size_bytes}, entries={cache._stats.current_entries}")
print(f"Keys: {list(cache._cache.keys())}")

print("\nSetting key2...")
cache.set('key2', b'B' * 30)
print(f"After key2: size={cache._stats.current_size_bytes}/{cache._max_size_bytes}, entries={cache._stats.current_entries}")
print(f"Keys: {list(cache._cache.keys())}")

print("\nSetting key3...")
cache.set('key3', b'C' * 30)
print(f"After key3: size={cache._stats.current_size_bytes}/{cache._max_size_bytes}, entries={cache._stats.current_entries}")
print(f"Keys: {list(cache._cache.keys())}")
print(f"Evictions: {cache._stats.evictions}")

print(f"\nkey1 exists: {cache.exists('key1')}")
print(f"key2 exists: {cache.exists('key2')}")
print(f"key3 exists: {cache.exists('key3')}")


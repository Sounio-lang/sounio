# stdlib/cache

Fixed-capacity LRU-style key-value cache.

## Architecture

- `pure/types.sio` - LruCache type
- `lib.sio` - Public API

## Storage Model

- Fixed capacity: 64 entries
- Keys stored as strings
- Values stored as i64
- Validity markers for slot tracking

## Capabilities

- cache_new - Create empty cache
- cache_put - Insert or update key-value pair
- cache_get - Retrieve value by key (returns -1 on miss)
- cache_size - Get current entry count
- cache_is_empty - Check if cache has entries

## Usage

```
use cache::lib

var c = cache_new()
cache_put(&! c, "user:1", 100)
cache_put(&! c, "user:2", 200)
let v = cache_get(&c, "user:1")  // 100
```

## Tests

`tests/stdlib/cache/test_cache_core.sio` (check-only, Madaros gate)
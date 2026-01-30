# Phase 1: Quick Wins - Status Report

## Overview
Phase 1 aims to implement four core modules for data structures and search algorithms:
1. `stdlib/collections/heap.sio` - Binary heap / priority queue
2. `stdlib/collections/bitset.sio` - Bit vector operations
3. `stdlib/search/basic.sio` - Binary search, linear search, bounds
4. `stdlib/search/pattern.sio` - String pattern matching

Additionally, the memory allocator modules from the session preparation are included:
- `stdlib/mem/arena.sio` - Bump allocator
- `stdlib/mem/pool.sio` - Object pool allocator
- `stdlib/mem/stack.sio` - LIFO allocator

## Completed

### ✅ stdlib/mem/ - Memory Allocators (~400 LOC)
**Status**: Complete and committed
- `arena.sio` (110 LOC): Bump allocator with O(1) allocation, batch deallocation
- `pool.sio` (170 LOC): Fixed-size object pool with free list, O(1) alloc/free
- `stack.sio` (157 LOC): LIFO allocator with marker-based reset
- Pattern: Uses raw pointers (malloc/realloc/free) and proper effect declarations
- Commits: 3 commits (from earlier session)

### ✅ stdlib/collections/heap.sio (~190 LOC)
**Status**: Complete and committed (commit d50c852)
- Binary min-heap using raw i64 pointers
- Core functions: `heap_new()`, `heap_push()`, `heap_pop()`, `heap_peek()`
- Functions use effects: `with Alloc, Panic, Div`
- Freestanding function pattern with namespace prefix
- Properly handles dynamic resizing with malloc/realloc

## Blocking Issues

### ❌ stdlib/collections/bitset.sio
**Status**: BLOCKED - Rust syntax not compatible with Sounio
- **Issue**: Uses `pub struct BitSet<T>` and `impl` blocks for methods
- **Error**: P0002 "Expected an expression"
- **Fix Needed**: Rewrite with:
  - Freestanding functions: `bitset_set()`, `bitset_get()`, etc.
  - Raw pointer-based storage like heap.sio
  - Type-specific versions (e.g., `bitset_u64_set()`) instead of generics

### ❌ stdlib/collections/btree.sio
**Status**: BLOCKED - Generic impl<K,V> syntax not supported
- **Issue**: Uses Rust-style B-tree with trait implementations
- **Fix Needed**: Same as bitset - convert to freestanding functions

### ❌ stdlib/collections/trie.sio
**Status**: BLOCKED - Generic trie<V> syntax not supported
- **Issue**: Uses impl<V> TrieNode pattern
- **Fix Needed**: Same as bitset - convert to freestanding functions

### ⚠️ stdlib/search/basic.sio
**Status**: BLOCKED - Generic function syntax issues
- **Issue**: `linear_search_by<T>()` and other generic functions with impl blocks
- **Error**: P0006 "Expected a pattern"
- **Fix Needed**: Provide only type-specific versions (i64, f64) without generics
- **Note**: Type-specific functions already exist (linear_search_i64, linear_search_f64)

### ⚠️ stdlib/search/pattern.sio
**Status**: UNKNOWN - Not yet tested
- **Likely Issue**: May have similar generic function issues
- **Fix Needed**: Verify and test for compilation

## Root Cause Analysis

The main blocker is **Sounio's lack of support for generic type parameters and impl blocks**:

1. **Generic syntax not supported**:
   - `pub struct Heap<T>` - Type parameters in struct definitions
   - `impl<T> Heap<T> { }` - impl blocks for structs
   - `fn search<T>()` - Generic function calls

2. **Method syntax not supported**:
   - `heap.push(item)` - Method call syntax
   - `&self` - self-reference in methods
   - `self.field` - self-field access

3. **Current Sounio strengths**:
   - Freestanding functions with namespace prefixes
   - Raw pointers and manual memory management
   - Proper effect system (Alloc, IO, Panic, etc.)
   - Type-specific function implementations

## Recommended Fixes

For each blocking module, follow the heap.sio pattern:

1. **Remove impl blocks** - Convert methods to freestanding functions
2. **Use type-specific versions** - `bitset_set_u64()`, `bitset_get_u64()`, etc.
3. **Replace generics** - Use raw pointers or array-based storage
4. **Add effect declarations** - Proper `with` clauses for operations
5. **Raw memory management** - Use malloc/realloc/free directly

Example conversion:
```sio
// BEFORE (doesn't work)
impl<T> BinaryHeap<T> {
    pub fn push(&mut self, item: T) { ... }
}

// AFTER (works in Sounio)
pub fn heap_push(h: &!BinaryHeap, item: i64) with Alloc, Panic, Div { ... }
```

## Effort Estimate

To complete Phase 1:
- bitset.sio: ~3-4 hours (250 LOC, straightforward conversion)
- btree.sio: ~5-6 hours (400 LOC, more complex algorithms)
- trie.sio: ~4-5 hours (250 LOC, string handling)
- search/basic.sio: ~2-3 hours (fix or remove generics)
- search/pattern.sio: ~2-3 hours (verify and test)

**Total**: ~16-21 hours focused implementation

## Next Steps

1. **Immediate**: Complete the conversion template for bitset.sio as proof-of-concept
2. **Short-term**: Apply the same pattern to btree and trie
3. **Validation**: Ensure all Phase 1 modules compile and can be tested
4. **Integration**: Update stdlib/collections/lib.sio and stdlib/search/lib.sio exports

## Learning for Future Phases

- **Phase 2 (Stats & Algorithms)**: Expect distributions and sorting to have similar issues
- **Phase 3 (Data Serialization)**: TOML/YAML/MsgPack may require significant rewrites
- **Phase 4-5 (Advanced structures)**: Generics are fundamental - will need careful design

## Files Modified

- `stdlib/collections/heap.sio` - ✅ Rewritten and tested
- `stdlib/mem/arena.sio` - ✅ Previously committed
- `stdlib/mem/pool.sio` - ✅ Previously committed
- `stdlib/mem/stack.sio` - ✅ Previously committed
- `PHASE_1_STATUS.md` - 📄 This file (new)

## Git Commits

```
d50c852 [stdlib] Rewrite collections/heap with proper Sounio syntax
```

(Plus 3 earlier commits for mem/ modules from previous session)

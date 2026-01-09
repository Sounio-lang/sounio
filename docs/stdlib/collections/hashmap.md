---
title: HashMap<K, V>
description: Hash-based key-value map collection
---

# HashMap<K, V>

`HashMap<K, V>` is a hash map implemented with quadratic probing. It provides fast O(1) average-case insertion, lookup, and removal of key-value pairs.

## Type Definition

```sio
pub struct HashMap<K, V> {
    buckets: Vec<Bucket<K, V>>,
    len: usize,
}
```

## Constructors

### new

```sio
pub fn new() -> HashMap<K, V>
where K: Hash + Eq
```

Creates a new empty `HashMap`.

**Example:**

```sio
let map: HashMap<String, i32> = HashMap::new()
```

### with_capacity

```sio
pub fn with_capacity(capacity: usize) -> HashMap<K, V> with Alloc
where K: Hash + Eq
```

Creates a `HashMap` with the specified capacity. The map will be able to hold at least `capacity` elements without reallocating.

**Parameters:**
- `capacity` - The minimum number of elements to allocate space for

**Example:**

```sio
let map: HashMap<String, i32> = HashMap::with_capacity(100)
```

## Methods

### Basic Properties

#### len

```sio
pub fn len(&self) -> usize
```

Returns the number of elements in the map.

**Example:**

```sio
var map: HashMap<&str, i32> = HashMap::new()
map.insert("a", 1)
map.insert("b", 2)
map.len()  // 2
```

#### is_empty

```sio
pub fn is_empty(&self) -> bool
```

Returns `true` if the map contains no elements.

**Example:**

```sio
let map: HashMap<&str, i32> = HashMap::new()
map.is_empty()  // true
```

#### capacity

```sio
pub fn capacity(&self) -> usize
```

Returns the number of elements the map can hold without reallocating.

### Insertion and Removal

#### insert

```sio
pub fn insert(&!self, key: K, value: V) -> Option<V> with Alloc
```

Inserts a key-value pair into the map. If the key already exists, the old value is replaced and returned.

**Parameters:**
- `key` - The key to insert
- `value` - The value to insert

**Returns:** `Some(old_value)` if the key existed, `None` otherwise.

**Example:**

```sio
var scores: HashMap<&str, i32> = HashMap::new()
scores.insert("Alice", 100)   // None
scores.insert("Bob", 85)      // None
scores.insert("Alice", 95)    // Some(100) - replaced
```

#### remove

```sio
pub fn remove(&!self, key: &K) -> Option<V>
```

Removes a key from the map, returning the value if the key was present.

**Parameters:**
- `key` - The key to remove

**Returns:** `Some(value)` if the key existed, `None` otherwise.

**Example:**

```sio
var map: HashMap<&str, i32> = HashMap::new()
map.insert("key", 42)
let removed = map.remove(&"key")  // Some(42)
let missing = map.remove(&"key")  // None
```

#### remove_entry

```sio
pub fn remove_entry(&!self, key: &K) -> Option<(K, V)>
```

Removes a key from the map, returning the key-value pair if present.

**Example:**

```sio
var map: HashMap<String, i32> = HashMap::new()
map.insert("key".to_string(), 42)
let entry = map.remove_entry(&"key".to_string())  // Some(("key", 42))
```

#### clear

```sio
pub fn clear(&!self)
```

Removes all elements from the map.

**Example:**

```sio
var map: HashMap<&str, i32> = HashMap::new()
map.insert("a", 1)
map.insert("b", 2)
map.clear()
map.is_empty()  // true
```

### Lookup

#### get

```sio
pub fn get(&self, key: &K) -> Option<&V>
```

Returns a reference to the value corresponding to the key.

**Parameters:**
- `key` - The key to look up

**Returns:** `Some(&value)` if found, `None` otherwise.

**Example:**

```sio
var map: HashMap<&str, i32> = HashMap::new()
map.insert("key", 42)

match map.get(&"key") {
    Some(value) => println("Found: " ++ value.to_string()),
    None => println("Not found"),
}
```

#### get_mut

```sio
pub fn get_mut(&!self, key: &K) -> Option<&!V>
```

Returns a mutable reference to the value corresponding to the key.

**Example:**

```sio
var map: HashMap<&str, i32> = HashMap::new()
map.insert("counter", 0)

if let Some(counter) = map.get_mut(&"counter") {
    *counter = *counter + 1
}
```

#### contains_key

```sio
pub fn contains_key(&self, key: &K) -> bool
```

Returns `true` if the map contains the specified key.

**Example:**

```sio
var map: HashMap<&str, i32> = HashMap::new()
map.insert("key", 42)

map.contains_key(&"key")      // true
map.contains_key(&"missing")  // false
```

### Index Access

Maps support indexing with `[&key]`:

```sio
var map: HashMap<&str, i32> = HashMap::new()
map.insert("key", 42)

let value = map[&"key"]  // 42
// map[&"missing"]       // panics!
```

**Panics:** If the key is not found.

### Entry API

The entry API provides a way to do in-place modifications efficiently.

#### entry

```sio
pub fn entry(&!self, key: K) -> Entry<K, V> with Alloc
```

Gets the entry for the given key for in-place manipulation.

**Example:**

```sio
var word_counts: HashMap<String, i32> = HashMap::new()

// Count word occurrences
for word in words {
    word_counts.entry(word.to_string())
        .and_modify(|count| *count = *count + 1)
        .or_insert(1);
}
```

### Entry Methods

#### or_insert

```sio
pub fn or_insert(self, default: V) -> &!V with Alloc
```

Ensures a value is in the entry by inserting the default if empty.

**Example:**

```sio
var map: HashMap<&str, Vec<i32>> = HashMap::new()
map.entry("numbers").or_insert(Vec::new()).push(1)
map.entry("numbers").or_insert(Vec::new()).push(2)
// map["numbers"] = [1, 2]
```

#### or_insert_with

```sio
pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &!V with Alloc
```

Ensures a value is in the entry by inserting the result of the closure if empty.

**Example:**

```sio
var map: HashMap<&str, String> = HashMap::new()
map.entry("greeting")
    .or_insert_with(|| {
        expensive_computation()
    });
```

#### or_default

```sio
pub fn or_default(self) -> &!V with Alloc
where V: Default
```

Ensures a value is in the entry by inserting the default value if empty.

**Example:**

```sio
var map: HashMap<&str, i32> = HashMap::new()
let counter = map.entry("count").or_default()  // 0
```

#### and_modify

```sio
pub fn and_modify<F: FnOnce(&!V)>(self, f: F) -> Self
```

Modifies an existing entry value before potentially inserting.

**Example:**

```sio
var map: HashMap<&str, i32> = HashMap::new()
map.insert("count", 0)

map.entry("count")
    .and_modify(|n| *n = *n + 1)
    .or_insert(0);
// map["count"] = 1
```

#### key

```sio
pub fn key(&self) -> &K
```

Returns a reference to the key.

### OccupiedEntry Methods

For entries where the key exists:

#### get / get_mut

```sio
pub fn get(&self) -> &V
pub fn get_mut(&!self) -> &!V
```

Access the value.

#### into_mut

```sio
pub fn into_mut(self) -> &'a !V
```

Converts to a mutable reference to the value.

#### insert

```sio
pub fn insert(&!self, value: V) -> V
```

Replaces the value and returns the old one.

#### remove / remove_entry

```sio
pub fn remove(self) -> V
pub fn remove_entry(self) -> (K, V)
```

Removes the entry.

### VacantEntry Methods

For entries where the key doesn't exist:

#### key / into_key

```sio
pub fn key(&self) -> &K
pub fn into_key(self) -> K
```

Access or consume the key.

#### insert

```sio
pub fn insert(self, value: V) -> &'a !V
```

Inserts a value and returns a mutable reference to it.

### Iteration

#### keys

```sio
pub fn keys(&self) -> Keys<K, V>
```

Returns an iterator over the keys.

**Example:**

```sio
let map = create_map()  // {a: 1, b: 2, c: 3}

for key in map.keys() {
    println(key)
}
```

#### values

```sio
pub fn values(&self) -> Values<K, V>
```

Returns an iterator over the values.

**Example:**

```sio
for value in map.values() {
    println(value.to_string())
}
```

#### values_mut

```sio
pub fn values_mut(&!self) -> ValuesMut<K, V>
```

Returns an iterator over mutable values.

**Example:**

```sio
var map: HashMap<&str, i32> = HashMap::new()
map.insert("a", 1)
map.insert("b", 2)

for value in map.values_mut() {
    *value = *value * 2
}
// All values doubled
```

#### iter

```sio
pub fn iter(&self) -> Iter<K, V>
```

Returns an iterator over key-value pairs.

**Example:**

```sio
for (key, value) in map.iter() {
    println(key ++ ": " ++ value.to_string())
}
```

#### iter_mut

```sio
pub fn iter_mut(&!self) -> IterMut<K, V>
```

Returns an iterator over key-value pairs with mutable values.

**Example:**

```sio
for (key, value) in map.iter_mut() {
    *value = process(key, *value)
}
```

### Filtering

#### retain

```sio
pub fn retain<F>(&!self, f: F)
where F: FnMut(&K, &!V) -> bool
```

Retains only the elements for which the predicate returns `true`.

**Example:**

```sio
var scores: HashMap<&str, i32> = HashMap::new()
scores.insert("Alice", 100)
scores.insert("Bob", 45)
scores.insert("Carol", 78)

// Keep only passing scores
scores.retain(|_, score| *score >= 50)
// scores = {Alice: 100, Carol: 78}
```

## Trait Implementations

### Default

```sio
let map: HashMap<String, i32> = HashMap::default()  // Empty map
```

### Clone

```sio
let map2 = map1.clone() with Alloc
```

### Eq

```sio
let a = create_map()
let b = create_map()
a == b  // true if same key-value pairs
```

### Debug

```sio
let map = create_map()
// Debug format: {"key1": value1, "key2": value2}
```

### Index

```sio
let value = map[&key]  // Panics if not found
```

### IntoIterator

```sio
for (key, value) in map {  // Consumes map
    process(key, value)
}
```

### FromIterator

```sio
let map: HashMap<&str, i32> = [("a", 1), ("b", 2)].iter().collect()
```

### Extend

```sio
var map: HashMap<&str, i32> = HashMap::new()
map.extend([("a", 1), ("b", 2)])
```

## Common Patterns

### Word Counting

```sio
fn count_words(text: &str) -> HashMap<String, i32> with Alloc {
    var counts: HashMap<String, i32> = HashMap::new()

    for word in text.split_whitespace() {
        counts.entry(word.to_string())
            .and_modify(|c| *c = *c + 1)
            .or_insert(1);
    }

    counts
}
```

### Grouping by Key

```sio
fn group_by_first_char(words: &[String]) -> HashMap<char, Vec<String>> with Alloc {
    var groups: HashMap<char, Vec<String>> = HashMap::new()

    for word in words {
        if let Some(first) = word.chars().next() {
            groups.entry(first)
                .or_insert(Vec::new())
                .push(word.clone());
        }
    }

    groups
}
```

### Caching / Memoization

```sio
fn memoized_fib() -> impl Fn(u64) -> u64 {
    var cache: HashMap<u64, u64> = HashMap::new()

    move |n: u64| -> u64 {
        if let Some(result) = cache.get(&n) {
            return *result
        }

        let result = if n <= 1 {
            n
        } else {
            fib(n - 1) + fib(n - 2)
        }

        cache.insert(n, result);
        result
    }
}
```

### Two-Way Lookup

```sio
struct BiMap<K, V> {
    forward: HashMap<K, V>,
    reverse: HashMap<V, K>,
}

impl<K: Hash + Eq + Clone, V: Hash + Eq + Clone> BiMap<K, V> {
    fn insert(&!self, key: K, value: V) with Alloc {
        self.forward.insert(key.clone(), value.clone());
        self.reverse.insert(value, key);
    }

    fn get_by_key(&self, key: &K) -> Option<&V> {
        self.forward.get(key)
    }

    fn get_by_value(&self, value: &V) -> Option<&K> {
        self.reverse.get(value)
    }
}
```

### Default Values

```sio
fn get_or_default(map: &HashMap<&str, i32>, key: &str) -> i32 {
    map.get(&key).copied().unwrap_or(0)
}

// Or using entry API for mutable access:
var map: HashMap<&str, Vec<i32>> = HashMap::new()
let list = map.entry("items").or_default()
list.push(42)
```

## Hashing

HashMap uses FNV-1a as the default hash algorithm. Keys must implement the `Hash` trait:

```sio
pub trait Hash {
    fn hash<H: Hasher>(&self, state: &!H);
}
```

Custom types can implement `Hash`:

```sio
struct Point {
    x: i32,
    y: i32,
}

impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &!H) {
        self.x.hash(state)
        self.y.hash(state)
    }
}
```

## Performance Considerations

- Average O(1) for insert, lookup, and remove
- Worst case O(n) if many hash collisions
- Grows at 75% load factor
- Uses quadratic probing for collision resolution
- Pre-allocate with `with_capacity` when size is known

## See Also

- [Vec<T>](vec.md) - For ordered collections
- [Iterator](../iter.md) - For lazy processing of map contents
- [Option](../core/option.md) - Return type for lookups

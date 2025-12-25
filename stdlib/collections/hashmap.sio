// stdlib/collections/hashmap.d - Hash map collection
//
// A hash table with quadratic probing.

module std.collections.hashmap;

import std.core.option;
import std.core.result;
import std.mem.allocator;
import std.iter.iterator;
import std.hash;
import std.cmp;
import std.fmt;

/// A hash map implemented with quadratic probing.
///
/// # Examples
/// ```
/// let mut map = HashMap.new();
/// map.insert("hello", 1);
/// map.insert("world", 2);
/// assert_eq!(map.get(&"hello"), Some(&1));
/// ```
pub struct HashMap<K, V> {
    buckets: Vec<Bucket<K, V>>,
    len: usize,
}

/// Internal bucket state.
enum Bucket<K, V> {
    Empty,
    Deleted,
    Occupied { key: K, value: V, hash: u64 },
}

impl<K, V> HashMap<K, V>
where
    K: Hash + Eq
{
    /// Creates a new empty HashMap.
    pub fn new() -> HashMap<K, V> {
        HashMap {
            buckets: Vec.new(),
            len: 0,
        }
    }

    /// Creates a HashMap with the specified capacity.
    pub fn with_capacity(capacity: usize) -> HashMap<K, V> with Alloc {
        let actual_cap = next_power_of_two(capacity.max(8));
        let mut buckets = Vec.with_capacity(actual_cap);
        for _ in 0..actual_cap {
            buckets.push(Bucket.Empty);
        }
        HashMap { buckets, len: 0 }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the capacity.
    pub fn capacity(&self) -> usize {
        self.buckets.len()
    }

    /// Inserts a key-value pair, returning the old value if present.
    pub fn insert(&!self, key: K, value: V) -> Option<V> with Alloc {
        if self.should_grow() {
            self.grow();
        }

        let hash = self.hash_key(&key);
        let idx = self.find_slot_for_insert(&key, hash);

        match &!self.buckets[idx] {
            Bucket.Occupied { value: old_value, .. } => {
                let old = mem.replace(old_value, value);
                Some(old)
            }
            _ => {
                self.buckets[idx] = Bucket.Occupied { key, value, hash };
                self.len += 1;
                None
            }
        }
    }

    /// Gets a reference to the value for a key.
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.is_empty() {
            return None;
        }

        let hash = self.hash_key(key);
        match self.find_slot(key, hash) {
            Some(idx) => {
                match &self.buckets[idx] {
                    Bucket.Occupied { value, .. } => Some(value),
                    _ => None,
                }
            }
            None => None,
        }
    }

    /// Gets a mutable reference to the value for a key.
    pub fn get_mut(&!self, key: &K) -> Option<&!V> {
        if self.is_empty() {
            return None;
        }

        let hash = self.hash_key(key);
        match self.find_slot(key, hash) {
            Some(idx) => {
                match &!self.buckets[idx] {
                    Bucket.Occupied { value, .. } => Some(value),
                    _ => None,
                }
            }
            None => None,
        }
    }

    /// Returns true if the map contains the key.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Removes a key from the map, returning the value if present.
    pub fn remove(&!self, key: &K) -> Option<V> {
        if self.is_empty() {
            return None;
        }

        let hash = self.hash_key(key);
        match self.find_slot(key, hash) {
            Some(idx) => {
                let bucket = mem.replace(&!self.buckets[idx], Bucket.Deleted);
                match bucket {
                    Bucket.Occupied { value, .. } => {
                        self.len -= 1;
                        Some(value)
                    }
                    _ => None,
                }
            }
            None => None,
        }
    }

    /// Removes a key-value pair and returns it.
    pub fn remove_entry(&!self, key: &K) -> Option<(K, V)> {
        if self.is_empty() {
            return None;
        }

        let hash = self.hash_key(key);
        match self.find_slot(key, hash) {
            Some(idx) => {
                let bucket = mem.replace(&!self.buckets[idx], Bucket.Deleted);
                match bucket {
                    Bucket.Occupied { key: k, value: v, .. } => {
                        self.len -= 1;
                        Some((k, v))
                    }
                    _ => None,
                }
            }
            None => None,
        }
    }

    /// Clears the map.
    pub fn clear(&!self) {
        for bucket in self.buckets.iter_mut() {
            *bucket = Bucket.Empty;
        }
        self.len = 0;
    }

    /// Gets entry for in-place manipulation.
    pub fn entry(&!self, key: K) -> Entry<K, V> with Alloc {
        if self.should_grow() {
            self.grow();
        }

        let hash = self.hash_key(&key);
        let idx = self.find_slot_for_insert(&key, hash);

        match &self.buckets[idx] {
            Bucket.Occupied { .. } => {
                Entry.Occupied(OccupiedEntry {
                    map: self,
                    idx
                })
            }
            _ => {
                Entry.Vacant(VacantEntry {
                    map: self,
                    key,
                    hash,
                    idx
                })
            }
        }
    }

    /// Returns an iterator over keys.
    pub fn keys(&self) -> Keys<K, V> {
        Keys { iter: self.iter() }
    }

    /// Returns an iterator over values.
    pub fn values(&self) -> Values<K, V> {
        Values { iter: self.iter() }
    }

    /// Returns an iterator over mutable values.
    pub fn values_mut(&!self) -> ValuesMut<K, V> {
        ValuesMut { iter: self.iter_mut() }
    }

    /// Returns an iterator over key-value pairs.
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            buckets: self.buckets.iter(),
        }
    }

    /// Returns an iterator over mutable key-value pairs.
    pub fn iter_mut(&!self) -> IterMut<K, V> {
        IterMut {
            buckets: self.buckets.iter_mut(),
        }
    }

    /// Retains only elements for which the predicate returns true.
    pub fn retain<F>(&!self, mut f: F)
    where
        F: FnMut(&K, &!V) -> bool
    {
        for bucket in self.buckets.iter_mut() {
            if let Bucket.Occupied { key, value, .. } = bucket {
                if !f(key, value) {
                    *bucket = Bucket.Deleted;
                    self.len -= 1;
                }
            }
        }
    }

    // Internal: Hash a key
    fn hash_key(&self, key: &K) -> u64 {
        let mut hasher = DefaultHasher.new();
        key.hash(&!hasher);
        hasher.finish()
    }

    // Internal: Find slot for a key
    fn find_slot(&self, key: &K, hash: u64) -> Option<usize> {
        if self.buckets.is_empty() {
            return None;
        }

        let mask = self.buckets.len() - 1;
        let mut idx = (hash as usize) & mask;
        let mut probe = 1;

        loop {
            match &self.buckets[idx] {
                Bucket.Empty => return None,
                Bucket.Deleted => {}
                Bucket.Occupied { key: k, hash: h, .. } => {
                    if *h == hash && k == key {
                        return Some(idx);
                    }
                }
            }

            idx = (idx + probe) & mask;
            probe += 1;

            if probe > self.buckets.len() {
                return None;
            }
        }
    }

    // Internal: Find slot for insertion
    fn find_slot_for_insert(&self, key: &K, hash: u64) -> usize {
        let mask = self.buckets.len() - 1;
        let mut idx = (hash as usize) & mask;
        let mut probe = 1;
        let mut first_deleted: Option<usize> = None;

        loop {
            match &self.buckets[idx] {
                Bucket.Empty => {
                    return first_deleted.unwrap_or(idx);
                }
                Bucket.Deleted => {
                    if first_deleted.is_none() {
                        first_deleted = Some(idx);
                    }
                }
                Bucket.Occupied { key: k, hash: h, .. } => {
                    if *h == hash && k == key {
                        return idx;
                    }
                }
            }

            idx = (idx + probe) & mask;
            probe += 1;
        }
    }

    // Internal: Check if we should grow
    fn should_grow(&self) -> bool {
        if self.buckets.is_empty() {
            return true;
        }
        // Grow at 75% load factor
        self.len * 4 >= self.buckets.len() * 3
    }

    // Internal: Grow the table
    fn grow(&!self) with Alloc {
        let new_cap = if self.buckets.is_empty() {
            8
        } else {
            self.buckets.len() * 2
        };

        let old_buckets = mem.replace(&!self.buckets, Vec.new());

        self.buckets = Vec.with_capacity(new_cap);
        for _ in 0..new_cap {
            self.buckets.push(Bucket.Empty);
        }
        self.len = 0;

        for bucket in old_buckets {
            if let Bucket.Occupied { key, value, .. } = bucket {
                self.insert(key, value);
            }
        }
    }
}

// ============================================================================
// Entry API
// ============================================================================

/// Entry for in-place manipulation.
pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K, V> Entry<'a, K, V>
where
    K: Hash + Eq
{
    /// Returns a reference to the entry's value, inserting default if vacant.
    pub fn or_insert(self, default: V) -> &!V with Alloc {
        match self {
            Entry.Occupied(entry) => entry.into_mut(),
            Entry.Vacant(entry) => entry.insert(default),
        }
    }

    /// Returns a reference to the entry's value, inserting from closure if vacant.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &!V with Alloc {
        match self {
            Entry.Occupied(entry) => entry.into_mut(),
            Entry.Vacant(entry) => entry.insert(default()),
        }
    }

    /// Returns a reference to the entry's value, inserting default if vacant.
    pub fn or_default(self) -> &!V
    where
        V: Default
    with Alloc {
        self.or_insert_with(V.default)
    }

    /// Modifies the entry if occupied.
    pub fn and_modify<F: FnOnce(&!V)>(self, f: F) -> Self {
        match self {
            Entry.Occupied(mut entry) => {
                f(entry.get_mut());
                Entry.Occupied(entry)
            }
            Entry.Vacant(entry) => Entry.Vacant(entry),
        }
    }

    /// Returns the key.
    pub fn key(&self) -> &K {
        match self {
            Entry.Occupied(entry) => entry.key(),
            Entry.Vacant(entry) => &entry.key,
        }
    }
}

/// An occupied entry.
pub struct OccupiedEntry<'a, K, V> {
    map: &'a !HashMap<K, V>,
    idx: usize,
}

impl<'a, K, V> OccupiedEntry<'a, K, V>
where
    K: Hash + Eq
{
    /// Returns a reference to the key.
    pub fn key(&self) -> &K {
        match &self.map.buckets[self.idx] {
            Bucket.Occupied { key, .. } => key,
            _ => unreachable!(),
        }
    }

    /// Returns a reference to the value.
    pub fn get(&self) -> &V {
        match &self.map.buckets[self.idx] {
            Bucket.Occupied { value, .. } => value,
            _ => unreachable!(),
        }
    }

    /// Returns a mutable reference to the value.
    pub fn get_mut(&!self) -> &!V {
        match &!self.map.buckets[self.idx] {
            Bucket.Occupied { value, .. } => value,
            _ => unreachable!(),
        }
    }

    /// Converts to a mutable reference to the value.
    pub fn into_mut(self) -> &'a !V {
        match &!self.map.buckets[self.idx] {
            Bucket.Occupied { value, .. } => value,
            _ => unreachable!(),
        }
    }

    /// Replaces the value and returns the old value.
    pub fn insert(&!self, value: V) -> V {
        match &!self.map.buckets[self.idx] {
            Bucket.Occupied { value: old, .. } => mem.replace(old, value),
            _ => unreachable!(),
        }
    }

    /// Removes the entry and returns the value.
    pub fn remove(self) -> V {
        let bucket = mem.replace(&!self.map.buckets[self.idx], Bucket.Deleted);
        self.map.len -= 1;
        match bucket {
            Bucket.Occupied { value, .. } => value,
            _ => unreachable!(),
        }
    }

    /// Removes the entry and returns the key-value pair.
    pub fn remove_entry(self) -> (K, V) {
        let bucket = mem.replace(&!self.map.buckets[self.idx], Bucket.Deleted);
        self.map.len -= 1;
        match bucket {
            Bucket.Occupied { key, value, .. } => (key, value),
            _ => unreachable!(),
        }
    }
}

/// A vacant entry.
pub struct VacantEntry<'a, K, V> {
    map: &'a !HashMap<K, V>,
    key: K,
    hash: u64,
    idx: usize,
}

impl<'a, K, V> VacantEntry<'a, K, V>
where
    K: Hash + Eq
{
    /// Returns a reference to the key.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Consumes the entry and returns the key.
    pub fn into_key(self) -> K {
        self.key
    }

    /// Inserts a value and returns a mutable reference.
    pub fn insert(self, value: V) -> &'a !V {
        self.map.buckets[self.idx] = Bucket.Occupied {
            key: self.key,
            value,
            hash: self.hash
        };
        self.map.len += 1;

        match &!self.map.buckets[self.idx] {
            Bucket.Occupied { value, .. } => value,
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Iterators
// ============================================================================

/// Iterator over key-value pairs.
pub struct Iter<'a, K, V> {
    buckets: vec.Iter<'a, Bucket<K, V>>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&!self) -> Option<(&'a K, &'a V)> {
        loop {
            match self.buckets.next()? {
                Bucket.Occupied { key, value, .. } => {
                    return Some((key, value));
                }
                _ => continue,
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.buckets.size_hint().1)
    }
}

/// Iterator over mutable key-value pairs.
pub struct IterMut<'a, K, V> {
    buckets: vec.IterMut<'a, Bucket<K, V>>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a !V);

    fn next(&!self) -> Option<(&'a K, &'a !V)> {
        loop {
            match self.buckets.next()? {
                Bucket.Occupied { key, value, .. } => {
                    return Some((key, value));
                }
                _ => continue,
            }
        }
    }
}

/// Iterator over keys.
pub struct Keys<'a, K, V> {
    iter: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&!self) -> Option<&'a K> {
        self.iter.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

/// Iterator over values.
pub struct Values<'a, K, V> {
    iter: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&!self) -> Option<&'a V> {
        self.iter.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

/// Iterator over mutable values.
pub struct ValuesMut<'a, K, V> {
    iter: IterMut<'a, K, V>,
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a !V;

    fn next(&!self) -> Option<&'a !V> {
        self.iter.next().map(|(_, v)| v)
    }
}

/// Owning iterator.
pub struct IntoIter<K, V> {
    buckets: vec.IntoIter<Bucket<K, V>>,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&!self) -> Option<(K, V)> {
        loop {
            match self.buckets.next()? {
                Bucket.Occupied { key, value, .. } => {
                    return Some((key, value));
                }
                _ => continue,
            }
        }
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl<K, V> Default for HashMap<K, V>
where
    K: Hash + Eq
{
    fn default() -> HashMap<K, V> {
        HashMap.new()
    }
}

impl<K, V> Clone for HashMap<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone
{
    fn clone(&self) -> HashMap<K, V> with Alloc {
        let mut new_map = HashMap.with_capacity(self.len);
        for (k, v) in self.iter() {
            new_map.insert(k.clone(), v.clone());
        }
        new_map
    }
}

impl<K, V> Eq for HashMap<K, V>
where
    K: Hash + Eq,
    V: Eq
{
    fn eq(&self, other: &HashMap<K, V>) -> bool {
        if self.len != other.len {
            return false;
        }

        for (k, v) in self.iter() {
            match other.get(k) {
                Some(other_v) if v == other_v => continue,
                _ => return false,
            }
        }
        true
    }
}

impl<K, V> Debug for HashMap<K, V>
where
    K: Debug,
    V: Debug
{
    fn fmt(&self, f: &!Formatter) -> Result<(), Error> {
        write!(f, "{{")?;
        let mut first = true;
        for (k, v) in self.iter() {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{:?}: {:?}", k, v)?;
            first = false;
        }
        write!(f, "}}")
    }
}

impl<K, V> Index<&K> for HashMap<K, V>
where
    K: Hash + Eq
{
    type Output = V;

    fn index(&self, key: &K) -> &V with Panic {
        self.get(key).expect("key not found")
    }
}

impl<K, V> IntoIterator for HashMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        IntoIter { buckets: self.buckets.into_iter() }
    }
}

impl<K, V> FromIterator<(K, V)> for HashMap<K, V>
where
    K: Hash + Eq
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> HashMap<K, V> with Alloc {
        let mut map = HashMap.new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K, V> Extend<(K, V)> for HashMap<K, V>
where
    K: Hash + Eq
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&!self, iter: I) with Alloc {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Returns the next power of two >= n.
fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }

    let mut result = 1;
    while result < n {
        result *= 2;
    }
    result
}

// ============================================================================
// Default Hasher
// ============================================================================

/// A simple default hasher (FNV-1a).
pub struct DefaultHasher {
    state: u64,
}

impl DefaultHasher {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;

    pub fn new() -> DefaultHasher {
        DefaultHasher { state: Self.FNV_OFFSET }
    }
}

impl Hasher for DefaultHasher {
    fn write(&!self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= *byte as u64;
            self.state = self.state.wrapping_mul(Self.FNV_PRIME);
        }
    }

    fn finish(&self) -> u64 {
        self.state
    }
}

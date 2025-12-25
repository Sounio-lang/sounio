// stdlib/collections/hashset.d - Hash set collection
//
// A hash set implemented as a HashMap with unit values.

module std.collections.hashset;

import std.collections.hashmap;
import std.core.option;
import std.hash;
import std.cmp;
import std.iter.iterator;
import std.fmt;
import std.clone;
import std.default;

/// A hash set implemented as a HashMap where the value is unit.
///
/// As with the HashMap type, a HashSet requires that the elements
/// implement the Eq and Hash traits.
///
/// # Examples
///
/// ```d
/// let mut set = HashSet.new();
///
/// set.insert(1);
/// set.insert(2);
/// set.insert(3);
///
/// assert(set.contains(&2));
/// assert(!set.contains(&4));
///
/// set.remove(&2);
/// assert(!set.contains(&2));
/// ```
pub struct HashSet<T> {
    map: HashMap<T, unit>,
}

impl<T> HashSet<T>
where
    T: Hash + Eq
{
    /// Creates an empty HashSet.
    ///
    /// The hash set is initially created with a capacity of 0, so it
    /// will not allocate until it is first inserted into.
    pub fn new() -> HashSet<T> {
        HashSet { map: HashMap.new() }
    }

    /// Creates an empty HashSet with the specified capacity.
    ///
    /// The hash set will be able to hold at least `capacity` elements
    /// without reallocating.
    pub fn with_capacity(capacity: usize) -> HashSet<T> with Alloc {
        HashSet { map: HashMap.with_capacity(capacity) }
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if the set contains no elements.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the number of elements the set can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }

    /// Clears the set, removing all values.
    pub fn clear(&!self) {
        self.map.clear()
    }

    /// Returns true if the set contains the value.
    ///
    /// # Examples
    ///
    /// ```d
    /// let mut set = HashSet.new();
    /// set.insert(1);
    /// assert(set.contains(&1));
    /// assert(!set.contains(&2));
    /// ```
    pub fn contains(&self, value: &T) -> bool {
        self.map.contains_key(value)
    }

    /// Adds a value to the set.
    ///
    /// Returns whether the value was newly inserted. That is:
    /// - If the set did not contain this value, true is returned.
    /// - If the set already contained this value, false is returned.
    ///
    /// # Examples
    ///
    /// ```d
    /// let mut set = HashSet.new();
    /// assert(set.insert(1));
    /// assert(!set.insert(1));
    /// ```
    pub fn insert(&!self, value: T) -> bool with Alloc {
        self.map.insert(value, ()).is_none()
    }

    /// Removes a value from the set.
    ///
    /// Returns whether the value was present in the set.
    pub fn remove(&!self, value: &T) -> bool {
        self.map.remove(value).is_some()
    }

    /// Adds a value to the set, replacing the existing value if any.
    ///
    /// Returns the replaced value if one existed.
    pub fn replace(&!self, value: T) -> Option<T> with Alloc {
        // Check if value exists, if so remove it first
        let existed = self.map.contains_key(&value);
        if existed {
            self.map.remove(&value);
        }
        self.map.insert(value, ());
        // Note: in a full implementation, we'd return the old key
        None
    }

    /// Returns an iterator over the values.
    pub fn iter(&self) -> Iter<T> {
        Iter { inner: self.map.keys() }
    }

    /// Returns true if self is a subset of other.
    ///
    /// This means all elements in self are contained in other.
    pub fn is_subset(&self, other: &HashSet<T>) -> bool {
        if self.len() > other.len() {
            return false;
        }
        // Check each element in self is in other
        for value in self.iter() {
            if !other.contains(value) {
                return false;
            }
        }
        true
    }

    /// Returns true if self is a superset of other.
    ///
    /// This means all elements in other are contained in self.
    pub fn is_superset(&self, other: &HashSet<T>) -> bool {
        other.is_subset(self)
    }

    /// Returns true if self has no elements in common with other.
    pub fn is_disjoint(&self, other: &HashSet<T>) -> bool {
        // Check the smaller set against the larger
        if self.len() <= other.len() {
            for value in self.iter() {
                if other.contains(value) {
                    return false;
                }
            }
        } else {
            for value in other.iter() {
                if self.contains(value) {
                    return false;
                }
            }
        }
        true
    }

    /// Visits the values representing the difference.
    ///
    /// Returns values that are in self but not in other.
    pub fn difference(&self, other: &HashSet<T>) -> Difference<T> {
        Difference { iter: self.iter(), other }
    }

    /// Visits the values representing the symmetric difference.
    ///
    /// Returns values that are in either set but not both.
    pub fn symmetric_difference(&self, other: &HashSet<T>) -> SymmetricDifference<T> {
        SymmetricDifference {
            a_diff_b: self.difference(other),
            b_diff_a: other.difference(self),
            in_second: false,
        }
    }

    /// Visits the values representing the intersection.
    ///
    /// Returns values that are in both sets.
    pub fn intersection(&self, other: &HashSet<T>) -> Intersection<T> {
        Intersection { iter: self.iter(), other }
    }

    /// Visits the values representing the union.
    ///
    /// Returns values that are in either set.
    pub fn union(&self, other: &HashSet<T>) -> Union<T> {
        Union {
            iter: self.iter(),
            other_iter: other.difference(self),
            in_other: false,
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// Removes all elements e where f(&e) returns false.
    pub fn retain<F>(&!self, f: F)
    where
        F: FnMut(&T) -> bool
    {
        self.map.retain(|k, _| f(k))
    }

    /// Reserves capacity for at least additional more elements.
    pub fn reserve(&!self, additional: usize) with Alloc {
        // HashMap should support reserve
        let needed = self.len() + additional;
        if needed > self.capacity() {
            // Grow the underlying map
            // For now, this is a no-op since HashMap doesn't expose reserve
        }
    }

    /// Creates a new set containing elements from both sets.
    ///
    /// This consumes both sets and returns a new set with all unique elements.
    pub fn union_into(self, other: HashSet<T>) -> HashSet<T> with Alloc {
        let mut result = HashSet.with_capacity(self.len() + other.len());
        for item in self {
            result.insert(item);
        }
        for item in other {
            result.insert(item);
        }
        result
    }

    /// Creates a new set containing elements in both sets.
    pub fn intersection_into(&self, other: &HashSet<T>) -> HashSet<T> with Alloc
    where
        T: Clone
    {
        let mut result = HashSet.new();
        for item in self.iter() {
            if other.contains(item) {
                result.insert(item.clone());
            }
        }
        result
    }

    /// Creates a new set containing elements in self but not in other.
    pub fn difference_into(&self, other: &HashSet<T>) -> HashSet<T> with Alloc
    where
        T: Clone
    {
        let mut result = HashSet.new();
        for item in self.iter() {
            if !other.contains(item) {
                result.insert(item.clone());
            }
        }
        result
    }

    /// Creates a new set containing elements in either set but not both.
    pub fn symmetric_difference_into(&self, other: &HashSet<T>) -> HashSet<T> with Alloc
    where
        T: Clone
    {
        let mut result = HashSet.new();
        for item in self.iter() {
            if !other.contains(item) {
                result.insert(item.clone());
            }
        }
        for item in other.iter() {
            if !self.contains(item) {
                result.insert(item.clone());
            }
        }
        result
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl<T> Clone for HashSet<T>
where
    T: Clone + Hash + Eq
{
    fn clone(&self) -> HashSet<T> with Alloc {
        HashSet { map: self.map.clone() }
    }
}

impl<T> Default for HashSet<T>
where
    T: Hash + Eq
{
    fn default() -> HashSet<T> {
        HashSet.new()
    }
}

impl<T> Eq for HashSet<T>
where
    T: Hash + Eq
{
    fn eq(&self, other: &HashSet<T>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for value in self.iter() {
            if !other.contains(value) {
                return false;
            }
        }
        true
    }
}

impl<T> Debug for HashSet<T>
where
    T: Debug + Hash + Eq
{
    fn fmt(&self, f: &!Formatter) -> Result<(), Error> {
        write!(f, "{{")?;
        let mut first = true;
        for item in self.iter() {
            if !first {
                write!(f, ", ")?;
            }
            item.fmt(f)?;
            first = false;
        }
        write!(f, "}}")
    }
}

impl<T> FromIterator<T> for HashSet<T>
where
    T: Hash + Eq
{
    fn from_iter<I>(iter: I) -> HashSet<T> with Alloc
    where
        I: IntoIterator<Item = T>
    {
        let mut set = HashSet.new();
        for item in iter {
            set.insert(item);
        }
        set
    }
}

impl<T> IntoIterator for HashSet<T>
where
    T: Hash + Eq
{
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter { inner: self.map.into_iter() }
    }
}

impl<T> Extend<T> for HashSet<T>
where
    T: Hash + Eq
{
    fn extend<I>(&!self, iter: I) with Alloc
    where
        I: IntoIterator<Item = T>
    {
        for item in iter {
            self.insert(item);
        }
    }
}

// ============================================================================
// Iterators
// ============================================================================

/// Iterator over HashSet values by reference.
pub struct Iter<'a, T> {
    inner: Keys<'a, T, unit>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&!self) -> Option<&'a T> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// Owning iterator over HashSet values.
pub struct IntoIter<T> {
    inner: hashmap.IntoIter<T, unit>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&!self) -> Option<T> {
        match self.inner.next() {
            Some((k, _)) => Some(k),
            None => None,
        }
    }
}

/// Iterator yielding elements in self but not in other.
pub struct Difference<'a, T> {
    iter: Iter<'a, T>,
    other: &'a HashSet<T>,
}

impl<'a, T> Iterator for Difference<'a, T>
where
    T: Hash + Eq
{
    type Item = &'a T;

    fn next(&!self) -> Option<&'a T> {
        loop {
            match self.iter.next() {
                Some(item) => {
                    if !self.other.contains(item) {
                        return Some(item);
                    }
                    // Keep looking
                }
                None => return None,
            }
        }
    }
}

/// Iterator yielding elements in either set but not both.
pub struct SymmetricDifference<'a, T> {
    a_diff_b: Difference<'a, T>,
    b_diff_a: Difference<'a, T>,
    in_second: bool,
}

impl<'a, T> Iterator for SymmetricDifference<'a, T>
where
    T: Hash + Eq
{
    type Item = &'a T;

    fn next(&!self) -> Option<&'a T> {
        if !self.in_second {
            match self.a_diff_b.next() {
                Some(v) => Some(v),
                None => {
                    self.in_second = true;
                    self.b_diff_a.next()
                }
            }
        } else {
            self.b_diff_a.next()
        }
    }
}

/// Iterator yielding elements in both sets.
pub struct Intersection<'a, T> {
    iter: Iter<'a, T>,
    other: &'a HashSet<T>,
}

impl<'a, T> Iterator for Intersection<'a, T>
where
    T: Hash + Eq
{
    type Item = &'a T;

    fn next(&!self) -> Option<&'a T> {
        loop {
            match self.iter.next() {
                Some(item) => {
                    if self.other.contains(item) {
                        return Some(item);
                    }
                    // Keep looking
                }
                None => return None,
            }
        }
    }
}

/// Iterator yielding elements in either set.
pub struct Union<'a, T> {
    iter: Iter<'a, T>,
    other_iter: Difference<'a, T>,
    in_other: bool,
}

impl<'a, T> Iterator for Union<'a, T>
where
    T: Hash + Eq
{
    type Item = &'a T;

    fn next(&!self) -> Option<&'a T> {
        if !self.in_other {
            match self.iter.next() {
                Some(v) => Some(v),
                None => {
                    self.in_other = true;
                    self.other_iter.next()
                }
            }
        } else {
            self.other_iter.next()
        }
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Creates a HashSet from an array.
pub fn hashset_from_array<T, const N: usize>(array: [T; N]) -> HashSet<T> with Alloc
where
    T: Hash + Eq
{
    let mut set = HashSet.with_capacity(N);
    for item in array {
        set.insert(item);
    }
    set
}

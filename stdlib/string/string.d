// stdlib/string/string.d - UTF-8 String type
//
// A growable UTF-8 encoded string with owned storage.

module std.string;

import std.core.option;
import std.core.result;
import std.mem.allocator;
import std.iter.iterator;
import std.cmp;
import std.ops;
import std.fmt;
import std.hash;
import std.convert;

/// A UTF-8 encoded, growable string.
///
/// # Examples
/// ```
/// let mut s = String.new();
/// s.push_str("Hello");
/// s.push(' ');
/// s.push_str("World!");
/// assert_eq!(s.as_str(), "Hello World!");
/// ```
pub struct String {
    bytes: Vec<u8>,
}

impl String {
    /// Creates a new empty String.
    pub fn new() -> String {
        String { bytes: Vec.new() }
    }

    /// Creates a new empty String with the specified capacity.
    pub fn with_capacity(capacity: usize) -> String {
        String { bytes: Vec.with_capacity(capacity) }
    }

    /// Creates a String from a string slice.
    pub fn from_str(s: str) -> String {
        let mut string = String.with_capacity(s.len());
        string.push_str(s);
        string
    }

    /// Creates a String from raw UTF-8 bytes.
    ///
    /// # Safety
    /// The bytes must be valid UTF-8.
    pub fn from_utf8(bytes: Vec<u8>) -> Result<String, Utf8Error> {
        if validate_utf8(bytes.as_slice()) {
            Ok(String { bytes })
        } else {
            Err(Utf8Error { valid_up_to: find_utf8_error(bytes.as_slice()) })
        }
    }

    /// Creates a String from raw UTF-8 bytes without validation.
    ///
    /// # Safety
    /// Caller must ensure bytes are valid UTF-8.
    pub unsafe fn from_utf8_unchecked(bytes: Vec<u8>) -> String {
        String { bytes }
    }

    /// Returns the length in bytes.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns true if the string is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Returns the capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    /// Returns the string as a str slice.
    pub fn as_str(&self) -> str {
        // Safety: String maintains UTF-8 invariant
        unsafe { str.from_utf8_unchecked(self.bytes.as_slice()) }
    }

    /// Returns a mutable string slice.
    pub fn as_mut_str(&!self) -> &!str {
        // Safety: String maintains UTF-8 invariant
        unsafe { str.from_utf8_unchecked_mut(self.bytes.as_mut_slice()) }
    }

    /// Returns the underlying bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    /// Appends a character to the string.
    pub fn push(&!self, ch: char) with Alloc {
        let mut buf = [0u8; 4];
        let len = ch.encode_utf8(&!buf);
        for i in 0..len {
            self.bytes.push(buf[i]);
        }
    }

    /// Appends a string slice to the string.
    pub fn push_str(&!self, s: str) with Alloc {
        self.bytes.extend_from_slice(s.as_bytes());
    }

    /// Removes and returns the last character.
    pub fn pop(&!self) -> Option<char> {
        if self.is_empty() {
            return None;
        }

        let ch = self.chars().last()?;
        let new_len = self.len() - ch.len_utf8();
        self.bytes.truncate(new_len);
        Some(ch)
    }

    /// Inserts a character at a byte position.
    ///
    /// # Panics
    /// Panics if idx is not on a char boundary or is out of bounds.
    pub fn insert(&!self, idx: usize, ch: char) with Alloc, Panic {
        assert!(self.is_char_boundary(idx), "insert not on char boundary");

        let mut buf = [0u8; 4];
        let len = ch.encode_utf8(&!buf);

        // Make room
        for _ in 0..len {
            self.bytes.push(0);
        }

        // Shift bytes
        let old_len = self.len() - len;
        for i in (idx..old_len).rev() {
            self.bytes[i + len] = self.bytes[i];
        }

        // Insert new bytes
        for i in 0..len {
            self.bytes[idx + i] = buf[i];
        }
    }

    /// Inserts a string slice at a byte position.
    pub fn insert_str(&!self, idx: usize, s: str) with Alloc, Panic {
        assert!(self.is_char_boundary(idx), "insert_str not on char boundary");

        let insert_len = s.len();
        let old_len = self.len();

        // Grow capacity
        self.bytes.reserve(insert_len);

        // Make room
        for _ in 0..insert_len {
            self.bytes.push(0);
        }

        // Shift existing bytes
        for i in (idx..old_len).rev() {
            self.bytes[i + insert_len] = self.bytes[i];
        }

        // Copy new bytes
        let s_bytes = s.as_bytes();
        for i in 0..insert_len {
            self.bytes[idx + i] = s_bytes[i];
        }
    }

    /// Removes a character at a byte position and returns it.
    pub fn remove(&!self, idx: usize) -> char with Panic {
        assert!(self.is_char_boundary(idx), "remove not on char boundary");

        let ch = self.as_str()[idx..].chars().next()
            .expect("remove on empty string");
        let ch_len = ch.len_utf8();

        // Shift bytes left
        let len = self.len();
        for i in idx..(len - ch_len) {
            self.bytes[i] = self.bytes[i + ch_len];
        }

        self.bytes.truncate(len - ch_len);
        ch
    }

    /// Truncates the string to the specified length.
    ///
    /// # Panics
    /// Panics if new_len is not on a char boundary.
    pub fn truncate(&!self, new_len: usize) with Panic {
        if new_len < self.len() {
            assert!(self.is_char_boundary(new_len), "truncate not on char boundary");
            self.bytes.truncate(new_len);
        }
    }

    /// Clears the string.
    pub fn clear(&!self) {
        self.bytes.clear();
    }

    /// Reserves capacity for at least additional more bytes.
    pub fn reserve(&!self, additional: usize) with Alloc {
        self.bytes.reserve(additional);
    }

    /// Reserves exact capacity for additional more bytes.
    pub fn reserve_exact(&!self, additional: usize) with Alloc {
        self.bytes.reserve_exact(additional);
    }

    /// Shrinks capacity to fit the current length.
    pub fn shrink_to_fit(&!self) with Alloc {
        self.bytes.shrink_to_fit();
    }

    /// Returns true if idx is on a UTF-8 char boundary.
    pub fn is_char_boundary(&self, idx: usize) -> bool {
        if idx == 0 || idx == self.len() {
            return true;
        }
        if idx > self.len() {
            return false;
        }
        // UTF-8 continuation bytes start with 10xxxxxx
        (self.bytes[idx] & 0b11000000) != 0b10000000
    }

    /// Returns an iterator over the characters.
    pub fn chars(&self) -> Chars {
        Chars { bytes: self.as_bytes(), pos: 0 }
    }

    /// Returns an iterator over the characters with their byte positions.
    pub fn char_indices(&self) -> CharIndices {
        CharIndices { bytes: self.as_bytes(), pos: 0 }
    }

    /// Returns an iterator over the bytes.
    pub fn bytes(&self) -> Bytes {
        Bytes { iter: self.bytes.iter() }
    }

    /// Returns an iterator over lines.
    pub fn lines(&self) -> Lines {
        Lines { remaining: self.as_str() }
    }

    /// Splits the string by a pattern.
    pub fn split(&self, pattern: char) -> Split {
        Split { remaining: self.as_str(), pattern, finished: false }
    }

    /// Splits the string by whitespace.
    pub fn split_whitespace(&self) -> SplitWhitespace {
        SplitWhitespace { remaining: self.as_str() }
    }

    /// Returns true if the string contains the pattern.
    pub fn contains(&self, pattern: str) -> bool {
        self.find(pattern).is_some()
    }

    /// Returns true if the string starts with the pattern.
    pub fn starts_with(&self, pattern: str) -> bool {
        let s = self.as_str();
        if pattern.len() > s.len() {
            return false;
        }
        s.as_bytes()[..pattern.len()] == pattern.as_bytes()
    }

    /// Returns true if the string ends with the pattern.
    pub fn ends_with(&self, pattern: str) -> bool {
        let s = self.as_str();
        if pattern.len() > s.len() {
            return false;
        }
        let start = s.len() - pattern.len();
        s.as_bytes()[start..] == pattern.as_bytes()
    }

    /// Finds the first occurrence of a pattern.
    pub fn find(&self, pattern: str) -> Option<usize> {
        let s = self.as_bytes();
        let p = pattern.as_bytes();

        if p.is_empty() {
            return Some(0);
        }
        if p.len() > s.len() {
            return None;
        }

        for i in 0..=(s.len() - p.len()) {
            if s[i..(i + p.len())] == p {
                return Some(i);
            }
        }
        None
    }

    /// Finds the last occurrence of a pattern.
    pub fn rfind(&self, pattern: str) -> Option<usize> {
        let s = self.as_bytes();
        let p = pattern.as_bytes();

        if p.is_empty() {
            return Some(s.len());
        }
        if p.len() > s.len() {
            return None;
        }

        for i in (0..=(s.len() - p.len())).rev() {
            if s[i..(i + p.len())] == p {
                return Some(i);
            }
        }
        None
    }

    /// Replaces all occurrences of a pattern.
    pub fn replace(&self, from: str, to: str) -> String with Alloc {
        let mut result = String.new();
        let mut remaining = self.as_str();

        while let Some(idx) = remaining.find(from) {
            result.push_str(&remaining[..idx]);
            result.push_str(to);
            remaining = &remaining[(idx + from.len())..];
        }
        result.push_str(remaining);
        result
    }

    /// Returns a string with leading and trailing whitespace removed.
    pub fn trim(&self) -> str {
        self.trim_start().trim_end()
    }

    /// Returns a string with leading whitespace removed.
    pub fn trim_start(&self) -> str {
        let s = self.as_str();
        let mut start = 0;
        for ch in s.chars() {
            if !ch.is_whitespace() {
                break;
            }
            start += ch.len_utf8();
        }
        &s[start..]
    }

    /// Returns a string with trailing whitespace removed.
    pub fn trim_end(&self) -> str {
        let s = self.as_str();
        let mut end = s.len();
        for ch in s.chars().rev() {
            if !ch.is_whitespace() {
                break;
            }
            end -= ch.len_utf8();
        }
        &s[..end]
    }

    /// Converts the string to lowercase.
    pub fn to_lowercase(&self) -> String with Alloc {
        let mut result = String.with_capacity(self.len());
        for ch in self.chars() {
            result.push(ch.to_lowercase());
        }
        result
    }

    /// Converts the string to uppercase.
    pub fn to_uppercase(&self) -> String with Alloc {
        let mut result = String.with_capacity(self.len());
        for ch in self.chars() {
            result.push(ch.to_uppercase());
        }
        result
    }

    /// Repeats the string n times.
    pub fn repeat(&self, n: usize) -> String with Alloc {
        let mut result = String.with_capacity(self.len() * n);
        for _ in 0..n {
            result.push_str(self.as_str());
        }
        result
    }

    /// Consumes the String and returns the underlying byte vector.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

// ============================================================================
// Iterators
// ============================================================================

/// Iterator over characters in a string.
pub struct Chars {
    bytes: &[u8],
    pos: usize,
}

impl Iterator for Chars {
    type Item = char;

    fn next(&!self) -> Option<char> {
        if self.pos >= self.bytes.len() {
            return None;
        }

        let (ch, len) = decode_utf8(&self.bytes[self.pos..]);
        self.pos += len;
        Some(ch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.bytes.len() - self.pos;
        // At least remaining/4 chars, at most remaining chars
        (remaining / 4, Some(remaining))
    }
}

/// Iterator over characters with byte indices.
pub struct CharIndices {
    bytes: &[u8],
    pos: usize,
}

impl Iterator for CharIndices {
    type Item = (usize, char);

    fn next(&!self) -> Option<(usize, char)> {
        if self.pos >= self.bytes.len() {
            return None;
        }

        let idx = self.pos;
        let (ch, len) = decode_utf8(&self.bytes[self.pos..]);
        self.pos += len;
        Some((idx, ch))
    }
}

/// Iterator over bytes.
pub struct Bytes {
    iter: slice.Iter<u8>,
}

impl Iterator for Bytes {
    type Item = u8;

    fn next(&!self) -> Option<u8> {
        self.iter.next().copied()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

/// Iterator over lines.
pub struct Lines {
    remaining: str,
}

impl Iterator for Lines {
    type Item = str;

    fn next(&!self) -> Option<str> {
        if self.remaining.is_empty() {
            return None;
        }

        match self.remaining.find('\n') {
            Some(idx) => {
                let line = &self.remaining[..idx];
                self.remaining = &self.remaining[(idx + 1)..];
                // Strip \r if present
                if line.ends_with('\r') {
                    Some(&line[..(line.len() - 1)])
                } else {
                    Some(line)
                }
            }
            None => {
                let line = self.remaining;
                self.remaining = "";
                Some(line)
            }
        }
    }
}

/// Iterator over splits by a character.
pub struct Split {
    remaining: str,
    pattern: char,
    finished: bool,
}

impl Iterator for Split {
    type Item = str;

    fn next(&!self) -> Option<str> {
        if self.finished {
            return None;
        }

        match self.remaining.find_char(self.pattern) {
            Some(idx) => {
                let part = &self.remaining[..idx];
                self.remaining = &self.remaining[(idx + self.pattern.len_utf8())..];
                Some(part)
            }
            None => {
                self.finished = true;
                Some(self.remaining)
            }
        }
    }
}

/// Iterator over whitespace-separated parts.
pub struct SplitWhitespace {
    remaining: str,
}

impl Iterator for SplitWhitespace {
    type Item = str;

    fn next(&!self) -> Option<str> {
        // Skip leading whitespace
        self.remaining = self.remaining.trim_start();

        if self.remaining.is_empty() {
            return None;
        }

        // Find end of word
        let mut end = 0;
        for ch in self.remaining.chars() {
            if ch.is_whitespace() {
                break;
            }
            end += ch.len_utf8();
        }

        let word = &self.remaining[..end];
        self.remaining = &self.remaining[end..];
        Some(word)
    }
}

// ============================================================================
// UTF-8 Error
// ============================================================================

/// Error returned when bytes are not valid UTF-8.
pub struct Utf8Error {
    valid_up_to: usize,
}

impl Utf8Error {
    /// Returns the index of the first invalid byte.
    pub fn valid_up_to(&self) -> usize {
        self.valid_up_to
    }
}

// ============================================================================
// UTF-8 Helpers
// ============================================================================

/// Validates that bytes are valid UTF-8.
fn validate_utf8(bytes: &[u8]) -> bool {
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];

        if b < 0x80 {
            // ASCII
            i += 1;
        } else if b < 0xC0 {
            // Invalid: continuation byte at start
            return false;
        } else if b < 0xE0 {
            // 2-byte sequence
            if i + 1 >= bytes.len() { return false; }
            if !is_continuation(bytes[i + 1]) { return false; }
            i += 2;
        } else if b < 0xF0 {
            // 3-byte sequence
            if i + 2 >= bytes.len() { return false; }
            if !is_continuation(bytes[i + 1]) { return false; }
            if !is_continuation(bytes[i + 2]) { return false; }
            i += 3;
        } else if b < 0xF8 {
            // 4-byte sequence
            if i + 3 >= bytes.len() { return false; }
            if !is_continuation(bytes[i + 1]) { return false; }
            if !is_continuation(bytes[i + 2]) { return false; }
            if !is_continuation(bytes[i + 3]) { return false; }
            i += 4;
        } else {
            return false;
        }
    }
    true
}

/// Finds the first invalid UTF-8 byte position.
fn find_utf8_error(bytes: &[u8]) -> usize {
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];

        if b < 0x80 {
            i += 1;
        } else if b < 0xC0 {
            return i;
        } else if b < 0xE0 {
            if i + 1 >= bytes.len() || !is_continuation(bytes[i + 1]) {
                return i;
            }
            i += 2;
        } else if b < 0xF0 {
            if i + 2 >= bytes.len()
                || !is_continuation(bytes[i + 1])
                || !is_continuation(bytes[i + 2]) {
                return i;
            }
            i += 3;
        } else if b < 0xF8 {
            if i + 3 >= bytes.len()
                || !is_continuation(bytes[i + 1])
                || !is_continuation(bytes[i + 2])
                || !is_continuation(bytes[i + 3]) {
                return i;
            }
            i += 4;
        } else {
            return i;
        }
    }
    bytes.len()
}

/// Checks if byte is a UTF-8 continuation byte.
fn is_continuation(b: u8) -> bool {
    (b & 0b11000000) == 0b10000000
}

/// Decodes a UTF-8 character from bytes.
/// Returns the character and number of bytes consumed.
fn decode_utf8(bytes: &[u8]) -> (char, usize) {
    let b0 = bytes[0];

    if b0 < 0x80 {
        // ASCII
        (b0 as char, 1)
    } else if b0 < 0xE0 {
        // 2-byte sequence
        let cp = ((b0 & 0x1F) as u32) << 6
               | ((bytes[1] & 0x3F) as u32);
        (char.from_u32(cp).unwrap_or('\u{FFFD}'), 2)
    } else if b0 < 0xF0 {
        // 3-byte sequence
        let cp = ((b0 & 0x0F) as u32) << 12
               | ((bytes[1] & 0x3F) as u32) << 6
               | ((bytes[2] & 0x3F) as u32);
        (char.from_u32(cp).unwrap_or('\u{FFFD}'), 3)
    } else {
        // 4-byte sequence
        let cp = ((b0 & 0x07) as u32) << 18
               | ((bytes[1] & 0x3F) as u32) << 12
               | ((bytes[2] & 0x3F) as u32) << 6
               | ((bytes[3] & 0x3F) as u32);
        (char.from_u32(cp).unwrap_or('\u{FFFD}'), 4)
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl Clone for String {
    fn clone(&self) -> String with Alloc {
        String { bytes: self.bytes.clone() }
    }
}

impl Default for String {
    fn default() -> String {
        String.new()
    }
}

impl Eq for String {
    fn eq(&self, other: &String) -> bool {
        self.bytes == other.bytes
    }
}

impl Ord for String {
    fn cmp(&self, other: &String) -> Ordering {
        self.bytes.cmp(&other.bytes)
    }
}

impl PartialOrd for String {
    fn partial_cmp(&self, other: &String) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for String {
    fn hash<H: Hasher>(&self, state: &!H) {
        self.bytes.hash(state);
    }
}

impl Debug for String {
    fn fmt(&self, f: &!Formatter) -> Result<(), Error> {
        write!(f, "\"{}\"", self.as_str())
    }
}

impl Display for String {
    fn fmt(&self, f: &!Formatter) -> Result<(), Error> {
        f.write_str(self.as_str())
    }
}

impl Add<&str> for String {
    type Output = String;

    fn add(self, other: &str) -> String with Alloc {
        let mut result = self;
        result.push_str(other);
        result
    }
}

impl AddAssign<&str> for String {
    fn add_assign(&!self, other: &str) with Alloc {
        self.push_str(other);
    }
}

impl Index<Range<usize>> for String {
    type Output = str;

    fn index(&self, range: Range<usize>) -> &str with Panic {
        &self.as_str()[range]
    }
}

impl From<&str> for String {
    fn from(s: &str) -> String with Alloc {
        String.from_str(s)
    }
}

impl From<char> for String {
    fn from(ch: char) -> String with Alloc {
        let mut s = String.new();
        s.push(ch);
        s
    }
}

impl FromIterator<char> for String {
    fn from_iter<I: IntoIterator<Item = char>>(iter: I) -> String with Alloc {
        let mut s = String.new();
        for ch in iter {
            s.push(ch);
        }
        s
    }
}

impl FromIterator<&str> for String {
    fn from_iter<I: IntoIterator<Item = &str>>(iter: I) -> String with Alloc {
        let mut s = String.new();
        for part in iter {
            s.push_str(part);
        }
        s
    }
}

impl FromIterator<String> for String {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> String with Alloc {
        let mut result = String.new();
        for s in iter {
            result.push_str(s.as_str());
        }
        result
    }
}

impl IntoIterator for String {
    type Item = char;
    type IntoIter = IntoChars;

    fn into_iter(self) -> IntoChars {
        IntoChars { string: self, pos: 0 }
    }
}

/// Owning iterator over characters.
pub struct IntoChars {
    string: String,
    pos: usize,
}

impl Iterator for IntoChars {
    type Item = char;

    fn next(&!self) -> Option<char> {
        if self.pos >= self.string.len() {
            return None;
        }

        let (ch, len) = decode_utf8(&self.string.as_bytes()[self.pos..]);
        self.pos += len;
        Some(ch)
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Creates a formatted String (like format! macro).
pub fn format(args: Arguments) -> String with Alloc {
    let mut s = String.new();
    s.write_fmt(args).unwrap();
    s
}

/// Joins an iterator of strings with a separator.
pub fn join<I, S>(iter: I, sep: &str) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
with Alloc {
    let mut result = String.new();
    let mut first = true;

    for item in iter {
        if !first {
            result.push_str(sep);
        }
        result.push_str(item.as_ref());
        first = false;
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_string() {
        let s = String.new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_from_str() {
        let s = String.from_str("hello");
        assert_eq!(s.len(), 5);
        assert_eq!(s.as_str(), "hello");
    }

    #[test]
    fn test_push() {
        let mut s = String.new();
        s.push('H');
        s.push('i');
        assert_eq!(s.as_str(), "Hi");
    }

    #[test]
    fn test_push_str() {
        let mut s = String.from_str("Hello");
        s.push_str(" World");
        assert_eq!(s.as_str(), "Hello World");
    }

    #[test]
    fn test_pop() {
        let mut s = String.from_str("Hello");
        assert_eq!(s.pop(), Some('o'));
        assert_eq!(s.as_str(), "Hell");
    }

    #[test]
    fn test_chars() {
        let s = String.from_str("Hi!");
        let chars: Vec<char> = s.chars().collect();
        assert_eq!(chars, ['H', 'i', '!']);
    }

    #[test]
    fn test_unicode() {
        let s = String.from_str("héllo 世界");
        assert_eq!(s.chars().count(), 9);
    }

    #[test]
    fn test_contains() {
        let s = String.from_str("hello world");
        assert!(s.contains("world"));
        assert!(!s.contains("foo"));
    }

    #[test]
    fn test_starts_ends_with() {
        let s = String.from_str("hello world");
        assert!(s.starts_with("hello"));
        assert!(s.ends_with("world"));
    }

    #[test]
    fn test_trim() {
        let s = String.from_str("  hello  ");
        assert_eq!(s.trim(), "hello");
    }

    #[test]
    fn test_split() {
        let s = String.from_str("a,b,c");
        let parts: Vec<str> = s.split(',').collect();
        assert_eq!(parts, ["a", "b", "c"]);
    }

    #[test]
    fn test_replace() {
        let s = String.from_str("hello world");
        let replaced = s.replace("world", "universe");
        assert_eq!(replaced.as_str(), "hello universe");
    }

    #[test]
    fn test_to_case() {
        let s = String.from_str("Hello");
        assert_eq!(s.to_lowercase().as_str(), "hello");
        assert_eq!(s.to_uppercase().as_str(), "HELLO");
    }
}

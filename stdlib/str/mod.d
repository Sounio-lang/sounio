//! String Utilities Module for Demetrios
//!
//! Provides string manipulation functions and iterators.

// =============================================================================
// String Extensions
// =============================================================================

impl str {
    /// Check if string is empty
    ///
    /// # Example
    /// ```d
    /// assert("".is_empty());
    /// assert(!"hello".is_empty());
    /// ```
    pub fn is_empty(self: &str) -> bool {
        self.len() == 0
    }

    /// Get string length in bytes
    pub fn len(self: &str) -> usize {
        // Built-in implementation
        extern "C" {
            fn __demetrios_str_len(s: *const u8, s_len: i64) -> i64;
        }
        unsafe { __demetrios_str_len(self.as_ptr(), self.byte_len() as i64) as usize }
    }

    /// Trim whitespace from both ends
    ///
    /// # Example
    /// ```d
    /// let s = "  hello  ".trim();
    /// assert(s == "hello");
    /// ```
    pub fn trim(self: &str) -> &str {
        self.trim_start().trim_end()
    }

    /// Trim whitespace from start
    pub fn trim_start(self: &str) -> &str {
        var start: usize = 0;
        for c in self.chars() {
            match c {
                ' ' | '\t' | '\n' | '\r' => start = start + c.len_utf8(),
                _ => break,
            }
        }
        &self[start..]
    }

    /// Trim whitespace from end
    pub fn trim_end(self: &str) -> &str {
        var end = self.len();
        for c in self.chars().rev() {
            match c {
                ' ' | '\t' | '\n' | '\r' => end = end - c.len_utf8(),
                _ => break,
            }
        }
        &self[..end]
    }

    /// Split string by a delimiter
    ///
    /// # Example
    /// ```d
    /// let parts: Vec<&str> = "a,b,c".split(',').collect();
    /// assert(parts.len() == 3);
    /// assert(parts[0] == "a");
    /// ```
    pub fn split(self: &str, delimiter: char) -> Split {
        Split {
            remaining: self,
            delimiter: delimiter,
            finished: false,
        }
    }

    /// Split string by a string delimiter
    pub fn split_str(self: &str, delimiter: &str) -> SplitStr {
        SplitStr {
            remaining: self,
            delimiter: delimiter,
            finished: false,
        }
    }

    /// Iterate over lines
    ///
    /// Lines are split by '\n' or '\r\n'. The newline characters are not included.
    ///
    /// # Example
    /// ```d
    /// let lines: Vec<&str> = "line1\nline2\nline3".lines().collect();
    /// assert(lines.len() == 3);
    /// ```
    pub fn lines(self: &str) -> Lines {
        Lines {
            remaining: self,
            finished: false,
        }
    }

    /// Check if string starts with prefix
    ///
    /// # Example
    /// ```d
    /// assert("hello world".starts_with("hello"));
    /// assert(!"hello world".starts_with("world"));
    /// ```
    pub fn starts_with(self: &str, prefix: &str) -> bool {
        if prefix.len() > self.len() {
            return false;
        }
        &self[..prefix.len()] == prefix
    }

    /// Check if string ends with suffix
    ///
    /// # Example
    /// ```d
    /// assert("hello world".ends_with("world"));
    /// assert(!"hello world".ends_with("hello"));
    /// ```
    pub fn ends_with(self: &str, suffix: &str) -> bool {
        if suffix.len() > self.len() {
            return false;
        }
        &self[(self.len() - suffix.len())..] == suffix
    }

    /// Check if string contains substring
    ///
    /// # Example
    /// ```d
    /// assert("hello world".contains("lo wo"));
    /// assert(!"hello world".contains("xyz"));
    /// ```
    pub fn contains(self: &str, needle: &str) -> bool {
        self.find(needle).is_some()
    }

    /// Find first occurrence of substring
    ///
    /// Returns byte offset of first match, or None if not found.
    ///
    /// # Example
    /// ```d
    /// assert("hello".find("ll") == Some(2));
    /// assert("hello".find("x") == None);
    /// ```
    pub fn find(self: &str, needle: &str) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if needle.len() > self.len() {
            return None;
        }

        for i in 0..(self.len() - needle.len() + 1) {
            if &self[i..(i + needle.len())] == needle {
                return Some(i);
            }
        }
        None
    }

    /// Find last occurrence of substring
    pub fn rfind(self: &str, needle: &str) -> Option<usize> {
        if needle.is_empty() {
            return Some(self.len());
        }
        if needle.len() > self.len() {
            return None;
        }

        var i = self.len() - needle.len();
        loop {
            if &self[i..(i + needle.len())] == needle {
                return Some(i);
            }
            if i == 0 {
                break;
            }
            i = i - 1;
        }
        None
    }

    /// Find first occurrence of character
    pub fn find_char(self: &str, needle: char) -> Option<usize> {
        for (i, c) in self.char_indices() {
            if c == needle {
                return Some(i);
            }
        }
        None
    }

    /// Find last occurrence of character
    pub fn rfind_char(self: &str, needle: char) -> Option<usize> {
        var last: Option<usize> = None;
        for (i, c) in self.char_indices() {
            if c == needle {
                last = Some(i);
            }
        }
        last
    }

    /// Replace all occurrences of a pattern with replacement
    ///
    /// # Example
    /// ```d
    /// let s = "hello world".replace("world", "rust");
    /// assert(s == "hello rust");
    /// ```
    pub fn replace(self: &str, from: &str, to: &str) -> String {
        if from.is_empty() {
            return self.to_string();
        }

        var result = String::new();
        var remaining = self;

        while let Some(pos) = remaining.find(from) {
            result.push_str(&remaining[..pos]);
            result.push_str(to);
            remaining = &remaining[(pos + from.len())..];
        }
        result.push_str(remaining);

        result
    }

    /// Convert to uppercase
    pub fn to_uppercase(self: &str) -> String {
        var result = String::new();
        for c in self.chars() {
            result.push(c.to_uppercase());
        }
        result
    }

    /// Convert to lowercase
    pub fn to_lowercase(self: &str) -> String {
        var result = String::new();
        for c in self.chars() {
            result.push(c.to_lowercase());
        }
        result
    }

    /// Repeat string n times
    ///
    /// # Example
    /// ```d
    /// assert("ab".repeat(3) == "ababab");
    /// ```
    pub fn repeat(self: &str, n: usize) -> String {
        var result = String::with_capacity(self.len() * n);
        for _ in 0..n {
            result.push_str(self);
        }
        result
    }

    /// Parse string to a value
    ///
    /// # Example
    /// ```d
    /// let n: i32 = "42".parse().unwrap();
    /// let f: f64 = "3.14".parse().unwrap();
    /// ```
    pub fn parse<T>(self: &str) -> Result<T, ParseError>
    where T: FromStr
    {
        T::from_str(self)
    }

    /// Convert to owned String
    pub fn to_string(self: &str) -> String {
        String::from(self)
    }

    /// Get character iterator
    pub fn chars(self: &str) -> Chars {
        Chars {
            bytes: self.as_bytes(),
            pos: 0,
        }
    }

    /// Get character iterator with byte indices
    pub fn char_indices(self: &str) -> CharIndices {
        CharIndices {
            bytes: self.as_bytes(),
            pos: 0,
        }
    }

    /// Get byte iterator
    pub fn bytes(self: &str) -> impl Iterator<Item = u8> {
        self.as_bytes().iter().copied()
    }

    /// Get raw byte slice
    pub fn as_bytes(self: &str) -> &[u8] {
        // Built-in: str is stored as UTF-8 bytes
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Get raw pointer
    pub fn as_ptr(self: &str) -> *const u8 {
        // Built-in
        self as *const str as *const u8
    }
}

// =============================================================================
// Iterators
// =============================================================================

/// Iterator over string lines
pub struct Lines {
    remaining: &str,
    finished: bool,
}

impl Iterator for Lines {
    type Item = &str;

    fn next(self: &mut Lines) -> Option<&str> {
        if self.finished {
            return None;
        }

        match self.remaining.find_char('\n') {
            Some(pos) => {
                // Check for \r\n
                let end = if pos > 0 && self.remaining.as_bytes()[pos - 1] == '\r' as u8 {
                    pos - 1
                } else {
                    pos
                };
                let line = &self.remaining[..end];
                self.remaining = &self.remaining[(pos + 1)..];
                Some(line)
            },
            None => {
                if self.remaining.is_empty() {
                    self.finished = true;
                    None
                } else {
                    let line = self.remaining;
                    self.remaining = "";
                    self.finished = true;
                    Some(line)
                }
            },
        }
    }
}

/// Iterator over string split by character
pub struct Split {
    remaining: &str,
    delimiter: char,
    finished: bool,
}

impl Iterator for Split {
    type Item = &str;

    fn next(self: &mut Split) -> Option<&str> {
        if self.finished {
            return None;
        }

        match self.remaining.find_char(self.delimiter) {
            Some(pos) => {
                let part = &self.remaining[..pos];
                self.remaining = &self.remaining[(pos + self.delimiter.len_utf8())..];
                Some(part)
            },
            None => {
                let part = self.remaining;
                self.remaining = "";
                self.finished = true;
                Some(part)
            },
        }
    }
}

/// Iterator over string split by string delimiter
pub struct SplitStr {
    remaining: &str,
    delimiter: &str,
    finished: bool,
}

impl Iterator for SplitStr {
    type Item = &str;

    fn next(self: &mut SplitStr) -> Option<&str> {
        if self.finished {
            return None;
        }

        match self.remaining.find(self.delimiter) {
            Some(pos) => {
                let part = &self.remaining[..pos];
                self.remaining = &self.remaining[(pos + self.delimiter.len())..];
                Some(part)
            },
            None => {
                let part = self.remaining;
                self.remaining = "";
                self.finished = true;
                Some(part)
            },
        }
    }
}

/// Character iterator
pub struct Chars {
    bytes: &[u8],
    pos: usize,
}

impl Iterator for Chars {
    type Item = char;

    fn next(self: &mut Chars) -> Option<char> {
        if self.pos >= self.bytes.len() {
            return None;
        }

        let first = self.bytes[self.pos];
        let width = if first < 0x80 {
            1
        } else if first < 0xE0 {
            2
        } else if first < 0xF0 {
            3
        } else {
            4
        };

        if self.pos + width > self.bytes.len() {
            return None;
        }

        // Decode UTF-8
        let c = match width {
            1 => first as u32,
            2 => {
                ((first & 0x1F) as u32) << 6 |
                (self.bytes[self.pos + 1] & 0x3F) as u32
            },
            3 => {
                ((first & 0x0F) as u32) << 12 |
                ((self.bytes[self.pos + 1] & 0x3F) as u32) << 6 |
                (self.bytes[self.pos + 2] & 0x3F) as u32
            },
            4 => {
                ((first & 0x07) as u32) << 18 |
                ((self.bytes[self.pos + 1] & 0x3F) as u32) << 12 |
                ((self.bytes[self.pos + 2] & 0x3F) as u32) << 6 |
                (self.bytes[self.pos + 3] & 0x3F) as u32
            },
            _ => return None,
        };

        self.pos = self.pos + width;
        char::from_u32(c)
    }
}

/// Character iterator with byte indices
pub struct CharIndices {
    bytes: &[u8],
    pos: usize,
}

impl Iterator for CharIndices {
    type Item = (usize, char);

    fn next(self: &mut CharIndices) -> Option<(usize, char)> {
        let idx = self.pos;
        if self.pos >= self.bytes.len() {
            return None;
        }

        let first = self.bytes[self.pos];
        let width = if first < 0x80 {
            1
        } else if first < 0xE0 {
            2
        } else if first < 0xF0 {
            3
        } else {
            4
        };

        if self.pos + width > self.bytes.len() {
            return None;
        }

        // Decode UTF-8 (same as Chars)
        let c = match width {
            1 => first as u32,
            2 => {
                ((first & 0x1F) as u32) << 6 |
                (self.bytes[self.pos + 1] & 0x3F) as u32
            },
            3 => {
                ((first & 0x0F) as u32) << 12 |
                ((self.bytes[self.pos + 1] & 0x3F) as u32) << 6 |
                (self.bytes[self.pos + 2] & 0x3F) as u32
            },
            4 => {
                ((first & 0x07) as u32) << 18 |
                ((self.bytes[self.pos + 1] & 0x3F) as u32) << 12 |
                ((self.bytes[self.pos + 2] & 0x3F) as u32) << 6 |
                (self.bytes[self.pos + 3] & 0x3F) as u32
            },
            _ => return None,
        };

        self.pos = self.pos + width;
        match char::from_u32(c) {
            Some(ch) => Some((idx, ch)),
            None => None,
        }
    }
}

// =============================================================================
// String Builder
// =============================================================================

impl String {
    /// Create new empty string
    pub fn new() -> String {
        String::with_capacity(0)
    }

    /// Create string with initial capacity
    pub fn with_capacity(capacity: usize) -> String {
        // Built-in implementation
        extern "C" {
            fn __demetrios_string_new(capacity: i64) -> String;
        }
        unsafe { __demetrios_string_new(capacity as i64) }
    }

    /// Create string from str slice
    pub fn from(s: &str) -> String {
        var result = String::with_capacity(s.len());
        result.push_str(s);
        result
    }

    /// Create string from raw parts (unsafe)
    pub unsafe fn from_raw_parts(ptr: *mut u8, len: usize) -> String {
        // Built-in implementation
        extern "C" {
            fn __demetrios_string_from_raw(ptr: *mut u8, len: i64) -> String;
        }
        __demetrios_string_from_raw(ptr, len as i64)
    }

    /// Push a character
    pub fn push(self: &mut String, c: char) {
        var buf = [0u8; 4];
        let len = c.encode_utf8(&mut buf);
        for i in 0..len {
            self.push_byte(buf[i]);
        }
    }

    /// Push a byte (unsafe - must maintain UTF-8)
    fn push_byte(self: &mut String, b: u8) {
        extern "C" {
            fn __demetrios_string_push_byte(s: &mut String, b: u8);
        }
        unsafe { __demetrios_string_push_byte(self, b) }
    }

    /// Push a string slice
    pub fn push_str(self: &mut String, s: &str) {
        for c in s.chars() {
            self.push(c);
        }
    }

    /// Get as str slice
    pub fn as_str(self: &String) -> &str {
        // Built-in: String derefs to &str
        &self[..]
    }

    /// Clear the string
    pub fn clear(self: &mut String) {
        extern "C" {
            fn __demetrios_string_clear(s: &mut String);
        }
        unsafe { __demetrios_string_clear(self) }
    }

    /// Get length in bytes
    pub fn len(self: &String) -> usize {
        self.as_str().len()
    }

    /// Check if empty
    pub fn is_empty(self: &String) -> bool {
        self.len() == 0
    }

    /// Get capacity
    pub fn capacity(self: &String) -> usize {
        extern "C" {
            fn __demetrios_string_capacity(s: &String) -> i64;
        }
        unsafe { __demetrios_string_capacity(self) as usize }
    }

    /// Reserve capacity for additional bytes
    pub fn reserve(self: &mut String, additional: usize) {
        extern "C" {
            fn __demetrios_string_reserve(s: &mut String, additional: i64);
        }
        unsafe { __demetrios_string_reserve(self, additional as i64) }
    }
}

// =============================================================================
// String Concatenation
// =============================================================================

/// Concatenate two strings using ++ operator
impl Add<&str> for String {
    type Output = String;

    fn add(self: String, other: &str) -> String {
        var result = self;
        result.push_str(other);
        result
    }
}

impl Add<&str> for &str {
    type Output = String;

    fn add(self: &str, other: &str) -> String {
        var result = String::from(self);
        result.push_str(other);
        result
    }
}

// =============================================================================
// Character Extensions
// =============================================================================

impl char {
    /// Check if ASCII digit
    pub fn is_ascii_digit(self: char) -> bool {
        self >= '0' && self <= '9'
    }

    /// Check if ASCII hexadecimal digit
    pub fn is_ascii_hexdigit(self: char) -> bool {
        (self >= '0' && self <= '9') ||
        (self >= 'a' && self <= 'f') ||
        (self >= 'A' && self <= 'F')
    }

    /// Check if ASCII alphabetic
    pub fn is_ascii_alphabetic(self: char) -> bool {
        (self >= 'a' && self <= 'z') || (self >= 'A' && self <= 'Z')
    }

    /// Check if ASCII alphanumeric
    pub fn is_ascii_alphanumeric(self: char) -> bool {
        self.is_ascii_alphabetic() || self.is_ascii_digit()
    }

    /// Check if ASCII whitespace
    pub fn is_ascii_whitespace(self: char) -> bool {
        self == ' ' || self == '\t' || self == '\n' || self == '\r'
    }

    /// Convert to uppercase
    pub fn to_uppercase(self: char) -> char {
        if self >= 'a' && self <= 'z' {
            ((self as u32) - 32) as char
        } else {
            self
        }
    }

    /// Convert to lowercase
    pub fn to_lowercase(self: char) -> char {
        if self >= 'A' && self <= 'Z' {
            ((self as u32) + 32) as char
        } else {
            self
        }
    }

    /// Get UTF-8 encoded length
    pub fn len_utf8(self: char) -> usize {
        let c = self as u32;
        if c < 0x80 {
            1
        } else if c < 0x800 {
            2
        } else if c < 0x10000 {
            3
        } else {
            4
        }
    }

    /// Encode as UTF-8 into buffer
    pub fn encode_utf8(self: char, buf: &mut [u8; 4]) -> usize {
        let c = self as u32;
        if c < 0x80 {
            buf[0] = c as u8;
            1
        } else if c < 0x800 {
            buf[0] = (0xC0 | (c >> 6)) as u8;
            buf[1] = (0x80 | (c & 0x3F)) as u8;
            2
        } else if c < 0x10000 {
            buf[0] = (0xE0 | (c >> 12)) as u8;
            buf[1] = (0x80 | ((c >> 6) & 0x3F)) as u8;
            buf[2] = (0x80 | (c & 0x3F)) as u8;
            3
        } else {
            buf[0] = (0xF0 | (c >> 18)) as u8;
            buf[1] = (0x80 | ((c >> 12) & 0x3F)) as u8;
            buf[2] = (0x80 | ((c >> 6) & 0x3F)) as u8;
            buf[3] = (0x80 | (c & 0x3F)) as u8;
            4
        }
    }

    /// Convert code point to char
    pub fn from_u32(c: u32) -> Option<char> {
        // Valid Unicode scalar values
        if c <= 0x10FFFF && !(c >= 0xD800 && c <= 0xDFFF) {
            Some(c as char)
        } else {
            None
        }
    }

    /// Convert to string
    pub fn to_string(self: char) -> String {
        var s = String::with_capacity(4);
        s.push(self);
        s
    }
}

// =============================================================================
// Formatting
// =============================================================================

/// Format a string with arguments
///
/// Uses {} as placeholder.
///
/// # Example
/// ```d
/// let s = format("Hello, {}!", "world");
/// assert(s == "Hello, world!");
/// ```
pub fn format(template: &str, args: ...) -> String {
    var result = String::new();
    var arg_idx: usize = 0;
    var in_placeholder = false;

    for c in template.chars() {
        if in_placeholder {
            if c == '}' {
                // Insert argument
                if arg_idx < args.len() {
                    result.push_str(&args[arg_idx].to_string());
                    arg_idx = arg_idx + 1;
                } else {
                    result.push_str("{}");
                }
                in_placeholder = false;
            } else {
                result.push('{');
                result.push(c);
                in_placeholder = false;
            }
        } else if c == '{' {
            in_placeholder = true;
        } else {
            result.push(c);
        }
    }

    if in_placeholder {
        result.push('{');
    }

    result
}

// =============================================================================
// Tests
// =============================================================================

fn test_trim() {
    assert("  hello  ".trim() == "hello");
    assert("hello".trim() == "hello");
    assert("".trim() == "");
    assert("   ".trim() == "");
}

fn test_split() {
    let parts: Vec<&str> = "a,b,c".split(',').collect();
    assert(parts.len() == 3);
    assert(parts[0] == "a");
    assert(parts[1] == "b");
    assert(parts[2] == "c");
}

fn test_lines() {
    let lines: Vec<&str> = "line1\nline2\nline3".lines().collect();
    assert(lines.len() == 3);
    assert(lines[0] == "line1");
    assert(lines[1] == "line2");
    assert(lines[2] == "line3");
}

fn test_find() {
    assert("hello".find("ll") == Some(2));
    assert("hello".find("x") == None);
    assert("hello".find("") == Some(0));
}

fn test_replace() {
    assert("hello world".replace("world", "rust") == "hello rust");
    assert("aaa".replace("a", "bb") == "bbbbbb");
}

fn test_starts_ends_with() {
    assert("hello world".starts_with("hello"));
    assert(!"hello world".starts_with("world"));
    assert("hello world".ends_with("world"));
    assert(!"hello world".ends_with("hello"));
}

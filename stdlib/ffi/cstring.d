/// C String Types for FFI
///
/// This module provides types for working with null-terminated C strings,
/// ensuring safe interoperability with C code.

module ffi::cstring;

use ffi::ctypes::*;

// =============================================================================
// CStr - Borrowed C String
// =============================================================================

/// A borrowed reference to a null-terminated C string.
///
/// This type represents a borrowed reference to a C string slice.
/// It does not own the underlying data and is analogous to `&str` for D strings.
///
/// # Safety
///
/// The underlying memory must:
/// - Contain a valid null terminator
/// - Remain valid for the lifetime of the CStr
/// - Not be modified while the CStr exists
///
/// # Example
///
/// ```d
/// extern "C" {
///     fn getenv(name: *const c_char) -> *const c_char;
/// }
///
/// let name = CStr::from_bytes_with_nul(b"HOME\0").unwrap();
/// let value = unsafe { CStr::from_ptr(getenv(name.as_ptr())) };
/// ```
pub struct CStr {
    /// Pointer to the first byte
    ptr: *const c_char,
    /// Length not including null terminator
    len: usize,
}

impl CStr {
    /// Create a CStr from a pointer to a null-terminated string.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid null-terminated C string.
    /// The string must remain valid for the lifetime of the returned CStr.
    pub unsafe fn from_ptr(ptr: *const c_char) -> Self {
        if is_null(ptr) {
            return CStr { ptr: ptr, len: 0 };
        }
        let len = strlen(ptr);
        CStr { ptr: ptr, len: len }
    }

    /// Create a CStr from a byte slice that includes a null terminator.
    ///
    /// Returns None if the slice doesn't end with a null byte or contains
    /// interior null bytes.
    pub fn from_bytes_with_nul(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() {
            return None;
        }

        // Check that the last byte is null
        if bytes[bytes.len() - 1] != 0 {
            return None;
        }

        // Check for interior nulls
        for i in 0..bytes.len() - 1 {
            if bytes[i] == 0 {
                return None;
            }
        }

        Some(CStr {
            ptr: bytes.as_ptr() as *const c_char,
            len: bytes.len() - 1,
        })
    }

    /// Get the pointer to the C string.
    pub fn as_ptr(&self) -> *const c_char {
        self.ptr
    }

    /// Get the length of the string (not including null terminator).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the string is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Convert to a byte slice (not including null terminator).
    pub fn to_bytes(&self) -> &[u8] {
        unsafe {
            slice_from_raw_parts(self.ptr as *const u8, self.len)
        }
    }

    /// Convert to a byte slice including the null terminator.
    pub fn to_bytes_with_nul(&self) -> &[u8] {
        unsafe {
            slice_from_raw_parts(self.ptr as *const u8, self.len + 1)
        }
    }

    /// Try to convert to a D string.
    ///
    /// Returns None if the string contains invalid UTF-8.
    pub fn to_str(&self) -> Option<string> {
        let bytes = self.to_bytes();
        string::from_utf8(bytes)
    }

    /// Convert to a D string, replacing invalid UTF-8 with replacement characters.
    pub fn to_string_lossy(&self) -> string {
        let bytes = self.to_bytes();
        string::from_utf8_lossy(bytes)
    }
}

// =============================================================================
// CString - Owned C String
// =============================================================================

/// An owned, null-terminated C string.
///
/// This type owns a null-terminated string buffer and will deallocate it
/// when dropped. It is analogous to `String` for D strings.
///
/// # Example
///
/// ```d
/// let greeting = CString::new("Hello, World!").unwrap();
///
/// extern "C" {
///     fn puts(s: *const c_char) -> c_int;
/// }
///
/// unsafe { puts(greeting.as_ptr()); }
/// ```
pub linear struct CString {
    /// Pointer to the owned buffer
    ptr: *mut c_char,
    /// Length not including null terminator
    len: usize,
    /// Capacity of the buffer (including null terminator)
    cap: usize,
}

impl CString {
    /// Create a new CString from a D string.
    ///
    /// Returns None if the string contains interior null bytes.
    pub fn new(s: string) -> Option<Self> with Alloc {
        Self::from_bytes(s.as_bytes())
    }

    /// Create a new CString from a byte slice.
    ///
    /// Returns None if the slice contains null bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> with Alloc {
        // Check for interior nulls
        for byte in bytes {
            if *byte == 0 {
                return None;
            }
        }

        let len = bytes.len();
        let cap = len + 1;

        // Allocate buffer
        let ptr = malloc(cap) as *mut c_char;
        if is_null(ptr) {
            return None;
        }

        // Copy data
        unsafe {
            memcpy(ptr as *mut c_void, bytes.as_ptr() as *const c_void, len);
            // Add null terminator
            write(offset_mut(ptr, len as isize), 0 as c_char);
        }

        Some(CString { ptr: ptr, len: len, cap: cap })
    }

    /// Create a CString from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must have been allocated with the same allocator
    /// and must be null-terminated.
    pub unsafe fn from_raw(ptr: *mut c_char) -> Self {
        let len = strlen(ptr as *const c_char);
        CString { ptr: ptr, len: len, cap: len + 1 }
    }

    /// Create a CString from a byte vector, adding a null terminator.
    ///
    /// Returns None if the vector contains null bytes.
    pub fn from_vec(mut bytes: Vec<u8>) -> Option<Self> with Alloc {
        // Check for interior nulls
        for byte in bytes.iter() {
            if *byte == 0 {
                return None;
            }
        }

        bytes.push(0);
        let len = bytes.len() - 1;
        let cap = bytes.capacity();
        let ptr = bytes.into_raw_parts().0 as *mut c_char;

        Some(CString { ptr: ptr, len: len, cap: cap })
    }

    /// Consume the CString and return the raw pointer.
    ///
    /// The caller is responsible for freeing the memory.
    pub fn into_raw(self) -> *mut c_char {
        let ptr = self.ptr;
        // Prevent destructor from freeing
        std::mem::forget(self);
        ptr
    }

    /// Get the pointer to the C string.
    pub fn as_ptr(&self) -> *const c_char {
        self.ptr as *const c_char
    }

    /// Get a mutable pointer to the C string.
    pub fn as_mut_ptr(&mut self) -> *mut c_char {
        self.ptr
    }

    /// Get the length of the string (not including null terminator).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the string is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow as a CStr.
    pub fn as_cstr(&self) -> CStr {
        CStr { ptr: self.ptr as *const c_char, len: self.len }
    }

    /// Convert to a byte slice (not including null terminator).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            slice_from_raw_parts(self.ptr as *const u8, self.len)
        }
    }

    /// Convert to a byte slice including the null terminator.
    pub fn as_bytes_with_nul(&self) -> &[u8] {
        unsafe {
            slice_from_raw_parts(self.ptr as *const u8, self.len + 1)
        }
    }

    /// Convert to a D string, consuming the CString.
    ///
    /// Returns an error if the string contains invalid UTF-8.
    pub fn into_string(self) -> Result<string, CString> {
        match self.as_cstr().to_str() {
            Some(s) => {
                let result = s.clone();
                // self will be dropped here
                Ok(result)
            }
            None => Err(self)
        }
    }
}

impl Drop for CString {
    fn drop(&mut self) with Alloc {
        if is_not_null(self.ptr) {
            free(self.ptr as *mut c_void);
        }
    }
}

// =============================================================================
// NulError
// =============================================================================

/// Error type for CString creation when a null byte is found.
pub struct NulError {
    /// Position of the null byte
    pub position: usize,
    /// The original bytes
    pub bytes: Vec<u8>,
}

impl NulError {
    /// Get the position of the null byte that caused the error.
    pub fn nul_position(&self) -> usize {
        self.position
    }

    /// Consume the error, returning the original byte vector.
    pub fn into_vec(self) -> Vec<u8> {
        self.bytes
    }
}

// =============================================================================
// FromBytesWithNulError
// =============================================================================

/// Error type for CStr::from_bytes_with_nul
pub enum FromBytesWithNulError {
    /// The slice didn't end with a null byte
    NotNulTerminated,
    /// An interior null byte was found at the given position
    InteriorNul(usize),
}

// =============================================================================
// IntoStringError
// =============================================================================

/// Error type for CString::into_string when the string contains invalid UTF-8.
pub struct IntoStringError {
    /// The original CString
    pub cstring: CString,
    /// The UTF-8 error
    pub utf8_error: Utf8Error,
}

impl IntoStringError {
    /// Consume the error, returning the original CString.
    pub fn into_cstring(self) -> CString {
        self.cstring
    }
}

// =============================================================================
// Wide String Types (Windows)
// =============================================================================

/// C wide character type (wchar_t)
#[cfg(target_os = "windows")]
pub type wchar_t = u16;

#[cfg(not(target_os = "windows"))]
pub type wchar_t = i32;

/// Borrowed wide string (null-terminated UTF-16 on Windows)
pub struct WCStr {
    ptr: *const wchar_t,
    len: usize,
}

impl WCStr {
    /// Create a WCStr from a pointer to a null-terminated wide string.
    pub unsafe fn from_ptr(ptr: *const wchar_t) -> Self {
        if is_null(ptr) {
            return WCStr { ptr: ptr, len: 0 };
        }
        let len = wcslen(ptr);
        WCStr { ptr: ptr, len: len }
    }

    /// Get the pointer to the wide string.
    pub fn as_ptr(&self) -> *const wchar_t {
        self.ptr
    }

    /// Get the length of the string (not including null terminator).
    pub fn len(&self) -> usize {
        self.len
    }
}

/// Owned wide string
pub linear struct WCString {
    ptr: *mut wchar_t,
    len: usize,
    cap: usize,
}

impl WCString {
    /// Create a new WCString from a D string.
    #[cfg(target_os = "windows")]
    pub fn new(s: string) -> Option<Self> with Alloc {
        // Convert UTF-8 to UTF-16
        let utf16: Vec<u16> = s.encode_utf16().collect();
        Self::from_vec(utf16)
    }

    /// Create a WCString from a vector of wide characters.
    pub fn from_vec(mut chars: Vec<wchar_t>) -> Option<Self> with Alloc {
        // Check for interior nulls
        for ch in chars.iter() {
            if *ch == 0 {
                return None;
            }
        }

        chars.push(0);
        let len = chars.len() - 1;
        let cap = chars.capacity();
        let ptr = chars.into_raw_parts().0 as *mut wchar_t;

        Some(WCString { ptr: ptr, len: len, cap: cap })
    }

    /// Get the pointer to the wide string.
    pub fn as_ptr(&self) -> *const wchar_t {
        self.ptr as *const wchar_t
    }

    /// Get the length of the string.
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for WCString {
    fn drop(&mut self) with Alloc {
        if is_not_null(self.ptr) {
            free(self.ptr as *mut c_void);
        }
    }
}

// =============================================================================
// OS String Types
// =============================================================================

/// Platform-native string slice.
/// - On Windows: UTF-16
/// - On Unix: byte string (usually UTF-8)
#[cfg(target_os = "windows")]
pub type OsStr = WCStr;

#[cfg(not(target_os = "windows"))]
pub type OsStr = CStr;

/// Platform-native owned string.
#[cfg(target_os = "windows")]
pub type OsString = WCString;

#[cfg(not(target_os = "windows"))]
pub type OsString = CString;

// =============================================================================
// C String Functions (from libc)
// =============================================================================

extern "C" {
    /// Get the length of a null-terminated string
    fn strlen(s: *const c_char) -> c_size_t;

    /// Get the length of a null-terminated wide string
    fn wcslen(s: *const wchar_t) -> c_size_t;

    /// Copy a string
    fn strcpy(dst: *mut c_char, src: *const c_char) -> *mut c_char;

    /// Copy at most n characters
    fn strncpy(dst: *mut c_char, src: *const c_char, n: c_size_t) -> *mut c_char;

    /// Concatenate strings
    fn strcat(dst: *mut c_char, src: *const c_char) -> *mut c_char;

    /// Compare strings
    fn strcmp(a: *const c_char, b: *const c_char) -> c_int;

    /// Compare at most n characters
    fn strncmp(a: *const c_char, b: *const c_char, n: c_size_t) -> c_int;

    /// Find a character in a string
    fn strchr(s: *const c_char, c: c_int) -> *const c_char;

    /// Find the last occurrence of a character
    fn strrchr(s: *const c_char, c: c_int) -> *const c_char;

    /// Find a substring
    fn strstr(haystack: *const c_char, needle: *const c_char) -> *const c_char;

    /// Duplicate a string (allocates memory)
    fn strdup(s: *const c_char) -> *mut c_char;

    /// Allocate memory
    fn malloc(size: c_size_t) -> *mut c_void;

    /// Free memory
    fn free(ptr: *mut c_void);

    /// Reallocate memory
    fn realloc(ptr: *mut c_void, size: c_size_t) -> *mut c_void;

    /// Allocate zeroed memory
    fn calloc(count: c_size_t, size: c_size_t) -> *mut c_void;
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Create a slice from raw parts
unsafe fn slice_from_raw_parts<T>(ptr: *const T, len: usize) -> &[T] {
    // Implementation provided by compiler intrinsic
    __builtin_slice_from_raw_parts(ptr, len)
}

/// Convert a D string to a CString, for use in FFI calls
///
/// This is a convenience function that handles the common pattern of
/// converting a string for a single FFI call.
///
/// # Example
///
/// ```d
/// extern "C" {
///     fn puts(s: *const c_char) -> c_int;
/// }
///
/// with_cstring("Hello, World!", |s| {
///     unsafe { puts(s.as_ptr()) }
/// });
/// ```
pub fn with_cstring<T>(s: string, f: fn(CStr) -> T) -> Option<T> with Alloc {
    let cstring = CString::new(s)?;
    Some(f(cstring.as_cstr()))
}

/// Safely get a string from an environment variable
pub fn getenv_safe(name: &str) -> Option<string> with IO, Alloc {
    extern "C" {
        fn getenv(name: *const c_char) -> *const c_char;
    }

    let cname = CString::new(name.to_string())?;
    let result = unsafe { getenv(cname.as_ptr()) };

    if is_null(result) {
        None
    } else {
        let cstr = unsafe { CStr::from_ptr(result) };
        cstr.to_str()
    }
}

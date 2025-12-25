/// Foreign Function Interface (FFI) Module
///
/// This module provides comprehensive support for interoperating with foreign code,
/// primarily C libraries. It includes:
///
/// - C type aliases for portable FFI
/// - String types for C string interop (CStr, CString)
/// - Dynamic library loading and symbol resolution
/// - Function pointer and callback support
/// - Panic safety at FFI boundaries
///
/// # Quick Start
///
/// ```d
/// use ffi::*;
///
/// // Declare external C functions
/// extern "C" {
///     fn printf(format: *const c_char, ...) -> c_int;
///     fn malloc(size: c_size_t) -> *mut c_void;
///     fn free(ptr: *mut c_void);
/// }
///
/// // Use CString for safe string passing
/// let msg = CString::new("Hello, %s!\n").unwrap();
/// let name = CString::new("World").unwrap();
/// unsafe {
///     printf(msg.as_ptr(), name.as_ptr());
/// }
///
/// // Load a dynamic library
/// let lib = Library::open("libmath.so")?;
/// let sin: fn(f64) -> f64 = unsafe { lib.get_fn("sin")? };
/// println!("sin(0.5) = {}", sin(0.5));
/// ```
///
/// # Safety
///
/// FFI code is inherently unsafe. The types in this module help ensure safety
/// where possible, but careful attention to memory management and type
/// compatibility is required.
///
/// Key safety considerations:
///
/// - Always ensure C strings are null-terminated
/// - Never pass D strings directly to C functions
/// - Be careful with pointer lifetimes
/// - Handle panics at FFI boundaries to prevent undefined behavior
/// - Verify ABI compatibility when loading dynamic libraries

module ffi;

// Re-export all FFI modules
pub use ffi::ctypes::*;
pub use ffi::cstring::*;
pub use ffi::library::*;
pub use ffi::callback::*;

// =============================================================================
// Convenience Macros
// =============================================================================

/// Macro for declaring extern functions with proper attributes.
///
/// # Example
///
/// ```d
/// extern_fn! {
///     "C" {
///         fn puts(s: *const c_char) -> c_int;
///         fn getenv(name: *const c_char) -> *const c_char;
///     }
/// }
/// ```
#[macro_export]
macro_rules! extern_fn {
    ($abi:literal { $($fn_decl:tt)* }) => {
        extern $abi {
            $($fn_decl)*
        }
    };
}

/// Macro for creating a CString literal at compile time.
///
/// Note: This requires compiler support for const evaluation.
///
/// # Example
///
/// ```d
/// const GREETING: &CStr = c_str!("Hello, World!");
/// ```
#[macro_export]
macro_rules! c_str {
    ($s:literal) => {
        unsafe { CStr::from_bytes_with_nul_unchecked(concat!($s, "\0").as_bytes()) }
    };
}

/// Macro for catching panics at FFI boundaries.
///
/// # Example
///
/// ```d
/// extern "C" fn callback(data: *mut c_void) -> c_int {
///     ffi_safe! {
///         // Code that might panic
///         process(data);
///         0
///     } else {
///         -1  // Return value on panic
///     }
/// }
/// ```
#[macro_export]
macro_rules! ffi_safe {
    ($body:block else { $on_panic:expr }) => {
        catch_panic(|| $body, $on_panic)
    };
}

// =============================================================================
// Platform Detection
// =============================================================================

/// Platform information for FFI.
pub mod platform {
    /// Target operating system.
    #[cfg(target_os = "windows")]
    pub const OS: &str = "windows";

    #[cfg(target_os = "linux")]
    pub const OS: &str = "linux";

    #[cfg(target_os = "macos")]
    pub const OS: &str = "macos";

    #[cfg(target_os = "freebsd")]
    pub const OS: &str = "freebsd";

    /// Target architecture.
    #[cfg(target_arch = "x86_64")]
    pub const ARCH: &str = "x86_64";

    #[cfg(target_arch = "x86")]
    pub const ARCH: &str = "x86";

    #[cfg(target_arch = "aarch64")]
    pub const ARCH: &str = "aarch64";

    #[cfg(target_arch = "arm")]
    pub const ARCH: &str = "arm";

    /// Pointer size in bytes.
    #[cfg(target_pointer_width = "64")]
    pub const POINTER_SIZE: usize = 8;

    #[cfg(target_pointer_width = "32")]
    pub const POINTER_SIZE: usize = 4;

    /// Default library extension.
    #[cfg(target_os = "windows")]
    pub const LIB_EXTENSION: &str = ".dll";

    #[cfg(target_os = "macos")]
    pub const LIB_EXTENSION: &str = ".dylib";

    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    pub const LIB_EXTENSION: &str = ".so";

    /// Default library prefix.
    #[cfg(target_os = "windows")]
    pub const LIB_PREFIX: &str = "";

    #[cfg(not(target_os = "windows"))]
    pub const LIB_PREFIX: &str = "lib";
}

// =============================================================================
// Common FFI Patterns
// =============================================================================

/// Wrapper for nullable pointers that tracks ownership.
pub struct OwnedPtr<T> {
    ptr: *mut T,
    deleter: fn(*mut T),
}

impl<T> OwnedPtr<T> {
    /// Create a new owned pointer with a custom deleter.
    pub fn new(ptr: *mut T, deleter: fn(*mut T)) -> Self {
        OwnedPtr { ptr: ptr, deleter: deleter }
    }

    /// Create an owned pointer that uses `free` as the deleter.
    pub fn with_free(ptr: *mut T) -> Self {
        extern "C" {
            fn free(ptr: *mut c_void);
        }

        fn free_wrapper<T>(ptr: *mut T) {
            unsafe { free(ptr as *mut c_void); }
        }

        OwnedPtr { ptr: ptr, deleter: free_wrapper::<T> }
    }

    /// Get the raw pointer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Get a mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Check if the pointer is null.
    pub fn is_null(&self) -> bool {
        self.ptr as usize == 0
    }

    /// Release ownership and return the raw pointer.
    pub fn into_raw(self) -> *mut T {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }
}

impl<T> Drop for OwnedPtr<T> {
    fn drop(&mut self) {
        if !self.is_null() {
            (self.deleter)(self.ptr);
        }
    }
}

/// Wrapper for arrays allocated by C code.
pub struct OwnedArray<T> {
    ptr: *mut T,
    len: usize,
    deleter: fn(*mut T),
}

impl<T> OwnedArray<T> {
    /// Create a new owned array.
    pub fn new(ptr: *mut T, len: usize, deleter: fn(*mut T)) -> Self {
        OwnedArray { ptr: ptr, len: len, deleter: deleter }
    }

    /// Get the length of the array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a slice view of the array.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get a mutable slice view of the array.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for OwnedArray<T> {
    fn drop(&mut self) {
        if self.ptr as usize != 0 {
            (self.deleter)(self.ptr);
        }
    }
}

// =============================================================================
// Error Handling
// =============================================================================

/// Get the last OS error code.
#[cfg(target_os = "windows")]
pub fn get_last_error() -> c_int {
    extern "system" {
        fn GetLastError() -> u32;
    }
    unsafe { GetLastError() as c_int }
}

#[cfg(not(target_os = "windows"))]
pub fn get_last_error() -> c_int {
    extern "C" {
        fn __errno_location() -> *mut c_int;
    }
    unsafe { *__errno_location() }
}

/// Set the last OS error code.
#[cfg(target_os = "windows")]
pub fn set_last_error(code: c_int) {
    extern "system" {
        fn SetLastError(code: u32);
    }
    unsafe { SetLastError(code as u32); }
}

#[cfg(not(target_os = "windows"))]
pub fn set_last_error(code: c_int) {
    extern "C" {
        fn __errno_location() -> *mut c_int;
    }
    unsafe { *__errno_location() = code; }
}

/// Get the error message for an OS error code.
pub fn error_message(code: c_int) -> string with Alloc {
    #[cfg(target_os = "windows")]
    {
        use ffi::library::get_win32_error;
        get_win32_error(code as u32)
    }

    #[cfg(not(target_os = "windows"))]
    {
        extern "C" {
            fn strerror(errnum: c_int) -> *const c_char;
        }

        let msg = unsafe { strerror(code) };
        if msg as usize == 0 {
            format!("error code {}", code)
        } else {
            unsafe { CStr::from_ptr(msg).to_string_lossy() }
        }
    }
}

// =============================================================================
// Linkage Attributes
// =============================================================================

/// Specify the library to link against.
///
/// This attribute is used on extern blocks to indicate which library
/// provides the declared symbols.
///
/// # Example
///
/// ```d
/// #[link(name = "ssl")]
/// extern "C" {
///     fn SSL_new(ctx: *mut SSL_CTX) -> *mut SSL;
/// }
/// ```
#[attribute]
pub struct link {
    /// Library name (without lib prefix or extension)
    pub name: string,
    /// Library kind: "dylib", "static", "framework"
    pub kind: Option<string>,
}

/// Specify weak linkage for a symbol.
///
/// Weak symbols can be overridden and may not be present at runtime.
#[attribute]
pub struct weak;

/// Specify that a symbol should not be exported.
#[attribute]
pub struct hidden;

// =============================================================================
// Repr Attributes
// =============================================================================

/// C-compatible representation.
///
/// Structs with this attribute have the same layout as the equivalent C struct.
///
/// # Example
///
/// ```d
/// #[repr(C)]
/// pub struct Point {
///     x: c_int,
///     y: c_int,
/// }
/// ```
#[attribute]
pub struct repr_c;

/// Packed representation (no padding).
#[attribute]
pub struct repr_packed;

/// Transparent representation (same as the single field).
#[attribute]
pub struct repr_transparent;

/// Specific alignment requirement.
#[attribute]
pub struct repr_align(pub usize);

// =============================================================================
// Safety Utilities
// =============================================================================

/// Assert that a type is FFI-safe.
///
/// This is a compile-time check that verifies a type can be safely
/// passed across FFI boundaries.
pub fn assert_ffi_safe<T>() {
    // Compile-time check implemented by compiler
    static_assert!(
        is_ffi_safe::<T>(),
        "type is not FFI-safe"
    );
}

/// Check if a type is FFI-safe.
pub const fn is_ffi_safe<T>() -> bool {
    // Implementation provided by compiler
    __builtin_is_ffi_safe::<T>()
}

/// Marker trait for types that are safe to pass to C.
pub trait CType {}

// Implement CType for primitive types
impl CType for c_char {}
impl CType for c_schar {}
impl CType for c_uchar {}
impl CType for c_short {}
impl CType for c_ushort {}
impl CType for c_int {}
impl CType for c_uint {}
impl CType for c_long {}
impl CType for c_ulong {}
impl CType for c_longlong {}
impl CType for c_ulonglong {}
impl CType for c_float {}
impl CType for c_double {}
impl CType for c_size_t {}
impl CType for c_ssize_t {}
impl CType for bool {}
impl<T> CType for *const T {}
impl<T> CType for *mut T {}

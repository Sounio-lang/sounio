/// C Types Module for FFI
///
/// This module provides type aliases and types that correspond to C types,
/// ensuring proper size and alignment for FFI interoperability.

module ffi::ctypes;

// =============================================================================
// C Integer Types
// =============================================================================

/// C `char` type - 8-bit signed or unsigned depending on platform
pub type c_char = i8;

/// C `signed char` type - always 8-bit signed
pub type c_schar = i8;

/// C `unsigned char` type - always 8-bit unsigned
pub type c_uchar = u8;

/// C `short` type - typically 16-bit signed
pub type c_short = i16;

/// C `unsigned short` type - typically 16-bit unsigned
pub type c_ushort = u16;

/// C `int` type - typically 32-bit signed
pub type c_int = i32;

/// C `unsigned int` type - typically 32-bit unsigned
pub type c_uint = u32;

/// C `long` type - platform dependent (32-bit on Windows, 64-bit on Unix)
#[cfg(target_os = "windows")]
pub type c_long = i32;

#[cfg(not(target_os = "windows"))]
pub type c_long = i64;

/// C `unsigned long` type - platform dependent
#[cfg(target_os = "windows")]
pub type c_ulong = u32;

#[cfg(not(target_os = "windows"))]
pub type c_ulong = u64;

/// C `long long` type - typically 64-bit signed
pub type c_longlong = i64;

/// C `unsigned long long` type - typically 64-bit unsigned
pub type c_ulonglong = u64;

// =============================================================================
// C Floating Point Types
// =============================================================================

/// C `float` type - 32-bit IEEE 754
pub type c_float = f32;

/// C `double` type - 64-bit IEEE 754
pub type c_double = f64;

// =============================================================================
// C Size Types
// =============================================================================

/// C `size_t` type - unsigned pointer-sized integer
pub type c_size_t = usize;

/// C `ssize_t` type - signed pointer-sized integer (POSIX)
pub type c_ssize_t = isize;

/// C `ptrdiff_t` type - signed pointer difference
pub type c_ptrdiff_t = isize;

/// C `intptr_t` type - signed pointer-sized integer
pub type c_intptr_t = isize;

/// C `uintptr_t` type - unsigned pointer-sized integer
pub type c_uintptr_t = usize;

// =============================================================================
// C Fixed-Width Types (from stdint.h)
// =============================================================================

/// C `int8_t` type
pub type c_int8_t = i8;

/// C `int16_t` type
pub type c_int16_t = i16;

/// C `int32_t` type
pub type c_int32_t = i32;

/// C `int64_t` type
pub type c_int64_t = i64;

/// C `uint8_t` type
pub type c_uint8_t = u8;

/// C `uint16_t` type
pub type c_uint16_t = u16;

/// C `uint32_t` type
pub type c_uint32_t = u32;

/// C `uint64_t` type
pub type c_uint64_t = u64;

// =============================================================================
// C Void Type
// =============================================================================

/// C `void` type - used for opaque pointers
/// Represented as an empty enum that cannot be instantiated
pub enum c_void {}

// =============================================================================
// Pointer Types
// =============================================================================

/// Raw pointer to a C type (mutable)
pub type *mut T = &!T;

/// Raw pointer to a C type (const)
pub type *const T = &T;

/// Null pointer constant
pub const NULL: *const c_void = 0 as *const c_void;

/// Check if a pointer is null
pub fn is_null<T>(ptr: *const T) -> bool {
    ptr as usize == 0
}

/// Check if a pointer is not null
pub fn is_not_null<T>(ptr: *const T) -> bool {
    ptr as usize != 0
}

// =============================================================================
// C Boolean Type
// =============================================================================

/// C `_Bool` type (C99)
pub type c_bool = bool;

// =============================================================================
// Platform-Specific Types
// =============================================================================

/// Windows `HANDLE` type
#[cfg(target_os = "windows")]
pub type HANDLE = *mut c_void;

/// Windows `HWND` type
#[cfg(target_os = "windows")]
pub type HWND = *mut c_void;

/// Windows `HMODULE` type
#[cfg(target_os = "windows")]
pub type HMODULE = *mut c_void;

/// Windows `DWORD` type
#[cfg(target_os = "windows")]
pub type DWORD = c_ulong;

/// Windows `BOOL` type
#[cfg(target_os = "windows")]
pub type BOOL = c_int;

/// Windows constants
#[cfg(target_os = "windows")]
pub const TRUE: BOOL = 1;

#[cfg(target_os = "windows")]
pub const FALSE: BOOL = 0;

/// Unix file descriptor
#[cfg(not(target_os = "windows"))]
pub type fd_t = c_int;

/// Unix pid_t type
#[cfg(not(target_os = "windows"))]
pub type pid_t = c_int;

/// Unix uid_t type
#[cfg(not(target_os = "windows"))]
pub type uid_t = c_uint;

/// Unix gid_t type
#[cfg(not(target_os = "windows"))]
pub type gid_t = c_uint;

/// Unix mode_t type
#[cfg(not(target_os = "windows"))]
pub type mode_t = c_uint;

/// Unix off_t type
#[cfg(not(target_os = "windows"))]
pub type off_t = c_long;

/// Unix time_t type
pub type time_t = c_long;

// =============================================================================
// Alignment Helpers
// =============================================================================

/// Get the alignment of a type in bytes
pub fn align_of<T>() -> usize with Alloc {
    // Implementation provided by compiler intrinsic
    __builtin_align_of::<T>()
}

/// Get the size of a type in bytes
pub fn size_of<T>() -> usize {
    // Implementation provided by compiler intrinsic
    __builtin_size_of::<T>()
}

/// Align a value up to the nearest alignment boundary
pub fn align_up(value: usize, align: usize) -> usize {
    let mask = align - 1;
    (value + mask) & !mask
}

/// Align a value down to the nearest alignment boundary
pub fn align_down(value: usize, align: usize) -> usize {
    let mask = align - 1;
    value & !mask
}

/// Check if a value is aligned to a given alignment
pub fn is_aligned(value: usize, align: usize) -> bool {
    value & (align - 1) == 0
}

// =============================================================================
// Pointer Arithmetic
// =============================================================================

/// Offset a pointer by a number of elements
pub fn offset<T>(ptr: *const T, count: isize) -> *const T {
    (ptr as isize + count * size_of::<T>() as isize) as *const T
}

/// Offset a mutable pointer by a number of elements
pub fn offset_mut<T>(ptr: *mut T, count: isize) -> *mut T {
    (ptr as isize + count * size_of::<T>() as isize) as *mut T
}

/// Add to a pointer (unsigned offset)
pub fn add<T>(ptr: *const T, count: usize) -> *const T {
    offset(ptr, count as isize)
}

/// Subtract from a pointer
pub fn sub<T>(ptr: *const T, count: usize) -> *const T {
    offset(ptr, -(count as isize))
}

/// Calculate the difference between two pointers in elements
pub fn diff<T>(a: *const T, b: *const T) -> isize {
    (a as isize - b as isize) / size_of::<T>() as isize
}

// =============================================================================
// Memory Operations
// =============================================================================

/// Copy memory from src to dst (may overlap)
pub fn memmove(dst: *mut c_void, src: *const c_void, len: c_size_t) -> *mut c_void with IO {
    extern "C" {
        fn memmove(dst: *mut c_void, src: *const c_void, len: c_size_t) -> *mut c_void;
    }
    memmove(dst, src, len)
}

/// Copy memory from src to dst (must not overlap)
pub fn memcpy(dst: *mut c_void, src: *const c_void, len: c_size_t) -> *mut c_void with IO {
    extern "C" {
        fn memcpy(dst: *mut c_void, src: *const c_void, len: c_size_t) -> *mut c_void;
    }
    memcpy(dst, src, len)
}

/// Set memory to a value
pub fn memset(dst: *mut c_void, val: c_int, len: c_size_t) -> *mut c_void with IO {
    extern "C" {
        fn memset(dst: *mut c_void, val: c_int, len: c_size_t) -> *mut c_void;
    }
    memset(dst, val, len)
}

/// Compare memory
pub fn memcmp(a: *const c_void, b: *const c_void, len: c_size_t) -> c_int {
    extern "C" {
        fn memcmp(a: *const c_void, b: *const c_void, len: c_size_t) -> c_int;
    }
    memcmp(a, b, len)
}

/// Find a byte in memory
pub fn memchr(ptr: *const c_void, val: c_int, len: c_size_t) -> *const c_void {
    extern "C" {
        fn memchr(ptr: *const c_void, val: c_int, len: c_size_t) -> *const c_void;
    }
    memchr(ptr, val, len)
}

// =============================================================================
// Transmutation (unsafe)
// =============================================================================

/// Transmute a value to a different type
/// This is extremely unsafe and should only be used when necessary for FFI
pub unsafe fn transmute<From, To>(value: From) -> To {
    // Implementation provided by compiler intrinsic
    __builtin_transmute::<From, To>(value)
}

/// Read a value from a raw pointer
pub unsafe fn read<T>(ptr: *const T) -> T with IO {
    // Implementation provided by compiler intrinsic
    __builtin_read(ptr)
}

/// Write a value to a raw pointer
pub unsafe fn write<T>(ptr: *mut T, value: T) with IO {
    // Implementation provided by compiler intrinsic
    __builtin_write(ptr, value)
}

/// Read a value from a raw pointer without moving
pub unsafe fn read_volatile<T>(ptr: *const T) -> T with IO {
    // Implementation provided by compiler intrinsic
    __builtin_read_volatile(ptr)
}

/// Write a value to a raw pointer without optimizing away
pub unsafe fn write_volatile<T>(ptr: *mut T, value: T) with IO {
    // Implementation provided by compiler intrinsic
    __builtin_write_volatile(ptr, value)
}

// =============================================================================
// Zeroing
// =============================================================================

/// Create a zeroed value of type T
/// Only safe for types with no invalid bit patterns (e.g., integers)
pub unsafe fn zeroed<T>() -> T {
    // Implementation provided by compiler intrinsic
    __builtin_zeroed::<T>()
}

/// Create an uninitialized value of type T
/// Using this value before initialization is undefined behavior
pub unsafe fn uninit<T>() -> T {
    // Implementation provided by compiler intrinsic
    __builtin_uninit::<T>()
}

# Foreign Function Interface (FFI)

This chapter covers Sounio's Foreign Function Interface for interoperating with C libraries and other foreign code.

## Overview

Sounio's FFI enables:

- **Calling C functions**: Invoke functions from C libraries
- **Exposing Sounio functions**: Make Sounio functions callable from C
- **Type compatibility**: Map between Sounio and C types
- **Memory safety**: Tools for safe pointer handling
- **Dynamic loading**: Load shared libraries at runtime

## Declaring External Functions

### extern Blocks

Use `extern "C"` blocks to declare C functions:

```sio
extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
    fn strlen(s: *const i8) -> usize;
    fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
}
```

### ABI Specifications

Sounio supports multiple ABI conventions:

```sio
// C calling convention (default for most platforms)
extern "C" {
    fn c_function(x: i32) -> i32;
}

// System calling convention (Windows API uses this)
extern "system" {
    fn GetLastError() -> u32;
    fn SetLastError(code: u32);
}
```

## Pointer Types

### Raw Pointers

Sounio has two raw pointer types for FFI:

```sio
*const T   // Read-only pointer (C: const T*)
*mut T     // Read-write pointer (C: T*)
```

### Null Pointers

Create and check null pointers:

```sio
import ffi::*

// Create null pointers
let const_ptr: *const i32 = null_ptr()
let mut_ptr: *mut i32 = null_mut()

// Check for null
if is_null(ptr) {
    println("Pointer is null")
}

// Compare pointers
if ptr_eq(ptr1, ptr2) {
    println("Pointers are equal")
}
```

### Pointer Operations

```sio
import ffi::ctypes::*

// Get address as integer
let addr: usize = ptr as usize

// Create pointer from address
let ptr: *const u8 = addr as *const u8

// Pointer arithmetic
let next = offset(ptr, 8)     // Offset by 8 elements
let elem = add(ptr, 1)        // Add 1 element
let prev = sub(ptr, 1)        // Subtract 1 element

// Pointer difference
let diff: isize = diff(ptr2, ptr1)
```

## C Type Mappings

### ctypes Module

The `ffi::ctypes` module provides C-compatible type aliases:

```sio
import ffi::ctypes::*

// Integer types
c_char      // i8 (signed char)
c_uchar     // u8 (unsigned char)
c_short     // i16
c_ushort    // u16
c_int       // i32
c_uint      // u32
c_long      // i64 on Unix, i32 on Windows
c_ulong     // u64 on Unix, u32 on Windows
c_longlong  // i64
c_ulonglong // u64

// Floating point
c_float     // f32
c_double    // f64

// Size types
c_size_t    // usize
c_ssize_t   // isize
c_ptrdiff_t // isize

// Void (for opaque pointers)
c_void      // empty enum, cannot be instantiated
```

### Type Mapping Table

| C Type | Sounio Type |
|--------|-------------|
| `char` | `c_char` (i8) |
| `unsigned char` | `c_uchar` (u8) |
| `short` | `c_short` (i16) |
| `int` | `c_int` (i32) |
| `long` | `c_long` (platform-dependent) |
| `long long` | `c_longlong` (i64) |
| `size_t` | `c_size_t` (usize) |
| `float` | `c_float` (f32) |
| `double` | `c_double` (f64) |
| `void*` | `*mut c_void` |
| `const void*` | `*const c_void` |
| `T*` | `*mut T` |
| `const T*` | `*const T` |

## C Strings

### CStr and CString

Use the `ffi::cstring` module for C string handling:

```sio
import ffi::cstring::*

// Create a CString from a Sounio string
let msg = CString::new("Hello, World!").unwrap()

// Get a pointer to pass to C
let ptr: *const c_char = msg.as_ptr()

extern "C" {
    fn puts(s: *const c_char) -> c_int;
}

// Pass to C function
puts(ptr)

// Create a CStr from a C pointer (borrowed)
let c_str = CStr::from_ptr(ptr)
let sounio_string = c_str.to_string()
```

### Safety with C Strings

```sio
import ffi::cstring::*

fn safe_strlen(s: *const c_char) -> usize {
    // Always check for null
    if is_null(s) {
        return 0
    }

    extern "C" {
        fn strlen(s: *const c_char) -> usize;
    }

    return strlen(s)
}
```

## Memory Management

### Allocating and Freeing

```sio
extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
    fn realloc(ptr: *mut c_void, size: usize) -> *mut c_void;
}

fn allocate_buffer(size: usize) -> *mut u8 {
    let ptr = malloc(size) as *mut u8
    if is_null(ptr) {
        panic("allocation failed")
    }
    return ptr
}

fn deallocate_buffer(ptr: *mut u8) {
    free(ptr as *mut c_void)
}
```

### Owned Pointers

Use `OwnedPtr` for automatic cleanup:

```sio
import ffi::OwnedPtr

fn with_managed_memory() {
    // Will be freed automatically when dropped
    let ptr = OwnedPtr::with_free(malloc(1024) as *mut u8)

    // Use the pointer
    let raw = ptr.as_ptr()

    // Automatically freed at end of scope
}

// Custom deleter
extern "C" {
    fn custom_free(ptr: *mut MyType);
}

let ptr = OwnedPtr::new(create_object(), custom_free)
```

## Slices from Raw Pointers

Convert C arrays to Sounio slices:

```sio
import ffi::{slice_from_raw_parts, slice_from_raw_parts_mut}

extern "C" {
    fn get_buffer(len: *mut usize) -> *const u8;
    fn get_mut_buffer(len: usize) -> *mut f32;
}

fn process_buffer() {
    var len: usize = 0
    let ptr = get_buffer(&mut len)

    // Create a slice from raw pointer (unsafe)
    unsafe {
        let slice: &[u8] = slice_from_raw_parts(ptr, len)
        for byte in slice {
            println("{}", byte)
        }
    }
}

fn fill_buffer(n: usize) {
    let ptr = get_mut_buffer(n)

    // Create a mutable slice
    unsafe {
        let slice: &![f32] = slice_from_raw_parts_mut(ptr, n)
        for i in 0..n {
            slice[i] = i as f32
        }
    }
}
```

## Unsafe Operations

### The unsafe Keyword

Mark blocks that bypass safety checks:

```sio
fn read_from_pointer(ptr: *const i32) -> i32 {
    unsafe {
        // Dereference a raw pointer
        return *ptr
    }
}

fn write_to_pointer(ptr: *mut i32, value: i32) {
    unsafe {
        *ptr = value
    }
}
```

### When unsafe Is Required

- Dereferencing raw pointers
- Calling C functions
- Implementing low-level abstractions
- Type transmutation
- Accessing mutable statics

## Calling C Libraries

### Complete Example

```sio
import ffi::*
import ffi::ctypes::*
import ffi::cstring::*

// Declare the C interface
extern "C" {
    fn sqlite3_open(filename: *const c_char, db: *mut *mut c_void) -> c_int;
    fn sqlite3_close(db: *mut c_void) -> c_int;
    fn sqlite3_exec(
        db: *mut c_void,
        sql: *const c_char,
        callback: *const c_void,
        arg: *mut c_void,
        errmsg: *mut *mut c_char
    ) -> c_int;
}

const SQLITE_OK: c_int = 0

fn open_database(path: string) -> Result<*mut c_void, string> {
    let c_path = CString::new(path).unwrap()
    var db: *mut c_void = null_mut()

    let rc = sqlite3_open(c_path.as_ptr(), &mut db)
    if rc != SQLITE_OK {
        return Err("Failed to open database")
    }

    return Ok(db)
}

fn close_database(db: *mut c_void) {
    sqlite3_close(db)
}

fn execute_sql(db: *mut c_void, sql: string) -> Result<(), string> {
    let c_sql = CString::new(sql).unwrap()
    var errmsg: *mut c_char = null_mut()

    let rc = sqlite3_exec(
        db,
        c_sql.as_ptr(),
        null_ptr(),
        null_mut(),
        &mut errmsg
    )

    if rc != SQLITE_OK {
        let err = if is_null(errmsg) {
            "Unknown error".to_string()
        } else {
            CStr::from_ptr(errmsg).to_string()
        }
        return Err(err)
    }

    return Ok(())
}
```

## Exposing Sounio Functions to C

### extern Functions

Mark functions as callable from C:

```sio
// This function can be called from C
#[export_name = "sounio_add"]
pub extern "C" fn add(a: c_int, b: c_int) -> c_int {
    return a + b
}

// With no_mangle to preserve the name
#[no_mangle]
pub extern "C" fn process_data(ptr: *const u8, len: usize) -> c_int {
    if is_null(ptr) {
        return -1
    }

    unsafe {
        let slice = slice_from_raw_parts(ptr, len)
        // Process the data
        return 0
    }
}
```

### Callback Functions

Pass Sounio functions as callbacks to C:

```sio
// C function that takes a callback
extern "C" {
    fn register_callback(cb: fn(c_int) -> c_int);
    fn qsort(
        base: *mut c_void,
        nmemb: usize,
        size: usize,
        compar: fn(*const c_void, *const c_void) -> c_int
    );
}

// Sounio callback function
extern "C" fn my_callback(x: c_int) -> c_int {
    return x * 2
}

fn setup() {
    register_callback(my_callback)
}

// Comparison function for qsort
extern "C" fn compare_ints(a: *const c_void, b: *const c_void) -> c_int {
    unsafe {
        let x = *(a as *const c_int)
        let y = *(b as *const c_int)
        return x - y
    }
}
```

## Dynamic Library Loading

### Loading at Runtime

```sio
import ffi::library::*

fn load_plugin() -> Result<(), string> {
    // Load a shared library
    let lib = Library::open("libplugin.so")?

    // Get a function pointer
    let init_fn: fn() -> c_int = unsafe {
        lib.get_fn("plugin_init")?
    }

    // Call the function
    let result = init_fn()
    if result != 0 {
        return Err("Plugin init failed")
    }

    return Ok(())
}
```

### Platform-Specific Extensions

```sio
import ffi::platform::*

// Library extension varies by platform
let lib_name = format("{}mylib{}", LIB_PREFIX, LIB_EXTENSION)
// Linux: "libmylib.so"
// macOS: "libmylib.dylib"
// Windows: "mylib.dll"
```

## Struct Layout

### repr(C) Attribute

Ensure C-compatible struct layout:

```sio
#[repr(C)]
pub struct Point {
    x: c_double,
    y: c_double,
}

#[repr(C)]
pub struct Buffer {
    data: *mut u8,
    len: usize,
    capacity: usize,
}

// Pass struct to C
extern "C" {
    fn process_point(p: *const Point) -> c_int;
    fn init_buffer(buf: *mut Buffer);
}

fn example() {
    let p = Point { x: 1.0, y: 2.0 }
    process_point(&p)

    var buf = Buffer { data: null_mut(), len: 0, capacity: 0 }
    init_buffer(&mut buf)
}
```

### Packed Structs

Remove padding between fields:

```sio
#[repr(C, packed)]
pub struct PackedData {
    flag: u8,
    value: u32,  // No padding before this
}
```

## Error Handling

### C Error Codes

```sio
import ffi::*

// Get the last OS error
fn check_error() {
    let errno = get_last_error()
    if errno != 0 {
        let msg = error_message(errno)
        println("Error {}: {}", errno, msg)
    }
}

// Set an error code (for exposing to C)
fn set_error(code: c_int) {
    set_last_error(code)
}
```

### Panic Safety

Prevent panics from crossing FFI boundaries:

```sio
import ffi::catch_panic

// Safe callback that won't panic across FFI
extern "C" fn safe_callback(data: *mut c_void) -> c_int {
    catch_panic(|| {
        // Code that might panic
        process(data)
        0  // Return value on success
    }, -1)  // Return value on panic
}
```

## Common FFI Patterns

### Opaque Pointers

Hide implementation details from C:

```sio
// Opaque handle type
pub struct Handle {
    // Private internals
    inner: Box<InternalState>,
}

// C sees only a pointer
type HandlePtr = *mut Handle

#[no_mangle]
pub extern "C" fn create_handle() -> HandlePtr {
    let handle = Box::new(Handle { inner: Box::new(InternalState::new()) })
    Box::into_raw(handle)
}

#[no_mangle]
pub extern "C" fn destroy_handle(ptr: HandlePtr) {
    if !is_null(ptr) {
        unsafe {
            let _ = Box::from_raw(ptr)
            // Automatically dropped
        }
    }
}
```

### Out Parameters

Return values through pointers:

```sio
extern "C" {
    // C function that returns through out parameter
    fn get_value(out: *mut c_int) -> c_int;
}

fn call_with_out_param() -> Option<c_int> {
    var result: c_int = 0

    let status = get_value(&mut result)
    if status == 0 {
        return Some(result)
    }
    return None
}
```

### Array Parameters

Pass arrays to C:

```sio
extern "C" {
    fn process_array(data: *const f64, len: usize) -> f64;
}

fn process_sounio_array(arr: &[f64]) -> f64 {
    return process_array(arr.as_ptr(), arr.len())
}
```

## Safety Guidelines

1. **Always check for null**: Before dereferencing any pointer

2. **Match allocator/deallocator pairs**: Memory allocated with `malloc` must be freed with `free`

3. **Respect ownership**: Know who owns what memory

4. **Validate array bounds**: Ensure length parameters are accurate

5. **Handle errors**: Check return values from C functions

6. **Minimize unsafe blocks**: Keep unsafe code small and well-documented

7. **Use wrapper types**: Create safe Sounio APIs around unsafe FFI

```sio
// Good pattern: safe wrapper around unsafe FFI
pub struct SafeBuffer {
    ptr: *mut u8,
    len: usize,
}

impl SafeBuffer {
    pub fn new(size: usize) -> Option<SafeBuffer> {
        let ptr = malloc(size) as *mut u8
        if is_null(ptr) {
            return None
        }
        Some(SafeBuffer { ptr, len: size })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice_from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for SafeBuffer {
    fn drop(&mut self) {
        free(self.ptr as *mut c_void)
    }
}
```

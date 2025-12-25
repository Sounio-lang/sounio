//! I/O Runtime Support for Sounio
//!
//! This module provides C-compatible runtime functions for file I/O operations
//! that are called from compiled Sounio programs. These functions match the
//! extern "C" declarations in `stdlib/io/mod.d`.
//!
//! # Error Codes
//! - 0: Success
//! - 1: Not found / Does not exist
//! - 2: Permission denied
//! - 3: Invalid input / UTF-8 encoding error
//! - 4: Other error
//!
//! # Memory Management
//! Strings returned through FFI use Box-based allocation. The caller is
//! responsible for calling `__sounio_free_string` to deallocate.

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::OnceLock;

// ============================================================================
// Global State
// ============================================================================

/// Global storage for command-line arguments (set once at program start)
static GLOBAL_ARGS: OnceLock<Vec<String>> = OnceLock::new();

/// Initialize command-line arguments from environment
pub fn init_args() {
    GLOBAL_ARGS.get_or_init(|| env::args().collect());
}

// ============================================================================
// File Operations
// ============================================================================

/// Read entire file contents into a string
///
/// # Safety
/// - `path` must be a valid pointer to UTF-8 encoded bytes
/// - `path_len` must be the exact length of the path string
/// - `out_ptr` and `out_len` must be valid pointers
///
/// # Returns
/// - 0 on success (content written to out_ptr/out_len)
/// - 1 if file not found
/// - 2 if permission denied
/// - 3 if path is not valid UTF-8
/// - 4 for other errors
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_read_file(
    path: *const u8,
    path_len: i64,
    out_ptr: *mut *mut u8,
    out_len: *mut i64,
) -> i32 {
    // Safety: validate inputs
    if path.is_null() || out_ptr.is_null() || out_len.is_null() {
        return 3; // Invalid input
    }

    let path_slice = unsafe { std::slice::from_raw_parts(path, path_len as usize) };
    let path_str = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(_) => return 3, // UTF-8 error
    };

    match fs::read_to_string(path_str) {
        Ok(content) => {
            let bytes = content.into_bytes();
            let len = bytes.len();
            let boxed = bytes.into_boxed_slice();
            let ptr = Box::into_raw(boxed) as *mut u8;

            unsafe {
                *out_ptr = ptr;
                *out_len = len as i64;
            }
            0 // Success
        }
        Err(e) => match e.kind() {
            io::ErrorKind::NotFound => 1,
            io::ErrorKind::PermissionDenied => 2,
            _ => 4,
        },
    }
}

/// Write string contents to a file (creates or overwrites)
///
/// # Safety
/// - `path` and `content` must be valid pointers to UTF-8 bytes
/// - lengths must match the actual data
///
/// # Returns
/// - 0 on success
/// - 2 if permission denied
/// - 3 if path is not valid UTF-8
/// - 4 for other errors
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_write_file(
    path: *const u8,
    path_len: i64,
    content: *const u8,
    content_len: i64,
) -> i32 {
    if path.is_null() || content.is_null() {
        return 3;
    }

    let path_slice = unsafe { std::slice::from_raw_parts(path, path_len as usize) };
    let path_str = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(_) => return 3,
    };

    let content_slice = unsafe { std::slice::from_raw_parts(content, content_len as usize) };

    match fs::write(path_str, content_slice) {
        Ok(()) => 0,
        Err(e) => match e.kind() {
            io::ErrorKind::PermissionDenied => 2,
            _ => 4,
        },
    }
}

/// Append string contents to a file (creates if doesn't exist)
///
/// # Returns
/// - 0 on success
/// - 2 if permission denied
/// - 3 if path is not valid UTF-8
/// - 4 for other errors
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_append_file(
    path: *const u8,
    path_len: i64,
    content: *const u8,
    content_len: i64,
) -> i32 {
    if path.is_null() || content.is_null() {
        return 3;
    }

    let path_slice = unsafe { std::slice::from_raw_parts(path, path_len as usize) };
    let path_str = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(_) => return 3,
    };

    let content_slice = unsafe { std::slice::from_raw_parts(content, content_len as usize) };

    use std::fs::OpenOptions;
    match OpenOptions::new().create(true).append(true).open(path_str) {
        Ok(mut file) => match file.write_all(content_slice) {
            Ok(()) => 0,
            Err(_) => 4,
        },
        Err(e) => match e.kind() {
            io::ErrorKind::PermissionDenied => 2,
            _ => 4,
        },
    }
}

/// Check if a file exists
///
/// # Returns
/// - 1 if file exists
/// - 0 if file does not exist
/// - -1 if path is invalid
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_file_exists(path: *const u8, path_len: i64) -> i32 {
    if path.is_null() {
        return -1;
    }

    let path_slice = unsafe { std::slice::from_raw_parts(path, path_len as usize) };
    let path_str = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    if Path::new(path_str).exists() { 1 } else { 0 }
}

/// Remove a file
///
/// # Returns
/// - 0 on success
/// - 1 if file not found
/// - 2 if permission denied
/// - 4 for other errors
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_remove_file(path: *const u8, path_len: i64) -> i32 {
    if path.is_null() {
        return 3;
    }

    let path_slice = unsafe { std::slice::from_raw_parts(path, path_len as usize) };
    let path_str = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(_) => return 3,
    };

    match fs::remove_file(path_str) {
        Ok(()) => 0,
        Err(e) => match e.kind() {
            io::ErrorKind::NotFound => 1,
            io::ErrorKind::PermissionDenied => 2,
            _ => 4,
        },
    }
}

// ============================================================================
// Process Control
// ============================================================================

/// Exit the program with a status code
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_exit(code: i32) -> ! {
    std::process::exit(code)
}

// ============================================================================
// Environment Access
// ============================================================================

/// Get the number of command-line arguments
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_get_argc() -> i32 {
    init_args();
    GLOBAL_ARGS.get().map(|v| v.len() as i32).unwrap_or(0)
}

/// Get a command-line argument by index
///
/// # Safety
/// - `out_ptr` and `out_len` must be valid pointers
///
/// # Returns
/// - 0 on success
/// - 1 if index out of bounds
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_get_argv(
    index: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut i64,
) -> i32 {
    if out_ptr.is_null() || out_len.is_null() {
        return 3;
    }

    init_args();

    match GLOBAL_ARGS.get() {
        Some(args) => {
            if index < 0 || (index as usize) >= args.len() {
                return 1; // Out of bounds
            }

            let arg = &args[index as usize];
            let bytes = arg.as_bytes().to_vec().into_boxed_slice();
            let len = bytes.len();
            let ptr = Box::into_raw(bytes) as *mut u8;

            unsafe {
                *out_ptr = ptr;
                *out_len = len as i64;
            }
            0
        }
        None => 1,
    }
}

/// Get an environment variable
///
/// # Returns
/// - 0 on success
/// - 1 if variable not found
/// - 3 if name is invalid UTF-8
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_get_env(
    name: *const u8,
    name_len: i64,
    out_ptr: *mut *mut u8,
    out_len: *mut i64,
) -> i32 {
    if name.is_null() || out_ptr.is_null() || out_len.is_null() {
        return 3;
    }

    let name_slice = unsafe { std::slice::from_raw_parts(name, name_len as usize) };
    let name_str = match std::str::from_utf8(name_slice) {
        Ok(s) => s,
        Err(_) => return 3,
    };

    match env::var(name_str) {
        Ok(value) => {
            let bytes = value.into_bytes().into_boxed_slice();
            let len = bytes.len();
            let ptr = Box::into_raw(bytes) as *mut u8;

            unsafe {
                *out_ptr = ptr;
                *out_len = len as i64;
            }
            0
        }
        Err(_) => 1,
    }
}

/// Set an environment variable
///
/// # Returns
/// - 0 on success
/// - 3 if name or value is invalid UTF-8
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_set_env(
    name: *const u8,
    name_len: i64,
    value: *const u8,
    value_len: i64,
) -> i32 {
    if name.is_null() || value.is_null() {
        return 3;
    }

    let name_slice = unsafe { std::slice::from_raw_parts(name, name_len as usize) };
    let name_str = match std::str::from_utf8(name_slice) {
        Ok(s) => s,
        Err(_) => return 3,
    };

    let value_slice = unsafe { std::slice::from_raw_parts(value, value_len as usize) };
    let value_str = match std::str::from_utf8(value_slice) {
        Ok(s) => s,
        Err(_) => return 3,
    };

    // SAFETY: set_var is unsafe in Rust 2024 edition because modifying
    // environment variables can cause data races in multi-threaded programs.
    // This function is intended to be called from D code which expects
    // single-threaded semantics for env var access.
    unsafe {
        env::set_var(name_str, value_str);
    }
    0
}

/// Get the current working directory
///
/// # Returns
/// - 0 on success
/// - 4 on error
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_current_dir(out_ptr: *mut *mut u8, out_len: *mut i64) -> i32 {
    if out_ptr.is_null() || out_len.is_null() {
        return 3;
    }

    match env::current_dir() {
        Ok(path) => {
            let path_str = path.to_string_lossy().into_owned();
            let bytes = path_str.into_bytes().into_boxed_slice();
            let len = bytes.len();
            let ptr = Box::into_raw(bytes) as *mut u8;

            unsafe {
                *out_ptr = ptr;
                *out_len = len as i64;
            }
            0
        }
        Err(_) => 4,
    }
}

// ============================================================================
// Standard Streams
// ============================================================================

/// Print a string to stdout (no newline)
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_print(s: *const u8, len: i64) {
    if s.is_null() {
        return;
    }

    let slice = unsafe { std::slice::from_raw_parts(s, len as usize) };
    if let Ok(text) = std::str::from_utf8(slice) {
        print!("{}", text);
        let _ = io::stdout().flush();
    }
}

/// Print a string to stderr (no newline)
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_eprint(s: *const u8, len: i64) {
    if s.is_null() {
        return;
    }

    let slice = unsafe { std::slice::from_raw_parts(s, len as usize) };
    if let Ok(text) = std::str::from_utf8(slice) {
        eprint!("{}", text);
        let _ = io::stderr().flush();
    }
}

/// Read a line from stdin
///
/// # Returns
/// - 0 on success
/// - 1 on EOF
/// - 4 on error
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_read_line(out_ptr: *mut *mut u8, out_len: *mut i64) -> i32 {
    if out_ptr.is_null() || out_len.is_null() {
        return 3;
    }

    let stdin = io::stdin();
    let mut line = String::new();

    match stdin.lock().read_line(&mut line) {
        Ok(0) => 1, // EOF
        Ok(_) => {
            // Remove trailing newline
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }

            let bytes = line.into_bytes().into_boxed_slice();
            let len = bytes.len();
            let ptr = Box::into_raw(bytes) as *mut u8;

            unsafe {
                *out_ptr = ptr;
                *out_len = len as i64;
            }
            0
        }
        Err(_) => 4,
    }
}

// ============================================================================
// Memory Management
// ============================================================================

/// Free a string allocated by the I/O runtime
///
/// # Safety
/// - `ptr` must be a pointer returned by one of the I/O runtime functions
/// - `len` must be the length that was returned with the pointer
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_free_string(ptr: *mut u8, len: i64) {
    if !ptr.is_null() && len > 0 {
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr, len as usize);
            let _ = Box::from_raw(slice as *mut [u8]);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_file_exists() {
        // Test with a known existing path
        let path = ".";
        let result = __sounio_file_exists(path.as_ptr(), path.len() as i64);
        assert_eq!(result, 1); // Directory exists

        // Test with non-existent path
        let path = "/nonexistent/path/12345";
        let result = __sounio_file_exists(path.as_ptr(), path.len() as i64);
        assert_eq!(result, 0);
    }

    #[test]
    #[cfg(not(windows))] // Skip on Windows - temp file handling differs, see issue tracking
    fn test_read_write_file() {
        // Use cross-platform temp directory
        let temp_dir = std::env::temp_dir();
        let test_path_buf = temp_dir.join("sounio_io_test.txt");
        let test_path = test_path_buf.to_str().unwrap();
        let content = "Hello, Sounio!";

        // Write file
        let write_result = __sounio_write_file(
            test_path.as_ptr(),
            test_path.len() as i64,
            content.as_ptr(),
            content.len() as i64,
        );
        assert_eq!(
            write_result, 0,
            "write_file failed with error code {}",
            write_result
        );

        // Read file
        let mut out_ptr: *mut u8 = ptr::null_mut();
        let mut out_len: i64 = 0;
        let read_result = __sounio_read_file(
            test_path.as_ptr(),
            test_path.len() as i64,
            &mut out_ptr,
            &mut out_len,
        );
        assert_eq!(
            read_result, 0,
            "read_file failed with error code {}",
            read_result
        );
        assert!(!out_ptr.is_null());
        assert_eq!(out_len, content.len() as i64);

        // Verify content
        let read_content =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(out_ptr, out_len as usize)) };
        assert_eq!(read_content.unwrap(), content);

        // Free the string
        __sounio_free_string(out_ptr, out_len);

        // Clean up
        let _ = __sounio_remove_file(test_path.as_ptr(), test_path.len() as i64);
    }

    #[test]
    fn test_get_env() {
        // Set a test env var
        unsafe {
            env::set_var("SOUNIO_TEST_VAR", "test_value");
        }

        let name = "SOUNIO_TEST_VAR";
        let mut out_ptr: *mut u8 = ptr::null_mut();
        let mut out_len: i64 = 0;

        let result =
            __sounio_get_env(name.as_ptr(), name.len() as i64, &mut out_ptr, &mut out_len);
        assert_eq!(result, 0);

        let value =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(out_ptr, out_len as usize)) };
        assert_eq!(value.unwrap(), "test_value");

        __sounio_free_string(out_ptr, out_len);
    }

    #[test]
    fn test_current_dir() {
        let mut out_ptr: *mut u8 = ptr::null_mut();
        let mut out_len: i64 = 0;

        let result = __sounio_current_dir(&mut out_ptr, &mut out_len);
        assert_eq!(result, 0);
        assert!(!out_ptr.is_null());
        assert!(out_len > 0);

        __sounio_free_string(out_ptr, out_len);
    }

    #[test]
    fn test_argc_argv() {
        init_args();

        let argc = __sounio_get_argc();
        assert!(argc > 0); // At least the program name

        // Get first argument (program name)
        let mut out_ptr: *mut u8 = ptr::null_mut();
        let mut out_len: i64 = 0;
        let result = __sounio_get_argv(0, &mut out_ptr, &mut out_len);
        assert_eq!(result, 0);

        __sounio_free_string(out_ptr, out_len);

        // Out of bounds should return 1
        let result = __sounio_get_argv(9999, &mut out_ptr, &mut out_len);
        assert_eq!(result, 1);
    }
}

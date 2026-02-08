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
// File Operations (moved to runtime/ffi/ffi_io.rs to avoid duplication)
// ============================================================================

// NOTE: __sounio_read_file, __sounio_write_file, __sounio_append_file
// are now defined in runtime/ffi/ffi_io.rs with tracing support.
// The old definitions are preserved below for reference but commented out.

// NOTE: __sounio_file_exists and __sounio_remove_file moved to ffi/ffi_io.rs

// ============================================================================
// Process Control (moved to runtime/ffi/ffi_process.rs)
// ============================================================================

// NOTE: __sounio_exit moved to ffi_process.rs

// ============================================================================
// Environment Access (moved to runtime/ffi/ffi_process.rs)
// ============================================================================

// NOTE: __sounio_get_argc, __sounio_get_argv, __sounio_get_env, __sounio_set_env
// moved to ffi_process.rs

// NOTE: __sounio_current_dir moved to runtime/ffi/ffi_process.rs to avoid duplication
// /// Get the current working directory
// ///
// /// # Returns
// /// - 0 on success
// /// - 4 on error
// #[unsafe(no_mangle)]
// pub extern "C" fn __sounio_current_dir(out_ptr: *mut *mut u8, out_len: *mut i64) -> i32 {
//     if out_ptr.is_null() || out_len.is_null() {
//         return 3;
//     }
//
//     match env::current_dir() {
//         Ok(path) => {
//             let path_str = path.to_string_lossy().into_owned();
//             let bytes = path_str.into_bytes().into_boxed_slice();
//             let len = bytes.len();
//             let ptr = Box::into_raw(bytes) as *mut u8;
//
//             unsafe {
//                 *out_ptr = ptr;
//                 *out_len = len as i64;
//             }
//             0
//         }
//         Err(_) => 4,
//     }
// }

// ============================================================================
// Standard Streams (moved to runtime/ffi/ffi_stdio.rs)
// ============================================================================

// NOTE: __sounio_print, __sounio_eprint, __sounio_read_line moved to ffi_stdio.rs

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

    // NOTE: test_file_exists moved to ffi_io.rs tests
    // #[test]
    // fn test_file_exists() { ... }

    // NOTE: __sounio_read_file and __sounio_write_file moved to runtime/ffi/ffi_io.rs
    // #[test]
    // #[cfg(not(windows))] // Skip on Windows - temp file handling differs, see issue tracking
    // fn test_read_write_file() {
    //     // Use cross-platform temp directory
    //     let temp_dir = std::env::temp_dir();
    //     let test_path_buf = temp_dir.join("sounio_io_test.txt");
    //     let test_path = test_path_buf.to_str().unwrap();
    //     let content = "Hello, Sounio!";
    //
    //     // Write file
    //     let write_result = __sounio_write_file(
    //         test_path.as_ptr(),
    //         test_path.len() as i64,
    //         content.as_ptr(),
    //         content.len() as i64,
    //     );
    //     assert_eq!(
    //         write_result, 0,
    //         "write_file failed with error code {}",
    //         write_result
    //     );
    //
    //     // Read file
    //     let mut out_ptr: *mut u8 = ptr::null_mut();
    //     let mut out_len: i64 = 0;
    //     let read_result = __sounio_read_file(
    //         test_path.as_ptr(),
    //         test_path.len() as i64,
    //         &mut out_ptr,
    //         &mut out_len,
    //     );
    //     assert_eq!(
    //         read_result, 0,
    //         "read_file failed with error code {}",
    //         read_result
    //     );
    //     assert!(!out_ptr.is_null());
    //     assert_eq!(out_len, content.len() as i64);
    //
    //     // Verify content
    //     let read_content =
    //         unsafe { std::str::from_utf8(std::slice::from_raw_parts(out_ptr, out_len as usize)) };
    //     assert_eq!(read_content.unwrap(), content);
    //
    //     // Free the string
    //     __sounio_free_string(out_ptr, out_len);
    //
    //     // Clean up
    //     let _ = __sounio_remove_file(test_path.as_ptr(), test_path.len() as i64);
    // }

    // NOTE: Tests for FFI functions moved to ffi/ directory
    // #[test]
    // fn test_get_env() { ... }
    //
    // #[test]
    // fn test_argc_argv() { ... }
}

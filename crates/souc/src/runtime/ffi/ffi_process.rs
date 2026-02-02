//! Process & Environment FFI Functions
//!
//! Implements 8 process/environment functions:
//! - __sounio_exit
//! - __sounio_get_argc
//! - __sounio_get_argv
//! - __sounio_get_env
//! - __sounio_set_env
//! - __sounio_current_dir
//! - __sounio_set_current_dir
//! - __sounio_env_vars

use std::env;
use std::process;
use std::slice;

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_exit(code: i32) -> ! {
    let span = tracing::trace_span!("ffi::exit", code);
    let _guard = span.enter();
    tracing::trace!("exiting with code {}", code);
    process::exit(code);
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_get_argc() -> i32 {
    let span = tracing::trace_span!("ffi::get_argc");
    let _guard = span.enter();
    let count = std::env::args().count() as i32;
    tracing::trace!("argc: {}", count);
    count
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_get_argv(idx: i32, out_len: *mut usize) -> *mut u8 {
    let span = tracing::trace_span!("ffi::get_argv", idx);
    let _guard = span.enter();

    if out_len.is_null() {
        tracing::error!("null out_len");
        return std::ptr::null_mut();
    }

    if idx < 0 {
        tracing::error!("negative index");
        return std::ptr::null_mut();
    }

    match std::env::args().nth(idx as usize) {
        Some(arg) => {
            let bytes = arg.into_bytes();
            let len = bytes.len();
            unsafe { *out_len = len };
            let ptr = bytes.leak().as_mut_ptr();
            tracing::trace!("argv[{}]: {} bytes", idx, len);
            ptr
        }
        None => {
            tracing::error!("argv index out of bounds: {}", idx);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_get_env(
    key_ptr: *const u8,
    key_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let span = tracing::trace_span!("ffi::get_env", key_len);
    let _guard = span.enter();

    if key_ptr.is_null() || out_len.is_null() {
        tracing::error!("null pointer");
        return std::ptr::null_mut();
    }

    let key_slice = unsafe { slice::from_raw_parts(key_ptr, key_len) };
    let key_str = match std::str::from_utf8(key_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid utf8: {}", e);
            return std::ptr::null_mut();
        }
    };

    match env::var(key_str) {
        Ok(value) => {
            let bytes = value.into_bytes();
            let len = bytes.len();
            unsafe { *out_len = len };
            let ptr = bytes.leak().as_mut_ptr();
            tracing::trace!("env[{}]: {} bytes", key_str, len);
            ptr
        }
        Err(_) => {
            tracing::trace!("env var not found: {}", key_str);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_set_env(
    key_ptr: *const u8,
    key_len: usize,
    value_ptr: *const u8,
    value_len: usize,
) -> i32 {
    let span = tracing::trace_span!("ffi::set_env", key_len, value_len);
    let _guard = span.enter();

    if key_ptr.is_null() || value_ptr.is_null() {
        tracing::error!("null pointer");
        return -1;
    }

    let key_slice = unsafe { slice::from_raw_parts(key_ptr, key_len) };
    let key_str = match std::str::from_utf8(key_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid key utf8: {}", e);
            return -1;
        }
    };

    let value_slice = unsafe { slice::from_raw_parts(value_ptr, value_len) };
    let value_str = match std::str::from_utf8(value_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid value utf8: {}", e);
            return -1;
        }
    };

    unsafe {
        env::set_var(key_str, value_str);
    }
    tracing::trace!("set env: {}={}", key_str, value_str);
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_current_dir(out_len: *mut usize) -> *mut u8 {
    let span = tracing::trace_span!("ffi::current_dir");
    let _guard = span.enter();

    if out_len.is_null() {
        tracing::error!("null out_len");
        return std::ptr::null_mut();
    }

    match env::current_dir() {
        Ok(path) => {
            let path_bytes = path.to_string_lossy().into_owned().into_bytes();
            let len = path_bytes.len();
            unsafe { *out_len = len };
            let ptr = path_bytes.leak().as_mut_ptr();
            tracing::trace!("current_dir: {} bytes", len);
            ptr
        }
        Err(e) => {
            tracing::error!("current_dir failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_set_current_dir(path_ptr: *const u8, path_len: usize) -> i32 {
    let span = tracing::trace_span!("ffi::set_current_dir", path_len);
    let _guard = span.enter();

    if path_ptr.is_null() {
        tracing::error!("null pointer");
        return -1;
    }

    let path_slice = unsafe { slice::from_raw_parts(path_ptr, path_len) };
    let path_str = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid utf8: {}", e);
            return -1;
        }
    };

    match env::set_current_dir(path_str) {
        Ok(_) => {
            tracing::trace!("set current_dir: {}", path_str);
            0
        }
        Err(e) => {
            tracing::error!("set_current_dir failed: {}", e);
            -1
        }
    }
}

// Stub for env_vars - returns count of environment variables
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_env_vars_count() -> i32 {
    let span = tracing::trace_span!("ffi::env_vars_count");
    let _guard = span.enter();
    let count = env::vars().count() as i32;
    tracing::trace!("env vars count: {}", count);
    count
}

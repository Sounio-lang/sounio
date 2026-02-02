//! File I/O FFI Functions
//!
//! Implements 8 file I/O functions:
//! - __sounio_read_file
//! - __sounio_write_file
//! - __sounio_append_file
//! - __sounio_remove_file
//! - __sounio_copy_file
//! - __sounio_rename_file
//! - __sounio_file_exists
//! - __sounio_file_metadata

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::slice;

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_read_file(
    path_ptr: *const u8,
    path_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let span = tracing::trace_span!("ffi::read_file", len = path_len);
    let _guard = span.enter();

    // Safety checks
    if path_ptr.is_null() || out_len.is_null() {
        tracing::error!("null pointer");
        return std::ptr::null_mut();
    }

    let path_slice = unsafe { slice::from_raw_parts(path_ptr, path_len) };
    let path_str = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid utf8: {}", e);
            return std::ptr::null_mut();
        }
    };

    match fs::read(path_str) {
        Ok(bytes) => {
            let len = bytes.len();
            unsafe { *out_len = len };
            let ptr = bytes.leak().as_mut_ptr();
            tracing::trace!("read {} bytes", len);
            ptr
        }
        Err(e) => {
            tracing::error!("read failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_write_file(
    path_ptr: *const u8,
    path_len: usize,
    data_ptr: *const u8,
    data_len: usize,
) -> i32 {
    let span = tracing::trace_span!("ffi::write_file", path_len, data_len);
    let _guard = span.enter();

    // Safety checks
    if path_ptr.is_null() || (data_ptr.is_null() && data_len > 0) {
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

    let data_slice = if data_len > 0 {
        unsafe { slice::from_raw_parts(data_ptr, data_len) }
    } else {
        &[]
    };

    match fs::write(path_str, data_slice) {
        Ok(_) => {
            tracing::trace!("wrote {} bytes", data_len);
            0
        }
        Err(e) => {
            tracing::error!("write failed: {}", e);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_append_file(
    path_ptr: *const u8,
    path_len: usize,
    data_ptr: *const u8,
    data_len: usize,
) -> i32 {
    let span = tracing::trace_span!("ffi::append_file", path_len, data_len);
    let _guard = span.enter();

    // Safety checks
    if path_ptr.is_null() || (data_ptr.is_null() && data_len > 0) {
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

    let data_slice = if data_len > 0 {
        unsafe { slice::from_raw_parts(data_ptr, data_len) }
    } else {
        &[]
    };

    let mut options = OpenOptions::new();
    options.append(true).create(true);
    match options.open(path_str) {
        Ok(mut file) => match file.write_all(data_slice) {
            Ok(_) => {
                tracing::trace!("appended {} bytes", data_len);
                0
            }
            Err(e) => {
                tracing::error!("append write failed: {}", e);
                -1
            }
        },
        Err(e) => {
            tracing::error!("append open failed: {}", e);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_remove_file(path_ptr: *const u8, path_len: usize) -> i32 {
    let span = tracing::trace_span!("ffi::remove_file", len = path_len);
    let _guard = span.enter();

    // Safety checks
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

    match fs::remove_file(path_str) {
        Ok(_) => {
            tracing::trace!("removed file");
            0
        }
        Err(e) => {
            tracing::error!("remove failed: {}", e);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_copy_file(
    src_ptr: *const u8,
    src_len: usize,
    dst_ptr: *const u8,
    dst_len: usize,
) -> i32 {
    let span = tracing::trace_span!("ffi::copy_file", src_len, dst_len);
    let _guard = span.enter();

    // Safety checks
    if src_ptr.is_null() || dst_ptr.is_null() {
        tracing::error!("null pointer");
        return -1;
    }

    let src_slice = unsafe { slice::from_raw_parts(src_ptr, src_len) };
    let src_str = match std::str::from_utf8(src_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid src utf8: {}", e);
            return -1;
        }
    };

    let dst_slice = unsafe { slice::from_raw_parts(dst_ptr, dst_len) };
    let dst_str = match std::str::from_utf8(dst_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid dst utf8: {}", e);
            return -1;
        }
    };

    match fs::copy(src_str, dst_str) {
        Ok(bytes_copied) => {
            tracing::trace!("copied {} bytes", bytes_copied);
            0
        }
        Err(e) => {
            tracing::error!("copy failed: {}", e);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_rename_file(
    old_ptr: *const u8,
    old_len: usize,
    new_ptr: *const u8,
    new_len: usize,
) -> i32 {
    let span = tracing::trace_span!("ffi::rename_file", old_len, new_len);
    let _guard = span.enter();

    // Safety checks
    if old_ptr.is_null() || new_ptr.is_null() {
        tracing::error!("null pointer");
        return -1;
    }

    let old_slice = unsafe { slice::from_raw_parts(old_ptr, old_len) };
    let old_str = match std::str::from_utf8(old_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid old utf8: {}", e);
            return -1;
        }
    };

    let new_slice = unsafe { slice::from_raw_parts(new_ptr, new_len) };
    let new_str = match std::str::from_utf8(new_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid new utf8: {}", e);
            return -1;
        }
    };

    match fs::rename(old_str, new_str) {
        Ok(_) => {
            tracing::trace!("renamed file");
            0
        }
        Err(e) => {
            tracing::error!("rename failed: {}", e);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_file_exists(path_ptr: *const u8, path_len: usize) -> bool {
    let span = tracing::trace_span!("ffi::file_exists", len = path_len);
    let _guard = span.enter();

    // Safety checks
    if path_ptr.is_null() {
        tracing::error!("null pointer");
        return false;
    }

    let path_slice = unsafe { slice::from_raw_parts(path_ptr, path_len) };
    let path_str = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid utf8: {}", e);
            return false;
        }
    };

    let exists = Path::new(path_str).exists();
    tracing::trace!("file exists: {}", exists);
    exists
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_file_metadata(path_ptr: *const u8, path_len: usize) -> i64 {
    let span = tracing::trace_span!("ffi::file_metadata", len = path_len);
    let _guard = span.enter();

    // Safety checks
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

    match fs::metadata(path_str) {
        Ok(metadata) => {
            let size = metadata.len() as i64;
            tracing::trace!("file size: {}", size);
            size
        }
        Err(e) => {
            tracing::error!("metadata failed: {}", e);
            -1
        }
    }
}

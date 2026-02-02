//! Path Utility FFI Functions
//! Implements 6 path functions: join, file_name, parent, extension, is_absolute, canonicalize

use std::path::Path;
use std::slice;

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_path_join(
    base_ptr: *const u8,
    base_len: usize,
    comp_ptr: *const u8,
    comp_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let span = tracing::trace_span!("ffi::path_join", base_len, comp_len);
    let _guard = span.enter();

    if base_ptr.is_null() || comp_ptr.is_null() || out_len.is_null() {
        tracing::error!("null pointer");
        return std::ptr::null_mut();
    }

    let base_slice = unsafe { slice::from_raw_parts(base_ptr, base_len) };
    let base_str = match std::str::from_utf8(base_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid base utf8: {}", e);
            return std::ptr::null_mut();
        }
    };

    let comp_slice = unsafe { slice::from_raw_parts(comp_ptr, comp_len) };
    let comp_str = match std::str::from_utf8(comp_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid component utf8: {}", e);
            return std::ptr::null_mut();
        }
    };

    let mut path = std::path::PathBuf::from(base_str);
    path.push(comp_str);
    let result_str = path.to_string_lossy().into_owned();
    let bytes = result_str.into_bytes();
    let len = bytes.len();
    unsafe { *out_len = len };
    let ptr = bytes.leak().as_mut_ptr();
    tracing::trace!("joined paths: {} bytes", len);
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_path_file_name(
    path_ptr: *const u8,
    path_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let span = tracing::trace_span!("ffi::path_file_name", path_len);
    let _guard = span.enter();

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

    let path = Path::new(path_str);
    match path.file_name() {
        Some(name) => {
            let name_str = name.to_string_lossy().into_owned();
            let bytes = name_str.into_bytes();
            let len = bytes.len();
            unsafe { *out_len = len };
            let ptr = bytes.leak().as_mut_ptr();
            tracing::trace!("file_name: {} bytes", len);
            ptr
        }
        None => {
            tracing::trace!("no file name");
            unsafe { *out_len = 0 };
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_path_parent(
    path_ptr: *const u8,
    path_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let span = tracing::trace_span!("ffi::path_parent", path_len);
    let _guard = span.enter();

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

    let path = Path::new(path_str);
    match path.parent() {
        Some(parent) => {
            let parent_str = parent.to_string_lossy().into_owned();
            let bytes = parent_str.into_bytes();
            let len = bytes.len();
            unsafe { *out_len = len };
            let ptr = bytes.leak().as_mut_ptr();
            tracing::trace!("parent: {} bytes", len);
            ptr
        }
        None => {
            tracing::trace!("no parent");
            unsafe { *out_len = 0 };
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_path_extension(
    path_ptr: *const u8,
    path_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let span = tracing::trace_span!("ffi::path_extension", path_len);
    let _guard = span.enter();

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

    let path = Path::new(path_str);
    match path.extension() {
        Some(ext) => {
            let ext_str = ext.to_string_lossy().into_owned();
            let bytes = ext_str.into_bytes();
            let len = bytes.len();
            unsafe { *out_len = len };
            let ptr = bytes.leak().as_mut_ptr();
            tracing::trace!("extension: {} bytes", len);
            ptr
        }
        None => {
            tracing::trace!("no extension");
            unsafe { *out_len = 0 };
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_path_is_absolute(path_ptr: *const u8, path_len: usize) -> bool {
    let span = tracing::trace_span!("ffi::path_is_absolute", path_len);
    let _guard = span.enter();

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

    let is_abs = Path::new(path_str).is_absolute();
    tracing::trace!("is_absolute: {}", is_abs);
    is_abs
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_path_canonicalize(
    path_ptr: *const u8,
    path_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let span = tracing::trace_span!("ffi::path_canonicalize", path_len);
    let _guard = span.enter();

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

    match std::fs::canonicalize(path_str) {
        Ok(canonical) => {
            let result_str = canonical.to_string_lossy().into_owned();
            let bytes = result_str.into_bytes();
            let len = bytes.len();
            unsafe { *out_len = len };
            let ptr = bytes.leak().as_mut_ptr();
            tracing::trace!("canonicalize: {} bytes", len);
            ptr
        }
        Err(e) => {
            tracing::error!("canonicalize failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

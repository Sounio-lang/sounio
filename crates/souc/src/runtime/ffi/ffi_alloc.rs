//! Memory Allocation FFI Functions
//! Implements 6 allocation functions: alloc, dealloc, realloc, alloc_zeroed, alloc_array, dealloc_array

use std::alloc::{Layout, alloc, dealloc, realloc};

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_alloc(size: usize, align: usize) -> *mut u8 {
    let span = tracing::trace_span!("ffi::alloc", size, align);
    let _guard = span.enter();

    if align == 0 || !align.is_power_of_two() {
        tracing::error!("invalid alignment: {}", align);
        return std::ptr::null_mut();
    }

    match Layout::from_size_align(size, align) {
        Ok(layout) => {
            let ptr = unsafe { alloc(layout) };
            if ptr.is_null() {
                tracing::error!("allocation failed: size={}, align={}", size, align);
            } else {
                tracing::trace!("allocated {} bytes", size);
            }
            ptr
        }
        Err(e) => {
            tracing::error!("layout error: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dealloc(ptr: *mut u8, size: usize, align: usize) {
    let span = tracing::trace_span!("ffi::dealloc", size, align);
    let _guard = span.enter();

    if ptr.is_null() {
        tracing::warn!("dealloc null pointer");
        return;
    }

    if align == 0 || !align.is_power_of_two() {
        tracing::error!("invalid alignment: {}", align);
        return;
    }

    if let Ok(layout) = Layout::from_size_align(size, align) {
        unsafe {
            dealloc(ptr, layout);
            tracing::trace!("deallocated {} bytes", size);
        }
    } else {
        tracing::error!("layout error for dealloc");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_realloc(
    ptr: *mut u8,
    old_size: usize,
    new_size: usize,
    align: usize,
) -> *mut u8 {
    let span = tracing::trace_span!("ffi::realloc", old_size, new_size, align);
    let _guard = span.enter();

    if ptr.is_null() || old_size == 0 {
        tracing::trace!("realloc with null/empty ptr, using alloc");
        return __sounio_alloc(new_size, align);
    }

    if align == 0 || !align.is_power_of_two() {
        tracing::error!("invalid alignment: {}", align);
        return std::ptr::null_mut();
    }

    match Layout::from_size_align(old_size, align) {
        Ok(layout) => {
            let new_ptr = unsafe { realloc(ptr, layout, new_size) };
            if new_ptr.is_null() {
                tracing::error!(
                    "realloc failed: old_size={}, new_size={}",
                    old_size,
                    new_size
                );
            } else {
                tracing::trace!("reallocated {} -> {} bytes", old_size, new_size);
            }
            new_ptr
        }
        Err(e) => {
            tracing::error!("layout error: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
    let span = tracing::trace_span!("ffi::alloc_zeroed", size, align);
    let _guard = span.enter();

    let ptr = __sounio_alloc(size, align);
    if !ptr.is_null() {
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
            tracing::trace!("allocated and zeroed {} bytes", size);
        }
    }
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_alloc_array(
    count: usize,
    elem_size: usize,
    elem_align: usize,
) -> *mut u8 {
    let span = tracing::trace_span!("ffi::alloc_array", count, elem_size);
    let _guard = span.enter();

    match count.checked_mul(elem_size) {
        Some(total_size) => {
            let ptr = __sounio_alloc(total_size, elem_align);
            if !ptr.is_null() {
                tracing::trace!("allocated array of {} elements", count);
            }
            ptr
        }
        None => {
            tracing::error!(
                "array size overflow: count={}, elem_size={}",
                count,
                elem_size
            );
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dealloc_array(
    ptr: *mut u8,
    count: usize,
    elem_size: usize,
    elem_align: usize,
) {
    let span = tracing::trace_span!("ffi::dealloc_array", count, elem_size);
    let _guard = span.enter();

    if let Some(total_size) = count.checked_mul(elem_size) {
        __sounio_dealloc(ptr, total_size, elem_align);
        tracing::trace!("deallocated array of {} elements", count);
    } else {
        tracing::error!("array size overflow on dealloc");
    }
}

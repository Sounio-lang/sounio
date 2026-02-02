//! Stdio FFI Functions
//! Implements 6 stdio functions: print, println, eprint, eprintln, read_line, flush_stdout

use std::io::{self, Write};
use std::slice;

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_print(data_ptr: *const u8, data_len: usize) {
    let span = tracing::trace_span!("ffi::print", data_len);
    let _guard = span.enter();

    if data_ptr.is_null() {
        tracing::error!("null pointer");
        return;
    }

    let data_slice = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    if let Ok(s) = std::str::from_utf8(data_slice) {
        print!("{}", s);
        tracing::trace!("printed {} bytes", data_len);
    } else {
        tracing::error!("invalid utf8");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_println(data_ptr: *const u8, data_len: usize) {
    let span = tracing::trace_span!("ffi::println", data_len);
    let _guard = span.enter();

    if data_ptr.is_null() {
        println!();
        return;
    }

    let data_slice = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    if let Ok(s) = std::str::from_utf8(data_slice) {
        println!("{}", s);
        tracing::trace!("printed line {} bytes", data_len);
    } else {
        tracing::error!("invalid utf8");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_eprint(data_ptr: *const u8, data_len: usize) {
    let span = tracing::trace_span!("ffi::eprint", data_len);
    let _guard = span.enter();

    if data_ptr.is_null() {
        tracing::error!("null pointer");
        return;
    }

    let data_slice = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    if let Ok(s) = std::str::from_utf8(data_slice) {
        eprint!("{}", s);
        tracing::trace!("eprinted {} bytes", data_len);
    } else {
        tracing::error!("invalid utf8");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_eprintln(data_ptr: *const u8, data_len: usize) {
    let span = tracing::trace_span!("ffi::eprintln", data_len);
    let _guard = span.enter();

    if data_ptr.is_null() {
        eprintln!();
        return;
    }

    let data_slice = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    if let Ok(s) = std::str::from_utf8(data_slice) {
        eprintln!("{}", s);
        tracing::trace!("eprintln {} bytes", data_len);
    } else {
        tracing::error!("invalid utf8");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_read_line(out_len: *mut usize) -> *mut u8 {
    let span = tracing::trace_span!("ffi::read_line");
    let _guard = span.enter();

    if out_len.is_null() {
        tracing::error!("null out_len");
        return std::ptr::null_mut();
    }

    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(_) => {
            let bytes = line.into_bytes();
            let len = bytes.len();
            unsafe { *out_len = len };
            let ptr = bytes.leak().as_mut_ptr();
            tracing::trace!("read line: {} bytes", len);
            ptr
        }
        Err(e) => {
            tracing::error!("read_line failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_flush_stdout() {
    let span = tracing::trace_span!("ffi::flush_stdout");
    let _guard = span.enter();

    if let Err(e) = io::stdout().flush() {
        tracing::error!("flush_stdout failed: {}", e);
    } else {
        tracing::trace!("flushed stdout");
    }
}

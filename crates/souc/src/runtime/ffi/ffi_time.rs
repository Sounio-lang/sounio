//! Time FFI Functions
//! Implements 4 time functions: now, sleep, format, parse

use std::slice;
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_time_now() -> i64 {
    let span = tracing::trace_span!("ffi::time_now");
    let _guard = span.enter();

    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            let millis = (duration.as_secs() * 1000) + (duration.subsec_millis() as u64);
            tracing::trace!("time_now: {} ms", millis);
            millis as i64
        }
        Err(e) => {
            tracing::error!("time_now failed: {}", e);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_time_sleep(millis: u64) {
    let span = tracing::trace_span!("ffi::time_sleep", millis);
    let _guard = span.enter();

    thread::sleep(Duration::from_millis(millis));
    tracing::trace!("slept {} ms", millis);
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_time_format(
    timestamp: i64,
    fmt_ptr: *const u8,
    fmt_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let span = tracing::trace_span!("ffi::time_format", timestamp, fmt_len);
    let _guard = span.enter();

    if fmt_ptr.is_null() || out_len.is_null() {
        tracing::error!("null pointer");
        return std::ptr::null_mut();
    }

    if timestamp < 0 {
        tracing::error!("negative timestamp");
        return std::ptr::null_mut();
    }

    let fmt_slice = unsafe { slice::from_raw_parts(fmt_ptr, fmt_len) };
    let fmt_str = match std::str::from_utf8(fmt_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid fmt utf8: {}", e);
            return std::ptr::null_mut();
        }
    };

    // Simple ISO 8601 format (basic implementation)
    let duration = Duration::from_millis(timestamp as u64);
    let secs = duration.as_secs();
    let days = secs / 86400;
    let remaining_secs = secs % 86400;
    let hours = remaining_secs / 3600;
    let minutes = (remaining_secs % 3600) / 60;
    let seconds = remaining_secs % 60;
    let millis = duration.subsec_millis();

    let result = if fmt_str.contains("iso") || fmt_str.is_empty() {
        format!(
            "{:04}-01-01T{:02}:{:02}:{:02}.{:03}Z",
            1970 + days / 365,
            hours,
            minutes,
            seconds,
            millis
        )
    } else if fmt_str.contains("ms") {
        format!("{}ms", timestamp)
    } else if fmt_str.contains("s") {
        format!("{}s", secs)
    } else {
        format!("{}", timestamp)
    };

    let bytes = result.into_bytes();
    let len = bytes.len();
    unsafe { *out_len = len };
    let ptr = bytes.leak().as_mut_ptr();
    tracing::trace!("time_format: {} bytes", len);
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_time_parse(
    str_ptr: *const u8,
    str_len: usize,
    fmt_ptr: *const u8,
    fmt_len: usize,
) -> i64 {
    let span = tracing::trace_span!("ffi::time_parse", str_len, fmt_len);
    let _guard = span.enter();

    if str_ptr.is_null() || fmt_ptr.is_null() {
        tracing::error!("null pointer");
        return -1;
    }

    let str_slice = unsafe { slice::from_raw_parts(str_ptr, str_len) };
    let str_val = match std::str::from_utf8(str_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid str utf8: {}", e);
            return -1;
        }
    };

    let fmt_slice = unsafe { slice::from_raw_parts(fmt_ptr, fmt_len) };
    let fmt_str = match std::str::from_utf8(fmt_slice) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("invalid fmt utf8: {}", e);
            return -1;
        }
    };

    // Simple parser for basic formats
    if fmt_str.contains("ms") {
        match str_val.trim_end_matches("ms").parse::<i64>() {
            Ok(ms) => {
                tracing::trace!("parsed ms: {}", ms);
                ms
            }
            Err(e) => {
                tracing::error!("parse failed: {}", e);
                -1
            }
        }
    } else if fmt_str.contains("s") {
        match str_val.trim_end_matches("s").parse::<i64>() {
            Ok(secs) => {
                let ms = secs * 1000;
                tracing::trace!("parsed s: {} -> {} ms", secs, ms);
                ms
            }
            Err(e) => {
                tracing::error!("parse failed: {}", e);
                -1
            }
        }
    } else {
        match str_val.parse::<i64>() {
            Ok(ms) => {
                tracing::trace!("parsed numeric: {}", ms);
                ms
            }
            Err(e) => {
                tracing::error!("parse failed: {}", e);
                -1
            }
        }
    }
}

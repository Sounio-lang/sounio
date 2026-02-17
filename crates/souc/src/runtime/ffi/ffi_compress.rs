//! Compression FFI Functions
//!
//! Implements zlib inflate/deflate via the `flate2` crate:
//! - __sounio_zlib_inflate
//! - __sounio_zlib_deflate

use std::io::{Read, Write};
use std::slice;

/// Decompress raw DEFLATE data into a caller-provided output buffer.
///
/// # Arguments
///
/// * `input` - Pointer to compressed input data
/// * `input_len` - Length of input data in bytes
/// * `output` - Pointer to output buffer (must be pre-allocated by caller)
/// * `output_len` - Capacity of the output buffer in bytes
/// * `actual_len` - Out-parameter: actual number of decompressed bytes written
///
/// # Returns
///
/// * `0` on success
/// * `-1` on null pointer or invalid input
/// * `-2` on decompression error
/// * `-5` if the output buffer is too small (matches Z_BUF_ERROR)
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_zlib_inflate(
    input: *const u8,
    input_len: i64,
    output: *mut u8,
    output_len: i64,
    actual_len: *mut i64,
) -> i32 {
    let span = tracing::trace_span!("ffi::zlib_inflate", input_len, output_len);
    let _guard = span.enter();

    // Safety checks
    if input.is_null() || output.is_null() || actual_len.is_null() {
        tracing::error!("null pointer");
        return -1;
    }

    if input_len < 0 || output_len < 0 {
        tracing::error!("negative length: input_len={}, output_len={}", input_len, output_len);
        return -1;
    }

    let input_slice = unsafe { slice::from_raw_parts(input, input_len as usize) };
    let output_slice = unsafe { slice::from_raw_parts_mut(output, output_len as usize) };

    let mut decoder = flate2::read::DeflateDecoder::new(input_slice);
    let mut total_read = 0usize;

    loop {
        if total_read >= output_slice.len() {
            // Check if there is more data to decompress
            let mut check = [0u8; 1];
            match decoder.read(&mut check) {
                Ok(0) => break, // Done, fits exactly
                Ok(_) => {
                    tracing::error!("output buffer too small");
                    return -5;
                }
                Err(_) => break, // EOF or stream end
            }
        }

        match decoder.read(&mut output_slice[total_read..]) {
            Ok(0) => break, // EOF
            Ok(n) => {
                total_read += n;
            }
            Err(e) => {
                tracing::error!("inflate error: {}", e);
                return -2;
            }
        }
    }

    unsafe { *actual_len = total_read as i64 };
    tracing::trace!("inflated {} -> {} bytes", input_len, total_read);
    0
}

/// Compress data using raw DEFLATE into a caller-provided output buffer.
///
/// # Arguments
///
/// * `input` - Pointer to uncompressed input data
/// * `input_len` - Length of input data in bytes
/// * `output` - Pointer to output buffer (must be pre-allocated by caller)
/// * `output_len` - Capacity of the output buffer in bytes
/// * `actual_len` - Out-parameter: actual number of compressed bytes written
///
/// # Returns
///
/// * `0` on success
/// * `-1` on null pointer or invalid input
/// * `-2` on compression error
/// * `-5` if the output buffer is too small
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_zlib_deflate(
    input: *const u8,
    input_len: i64,
    output: *mut u8,
    output_len: i64,
    actual_len: *mut i64,
) -> i32 {
    let span = tracing::trace_span!("ffi::zlib_deflate", input_len, output_len);
    let _guard = span.enter();

    // Safety checks
    if input.is_null() || output.is_null() || actual_len.is_null() {
        tracing::error!("null pointer");
        return -1;
    }

    if input_len < 0 || output_len < 0 {
        tracing::error!("negative length: input_len={}, output_len={}", input_len, output_len);
        return -1;
    }

    let input_slice = unsafe { slice::from_raw_parts(input, input_len as usize) };

    // Compress into a Vec first, then copy to the output buffer.
    // Using a Vec intermediary avoids partial-write edge cases with the
    // fixed-size output slice.
    let mut encoder = flate2::write::DeflateEncoder::new(
        Vec::with_capacity(input_len as usize),
        flate2::Compression::default(),
    );

    if let Err(e) = encoder.write_all(input_slice) {
        tracing::error!("deflate write error: {}", e);
        return -2;
    }

    let compressed = match encoder.finish() {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("deflate finish error: {}", e);
            return -2;
        }
    };

    if compressed.len() > output_len as usize {
        tracing::error!(
            "output buffer too small: need {} but have {}",
            compressed.len(),
            output_len
        );
        return -5;
    }

    let output_slice = unsafe { slice::from_raw_parts_mut(output, output_len as usize) };
    output_slice[..compressed.len()].copy_from_slice(&compressed);
    unsafe { *actual_len = compressed.len() as i64 };

    tracing::trace!("deflated {} -> {} bytes", input_len, compressed.len());
    0
}

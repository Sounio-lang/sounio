//! Fuzz target for SOIR deserialization
//!
//! Goals:
//! - No panics on arbitrary input
//! - Proper error handling for invalid bytecode
//! - No buffer overruns or memory errors

#![no_main]

use libfuzzer_sys::fuzz_target;
use soir::deserialize;

fuzz_target!(|data: &[u8]| {
    // Should not panic on any input
    // It's OK to return an error
    let _ = deserialize(data);
});

//! Fuzz target for the Sounio lexer
//!
//! Tests that the lexer handles arbitrary byte sequences without panicking.
//! The lexer should either produce valid tokens or return an error, never crash.

#![no_main]

use libfuzzer_sys::fuzz_target;
use sounio::lexer;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string - skip invalid UTF-8 sequences
    if let Ok(source) = std::str::from_utf8(data) {
        // The lexer should handle any valid UTF-8 string without panicking
        // It's OK for it to return an error for invalid input
        let _ = lexer::lex(source);
    }
});

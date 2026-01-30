//! Lexer Fuzz Target
//!
//! INVARIANT: The lexer must NEVER panic on any input.
//! It may return errors, but it must handle all byte sequences gracefully.
//!
//! This is a crash-proof requirement: adversarial input must be tolerated forever.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Convert arbitrary bytes to a string (lossy is fine - we want to test edge cases)
    let input = String::from_utf8_lossy(data);

    // The lexer MUST NOT panic. It may return Err, but never panic.
    // This is the core invariant we're testing.
    let result = std::panic::catch_unwind(|| {
        sounio::lexer::lex(&input)
    });

    match result {
        Ok(Ok(_tokens)) => {
            // Successfully lexed - this is fine
        }
        Ok(Err(_e)) => {
            // Lexer returned an error - this is acceptable
            // The lexer is allowed to reject input, just not panic
        }
        Err(_panic) => {
            // LEXER PANICKED - THIS IS A BUG
            // libfuzzer will catch this and report it
            panic!("Lexer panicked on input: {:?}", &input[..input.len().min(100)]);
        }
    }
});

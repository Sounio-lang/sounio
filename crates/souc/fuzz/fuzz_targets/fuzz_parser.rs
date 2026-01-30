//! Parser Fuzz Target
//!
//! INVARIANT: The parser must NEVER panic on any token stream.
//! It may return errors, but it must handle all inputs gracefully.
//!
//! We first lex the input (which may fail), then parse the tokens.
//! Both stages must be crash-proof.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Convert arbitrary bytes to a string
    let input = String::from_utf8_lossy(data);
    let input_str: &str = &input;

    // Step 1: Lex the input (may fail, that's ok)
    let tokens = match std::panic::catch_unwind(|| {
        sounio::lexer::lex(input_str)
    }) {
        Ok(Ok(tokens)) => tokens,
        Ok(Err(_)) => return, // Lexer error - acceptable
        Err(_) => {
            panic!("Lexer panicked on input: {:?}", &input[..input.len().min(100)]);
        }
    };

    // Step 2: Parse the tokens - MUST NOT panic
    let result = std::panic::catch_unwind(|| {
        sounio::parser::parse(&tokens, input_str)
    });

    match result {
        Ok(Ok(_ast)) => {
            // Successfully parsed - this is fine
        }
        Ok(Err(_e)) => {
            // Parser returned an error - this is acceptable
            // The parser is allowed to reject input, just not panic
        }
        Err(_panic) => {
            // PARSER PANICKED - THIS IS A BUG
            panic!("Parser panicked on input: {:?}", &input[..input.len().min(100)]);
        }
    }
});

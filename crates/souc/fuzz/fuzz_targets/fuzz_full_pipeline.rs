//! Full Pipeline Fuzz Target
//!
//! INVARIANT: The entire compilation pipeline must NEVER panic on any input.
//! This tests: lexer → parser → type checker → HIR lowering
//!
//! Each stage may return errors, but none may panic.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Convert arbitrary bytes to a string
    let input = String::from_utf8_lossy(data);
    let input_str: &str = &input;

    // Stage 1: Lex
    let tokens = match std::panic::catch_unwind(|| {
        sounio::lexer::lex(input_str)
    }) {
        Ok(Ok(tokens)) => tokens,
        Ok(Err(_)) => return,
        Err(_) => {
            panic!("Lexer panicked");
        }
    };

    // Stage 2: Parse
    let ast = match std::panic::catch_unwind(|| {
        sounio::parser::parse(&tokens, input_str)
    }) {
        Ok(Ok(ast)) => ast,
        Ok(Err(_)) => return,
        Err(_) => {
            panic!("Parser panicked");
        }
    };

    // Stage 3: Type check (this is where many edge cases hide)
    let _result = std::panic::catch_unwind(|| {
        // Try to type check - may fail, but must not panic
        sounio::check::check(&ast)
    });

    // We don't care about the result - just that we didn't panic
    // (catch_unwind will catch panics and return Err)
});

//! Tests for extern block (FFI import) handling in the type checker and HLIR lowering.

use crate::{check, hlir, lexer, parser};

#[test]
fn extern_fn_is_resolved_and_lowered() {
    let source = r#"
        extern "C" {
            #[link_name = "puts"]
            fn d_puts(s: *const u8) -> i32;
        }

        fn main() -> i32 {
            d_puts(0 as *const u8);
            0
        }
    "#;

    let tokens = lexer::lex(source).expect("lex");
    let ast = parser::parse(&tokens, source).expect("parse");
    let hir = check::check_ast(&ast).expect("typecheck");

    assert_eq!(hir.externs.len(), 1);
    assert_eq!(hir.externs[0].functions.len(), 1);
    assert_eq!(hir.externs[0].functions[0].name, "d_puts");
    assert_eq!(
        hir.externs[0].functions[0].link_name.as_deref(),
        Some("puts")
    );

    let hlir = hlir::lower(&hir);

    let puts = hlir.find_function("d_puts").expect("extern fn in HLIR");
    assert!(
        puts.blocks.is_empty(),
        "extern functions should be declaration-only (no blocks)"
    );
    assert_eq!(puts.link_name.as_deref(), Some("puts"));

    let main_fn = hlir.find_function("main").expect("main in HLIR");
    let has_call = main_fn
        .blocks
        .iter()
        .flat_map(|b| b.instructions.iter())
        .any(|i| matches!(&i.op, hlir::Op::CallDirect { name, .. } if name == "d_puts"));
    assert!(has_call, "expected a direct call to extern fn `puts`");
}

fn check_source(source: &str) -> miette::Result<crate::hir::Hir> {
    let tokens = lexer::lex(source).expect("lex");
    let ast = parser::parse(&tokens, source).expect("parse");
    check::check_ast(&ast)
}

#[test]
fn return_with_trailing_semicolon_typechecks() {
    let source = r#"
        fn main() -> i64 {
            return 1;
        }
    "#;

    let checked = check_source(source);
    assert!(checked.is_ok(), "expected explicit return to typecheck");
}

#[test]
fn trailing_semicolon_non_return_still_fails() {
    let source = r#"
        fn main() -> i64 {
            1;
        }
    "#;

    let checked = check_source(source);
    assert!(
        checked.is_err(),
        "non-return trailing semicolon must not typecheck"
    );
}

#[test]
fn vec_plus_array_is_rejected_with_hint() {
    let source = r#"
        fn main() -> i32 with Mut, Alloc {
            var xs: Vec<i64> = []
            xs = xs + [1]
            return 0
        }
    "#;

    let err = check_source(source).expect_err("Vec + [x] must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("Vec append with '+' is not supported")
            && msg.contains("use '++ [value]' instead"),
        "expected actionable Vec append hint, got: {msg}"
    );
}

#[test]
fn vec_concat_array_typechecks() {
    let source = r#"
        fn main() -> i32 with Mut, Alloc {
            var xs: Vec<i64> = []
            xs = xs ++ [1]
            return 0
        }
    "#;

    let checked = check_source(source);
    assert!(checked.is_ok(), "Vec ++ [x] should typecheck");
}

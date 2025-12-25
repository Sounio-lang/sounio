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
    let hir = check::check(&ast).expect("typecheck");

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

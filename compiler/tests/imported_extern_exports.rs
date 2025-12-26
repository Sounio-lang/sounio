use std::fs;

use sounio::ast::Abi;
use sounio::{check, hlir, module_loader};

#[test]
fn extern_c_fn_from_imported_module_is_included() {
    let dir = tempfile::tempdir().expect("temp dir");
    let lib_path = dir.path().join("lib.sio");
    let ffi_path = dir.path().join("ffi.sio");

    fs::write(&lib_path, "pub import ffi;\n").expect("write lib.sio");
    fs::write(
        &ffi_path,
        "pub extern \"C\" fn test() -> usize {\n    return 42\n}\n",
    )
    .expect("write ffi.sio");

    let ast = module_loader::load_program_ast(&lib_path).expect("load program ast");
    let has_ffi_fn = ast.items.iter().any(|item| match item {
        sounio::ast::Item::Function(f) => f.name == "test" && f.modifiers.abi == Some(Abi::C),
        _ => false,
    });
    assert!(
        has_ffi_fn,
        "expected imported extern \"C\" fn in merged AST"
    );

    let hir = check::check(&ast).expect("type check");
    let hlir = hlir::lower(&hir);

    let func = hlir
        .functions
        .iter()
        .find(|func| func.name == "test")
        .expect("missing HLIR function");
    assert!(matches!(func.abi, Abi::C));
    assert!(func.is_exported);
}

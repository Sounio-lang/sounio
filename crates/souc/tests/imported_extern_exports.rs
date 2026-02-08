use std::fs;

use sounio::ast::{Abi, Item};
use sounio::{check, hlir, module_loader};

/// Helper: search for a function by name anywhere in the AST, including
/// inside `Module` wrappers (the module loader wraps imported modules).
fn find_extern_c_fn(items: &[Item], name: &str) -> bool {
    for item in items {
        match item {
            Item::Function(f) => {
                if f.name == name && f.modifiers.abi == Some(Abi::C) {
                    return true;
                }
            }
            Item::Module(m) => {
                if let Some(inner) = &m.items {
                    if find_extern_c_fn(inner, name) {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

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
    assert!(
        find_extern_c_fn(&ast.items, "test"),
        "expected imported extern \"C\" fn in merged AST"
    );

    let hir = check::check_ast(&ast).expect("type check");
    let hlir = hlir::lower(&hir);

    let func = hlir
        .functions
        .iter()
        .find(|func| func.name == "test")
        .expect("missing HLIR function");
    assert!(matches!(func.abi, Abi::C));
    assert!(func.is_exported);
}

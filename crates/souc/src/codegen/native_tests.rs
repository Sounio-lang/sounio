use super::native::NativeCodegen;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn parse_and_check(source: &str) -> Result<crate::hir::Hir, String> {
    let tokens = crate::lexer::lex(source).map_err(|e| e.to_string())?;
    let ast = crate::parser::parse(&tokens, source).map_err(|e| e.to_string())?;
    crate::check::check_ast(&ast).map_err(|e| e.to_string())
}

fn compile_only(source: &str) -> Result<Vec<u8>, String> {
    let hir = parse_and_check(source)?;
    let mut codegen = NativeCodegen::new();
    codegen.compile(&hir)
}

fn unique_test_path() -> PathBuf {
    let mut dir = std::env::temp_dir();
    let pid = std::process::id();
    let seq = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_nanos();
    dir.push(format!("sounio_native_test_{}_{}_{}", pid, seq, nanos));
    std::fs::create_dir_all(&dir).expect("create temp test dir");
    dir.join("a.out")
}

fn compile_and_run(source: &str) -> (i32, String) {
    let binary = compile_only(source).expect("native compile should succeed");

    let path = unique_test_path();
    std::fs::write(&path, &binary).expect("write executable bytes");

    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
        .expect("chmod executable");

    let output = {
        // Some CI/FS setups transiently return ETXTBSY ("Text file busy") when
        // executing a freshly-written binary. Retry briefly before failing.
        let mut attempt = 0;
        loop {
            match std::process::Command::new(&path).output() {
                Ok(output) => break output,
                Err(err)
                    if err.kind() == std::io::ErrorKind::ExecutableFileBusy && attempt < 20 =>
                {
                    attempt += 1;
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(err) => panic!("run compiled executable: {err}"),
            }
        }
    };
    let _ = std::fs::remove_file(&path);
    if let Some(parent) = path.parent() {
        let _ = std::fs::remove_dir(parent);
    }

    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).to_string(),
    )
}

fn compile_error(source: &str) -> String {
    match compile_only(source) {
        Ok(_) => panic!("expected native compile to fail"),
        Err(err) => err,
    }
}

#[test]
fn test_native_exit_code_constant() {
    let (code, _) = compile_and_run("fn main() -> i64 { 42 }");
    assert_eq!(code, 42);
}

#[test]
fn test_native_arithmetic_add() {
    let (code, _) = compile_and_run("fn main() -> i64 { 10 + 32 }");
    assert_eq!(code, 42);
}

#[test]
fn test_native_arithmetic_mul_sub() {
    let (code, _) = compile_and_run("fn main() -> i64 { (10 * 5) - 8 }");
    assert_eq!(code, 42);
}

#[test]
fn test_native_arithmetic_div_rem() {
    let (code, _) = compile_and_run("fn main() -> i64 { (100 / 4) + (10 % 3) + 16 }");
    assert_eq!(code, 42);
}

#[test]
fn test_native_bool_literal_in_if() {
    let (code, _) = compile_and_run("fn main() -> i64 { if true { 42 } else { 0 } }");
    assert_eq!(code, 42);
}

#[test]
fn test_native_comparisons_eq_lt_gt() {
    let (code_eq, _) = compile_and_run("fn main() -> i64 { if 10 == 10 { 42 } else { 0 } }");
    assert_eq!(code_eq, 42);

    let (code_lt, _) = compile_and_run("fn main() -> i64 { if 10 < 32 { 42 } else { 0 } }");
    assert_eq!(code_lt, 42);

    let (code_gt, _) = compile_and_run("fn main() -> i64 { if 32 > 10 { 42 } else { 0 } }");
    assert_eq!(code_gt, 42);
}

#[test]
fn test_native_local_variables() {
    let (code, _) = compile_and_run("fn main() -> i64 { let x: i64 = 40; let y: i64 = 2; x + y }");
    assert_eq!(code, 42);
}

#[test]
fn test_native_function_call() {
    let (code, _) = compile_and_run(
        "fn add(a: i64, b: i64) -> i64 { a + b }\nfn main() -> i64 { add(10, 32) }",
    );
    assert_eq!(code, 42);
}

#[test]
fn test_native_call_inside_binary_expression() {
    // Regression guard: evaluating a binary op must preserve stack alignment so nested calls remain
    // ABI-correct on System V platforms.
    let (code, _) =
        compile_and_run("fn aux() -> i64 { 39 }\nfn main() -> i64 { aux() + 3 }");
    assert_eq!(code, 42);
}

#[test]
fn test_native_if_else() {
    let (code, _) = compile_and_run("fn main() -> i64 { if true { 42 } else { 0 } }");
    assert_eq!(code, 42);
}

#[test]
fn test_native_logical_and_short_circuit() {
    let (code, _) =
        compile_and_run("fn main() -> i64 { if false && ((1 / 0) == 0) { 0 } else { 42 } }");
    assert_eq!(code, 42);
}

#[test]
fn test_native_logical_or_short_circuit() {
    let (code, _) =
        compile_and_run("fn main() -> i64 { if true || ((1 / 0) == 0) { 42 } else { 0 } }");
    assert_eq!(code, 42);
}

#[test]
fn test_native_print_builtin() {
    let (code, stdout) = compile_and_run(r#"fn main() -> i64 with IO { print("hello\n"); 0 }"#);
    assert_eq!(code, 0);
    assert_eq!(stdout, "hello\n");
}

#[test]
fn test_native_print_non_string_error_is_explicit() {
    let err = compile_error("fn main() -> i64 with IO { print(42); 0 }");
    assert!(
        err.contains("print currently supports only string literals"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_native_while_loop() {
    let (code, _) =
        compile_and_run("fn main() -> i64 { let x: i64 = 0; while x < 42 { x = x + 1 }; x }");
    assert_eq!(code, 42);
}

#[test]
fn test_native_user_defined_print_is_rejected_with_clear_error() {
    let err = compile_error("fn print(x: i64) -> i64 { x + 1 }\nfn main() -> i64 { print(41) }");
    assert!(
        err.contains("Duplicate definition: print"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_native_entry_uses_builtin_exit_even_if_user_exit_exists() {
    let (code, _) = compile_and_run("fn exit(code: i64) -> i64 { code + 1 }\nfn main() -> i64 { exit(41) }");
    assert_eq!(code, 42);
}

#[test]
#[ignore = "TODO(native): add lowering and codegen support for structs"]
fn test_native_structs_todo() {
    let _ = compile_and_run("struct S { x: i64 }\nfn main() -> i64 { let s = S { x: 42 }; s.x }");
}

#[test]
#[ignore = "TODO(native): add lowering and codegen support for enums + match"]
fn test_native_enums_match_todo() {
    let _ = compile_and_run(
        "enum E { A(i64), B(i64) }\nfn main() -> i64 { let e = E::A(42); match e { E::A(x) => x, E::B(x) => x } }",
    );
}

#[test]
#[ignore = "TODO(native): add lowering and codegen support for arrays"]
fn test_native_arrays_todo() {
    let _ = compile_and_run("fn main() -> i64 { let xs = [40, 2]; xs[0] + xs[1] }");
}

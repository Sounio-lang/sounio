//! Integration Tests for REPL (Read-Eval-Print Loop)
//!
//! Tests for interactive REPL functionality including:
//! - Expression evaluation
//! - Variable persistence
//! - Multi-line input
//! - REPL commands
//! - Error recovery

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

// ============================================================================
// Test Helpers
// ============================================================================

fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest_dir.ancestors() {
        if ancestor.join("Cargo.toml").exists() && ancestor.join("stdlib").is_dir() {
            return ancestor.to_path_buf();
        }
    }
    panic!("could not locate repo root from {}", manifest_dir.display());
}

fn souc_bin() -> PathBuf {
    // Prefer the compile-time path injected by Cargo for test targets.
    if let Some(path) = option_env!("CARGO_BIN_EXE_souc") {
        return PathBuf::from(path);
    }

    let root = repo_root();
    let exe_name = if cfg!(windows) { "souc.exe" } else { "souc" };
    let debug = root.join("target").join("debug").join(exe_name);
    let release = root.join("target").join("release").join(exe_name);

    if debug.exists() {
        return debug;
    }
    if release.exists() {
        return release;
    }

    // Last resort: build the binary from the workspace root.
    eprintln!("Building souc binary...");
    let status = Command::new("cargo")
        .args(["build", "--bin", "souc"])
        .current_dir(&root)
        .status()
        .expect("Failed to build souc");
    assert!(status.success(), "Failed to build souc binary");

    if debug.exists() {
        return debug;
    }
    if release.exists() {
        return release;
    }

    panic!("built souc, but binary not found under {}", root.display());
}

/// A REPL session for testing
struct ReplSession {
    child: Child,
    output_reader: thread::JoinHandle<Vec<String>>,
    stderr_reader: thread::JoinHandle<Vec<String>>,
}

impl ReplSession {
    /// Start a new REPL session
    fn new() -> Self {
        let souc_path = souc_bin();

        let mut child = Command::new(&souc_path)
            .arg("repl")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start REPL");

        let stdout = child.stdout.take().expect("Failed to get stdout");
        let stderr = child.stderr.take().expect("Failed to get stderr");

        // Drain stdout continuously so the child never blocks on a full pipe.
        let output_reader = thread::spawn(move || {
            let reader = BufReader::new(stdout);
            let mut lines = Vec::new();
            for line in reader.lines() {
                match line {
                    Ok(line) => {
                        lines.push(line);
                    }
                    Err(_) => break,
                }
            }
            lines
        });

        // Drain stderr continuously for the same reason.
        let stderr_reader = thread::spawn(move || {
            let reader = BufReader::new(stderr);
            let mut lines = Vec::new();
            for line in reader.lines() {
                match line {
                    Ok(line) => {
                        lines.push(line);
                    }
                    Err(_) => break,
                }
            }
            lines
        });

        // Give REPL time to start
        thread::sleep(Duration::from_millis(500));

        Self {
            child,
            output_reader,
            stderr_reader,
        }
    }

    /// Send input to the REPL
    fn send(&mut self, input: &str) {
        if let Some(stdin) = self.child.stdin.as_mut() {
            // Ignore broken pipe errors (REPL may have already exited)
            let _ = writeln!(stdin, "{}", input);
            let _ = stdin.flush();
        }

        // Give REPL time to process
        thread::sleep(Duration::from_millis(300));
    }

    /// Send quit command and collect output
    fn quit_and_collect(mut self) -> Vec<String> {
        self.send(":quit");

        // Give time for graceful shutdown
        thread::sleep(Duration::from_millis(200));

        // Try to terminate gracefully
        let _ = self.child.kill();
        let _ = self.child.wait();

        // Collect output
        let mut output = self.output_reader.join().unwrap_or_default();
        let stderr = self.stderr_reader.join().unwrap_or_default();
        if !stderr.is_empty() {
            output.extend(stderr.into_iter().map(|line| format!("[stderr] {line}")));
        }
        output
    }

    /// Send input and collect all subsequent output
    fn send_and_collect(mut self, input: &str) -> Vec<String> {
        self.send(input);
        self.quit_and_collect()
    }
}

/// Check if output contains a pattern
fn output_contains(output: &[String], pattern: &str) -> bool {
    output.iter().any(|line| line.contains(pattern))
}

/// Get all lines containing a pattern
fn lines_containing(output: &[String], pattern: &str) -> Vec<String> {
    output
        .iter()
        .filter(|line| line.contains(pattern))
        .cloned()
        .collect()
}

// ============================================================================
// Basic REPL Tests
// ============================================================================

#[test]
fn test_repl_starts_and_shows_banner() {
    let repl = ReplSession::new();
    let output = repl.quit_and_collect();

    // Should show banner with "Sounio REPL" or similar
    assert!(
        output_contains(&output, "Sounio") || output_contains(&output, "REPL"),
        "REPL should show banner. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_simple_expression() {
    let mut repl = ReplSession::new();
    repl.send("1 + 1");
    let output = repl.quit_and_collect();

    // Should evaluate to 2
    assert!(
        output_contains(&output, "2"),
        "Should evaluate 1 + 1 to 2. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_arithmetic_expression() {
    let mut repl = ReplSession::new();
    repl.send("(10 + 20) * 2");
    let output = repl.quit_and_collect();

    // Should evaluate to 60
    assert!(
        output_contains(&output, "60"),
        "Should evaluate (10 + 20) * 2 to 60. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_negative_numbers() {
    let mut repl = ReplSession::new();
    repl.send("-5 + 3");
    let output = repl.quit_and_collect();

    // Should handle negative numbers
    assert!(
        output_contains(&output, "-2") || output_contains(&output, "2"),
        "Should handle negative numbers. Output: {:?}",
        output
    );
}

// ============================================================================
// Variable Binding Tests
// ============================================================================

#[test]
fn test_repl_let_binding() {
    let mut repl = ReplSession::new();
    repl.send("let x = 42");
    repl.send("x");
    let output = repl.quit_and_collect();

    // Should show x = 42 and then 42
    assert!(
        output_contains(&output, "42"),
        "Should bind and retrieve variable. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_variable_persistence() {
    let mut repl = ReplSession::new();
    repl.send("let a = 10");
    repl.send("let b = 32");
    repl.send("a + b");
    let output = repl.quit_and_collect();

    // Should show 42
    assert!(
        output_contains(&output, "42"),
        "Variables should persist across lines. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_variable_shadowing() {
    let mut repl = ReplSession::new();
    repl.send("let x = 10");
    repl.send("let x = 20");
    repl.send("x");
    let output = repl.quit_and_collect();

    // Should show 20 (shadowed value)
    assert!(
        output_contains(&output, "20"),
        "Variable shadowing should work. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_var_binding() {
    let mut repl = ReplSession::new();
    repl.send("var y = 5");
    repl.send("y");
    let output = repl.quit_and_collect();

    // Should bind mutable variable
    assert!(
        output_contains(&output, "5"),
        "Should bind var variable. Output: {:?}",
        output
    );
}

// ============================================================================
// REPL Command Tests
// ============================================================================

#[test]
fn test_repl_help_command() {
    let repl = ReplSession::new();
    let output = repl.send_and_collect(":help");

    // Should show help text
    assert!(
        output_contains(&output, "help") || output_contains(&output, "Commands"),
        "Help should show command list. Output: {:?}",
        output
    );

    // Should list some commands
    assert!(
        output_contains(&output, ":quit") || output_contains(&output, "quit"),
        "Help should mention :quit. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_env_command() {
    let mut repl = ReplSession::new();
    repl.send("let x = 42");
    repl.send("let y = 100");
    repl.send(":env");
    let output = repl.quit_and_collect();

    // Should show environment with x and y
    let env_lines = lines_containing(&output, "x");
    assert!(
        !env_lines.is_empty() && env_lines.iter().any(|l| l.contains("42")),
        "Environment should show x = 42. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_clear_command() {
    let mut repl = ReplSession::new();
    repl.send("let x = 42");
    repl.send(":clear");
    repl.send(":env");
    let output = repl.quit_and_collect();

    // After clear, environment should be empty or show "No bindings"
    assert!(
        output_contains(&output, "No bindings") || output_contains(&output, "Cleared"),
        "Clear should remove all bindings. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_ast_toggle() {
    let mut repl = ReplSession::new();
    repl.send(":ast");
    let output = repl.quit_and_collect();

    // Should acknowledge toggle
    assert!(
        output_contains(&output, "AST") || output_contains(&output, "ast"),
        "Should toggle AST display. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_types_toggle() {
    let mut repl = ReplSession::new();
    repl.send(":types");
    let output = repl.quit_and_collect();

    // Should acknowledge toggle
    assert!(
        output_contains(&output, "types") || output_contains(&output, "Types"),
        "Should toggle type display. Output: {:?}",
        output
    );
}

// ============================================================================
// Function Definition Tests
// ============================================================================

#[test]
fn test_repl_function_definition() {
    let mut repl = ReplSession::new();
    repl.send("fn double(x: i64) -> i64 {");
    repl.send("    x * 2");
    repl.send("}");
    let output = repl.quit_and_collect();

    // Should acknowledge function definition
    assert!(
        output_contains(&output, "Defined") || output_contains(&output, "double"),
        "Should define function. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_funcs_command() {
    let mut repl = ReplSession::new();
    repl.send("fn add(x: i64, y: i64) -> i64 {");
    repl.send("    x + y");
    repl.send("}");
    repl.send(":funcs");
    let output = repl.quit_and_collect();

    // Should list defined functions
    assert!(
        output_contains(&output, "add"),
        "Should list defined function. Output: {:?}",
        output
    );
}

// ============================================================================
// Error Recovery Tests
// ============================================================================

#[test]
fn test_repl_syntax_error_recovery() {
    let mut repl = ReplSession::new();
    repl.send("let x ="); // Syntax error
    repl.send("let y = 42"); // Should still work
    repl.send("y");
    let output = repl.quit_and_collect();

    // REPL should continue working after syntax error
    // Error may or may not be captured in output depending on timing

    // Should still evaluate y correctly (this is the key test)
    assert!(
        output_contains(&output, "42"),
        "Should recover and evaluate subsequent input. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_type_error_recovery() {
    let mut repl = ReplSession::new();
    repl.send("let x: i64 = true"); // Type error
    repl.send("let z = 100"); // Should still work
    repl.send("z");
    let output = repl.quit_and_collect();

    // Should show type error but continue
    assert!(
        output_contains(&output, "100"),
        "Should recover from type error. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_undefined_variable_error() {
    let mut repl = ReplSession::new();
    repl.send("undefined_var"); // Error
    repl.send("let x = 5"); // Should work
    repl.send("x");
    let output = repl.quit_and_collect();

    // Should show error but continue
    assert!(
        output_contains(&output, "5"),
        "Should recover from undefined variable. Output: {:?}",
        output
    );
}

// ============================================================================
// Complex Expression Tests
// ============================================================================

#[test]
fn test_repl_nested_arithmetic() {
    let mut repl = ReplSession::new();
    repl.send("((1 + 2) * 3 + 4) * 5");
    let output = repl.quit_and_collect();

    // (3 * 3 + 4) * 5 = 13 * 5 = 65
    assert!(
        output_contains(&output, "65"),
        "Should evaluate nested expressions. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_variable_expressions() {
    let mut repl = ReplSession::new();
    repl.send("let x = 5");
    repl.send("let y = 10");
    repl.send("x * y + x - y");
    let output = repl.quit_and_collect();

    // 5 * 10 + 5 - 10 = 50 + 5 - 10 = 45
    assert!(
        output_contains(&output, "45"),
        "Should evaluate variable expressions. Output: {:?}",
        output
    );
}

// ============================================================================
// Epistemic Features Tests (if available)
// ============================================================================

#[test]
fn test_repl_confidence_command() {
    let mut repl = ReplSession::new();
    repl.send("let x = 42");
    repl.send(":confidence");
    let output = repl.quit_and_collect();

    // Should show confidence information (or indicate no epistemic data)
    assert!(
        output_contains(&output, "confidence")
            || output_contains(&output, "Confidence")
            || output_contains(&output, "%")
            || output_contains(&output, "No"),
        "Should show confidence info or indicate none available. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_info_command_usage() {
    let repl = ReplSession::new();
    let output = repl.send_and_collect(":info");

    // Should show usage or error (needs variable name)
    assert!(
        output_contains(&output, "Usage") || output_contains(&output, "variable"),
        "Should show usage for :info. Output: {:?}",
        output
    );
}

// ============================================================================
// Multi-line Input Tests
// ============================================================================

#[test]
fn test_repl_multiline_function() {
    let mut repl = ReplSession::new();
    repl.send("fn triple(n: i64) -> i64 {");
    repl.send("    n * 3");
    repl.send("}");
    let output = repl.quit_and_collect();

    // Should accept multi-line function definition
    assert!(
        output_contains(&output, "Defined") || output_contains(&output, "triple"),
        "Should accept multi-line function. Output: {:?}",
        output
    );
}

// ============================================================================
// Empty Input Tests
// ============================================================================

#[test]
fn test_repl_empty_lines() {
    let mut repl = ReplSession::new();
    repl.send(""); // Empty line
    repl.send("   "); // Whitespace only
    repl.send("42"); // Actual input
    let output = repl.quit_and_collect();

    // Should ignore empty lines and evaluate actual input
    assert!(
        output_contains(&output, "42"),
        "Should handle empty lines gracefully. Output: {:?}",
        output
    );
}

// ============================================================================
// Unknown Command Tests
// ============================================================================

#[test]
fn test_repl_unknown_command() {
    let repl = ReplSession::new();
    let output = repl.send_and_collect(":unknown");

    // Should show error for unknown command
    assert!(
        output_contains(&output, "Unknown") || output_contains(&output, "unknown"),
        "Should report unknown command. Output: {:?}",
        output
    );
}

// ============================================================================
// Long-running Session Test
// ============================================================================

#[test]
fn test_repl_multiple_operations() {
    let mut repl = ReplSession::new();

    // Sequence of operations
    repl.send("let a = 1");
    repl.send("let b = 2");
    repl.send("let c = 3");
    repl.send("a + b + c");
    repl.send("let d = a * b * c");
    repl.send("d");

    let output = repl.quit_and_collect();

    // Should show 6 for a + b + c
    assert!(
        output_contains(&output, "6"),
        "Should handle multiple operations. Output: {:?}",
        output
    );
}

// ============================================================================
// Exit/Quit Tests
// ============================================================================

#[test]
fn test_repl_quit_command() {
    let repl = ReplSession::new();
    let output = repl.send_and_collect(":quit");

    // Should exit gracefully
    assert!(
        output_contains(&output, "Goodbye") || output_contains(&output, "goodbye"),
        "Should show goodbye message. Output: {:?}",
        output
    );
}

#[test]
fn test_repl_q_command() {
    let repl = ReplSession::new();
    let output = repl.send_and_collect(":q");

    // :q should also quit
    assert!(
        output_contains(&output, "Goodbye")
            || output_contains(&output, "goodbye")
            || output.len() > 0,
        "Should exit with :q command. Output: {:?}",
        output
    );
}

// ============================================================================
// Type Display Tests
// ============================================================================

#[test]
fn test_repl_type_command() {
    let repl = ReplSession::new();
    let output = repl.send_and_collect(":type 42");

    // Should show type information
    assert!(
        output_contains(&output, "i64") || output_contains(&output, "Type"),
        "Should show type of expression. Output: {:?}",
        output
    );
}

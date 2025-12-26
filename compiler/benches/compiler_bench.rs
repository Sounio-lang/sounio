//! Compiler Pipeline Benchmarks
//!
//! Comprehensive benchmarks for the Sounio compiler:
//! - Lexer performance (tokens/sec, lines/sec)
//! - Parser performance (AST nodes/sec)
//! - Type checker performance
//! - Interpreter execution speed
//! - JIT compilation time (when available)

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use sounio::{check, interp::Interpreter, lexer, parser};

// ============================================================================
// Sample Programs of Varying Sizes
// ============================================================================

/// Generate a simple program with N function definitions
fn generate_simple_program(n: usize) -> String {
    let functions: Vec<String> = (0..n)
        .map(|i| {
            format!(
                r#"fn func_{i}(x: i32) -> i32 {{
    return x + {i};
}}"#
            )
        })
        .collect();
    functions.join("\n\n")
}

/// Generate a program with arithmetic expressions
fn generate_arithmetic_program(depth: usize, breadth: usize) -> String {
    fn gen_expr(depth: usize) -> String {
        if depth == 0 {
            "1".to_string()
        } else {
            format!(
                "({} + {} * {} - {})",
                gen_expr(depth - 1),
                gen_expr(depth - 1),
                gen_expr(depth - 1),
                gen_expr(depth - 1),
            )
        }
    }

    let bindings: Vec<String> = (0..breadth)
        .map(|i| format!("    let x{} = {};", i, gen_expr(depth)))
        .collect();

    format!(
        r#"fn compute() -> i32 {{
{}
    return x0;
}}"#,
        bindings.join("\n")
    )
}

/// Generate a program with nested control flow
fn generate_control_flow_program(depth: usize) -> String {
    fn gen_if(depth: usize, indent: usize) -> String {
        let spaces = " ".repeat(indent);
        if depth == 0 {
            format!("{}return 1;", spaces)
        } else {
            format!(
                r#"{spaces}if true {{
{body1}
{spaces}}} else {{
{body2}
{spaces}}}"#,
                body1 = gen_if(depth - 1, indent + 4),
                body2 = gen_if(depth - 1, indent + 4),
            )
        }
    }

    format!(
        r#"fn nested_control() -> i32 {{
{}
}}"#,
        gen_if(depth, 4)
    )
}

/// Generate a program with struct definitions
fn generate_struct_program(n_structs: usize, n_fields: usize) -> String {
    let structs: Vec<String> = (0..n_structs)
        .map(|i| {
            let fields: Vec<String> = (0..n_fields)
                .map(|j| format!("    field_{}: i32", j))
                .collect();
            format!("struct Struct_{} {{\n{}\n}}", i, fields.join(",\n"))
        })
        .collect();
    structs.join("\n\n")
}

/// Generate a program with function calls
fn generate_call_program(depth: usize) -> String {
    let functions: Vec<String> = (0..depth)
        .map(|i| {
            if i == 0 {
                format!("fn func_0(x: i32) -> i32 {{ return x + 1; }}")
            } else {
                format!(
                    "fn func_{i}(x: i32) -> i32 {{ return func_{}(x) + {i}; }}",
                    i - 1
                )
            }
        })
        .collect();

    let main_fn = format!(
        "fn main() -> i32 {{ return func_{}(0); }}",
        depth.saturating_sub(1)
    );

    format!("{}\n\n{}", functions.join("\n\n"), main_fn)
}

/// Generate a program with effects
fn generate_effects_program(n_effects: usize) -> String {
    let effects = vec!["IO", "Mut", "Alloc", "Panic", "Async", "Prob"];
    let used_effects: Vec<&str> = effects.iter().take(n_effects).copied().collect();

    let functions: Vec<String> = (0..10)
        .map(|i| {
            format!(
                "fn func_{}() with {} {{ }}",
                i,
                used_effects.join(", ")
            )
        })
        .collect();
    functions.join("\n")
}

/// A realistic pharmacokinetics-style program
fn pk_model_program() -> String {
    r#"
// Pharmacokinetic Model
struct PKParams {
    ka: f64,
    ke: f64,
    vd: f64,
    dose: f64
}

fn one_compartment(params: PKParams, time: f64) -> f64 {
    let ka = params.ka;
    let ke = params.ke;
    let vd = params.vd;
    let dose = params.dose;

    let term1 = ka / (ka - ke);
    let exp_ke = 0.0 - ke * time;
    let exp_ka = 0.0 - ka * time;

    return (dose / vd) * term1;
}

fn compute_concentration() -> f64 {
    let params = PKParams {
        ka: 1.0,
        ke: 0.1,
        vd: 10.0,
        dose: 100.0
    };
    return one_compartment(params, 2.0);
}
"#
    .to_string()
}

// ============================================================================
// Lexer Benchmarks
// ============================================================================

fn bench_lexer(c: &mut Criterion) {
    let mut group = c.benchmark_group("lexer");

    // Small program
    let small = generate_simple_program(10);
    group.throughput(Throughput::Bytes(small.len() as u64));
    group.bench_with_input(BenchmarkId::new("small", small.len()), &small, |b, s| {
        b.iter(|| lexer::lex(black_box(s)))
    });

    // Medium program
    let medium = generate_simple_program(100);
    group.throughput(Throughput::Bytes(medium.len() as u64));
    group.bench_with_input(BenchmarkId::new("medium", medium.len()), &medium, |b, s| {
        b.iter(|| lexer::lex(black_box(s)))
    });

    // Large program
    let large = generate_simple_program(500);
    group.throughput(Throughput::Bytes(large.len() as u64));
    group.bench_with_input(BenchmarkId::new("large", large.len()), &large, |b, s| {
        b.iter(|| lexer::lex(black_box(s)))
    });

    // Lines per second benchmark
    let lines_test = (0..1000).map(|i| format!("let x{} = {};", i, i)).collect::<Vec<_>>().join("\n");
    let line_count = lines_test.lines().count();
    group.throughput(Throughput::Elements(line_count as u64));
    group.bench_with_input(
        BenchmarkId::new("lines_per_sec", line_count),
        &lines_test,
        |b, s| b.iter(|| lexer::lex(black_box(s))),
    );

    group.finish();
}

// ============================================================================
// Parser Benchmarks
// ============================================================================

fn bench_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser");

    // Simple functions
    for n in [10, 50, 100, 200] {
        let source = generate_simple_program(n);
        let tokens = lexer::lex(&source).unwrap();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("simple_functions", n),
            &(tokens.clone(), source.clone()),
            |b, (t, s)| b.iter(|| parser::parse(black_box(t), s)),
        );
    }

    // Arithmetic expressions
    for depth in [2, 3, 4] {
        let source = generate_arithmetic_program(depth, 5);
        let tokens = lexer::lex(&source).unwrap();
        group.bench_with_input(
            BenchmarkId::new("arithmetic_depth", depth),
            &(tokens.clone(), source.clone()),
            |b, (t, s)| b.iter(|| parser::parse(black_box(t), s)),
        );
    }

    // Control flow nesting
    for depth in [3, 5, 7] {
        let source = generate_control_flow_program(depth);
        let tokens = lexer::lex(&source).unwrap();
        group.bench_with_input(
            BenchmarkId::new("control_flow_depth", depth),
            &(tokens.clone(), source.clone()),
            |b, (t, s)| b.iter(|| parser::parse(black_box(t), s)),
        );
    }

    // Struct definitions
    for n in [10, 50, 100] {
        let source = generate_struct_program(n, 5);
        let tokens = lexer::lex(&source).unwrap();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("struct_definitions", n),
            &(tokens.clone(), source.clone()),
            |b, (t, s)| b.iter(|| parser::parse(black_box(t), s)),
        );
    }

    // Realistic PK model
    let pk_source = pk_model_program();
    let pk_tokens = lexer::lex(&pk_source).unwrap();
    group.bench_with_input(
        BenchmarkId::new("pk_model", 1),
        &(pk_tokens.clone(), pk_source.clone()),
        |b, (t, s)| b.iter(|| parser::parse(black_box(t), s)),
    );

    group.finish();
}

// ============================================================================
// Type Checker Benchmarks
// ============================================================================

fn bench_typecheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("typecheck");

    // Simple functions
    for n in [10, 50, 100] {
        let source = generate_simple_program(n);
        let tokens = lexer::lex(&source).unwrap();
        let ast = parser::parse(&tokens, &source).unwrap();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("simple_functions", n),
            &ast,
            |b, a| b.iter(|| check::check(black_box(a))),
        );
    }

    // Function call chains
    for depth in [5, 10, 20] {
        let source = generate_call_program(depth);
        if let Ok(tokens) = lexer::lex(&source) {
            if let Ok(ast) = parser::parse(&tokens, &source) {
                group.bench_with_input(
                    BenchmarkId::new("call_chain_depth", depth),
                    &ast,
                    |b, a| b.iter(|| check::check(black_box(a))),
                );
            }
        }
    }

    // Effects
    for n in [1, 3, 6] {
        let source = generate_effects_program(n);
        if let Ok(tokens) = lexer::lex(&source) {
            if let Ok(ast) = parser::parse(&tokens, &source) {
                group.bench_with_input(
                    BenchmarkId::new("with_effects", n),
                    &ast,
                    |b, a| b.iter(|| check::check(black_box(a))),
                );
            }
        }
    }

    // PK model
    let pk_source = pk_model_program();
    if let Ok(tokens) = lexer::lex(&pk_source) {
        if let Ok(ast) = parser::parse(&tokens, &pk_source) {
            group.bench_with_input(
                BenchmarkId::new("pk_model", 1),
                &ast,
                |b, a| b.iter(|| check::check(black_box(a))),
            );
        }
    }

    group.finish();
}

// ============================================================================
// Interpreter Benchmarks
// ============================================================================

fn bench_interpreter(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpreter");

    // Simple arithmetic
    let arith = r#"
fn main() -> i32 {
    let a = 1 + 2 * 3;
    let b = 4 - 5 / 1;
    return a + b;
}
"#;
    if let Ok(tokens) = lexer::lex(arith) {
        if let Ok(ast) = parser::parse(&tokens, arith) {
            if let Ok(hir) = check::check(&ast) {
                group.bench_function("simple_arithmetic", |b| {
                    b.iter(|| {
                        let mut interp = Interpreter::new();
                        interp.interpret(black_box(&hir))
                    })
                });
            }
        }
    }

    // Function calls
    let calls = generate_call_program(10);
    if let Ok(tokens) = lexer::lex(&calls) {
        if let Ok(ast) = parser::parse(&tokens, &calls) {
            if let Ok(hir) = check::check(&ast) {
                group.bench_function("function_calls_10", |b| {
                    b.iter(|| {
                        let mut interp = Interpreter::new();
                        interp.interpret(black_box(&hir))
                    })
                });
            }
        }
    }

    // Loop iterations
    for n in [10, 100, 1000] {
        let loop_prog = format!(
            r#"
fn main() -> i32 {{
    var sum = 0;
    var i = 0;
    while i < {} {{
        sum = sum + i;
        i = i + 1;
    }}
    return sum;
}}
"#,
            n
        );
        if let Ok(tokens) = lexer::lex(&loop_prog) {
            if let Ok(ast) = parser::parse(&tokens, &loop_prog) {
                if let Ok(hir) = check::check(&ast) {
                    group.throughput(Throughput::Elements(n as u64));
                    group.bench_with_input(
                        BenchmarkId::new("loop_iterations", n),
                        &hir,
                        |b, h| b.iter(|| {
                            let mut interp = Interpreter::new();
                            interp.interpret(black_box(h))
                        }),
                    );
                }
            }
        }
    }

    // Recursive fibonacci (if supported)
    let fib = r#"
fn fib(n: i32) -> i32 {
    if n < 2 {
        return n;
    } else {
        return fib(n - 1) + fib(n - 2);
    }
}

fn main() -> i32 {
    return fib(15);
}
"#;
    if let Ok(tokens) = lexer::lex(fib) {
        if let Ok(ast) = parser::parse(&tokens, fib) {
            if let Ok(hir) = check::check(&ast) {
                group.bench_function("fibonacci_15", |b| {
                    b.iter(|| {
                        let mut interp = Interpreter::new();
                        interp.interpret(black_box(&hir))
                    })
                });
            }
        }
    }

    group.finish();
}

// ============================================================================
// Full Pipeline Benchmarks
// ============================================================================

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    // End-to-end: source -> interpret
    for n in [10, 50, 100] {
        let source = generate_simple_program(n);
        group.throughput(Throughput::Bytes(source.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("lex_parse_check", n),
            &source,
            |b, s| {
                b.iter(|| {
                    let tokens = lexer::lex(black_box(s)).unwrap();
                    let ast = parser::parse(&tokens, s).unwrap();
                    check::check(&ast)
                })
            },
        );
    }

    // PK model full pipeline
    let pk_source = pk_model_program();
    group.bench_with_input(
        BenchmarkId::new("pk_model_full", 1),
        &pk_source,
        |b, s| {
            b.iter(|| {
                let tokens = lexer::lex(black_box(s)).unwrap();
                let ast = parser::parse(&tokens, s).unwrap();
                if let Ok(hir) = check::check(&ast) {
                    let mut interp = Interpreter::new();
                    interp.interpret(&hir)
                } else {
                    Err(miette::miette!("Failed to typecheck"))
                }
            })
        },
    );

    group.finish();
}

// ============================================================================
// Memory Benchmarks
// ============================================================================

fn bench_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");

    // Measure AST size for different program sizes
    for n in [10, 50, 100, 200] {
        let source = generate_simple_program(n);
        let tokens = lexer::lex(&source).unwrap();

        group.bench_with_input(
            BenchmarkId::new("ast_construction", n),
            &(tokens.clone(), source.clone()),
            |b, (t, s)| {
                b.iter(|| {
                    let ast = parser::parse(black_box(t), s).unwrap();
                    // Force the AST to be kept in memory
                    black_box(ast.items.len())
                })
            },
        );
    }

    // Token stream size
    for n in [100, 500, 1000] {
        let source = generate_simple_program(n);
        group.bench_with_input(
            BenchmarkId::new("token_stream", n),
            &source,
            |b, s| {
                b.iter(|| {
                    let tokens = lexer::lex(black_box(s)).unwrap();
                    black_box(tokens.len())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_lexer,
    bench_parser,
    bench_typecheck,
    bench_interpreter,
    bench_full_pipeline,
    bench_memory,
);

criterion_main!(benches);

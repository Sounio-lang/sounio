> **Status**: Production | **Last validated**: 2026-04-12 | **Source**: `tests/run-pass/`

# Effect System

Sounio uses algebraic effects to track side effects in the type system. Functions that perform side effects **must** declare them with `with`.

## Required Effects

| Effect | When required | Example |
|---|---|---|
| `IO` | `print()`, `println()`, file/env | `println("text")` |
| `Mut` | `&!` mutation, array assignment | `*x = 42` |
| `Div, Panic` | Division `/`, modulo `%` | `a / b` |
| `Panic` | Array access, `assert()`, `as` casts | `arr[i]` |
| `Alloc` | Heap allocation | rare |
| `Async` | Asynchronous operations | `spawn`, `await` |
| `GPU` | GPU kernel launch | `gpu.launch()` |
| `Observe` | Observation boundaries | comparison, FFI |
| `Prob` | Probabilistic operations | sampling |

## Examples

```sio
fn pure_add(a: i64, b: i64) -> i64 { a + b }
fn mutate(x: &!i32) with Mut { *x = 42 }
fn divide(a: f64, b: f64) -> f64 with Div, Panic { a / b }
fn hello() with IO { println("hi") }
fn process(arr: &![u8; 256]) with IO, Mut, Panic, Div { /* ... */ }
```

## Custom Effects

```sio
effect Choice {
    fn pick() -> bool
}

fn coin_flip() with Choice {
    // uses Choice effect
}
```

## Rules

1. Missing effects cause compile errors
2. Effects propagate: if `f` calls `g` with `IO`, then `f` must also declare `IO`
3. Pure functions (no `with` clause) cannot perform any side effects

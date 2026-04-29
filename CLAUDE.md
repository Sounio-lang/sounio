# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Recovery context**: [CLAUDE_HANDOFF.md](CLAUDE_HANDOFF.md) | **Syntax ref**: [docs/guide/LLM_PROGRAMMING_GUIDE.md](docs/guide/LLM_PROGRAMMING_GUIDE.md) | **LLM guide**: [docs/llm-guide/](docs/llm-guide/) | **Minimum viable**: [docs/guide/MINIMUM_VIABLE_SOUNIO.md](docs/guide/MINIMUM_VIABLE_SOUNIO.md)

## Project Identity

**Sounio** — a self-hosted systems + scientific programming language for epistemic computing, uncertainty propagation, and algebraic effects. NOT a Rust/Julia dialect; own syntax, semantics, philosophy. Linux x86-64 only.

## Session Bootstrap

Before non-trivial changes:

1. Read `CLAUDE_HANDOFF.md` — recovery history and workspace context
2. Verify current branch (should be `integration/sounio-dev-ready-base`)
3. Do not start from `main` until reconciliation is completed
4. Do not propose destructive reset/clean/rebase flows on this repo

## Build & Run

The compiler is **self-hosted** (written in Sounio, not Rust). `bin/souc` is a bash wrapper around the native self-hosted binary at `artifacts/self-hosted/souc-self-hosted-x86_64`.

```bash
SOUC=./bin/souc
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib   # needed when outside repo root

$SOUC --version                           # verify toolchain
$SOUC check file.sio                      # type-check only
$SOUC run file.sio                        # compile to temp ELF, execute, clean up
$SOUC compile file.sio -o output.elf      # emit named ELF binary
$SOUC info                                # compiler status

# Bootstrap chain (fixed-point verification)
make build      # boot4 → gen1 → gen2 → gen3, verifies gen2 == gen3
make clean      # remove gen1/gen2/gen3.elf
```

## Testing

```bash
# Full test suite (run-pass + compile-fail + stdlib)
bash scripts/run_sio_test_suite.sh
bash scripts/run_sio_test_suite.sh --verbose

# Single test by name pattern
bash scripts/run_sio_test_suite.sh covid
bash scripts/run_sio_test_suite.sh vancomycin --verbose

# Stdlib gates
bash scripts/stdlib_hyper_execution_gate.sh
bash scripts/stdlib_science_pipeline_gate.sh
bash scripts/stdlib_reliability_gate.sh

# Lint for Rust hallucinations in .sio files
make lint
make lint-fix FILE=path/to/file.sio

# Type-check compiler + CI gates
make check

# Doctor the workspace
bash scripts/dev/doctor_workspace.sh
```

### Test Annotations

Tests use header comments for the harness:

- `//@ run-pass` — expect exit 0
- `//@ compile-fail` — expect exit != 0
- `//@ expect-stdout: X` — stdout must contain X (run-pass only)
- `//@ error-pattern: X` — stderr/stdout must contain X (compile-fail only)
- `//@ ignore` — skip this test
- `//@ check-only` — compile only, do not execute

Test directories: `tests/run-pass/`, `tests/compile-fail/`, `tests/ui/`, `tests/stdlib/`, `tests/selfhost/`, `tests/native/`, `tests/regression/`

## Sounio Syntax (NOT Rust)

**Critical differences — these are compile errors:**

| Wrong (Rust) | Correct (Sounio) |
|---|---|
| `let x = 5;` | `let x = 5` (no semicolons) |
| `let mut y = 10` | `var y = 10` |
| `&mut T` | `&!T` |
| `assert!(cond)` | `assert(cond)` |
| `println!("hi")` | `println("hi")` |
| `#[test]`, `#[derive()]` | No attributes |
| `\|x\| x + 1` | Named fn refs: `let f = square` |
| `-42` | `0 - 42` (no unary minus) |
| `x >> 4` | `x >> 4u8` (bit shifts require u8) |

**Helpers must be defined before callers** — no forward references.

**Quick reference:**

```sio
let x = 5                              // immutable
var y = 10                             // mutable
var buf: [i64; 8] = [0; 8]            // fixed-size array
&T / &!T                               // shared / exclusive ref
fn f(x: i32) -> i32 with IO { }        // effects declaration
linear struct Handle { fd: i32 }       // linear types (consumed exactly once)
let dose: mg = 500.0                   // units
let arr2 = a ++ b                      // array concatenation
type Pos = { x: i32 | x > 0 }          // refinement type
let m: Knowledge<mg> = measure(500.0, uncertainty: 2.5)  // epistemic type
fn observe(x: Unobserved<f64>) -> bool with Observe { x > 0.0 }

// Effects: IO, Mut, Div, Panic, Alloc, Async, GPU, Prob, Observe
fn main() with IO, Mut, Panic, Div { }

// Methods use explicit self
impl MyStruct {
    fn get(self: &MyStruct) -> i64 { self.val }
    fn set(self: &!MyStruct, v: i64) with Mut { self.val = v }
}

// Control flow
for i in 0..10 { }       // exclusive range
for i in 0..=10 { }      // inclusive range
while cond { }
if x > 0 { "pos" } else { "neg" }   // if is an expression
```

## Architecture

**Pipeline:** Source → Lexer → Parser → AST → Check → HIR → SIR → HLIR (SSA) → Codegen (x86-64 ELF)

| Directory | Purpose |
|---|---|
| `self-hosted/lexer/`, `parser/` | Frontend (tokenizer, recursive descent) |
| `self-hosted/check/`, `types/` | Bidirectional type inference + algebraic effects |
| `self-hosted/ir/` | IR lowering, e-graph optimization (1000+ rewrite rules) |
| `self-hosted/native/` | x86-64 ELF emission |
| `self-hosted/compiler/` | Codegen drivers (lean, IR, GPU) |
| `self-hosted/gpu/` | PTX/GPU codegen (exists but no end-to-end CLI path) |
| `stdlib/epistemic/` | Knowledge\<T\>, uncertainty (GUM), provenance |
| `stdlib/units/` | Dimensional analysis |
| `bootstrap/` | stage0 (C) → boot2g → boot3 → boot4 → self-hosted chain |
| `formal/` | Lean 4 proofs (epistemic type invariants) |

**Bootstrap chain**: `bootstrap/stage0` (C, ~103KB) builds the first Sounio binary; successive stages (`boot0`→`boot4`, each written in Sounio) build the next until the compiler compiles itself. Fixed-point: stage N and N+1 produce bit-identical ELFs.

**Key file**: `self-hosted/compiler/lean_single.sio` — the main self-hosted compiler entrypoint.

## Commits

```text
[component] Brief description

Components: lexer, parser, ast, check, types, effects, hir, hlir,
           codegen, backend, cli, docs, stdlib, tests, ontology,
           epistemic, lsp, pkg, sir, units, refinement
```

No AI attribution in commits.

## Working Principles

1. **Sounio syntax** — `&!` not `&mut`, `var` not `let mut`, no Rust macros
2. **Atomic commits** — one logical change per commit
3. **Q1+ research first** — literature review before architecture decisions
4. **Epistemic honesty** — cite sources, acknowledge uncertainty
5. **Edge of novelty** — don't copy existing languages
6. **No drift to mean** — excellence only

## Known Limitations

- `Knowledge<T>` is monomorphic (f64 only) — struct-level generics in progress
- No closure literals — use named fn refs (`let f = square`)
- No unary minus — write `0 - x`
- No REPL/`--show-ast`/`--show-types` in native mode yet
- `&![T; N]` bare array mutation broken in JIT — use struct wrapper or `(*arr)[i]`
- GPU: PTX codegen exists but no end-to-end path from CLI

Full list: [docs/compiler/KNOWN_LIMITATIONS.md](docs/compiler/KNOWN_LIMITATIONS.md)

## Cluster GPU Jobs

The AI/HPC cluster control plane is at `/home/devsounio/beagle/k8s/hpc-sota`. Before GPU work, read:
1. `/home/devsounio/beagle/k8s/hpc-sota/AGENT_BOOTSTRAP.md`
2. `/home/devsounio/beagle/k8s/hpc-sota/DEV_WORKFLOW.md`

Prefer proven wrappers from `ops/lab-ops.sh` over ad hoc `sbatch` or `kubectl` commands.

## LLM Offload

Route bulk/routine tasks to external providers before using Claude:

```bash
llm-offload -t expand -p grok       # outline → prose
llm-offload -t scaffold -p glm      # boilerplate code
llm-offload -t review -p deepseek   # second opinion
llm-offload -t paraphrase -p minimax # rewrite
```

Routing config: `.claude/offload-routing.md`

## Session Persistence

Cross-session context lives in `.claude/`:
- `decisions.md` — architectural choices
- `pending.md` — open questions, WIP
- `session_state.json` — structured state

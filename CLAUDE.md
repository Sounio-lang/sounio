# CLAUDE.md

**START HERE**: [docs/MINIMUM_VIABLE_SOUNIO.md](docs/MINIMUM_VIABLE_SOUNIO.md) | **Syntax ref**: [docs/LLM_PROGRAMMING_GUIDE.md](docs/LLM_PROGRAMMING_GUIDE.md)

## Project Identity

**Sounio** — L0 systems + scientific programming language for epistemic computing. NOT a Rust/Julia dialect; own syntax, semantics, philosophy.

## Working Principles (MANDATORY)

1. **No AI attribution** — No "Co-Authored-By" or similar in commits
2. **Sounio syntax** — `&!` not `&mut`, `var` not `let mut`
3. **Atomic commits** — One logical change per commit
4. **Token efficiency** — Parallel agents, concise ops
5. **YOLO mode** — Execute routine ops without asking
6. **Q1+ research first** — Literature review before architecture decisions
7. **No drift to mean** — Excellence only
8. **Epistemic honesty** — Cite sources, acknowledge uncertainty
9. **Edge of novelty** — Don't copy existing languages

## Sounio Syntax (NOT Rust)

**CRITICAL — What doesn't work:**

- `&mut` → use `&!`
- `assert!()`, `println!()` → no Rust macros
- `#[test]`, `#[derive()]` → no attributes
- `let (a, b) = tuple` → no destructuring
- `pub` → not implemented
- Forward refs → helpers must precede callers

**Quick reference:**

```sio
let x = 5                              // immutable
var y = 10                             // mutable
&T / &!T                               // shared / exclusive ref
fn f(x: i32) -> i32 with IO { }        // effects
linear struct Handle { fd: i32 }       // linear types
let dose: mg = 500.0                   // units
let arr2 = a ++ b                      // concatenation
type Pos = { x: i32 | x > 0 }          // refinement
let m: Knowledge<mg> = measure(500.0, uncertainty: 2.5)  // epistemic
```

## Build Commands

```bash
cd compiler && cargo build [--release]
cargo test [test_name] [-- --nocapture]
cargo run -- check examples/file.sio [--show-ast --show-types]
cargo run --features jit -- run examples/file.sio
cargo run -- repl
cargo clippy && cargo fmt

# Features: jit, llvm, lsp, smt, gpu, ontology, pkg, full
```

**Stdlib path** (when outside repo): `export SOUNIO_STDLIB_PATH=/home/demetrios/sounio-1/stdlib`

## Architecture

**Pipeline:** Source → Lexer → Parser → AST → Check → HIR → SIR → HLIR (SSA) → Codegen

| Module | Purpose |
|--------|---------|
| `lexer/`, `parser/`, `ast/` | Frontend |
| `check/`, `types/` | Bidirectional inference |
| `effects/` | Algebraic effects (IO, Mut, Alloc, Panic, Async, GPU, Prob, Div) |
| `linear/`, `ownership/` | Linear/affine checking |
| `units/` | Dimensional analysis |
| `refinement/`, `smt/` | Z3 refinement types |
| `epistemic/` | Knowledge<T>, uncertainty |
| `ontology/` | 15M+ scientific terms |
| `sir/` | Domain-specific IR (ODEs, tensors, autodiff) |
| `backend/` | Native ELF/Mach-O |
| `codegen/` | LLVM, Cranelift, GPU |

See: [compiler/docs/KNOWN_LIMITATIONS.md](compiler/docs/KNOWN_LIMITATIONS.md)

## Tests

- `compiler/tests/` — Integration (Rust)
- `tests/ui/` — Error messages
- `tests/run-pass/` — Should run
- `tests/compile-fail/` — Should fail

Annotations: `//@ run-pass`, `//@ compile-fail`, `//@ error-pattern: <text>`, `//@ ignore`

## Standards

- `thiserror` for errors, `miette` for diagnostics
- No `unwrap()` in library code
- Doc comments on public items

## Commits

```text
[component] Brief description

Components: lexer, parser, ast, check, types, effects, hir, hlir,
           codegen, backend, cli, docs, stdlib, tests, ontology,
           epistemic, lsp, pkg, sir, units, refinement
```

## LLM Offload

**Providers**: Grok (`grok`), GLM-5 (`glm`), MiniMax (`minimax`), DeepSeek (`deepseek`), Ollama (`local`)

```bash
llm-offload -t expand -p grok       # outline → prose
llm-offload -t scaffold -p glm      # boilerplate code
llm-offload -t review -p deepseek   # second opinion
llm-offload -t paraphrase -p minimax # rewrite
llm-offload --list-providers         # status table
```

**Slash commands** (use inside Claude Code):
- `/offload-expand [provider] [file]` — expand outline → prose
- `/offload-scaffold [provider] [file]` — spec → boilerplate
- `/offload-review [provider] [file]` — independent code review
- `/offload-paraphrase [provider] [file]` — rewrite text

**Pipelines** (multi-model workflows):
```bash
llm-pipeline consensus review -i file.rs    # 3 providers review same code
llm-pipeline expand-critique outline.md     # Grok expands → DeepSeek critiques
llm-pipeline multi-scaffold spec.txt        # 2 providers scaffold → diff
```

**Flow**: Claude designs → `/offload-expand` expands → Claude critiques

## Session Persistence

Use `.claude/` for cross-session context:

- `decisions.md` — Architectural choices
- `pending.md` — Open questions, WIP
- `session_state.json` — Structured state

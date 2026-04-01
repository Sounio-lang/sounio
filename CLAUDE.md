# CLAUDE.md

**START HERE**: [docs/guide/MINIMUM_VIABLE_SOUNIO.md](docs/guide/MINIMUM_VIABLE_SOUNIO.md) | **Syntax ref**: [docs/guide/LLM_PROGRAMMING_GUIDE.md](docs/guide/LLM_PROGRAMMING_GUIDE.md)

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
- `assert!()`, `println!()` → no Rust macros (use `assert()`, `println()`)
- `#[test]`, `#[derive()]` → no attributes
- `|x| x + 1` → no closure literals (named fn refs work: `let f = square`)
- Bare `&![T; N]` array mutation → wrap in struct (see KNOWN_LIMITATIONS.md)

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
algebra Octonion over f64 { add: commutative, associative; mul: alternative, non_commutative; reassociate: fano_selective }
fn observe(x: Unobserved<f64>) -> bool with Observe { x > 0.0 }
```

## Build & Run

The compiler is **self-hosted** (written in Sounio). Use the native wrapper:

```bash
SOUC=./bin/souc

$SOUC check examples/file.sio          # type-check
$SOUC run examples/file.sio            # compile to temp ELF, execute, clean up
$SOUC compile file.sio -o output.elf   # direct native compilation
$SOUC repl                             # not yet supported in native mode

# Debug flags not yet supported in native mode:
#   --show-ast
#   --show-types

# Bootstrap chain
./artifacts/self-hosted/souc-self-hosted-x86_64 self-hosted/compiler/lean_single.sio gen1.elf
./gen1.elf self-hosted/compiler/lean_single.sio gen2.elf
```

**Stdlib path** (when outside repo): `export SOUNIO_STDLIB_PATH=$(pwd)/stdlib`

## Architecture

**Pipeline:** Source → Lexer → Parser → AST → Check → HIR → SIR → HLIR (SSA) → Codegen

| Module | Purpose |
|--------|---------|
| `self-hosted/lexer/`, `parser/` | Frontend (tokenizer, recursive descent) |
| `self-hosted/check/`, `types/` | Bidirectional inference + effects |
| `self-hosted/ir/` | IR lowering, optimization, e-graph |
| `self-hosted/native/` | x86-64 ELF emission |
| `self-hosted/compiler/` | Codegen drivers (lean, IR, GPU) |
| `stdlib/epistemic/` | Knowledge<T>, uncertainty (GUM) |
| `stdlib/units/` | Dimensional analysis |
| `bootstrap/` | stage0 (C) → boot2g → boot1 chain |

See: [docs/compiler/KNOWN_LIMITATIONS.md](docs/compiler/KNOWN_LIMITATIONS.md)

## Tests

- `tests/run-pass/` — Should compile and run
- `tests/compile-fail/` — Should fail to compile
- `tests/ui/` — Error message snapshots
- `tests/stdlib/` — Standard library validation

Annotations: `//@ run-pass`, `//@ compile-fail`, `//@ error-pattern: <text>`, `//@ ignore`

## Commits

```text
[component] Brief description

Components: lexer, parser, ast, check, types, effects, hir, hlir,
           codegen, backend, cli, docs, stdlib, tests, ontology,
           epistemic, lsp, pkg, sir, units, refinement
```

## LLM Offload

**Providers**: Grok (`grok`), GLM-5 (`glm`), MiniMax M2.7 (`minimax`, Anthropic SDK compatible), DeepSeek (`deepseek`), Ollama (`local`)

**Routing config**: `.claude/offload-routing.md` — provider table, MiniMax SDK setup, routing rules

**MiniMax note**: Supports Anthropic messages API via `ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic`. Models: M2.7 (204K ctx), M2.5, M2.1, M2. Supports tools, streaming, thinking.

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

## Session Hygiene (Token Efficiency)

- **`/clear`** — Use at the start of an unrelated task or after a long exploration session
- **`/compact`** — Use when context is large but still relevant (summarizes history)
- **Start sessions with**: read `.claude/session_state.json` — never re-explore what's already tracked
- **Offload first**: route review, expand, scaffold tasks to `llm-offload` before asking Claude
- **Grep before Read**: use Grep/Glob for targeted lookups; only Read when full file content is needed
- **Batch related changes**: group edits to the same module in one session turn
- **Routing**: see `.claude/offload-routing.md` for which tasks go to which offload provider

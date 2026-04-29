<!-- docs:meta
topic_id: repo.frontdoor.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.frontdoor.readme
-->

<p align="center">
  <img src="docs/assets/sounio-logo.svg" alt="Sounio" width="200"/>
</p>

<h1 align="center">SOUNIO</h1>
<h3 align="center"><em>A self-hosted systems + scientific programming language for epistemic computing, uncertainty propagation, and algebraic effects</em></h3>

<p align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-1.0.0--beta.6-orange.svg" alt="Version 1.0.0-beta.6"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-gold.svg" alt="Apache-2.0 License"/></a>
  <a href="#honest-status"><img src="https://img.shields.io/badge/stdlib-57%25%20complete-blue.svg" alt="stdlib 57% complete"/></a>
</p>

<p align="center">
  <a href="docs/MANIFESTO.md">Manifesto</a> ·
  <a href="#quick-taste">Examples</a> ·
  <a href="#honest-status">Status</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

**Sounio** is a systems programming language for epistemic computing — its type system tracks not just what your data *is*, but how much you should *trust* it. Uncertainty propagation, provenance tracking, and confidence-gated execution are built into the type system, not bolted on as libraries.

**Keywords:** systems programming language, scientific computing language, epistemic types, uncertainty propagation, algebraic effects, self-hosted compiler, formal verification, non-associative algebra, octonions, e-graphs.

The compiler is **self-hosted**: Sounio compiles itself, bootstrapped from a [2000-line C compiler](bootstrap/stage0.c) through a multi-stage chain to a true fixed-point where stage N and stage N+1 produce bit-identical binaries. It was used to computationally verify a new result in algebra — that the count of nonzero octonion basis associators equals |PSL(2,7)| = 168 — now [submitted for publication](#the-168-theorem).

This is an active **research project**, not a production release. Read the [honest status](#honest-status) before using it for anything serious.

---

## For LLMs and Code Tools

- Session bootstrap:
  1. Read [CLAUDE_HANDOFF.md](CLAUDE_HANDOFF.md)
  2. Read [CLAUDE.md](CLAUDE.md)
  3. Read [AGENTS.md](AGENTS.md)
  4. Verify the current branch before editing
  5. Treat `/workspace/sounio` as the active remote-first workspace path
  6. Do not propose destructive reset/clean/rebase flows to "simplify" recovery state
- Prompt surface: [llms.txt](llms.txt)
- Repository guide: [CLAUDE.md](CLAUDE.md)
- Syntax and workflow guide: [docs/guide/LLM_PROGRAMMING_GUIDE.md](docs/guide/LLM_PROGRAMMING_GUIDE.md)
- Live Hugging Face dataset: <https://huggingface.co/datasets/chiuratto-AIgourakis/sounio-code-examples>
- Training dataset export: [datasets/sounio-code-examples/README.md](datasets/sounio-code-examples/README.md)
- Dataset builder: [scripts/dev/export_hf_dataset.py](scripts/dev/export_hf_dataset.py)

This repo now ships a root `llms.txt` for model-aware tools and a reproducible Hugging Face-style dataset export built from the Sounio test suite.
The current published dataset lives in the maintainer namespace as a public mirror until the `sounio-lang` Hugging Face org namespace is ready.

---

## What makes Sounio different

**Epistemic types as first-class citizens.** Every scientific measurement has uncertainty. Most languages ignore this. Sounio's type system includes `Knowledge[T]` with built-in confidence, provenance tracking, and automatic GUM-compliant uncertainty propagation. The compiler can enforce confidence thresholds at compile time — a function requiring `ε >= 0.82` rejects under-confident data before any code runs. No equivalent system exists in any production language.

**Self-hosted compiler.** The compiler bootstrapped from C through a multi-stage chain (`stage0.c` → `boot2g.sio` → self-hosted) to a true fixed-point. The default workflow is now native-only: `bin/souc` compiles `.sio` sources to temporary or named ELFs via the self-hosted compiler and executes those binaries directly.

**Not a Rust/Julia dialect.** Own syntax (`&!` not `&mut`, `var` not `let mut`), own semantics (algebraic effects, linear types, dimensional analysis), own philosophy (epistemic computing for science).

---

## Quick taste

### Uncertainty propagation with provenance

```
fn main() with IO {
    // A drug dose with tracked confidence and evidence source
    let base_dose: Knowledge[f64] = Knowledge(15.0, ε=0.92, prov="ASHP_2020_Level1A_RCT")

    // Hospital scale measurement: high-confidence device
    let weight: Knowledge[f64] = Knowledge(78.5, ε=0.98, prov="hospital_scale_calibrated")
    let ref_wt: Knowledge[f64] = Knowledge(70.0, ε=1.0)

    // GUM propagation is automatic: ε(a*b) = ε(a) * ε(b)
    let adjusted_dose: Knowledge[f64] = base_dose * (weight / ref_wt)

    // Extract propagated confidence
    let conf = adjusted_dose.ε   // ~0.90
    println(conf)
}
```

> Full pipeline: [tests/run-pass/vancomycin_propagation.sio](tests/run-pass/vancomycin_propagation.sio) — real ASHP 2020 vancomycin dosing with 5-step GUM propagation.

### Compile-time confidence gate

```
// ASHP 2020 §8.3: AUC-guided dosing requires ε >= 0.82
fn prescribe_vancomycin(dose: Knowledge[f64, ε >= 0.82]) with IO {
    println("Vancomycin prescribed")
}

fn main() with IO {
    let risky_dose: Knowledge[f64, ε=0.40] = Knowledge { value: 500.0, epsilon: 0.40 }

    prescribe_vancomycin(risky_dose)  // COMPILE ERROR: ε=0.40 < required 0.82
}
```

> The compiler rejects this *before any code runs* — a hard patient-safety guarantee. See: [tests/compile-fail/vancomycin_low_conf.sio](tests/compile-fail/vancomycin_low_conf.sio)

### Effects and linear types

```
fn sqrt_approx(x: f64) -> f64 with Mut, Div, Panic {
    if x <= 0.0 { return 0.0 }
    var g = x / 2.0
    var i = 0
    while i < 50 {
        g = (g + x / g) / 2.0
        i = i + 1
    }
    return g
}

linear struct FileHandle { fd: i32 }   // must be consumed exactly once
```

> More examples: [examples/epistemic_bmi.sio](examples/epistemic_bmi.sio), [docs/guide/SOUNIO_QUICK_START.md](docs/guide/SOUNIO_QUICK_START.md)

---

## Honest Status

This is an active research repository. Here's what actually works and what doesn't.

### What WORKS (production-tested)

| Component | Status | Evidence |
|---|---|---|
| **Epistemic core** | `Knowledge[T]` + GUM propagation + provenance | 52 files, tested, dissertation-grade |
| **Self-hosted compiler** | Lexer, parser, checker, codegen — compiles itself | Fixed-point verified (stage2 == stage3) |
| **Algebra** | Clifford Cl(p,q), Cayley-Dickson CD(k), Jordan J₃(O), octonions | Verified the 168 theorem |
| **Ontology** | OWL2 model + reasoner + query engine | 40 tests passing |
| **Native codegen** | Linux ELF plus current Mach-O output lanes from the self-hosted lean driver | Linux fixed-point verified; checked macOS artifact lane present |
| **Core stdlib** | Stats, linalg, ODE solvers, signal processing, CSV, JSON | Gate: 81 pass / 0 fail / 5 skip |
| **Optimizer** | 1000+ e-graph rewrite rules, GVN, LICM, load sinking | 1003 tests, all FAIL=0 |

### What's SCAFFOLDING (looks big, mostly empty)

| Component | Reality |
|---|---|
| **Theorem prover** | 9,600 lines — but NO inference logic, just arena + data structures |
| **~70% of epistemic modules** | Function signatures with minimal bodies |
| **Neural networks** (quaternion/octonion) | Compilation errors, won't run |
| **Genomics** | 11 files are single-line stubs (disabled on parser limitations) |
| **Async runtime** | 12 files, mostly <10 lines each |
| **Geometry engine** | 100% disabled |

### What's MISSING entirely

| Gap | Detail |
|---|---|
| **`Knowledge<T>` is NOT generic** | Hardcoded as `Epistemic` struct (f64 only). Struct-level generics [in progress](docs/compiler/KNOWN_LIMITATIONS.md). |
| **Epistemic ODE solver** | Only does exponential decay, not general RHS (needed for PBPK) |
| **Ontology federation** | Has 8 hardcoded CURIEs, NOT 15M terms — federation is a stub |
| **GPU entry point** | `gpu/lib.sio` is empty. PTX codegen exists but no end-to-end path. |
| **Closure literals** | `\|x\| x + 1` not supported. Named fn refs work (`let f = square`). |
| **Windows** | No checked-in Windows compiler artifact in this checkout. |
| **AArch64 native-v2 parity** | Newer `aarch64` native-v2 lowering still has unsupported opcodes; checked macOS support currently uses the self-hosted Mach-O artifact lane. |

### Stdlib by the numbers

| Category | Files | Percentage |
|---|---|---|
| **Complete** (working, tested) | 402 | 57% |
| **Partial** (some functions work) | 175 | 25% |
| **Skeleton** (types only, no logic) | 95 | 13% |
| **Stub** (1-line placeholder) | 38 | 5% |
| **Total** | 710 | |

---

## The 168 Theorem

While developing Sounio's octonion multiplication backend, we discovered and proved a combinatorial fact that appears not to have been explicitly stated in the literature:

> *The number of ordered triples (i, j, k) in {1,...,7}^3 for which the octonion basis associator [e_i, e_j, e_k] is nonzero is exactly 168 = |PSL(2,7)|.*

The decomposition is 343 = 133 (repeated indices) + 42 (Fano-line triples) + **168** (non-collinear triples). We also report that sedenion nonzero associator counts are multiples of 168, and that the primitive zero-divisor pair count 336 = 2 x 168.

The result was verified computationally in Sounio and independently reproduced in Python/NumPy.

**Paper:** "The 168 Theorem: PSL(2,7) Governs Non-Associativity and Zero-Divisor Structure in the Cayley-Dickson Tower" — Agourakis & Gerenutti (2026). Submitted to *Advances in Applied Clifford Algebras*.

---

## Get started

This checkout ships checked self-hosted compiler artifacts for Linux `x86_64`, macOS `arm64`, and macOS `x86_64` behind the host-aware `bin/souc` launcher. No Rust build step is required for the default workflow.

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio

export SOUC="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

$SOUC --version                              # souc 1.0.0-beta.5
$SOUC info                                   # selected host artifact + wrapper contract
$SOUC check examples/hello.sio               # type-check via checked self-hosted lane
$SOUC compile self-hosted/compiler/lean_single.sio -o /tmp/souc-next
$SOUC run self-hosted/compiler/native_print_f64_smoke.sio
$SOUC compile examples/hello.sio -o /tmp/hello-macos --target aarch64-macos
```

For detailed setup: [INSTALL.md](INSTALL.md) · [docs/guide/MINIMUM_VIABLE_SOUNIO.md](docs/guide/MINIMUM_VIABLE_SOUNIO.md)

---

## Architecture

**Pipeline:** Source → Lexer → Parser → AST → Check → HIR → SIR → HLIR (SSA) → Codegen

| Directory | Purpose |
|---|---|
| `self-hosted/lexer/`, `parser/` | Frontend (tokenizer, recursive descent) |
| `self-hosted/check/`, `types/` | Bidirectional type inference + algebraic effects |
| `self-hosted/ir/` | IR lowering, optimization, e-graph equality saturation |
| `self-hosted/native/` | Native ELF and Mach-O emission in the current self-hosted lane |
| `self-hosted/compiler/` | Codegen drivers (lean, IR) |
| `stdlib/epistemic/` | `Knowledge[T]`, uncertainty (GUM), provenance |
| `stdlib/units/` | Dimensional analysis |
| `bootstrap/` | stage0 (C) → boot2g → self-hosted chain |
| `formal/` | Lean 4 proofs (epistemic type invariants) |
| `tests/` | `run-pass/`, `compile-fail/`, `ui/`, `stdlib/` |

---

## Design Principles

1. **Uncertainty is not optional** — Every scientific value has uncertainty. Ignoring it is a bug, not a simplification.
2. **Provenance matters** — Data without origin is data without trust.
3. **Propagation is automatic** — Manual uncertainty calculation is error-prone. The compiler handles it (GUM/ISO 17025).
4. **Confidence gates execution** — Low-confidence code paths require explicit acknowledgment.
5. **One type definition, compiler guarantees everything** — Define your epistemic constraints once; the compiler enforces them across all operations.

See [docs/MANIFESTO.md](docs/MANIFESTO.md) for the full philosophy.

---

## Known Limitations

**Platform.** This checkout ships checked self-hosted compiler artifacts for Linux `x86_64`, macOS `arm64`, and macOS `x86_64` behind the `bin/souc` launcher. Linux `x86_64` and macOS `arm64` are the active first-class host lanes. The newer native-v2 `aarch64` backend is still preview-grade, so Apple Silicon support in this checkout is via the self-hosted Mach-O artifact lane rather than the new preview emitter.

**Native startup cost.** Native execution still requires producing a host binary before launch, so there is a small startup cost compared with an in-process executor.

**Launcher contract.** `bin/souc` now provides compatibility commands for `check`, `run`, `compile`, and `build`, but broader omega workflows and JIT-oriented tooling still live outside the checked self-hosted launcher lane.

**No struct generics (yet).** `Knowledge<T>` is currently monomorphic (f64 only). Struct-level generics are the highest-priority language feature. Function-level generics work.

**No closure literals.** Named function references work (`let f = square`), but `|x| x + 1` lambda syntax is not supported.

**No REPL yet.** The checked self-hosted launcher does not support `repl`.

**Debug flags.** `--show-ast` and `--show-types` are supported as pass-through flags on the checked self-hosted launcher for `check`, `run`, `compile`, and `build`.

**FFI.** `extern "C"` remains limited in scope, but the old JIT-only integer FFI failure mode is gone on the native path.

**GPU.** PTX codegen exists in `self-hosted/gpu/` but there is no end-to-end compilation path from the CLI. SPIR-V/Metal/WGSL files exist as stubs.

Full list: [docs/compiler/KNOWN_LIMITATIONS.md](docs/compiler/KNOWN_LIMITATIONS.md)

---

## Citation

If you use Sounio in academic work:

```bibtex
@software{sounio2026,
  title     = {Sounio: A Systems Programming Language for Epistemic Computing},
  author    = {Agourakis, Demetrios Chiuratto and Gerenutti, Marli},
  year      = {2026},
  version   = {1.0.0-beta.6},
  doi       = {10.5281/zenodo.18726647},
  url       = {https://github.com/sounio-lang/sounio},
  note      = {Self-hosted compiler with epistemic types and Lean 4 verification}
}
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).

---

<p align="center"><em>At the horizon of certainty, where ancient columns meet the endless sea.</em></p>
<p align="center">SOUNIO</p>

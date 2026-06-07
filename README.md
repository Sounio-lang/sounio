<!-- docs:meta
topic_id: repo.frontdoor.readme
authority: repo_only
audience: users
last_validated: 2026-06-07
validated_by: claude-release-apparatus (mc = Madares v0.80.0, tip claude/release-apparatus)
source_of_truth: docs/governance/topic-registry.v1.json#repo.frontdoor.readme
-->

<p align="center">
  <img src="docs/assets/sounio-logo.svg" alt="Sounio" width="200"/>
</p>

<h1 align="center">SOUNIO</h1>
<h3 align="center"><em>A self-hosted systems + scientific programming language for epistemic computing, uncertainty propagation, and algebraic effects</em></h3>

<p align="center">
  <a href="https://www.souniolang.org"><img src="https://img.shields.io/badge/website-souniolang.org-blue.svg" alt="Sounio Website"/></a>
  <a href="https://www.souniolang.org/playground"><img src="https://img.shields.io/badge/playground-wasm-purple.svg" alt="Playground"/></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/compiler-Madares%20v0.80.0-orange.svg" alt="Self-hosted compiler: Madares v0.80.0"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-gold.svg" alt="Apache-2.0 License"/></a>
  <a href="#honest-status"><img src="https://img.shields.io/badge/scale-4.2k%20.sio%20files-informational.svg" alt="~4.2k tracked .sio files; see SCALE.md"/></a>
</p>

<p align="center">
  <a href="https://www.souniolang.org">Website</a> ·
  <a href="https://www.souniolang.org/playground">Playground</a> ·
  <a href="https://www.souniolang.org/docs/">Documentation</a> ·
  <a href="docs/MANIFESTO.md">Manifesto</a> ·
  <a href="#quick-taste">Examples</a> ·
  <a href="#honest-status">Status</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

**Sounio** is a systems programming language for epistemic computing — its type system tracks not just what your data *is*, but how much you should *trust* it. Uncertainty propagation, provenance tracking, and confidence-gated execution are built into the type system, not bolted on as libraries.

**Keywords:** systems programming language, scientific computing language, epistemic types, uncertainty propagation, algebraic effects, self-hosted compiler, formal verification, non-associative algebra, octonions, e-graphs.

### Technical Pillars & Core Keywords

| Pillar | Description | Key Search Terms |
| :--- | :--- | :--- |
| **Epistemic Computing** | Built-in confidence-gated execution tracking and provenance tracking. | `epistemic-computing`, `Knowledge[T]`, `confidence-threshold` |
| **Uncertainty Propagation** | GUM-compliant (Guide to the Expression of Uncertainty in Measurement) error propagation. | `uncertainty-propagation`, `GUM-compliance`, `error-propagation` |
| **Systems & Science** | Native x86_64 ELF compilation, self-hosted compiler loop, PTX/GPU acceleration. | `systems-programming`, `scientific-computing`, `ptx-codegen` |
| **Algebraic Effects** | Explicit side-effects declarations (`IO`, `Mut`, `Div`, `Panic`, `Alloc`). | `algebraic-effects`, `effect-system`, `effects-with` |
| **Mathematical Rigor** | Non-associative octonion basis associators, formalized Lean 4 proofs of invariants. | `non-associative-algebra`, `octonions`, `lean4-proofs` |
| **Dimensional Analysis** | Compile-time unit checking (`VAR_UNIT_DIM`) to prevent physical dimension errors. | `dimensional-analysis`, `unit-types`, `compile-time-units` |

The compiler is **self-hosted in Sounio**, bootstrapped from a [2000-line C compiler](bootstrap/stage0.c) through a multi-stage chain. The shipped bootstrap binary (`bin/souc`, a static `mini_native` ELF) builds the modular self-hosted compiler `self-hosted/compiler/main.sio` (which self-identifies as **Madares v0.80.0**). The legacy `lean_single.sio` lane reaches a bit-identical bootstrap fixed point; the modular `main.sio` compiler does **not** yet self-compile to a native ELF (see [Current limitations](#current-limitations)). Sounio was used to computationally verify a new result in algebra — that the count of nonzero octonion basis associators equals |PSL(2,7)| = 168 — now [submitted for publication](#the-168-theorem).

This is an active **research project**, not a production release. Read the [honest status](#honest-status) before using it for anything serious.

### Cross-Repo Example: Cognitive O-SSM on SWOW-EN

The canonical Sounio checkout now includes a bounded cross-repo example under:

- `examples/cognitive_ossm/`

This lane is paired with the repository:

- `github.com/agourakis82/hyperbolic-semantic-networks`

Workflow split:

- Sounio provides the executable parity path and canonical `.sio` implementation scaffolding.
- The hyperbolic repo exports the compact SWOW bundle in `data/cpc2026/sounio_input/`.
- The hyperbolic repo's Python mirror currently generates the full paper-scale O-SSM artifacts.

From the Sounio repo root:

```bash
./artifacts/omega/souc-bin/souc-linux-x86_64-gpu run examples/cognitive_ossm/cognitive_ossm.sio
./artifacts/omega/souc-bin/souc-linux-x86_64-gpu run examples/cognitive_ossm/run_regimes.sio -- --max-trajectories 8 --max-steps 64
./artifacts/omega/souc-bin/souc-linux-x86_64-gpu run examples/cognitive_ossm/export_results.sio
```

---

## For LLMs and Code Tools

- Session bootstrap:
  1. Run `./sounio-whereami --quick`
  2. Read [ONBOARDING.md](ONBOARDING.md)
  3. Read [CLAUDE_HANDOFF.md](CLAUDE_HANDOFF.md)
  4. Read [CLAUDE.md](CLAUDE.md)
  5. Read [AGENTS.md](AGENTS.md)
  6. Verify the current branch before editing
  7. Treat `/workspace/sounio` as the active remote-first workspace path
  8. Do not propose destructive reset/clean/rebase flows to "simplify" recovery state
- Prompt surface: [llms.txt](llms.txt)
- **Repository scale (read before estimating size):** [SCALE.md](SCALE.md) · [docs/audit/README.md](docs/audit/README.md)
- Regenerate numbers: `bash scripts/dev/measure_repo_scale.sh`
- Repository guide: [CLAUDE.md](CLAUDE.md)
- Syntax and workflow guide: [docs/guide/LLM_PROGRAMMING_GUIDE.md](docs/guide/LLM_PROGRAMMING_GUIDE.md)
- Live Hugging Face dataset: <https://huggingface.co/datasets/chiuratto-AIgourakis/sounio-code-examples>
- Training dataset export: [datasets/sounio-code-examples/README.md](datasets/sounio-code-examples/README.md)
- Dataset builder: [scripts/dev/export_hf_dataset.py](scripts/dev/export_hf_dataset.py)

This repo now ships a root `llms.txt` for model-aware tools and a reproducible Hugging Face-style dataset export built from the Sounio test suite.
The current published dataset lives in the maintainer namespace as a public mirror until the `sounio-lang` Hugging Face org namespace is ready.

---

## What makes Sounio different

**Epistemic types as a design goal.** Every scientific measurement has uncertainty. Most languages ignore this. Sounio's type system is being built around `Knowledge[T]` with confidence, provenance tracking, and GUM-compliant uncertainty propagation. The epistemic surface is **prototype-grade** on the current self-hosted compiler: the `Knowledge(...)` constructor and `ε >= 0.82` confidence-gate syntax do not yet typecheck/enforce on the modular compiler (single-module `--check` performs resolve-skip, so these forms parse but are not actually validated). See [Current limitations](#current-limitations) before relying on them.

**Self-hosted compiler.** The compiler bootstrapped from C through a multi-stage chain (`stage0.c` → `boot2g.sio` → self-hosted). The shipped `bin/souc` is the bootstrap binary (a static `mini_native` ELF) and uses the raw interface `souc <source.sio> <output> [flags]` — it is **not** a launcher with `check`/`compile`/`build`/`run` subcommands. The modular compiler it builds (`Madares v0.80.0`) compiles single-file `.sio` programs to a native x86-64 ELF via `--native-v2-compile`, with `main()`'s return value as the process exit code.

**Not a Rust/Julia dialect.** Own syntax (`&!` not `&mut`, `var` not `let mut`), own semantics (algebraic effects, linear types, dimensional analysis), own philosophy (epistemic computing for science).

---

## Quick taste

> The epistemic examples below show the **language design** (spec-level syntax). The
> `Knowledge(...)` constructor and `ε >= 0.82` confidence gates are **prototype** and do
> not yet typecheck or enforce on the current self-hosted compiler — see
> [Current limitations](#current-limitations). The effects/linear-types example further
> down is verified-working.

### Uncertainty propagation with provenance (design preview)

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

### Compile-time confidence gate (design preview — not yet enforced)

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

> This is the intended semantics — confidence-gated rejection *before any code runs*. It is a **design goal, not yet enforced** by the current self-hosted compiler. See: [tests/compile-fail/vancomycin_low_conf.sio](tests/compile-fail/vancomycin_low_conf.sio)

### Effects and linear types (verified-working)

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

This is an active research repository. Public claims are registry-backed; see [`docs/serious-language/public-claim-registry.v1.tsv`](docs/serious-language/public-claim-registry.v1.tsv) (authoritative for every feature's maturity tier).

**PL adoption audit (2026-05-27):** [`docs/audit/PL_ADOPTION_AUDIT_2026-05-27.md`](docs/audit/PL_ADOPTION_AUDIT_2026-05-27.md) — bone-honest stocktake of what a stranger cloning this repo will find, with live probes. The two biggest adopter-unlock gaps are (G1) closing the multi-module bundle compile and (G2) the CLI exit-code contract (G2 fixed 2026-05-27 in this commit).

**Registry rows you should read before drawing conclusions:**
`stdlib.surface = prototype` · `tooling.editor = prototype` (formatter, REPL, LSP) · `tooling.package = prototype` (no public registry) · `closures.lambdas = stale_conflicting` (spec §4.7.2 non-normative) · `generics.{structs,functions,traits} = prototype` · `binary.source = prototype` (`lean_single.sio` is the source of the checked binary, not the modular tree) · `platform.windows = prototype`.

**Scale (measured, 2026-05):** **4,233** tracked `.sio` files, **~1.84M** lines (`bash scripts/dev/measure_repo_scale.sh`). The self-hosted compiler alone is **~542k** lines — not a small experiment. Full audit: [docs/audit/README.md](docs/audit/README.md) · [SCALE.md](SCALE.md).

### What WORKS (evidence-backed, re-verified 2026-06-07 against `Madares v0.80.0`)

| Component | Status | Evidence |
|---|---|---|
| **Single-file source → native ELF** | mc `--native-v2-compile`; `main()` → exit code | `tests/native_v2_capgate/run.sh` = **32/32** |
| **Linear types (use-once)** | double-consume is a compile error | `--check` rejects second use of a `linear struct` |
| **Multi-module link + run** | imports resolve, link, execute | `tests/native_v2_multimodule_gate/run.sh` = 9/9 (one tracked import-typecheck-bypass class, documented) |
| **Cross-compile** | macOS `arm64` Mach-O + host `x86_64` ELF | `bin/souc src.sio out --target aarch64-macos`; verified Mach-O/ELF output |
| **Backend soundness** | field-store/load behavior gated | `tests/native_v2_backend_soundness_gate/run.sh` = 40/40, **1 tracked field-hash residual** (disclosed below) |
| **Effects** | `IO`/`Mut`/`Div`/`Panic`/`Alloc` declarations checked | `--check` + native-v2 gates |
| **Ontology** | Generated bundles + validation harness | `run_ontology_validation.sh` + compile gates |
| **Language server** | LSP 3.17 subset | Release binary + protocol tests (prototype per registry) |

> The legacy `lean_single.sio` lane reaches a **bit-identical bootstrap fixed point**; the
> modular `main.sio` compiler does not yet self-compile to ELF. The epistemic core
> (`Knowledge[T]` + GUM gates) is **prototype**, not in this table — see
> [Current limitations](#current-limitations).

### What's SCAFFOLDING or PARTIAL

| Component | Reality |
|---|---|
| **~46% of stdlib modules** | Classified scaffold in [audit A.2](docs/audit/README.md) — code without executable proof |
| **32 stdlib smoke tests** | Print `FOO_OK` only; do not exercise module logic |
| **129 CI gate scripts** | Most are **not** on `make check` / GitHub CI (audit A.4) |
| **GPU CLI path** | PTX/kaxi code exists; end-to-end CLI path incomplete |
| **Theorem prover / async / geometry** | Large or stub surfaces — see module audit JSON |

### Stdlib module audit (A.2, not file-count folklore)

| Tier | Modules | Meaning |
|---|---:|---|
| **works** | 66 | Tests, gates, or mass with executable evidence |
| **scaffold** | 59 | Code present; no direct executable proof in tree |
| **doc-only roots** | 3 | Non-module files at `stdlib/` root |

Do **not** cite **814/910 (89%)** as "stdlib completeness" — that is harness inventory, mixes real tests with smoke placeholders, and differs from the reliability gate inventory. See audit artifacts under `artifacts/audit/`.

---

## The 168 Theorem

While developing Sounio's octonion multiplication backend, we discovered and proved a combinatorial fact that appears not to have been explicitly stated in the literature:

> *The number of ordered triples (i, j, k) in {1,...,7}^3 for which the octonion basis associator [e_i, e_j, e_k] is nonzero is exactly 168 = |PSL(2,7)|.*

The decomposition is 343 = 133 (repeated indices) + 42 (Fano-line triples) + **168** (non-collinear triples). We also report that sedenion nonzero associator counts are multiples of 168, and that the primitive zero-divisor pair count 336 = 2 x 168.

The result was verified computationally in Sounio and independently reproduced in Python/NumPy.

**Paper:** "The 168 Theorem: PSL(2,7) Governs Non-Associativity and Zero-Divisor Structure in the Cayley-Dickson Tower" — Agourakis & Gerenutti (2026). Submitted to *Advances in Applied Clifford Algebras*.

---

## Get started

This checkout ships the checked-in bootstrap compiler `bin/souc` (a static `mini_native`
ELF). No Rust build step is required for the default workflow. `bin/souc` uses the **raw
compiler interface** — `souc <source.sio> <output> [flags]` — it is not a launcher with
`check`/`compile`/`build`/`run` subcommands.

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio

export SOUC="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

# bin/souc identifies as mini_native; usage is: souc <source.sio> <output> [flags]
$SOUC examples/hello.sio /tmp/hello.elf       # bootstrap: compile a host x86_64 ELF
chmod +x /tmp/hello.elf && /tmp/hello.elf     # main()'s return value is the exit code
$SOUC examples/hello.sio /tmp/hello-macos --target aarch64-macos   # cross to macOS arm64 Mach-O

# Build the modular self-hosted compiler (Madares v0.80.0), then drive it directly:
ulimit -s 1048576
$SOUC self-hosted/compiler/main.sio /tmp/mc.elf && chmod +x /tmp/mc.elf
/tmp/mc.elf --version                                          # Madares v0.80.0
/tmp/mc.elf --check examples/hello.sio                         # type-check
/tmp/mc.elf --native-v2-compile examples/hello.sio /tmp/h.elf  # single-file source → native ELF
```

For detailed setup: [INSTALL.md](INSTALL.md) · [docs/guide/MINIMUM_VIABLE_SOUNIO.md](docs/guide/MINIMUM_VIABLE_SOUNIO.md)

### Editor integration

The Sounio language server (`bin/sounio-lsp`) ships with the checkout and is the same binary published at
[`sounio-lsp-v0.3.0-r1`](https://github.com/Sounio-lang/sounio/releases/tag/sounio-lsp-v0.3.0-r1). Point any LSP-aware editor (VS Code, Neovim, Helix, Zed, etc.) at the binary with file-type `.sio`. Capabilities and the change log live in [`tools/lsp/CHANGELOG.md`](tools/lsp/CHANGELOG.md); Sprint-2 backlog in [`tools/lsp/SPRINT2_TODO.md`](tools/lsp/SPRINT2_TODO.md).

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

## Current limitations

Re-verified 2026-06-07 against the modular compiler (`Madares v0.80.0`). These are honest, probe-backed gaps — please do not read past them.

**Int literals need an explicit cast.** Integer literals do not yet coerce to `i32`: `fn main() -> i32 { 0 }` is **rejected** (`found i64`); write `{ 0 as i32 }`. Implicit literal coercion is in progress.

**Epistemic core is prototype.** The `Knowledge(15.0, ε=0.92, prov=...)` constructor and the `ε >= 0.82` confidence-gate syntax do **not** typecheck/enforce on the modular compiler. Single-module `--check` does resolve-skip, so these forms parse but are not validated (an undefined call passes `--check` identically). Confidence-gated rejection is a design goal, not a working guarantee.

**Closures / generics / nested control checks.** Closures and lambdas are **not** supported end-to-end: a lambda passed as a `fn` argument fails typecheck, and the backend does not reliably compile lambdas. Generics, nested `if let`, and `while let` currently produce false rejections. None of these should be treated as working.

**Self-compilation.** The modular `main.sio` compiler does **not** yet compile itself to a native ELF; only the legacy `lean_single.sio` lane reaches a bit-identical bootstrap fixed point. Multi-module also has a tracked import-typecheck-bypass class (ill-typed cross-module programs can slip through; see the multimodule gate).

**Field-hash residual (disclosed).** Struct field slots use a first-byte hash that can collide for fields sharing an initial byte; one such collision is a tracked residual in `tests/native_v2_backend_soundness_gate` (40/40 with exactly 1 known residual). Proper fix = declaration-order layout via struct-type tracking.

**Example coverage.** Roughly **252 of ~860** example programs are currently green; the rest hit unimplemented features or the checker (five example files currently SIGSEGV the checker).

**Platform.** Linux `x86_64` is the first-class host lane; macOS `arm64`/`x86_64` are supported via cross-compiled Mach-O output. The native-v2 `aarch64` backend is still preview-grade.

**Native startup cost.** Native execution requires producing a host binary before launch, so there is a small startup cost compared with an in-process executor.

**No subcommands / no REPL.** `bin/souc` is the raw `mini_native` compiler interface (`souc <source.sio> <output> [flags]`), not a `check`/`run`/`compile`/`build` launcher, and there is no `repl`. `--show-ast` and `--show-types` are pass-through debug flags.

**Windows cross-compile.** A PE/COFF backend exists and `--target x86_64-windows` emits Windows binaries, but the platform is **prototype** per the public-claim registry — not production-grade. No pre-built `.exe` is shipped.

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
  version   = {Madares v0.80.0 (self-hosted compiler)},
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

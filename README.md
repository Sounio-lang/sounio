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
  <a href="https://www.souniolang.org"><img src="https://img.shields.io/badge/website-souniolang.org-blue.svg" alt="Sounio Website"/></a>
  <a href="https://www.souniolang.org/playground"><img src="https://img.shields.io/badge/playground-wasm-purple.svg" alt="Playground"/></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-2.1.0-blue.svg" alt="Version 2.1.0"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-gold.svg" alt="Apache-2.0 License"/></a>
  <a href="#honest-status"><img src="https://img.shields.io/badge/scale-6.1k%20.sio%20files-informational.svg" alt="~6.1k tracked .sio files; see SCALE.md"/></a>
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

The compiler is **self-hosted**: Sounio compiles itself, bootstrapped from a [2000-line C compiler](bootstrap/stage0.c) through a multi-stage chain to a true fixed-point where stage N and stage N+1 produce bit-identical binaries. It was used to computationally verify a new result in algebra — that the count of nonzero octonion basis associators equals |PSL(2,7)| = 168 — now [submitted for publication](#the-168-theorem).

This is an active **research project**, not a production release. Read the [honest status](#honest-status) before using it for anything serious.

### Cross-Repo Example: Cognitive O-SSM on SWOW-EN

The canonical Sounio checkout now includes a bounded cross-repo example under:

- `examples/cognitive_ossm/`

This lane is paired with the repository:

- `github.com/agourakis82/hyperbolic-semantic-networks`

Workflow split and current boundary:

- Sounio provides checkable `.sio` implementation scaffolding and separately executable epistemic receipts.
- The hyperbolic repo exports the compact SWOW bundle in `data/cpc2026/sounio_input/`.
- The hyperbolic repo's Python mirror currently generates the full paper-scale O-SSM artifacts.
- The repaired native reference passes Madaros `check`, but current native-v2 compilation is blocked. Historical native n=100/n=1000 JSON files are excluded from parity claims.

From the Sounio repo root:

```bash
./bin/souc check examples/cognitive_ossm/run_ossm_native_reference.sio
CPC2026_SCIENTIFIC_REPO=/workspace/hyperbolic-semantic-networks \
  bash scripts/ci/cpc2026_yale_evidence_gate.sh
uv run --with numpy python scripts/research/cpc2026_ossm_subset_audit.py
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

**Epistemic types as first-class citizens.** Every scientific measurement has uncertainty. Most languages ignore this. Sounio's type system includes `Knowledge[T]` with built-in confidence, provenance tracking, and automatic GUM-compliant uncertainty propagation. The compiler can enforce confidence thresholds at compile time — a function requiring `ε >= 0.82` rejects under-confident data before any code runs. No equivalent system exists in any production language.

**Self-hosted compiler.** The compiler bootstrapped from C through a multi-stage chain (`stage0.c` → `boot2g.sio` → self-hosted) to a true fixed-point. The default workflow is now native-only: `bin/souc` compiles `.sio` sources to temporary or named ELFs via the Madaros self-hosted engine and executes those binaries directly.

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

This is an active research repository. Public claims are registry-backed; see [`docs/serious-language/public-claim-registry.v1.tsv`](docs/serious-language/public-claim-registry.v1.tsv) (authoritative for every feature's maturity tier).

**PL adoption audit (2026-05-27):** [`docs/audit/PL_ADOPTION_AUDIT_2026-05-27.md`](docs/audit/PL_ADOPTION_AUDIT_2026-05-27.md) — bone-honest stocktake of what a stranger cloning this repo will find, with live probes. The two biggest adopter-unlock gaps are (G1) closing the multi-module bundle compile and (G2) the CLI exit-code contract (G2 fixed 2026-05-27 in this commit).

**Registry rows you should read before drawing conclusions:**
`stdlib.surface = validated_research` (bounded support contract only; not broad all-file callability) · `tooling.editor = validated_research` (checked formatter, REPL, preview LSP, and editor wiring; not mature IDE support) · `tooling.package = validated_research` (local packages only; no public registry) · `closures.lambdas = stale_conflicting` (spec §4.7.2 non-normative) · `generics.{structs,functions,traits} = prototype` · `binary.source = validated_research` (checked Madaros prebuilt is built from the modular tree; `lean_single.sio` remains the bootstrap seed) · `platform.windows = prototype`.

**Scale (measured 2026-07-11):** **6,130** tracked `.sio` files, **~2.21M** lines (`bash scripts/dev/measure_repo_scale.sh`). The self-hosted compiler alone is **~555k** lines — not a small experiment. Full audit: [docs/audit/README.md](docs/audit/README.md) · [SCALE.md](SCALE.md).

### What WORKS (evidence-backed lanes)

| Component | Status | Evidence |
|---|---|---|
| **Epistemic core** | `Knowledge[T]` + GUM + provenance | Named package / conformance gates |
| **Self-hosted compiler** | Lexer → codegen; fixed-point bootstrap | `lean_single` fixed-point + native-v2 spine gates |
| **Ontology** | Generated bundles + validation harness | `run_ontology_validation.sh` + compile gates |
| **Native codegen** | Linux ELF; Mach-O artifact lane | Self-host + native-v2 gates |
| **Core stdlib slices** | Stats, linalg, ODE, etc. | `stdlib_science_pipeline_gate`, reliability inventory |
| **Language server** | LSP 3.17 subset | Release binary + protocol tests (prototype per registry) |

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

This checkout ships checked self-hosted compiler artifacts for Linux `x86_64` behind the host-aware `bin/souc` launcher, which is the official compiler entrypoint and routes to Madaros by default. The checked `bin/souc-*` binaries are Linux ELF artifacts; macOS is a cross-compile target rather than a host-native binary lane. No Rust build step is required for the default workflow.

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio

export SOUC="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

$SOUC --version                              # souc 2.1.0
$SOUC info                                   # selected host artifact + wrapper contract
$SOUC check examples/hello.sio               # type-check via checked self-hosted lane
$SOUC init hello_pkg && cd hello_pkg         # create a minimal sounio.toml project
$SOUC check && $SOUC run && $SOUC build      # project entrypoint -> ELF
$SOUC run examples/native/hello.sio          # compile to a temp ELF and execute it
$SOUC compile examples/hello.sio -o /tmp/souc-next
$SOUC compile examples/hello.sio -o /tmp/hello-macos --target aarch64-macos
```

If you need the legacy bootstrap path explicitly:

```bash
SOUNIO_SOUC_ENGINE=lean_single \
  $SOUC compile self-hosted/compiler/lean_single.sio -o /tmp/souc-next
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

## Known Limitations

**Platform.** This checkout ships checked self-hosted compiler artifacts for Linux `x86_64`, macOS `arm64`, and macOS `x86_64` behind the `bin/souc` launcher. Linux `x86_64` and macOS `arm64` are the active first-class host lanes. The newer native-v2 `aarch64` backend is still preview-grade, so Apple Silicon support in this checkout is via the self-hosted Mach-O artifact lane rather than the new preview emitter.

**Native startup cost.** Native execution still requires producing a host binary before launch, so there is a small startup cost compared with an in-process executor.

**Launcher contract.** `bin/souc` now provides compatibility commands for `check`, `run`, `compile`, `build`, and `init`. When invoked inside a directory with `sounio.toml`, `check`, `run`, and `build` resolve the project entrypoint from `[[bin]].path`, `[project].entry`, or `src/main.sio`. Broader omega workflows and JIT-oriented tooling still live outside the checked self-hosted launcher lane.

**Windows cross-compile.** The PE/COFF backend (3,508 lines) is production-grade. Use `--target x86_64-windows` to emit Windows binaries. No pre-built .exe is shipped in this checkout.

**REPL.** The checked self-hosted launcher supports `souc repl` for a file-backed interactive loop.

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
  version   = {2.1.0},
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

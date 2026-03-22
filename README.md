<p align="center">
  <img src="docs/assets/sounio-logo.svg" alt="Sounio" width="200"/>
</p>

<h1 align="center">SOUNIO</h1>
<h3 align="center"><em>A self-hosted language where types know what they don't know</em></h3>

<p align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-1.0.0--beta.6-orange.svg" alt="Version 1.0.0-beta.6"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-gold.svg" alt="Apache-2.0 License"/></a>
  <a href="#standard-library"><img src="https://img.shields.io/badge/stdlib-219K%2B%20lines-blue.svg" alt="stdlib"/></a>
  <a href="#formal-verification"><img src="https://img.shields.io/badge/verification-Lean%204-green.svg" alt="Lean 4"/></a>
  <a href="#gpu-codegen"><img src="https://img.shields.io/badge/GPU-PTX%20codegen-76b900.svg" alt="GPU"/></a>
</p>

<p align="center">
  <a href="https://souniolang.org">Documentation</a> ·
  <a href="docs/MANIFESTO.md">Manifesto</a> ·
  <a href="#quick-taste">Examples</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

**Sounio** is a systems programming language for epistemic computing. Its compiler doesn't just check what your data *is* — it tracks how much you should *trust* it. Uncertainty propagation, provenance tracking, and confidence-gated execution are built into the type system, not bolted on as libraries.

The compiler is **fully self-hosted**: Sounio compiles itself, bootstrapped from a [2000-line C compiler](bootstrap/stage0.c) to a complete toolchain. It was used to computationally verify a new result in algebra — that the count of nonzero octonion basis associators equals |PSL(2,7)| = 168 — now [submitted for publication](#the-168-theorem).

---

## What makes Sounio different

**Epistemic types as first-class citizens.** Every scientific measurement has uncertainty. Most languages ignore this. Sounio's type system includes `Knowledge[T]` with built-in confidence (ε), provenance tracking, and automatic GUM-compliant uncertainty propagation. The compiler can enforce confidence thresholds at compile time — `Knowledge[f64, ε >= 0.82]` rejects under-confident data before any code runs. No equivalent system exists in any production language.

**Self-hosted with a complete toolchain.** The compiler bootstrapped from C through a multi-stage chain (`stage0.c` → `boot2g.sio` → self-hosted) to a true fixed-point where Sounio compiles itself. The toolchain includes the compiler (`souc`), a language server (LSP), a source formatter, and a package manager.

**GPU codegen.** Sounio emits PTX for NVIDIA CUDA, with SPIR-V, Metal, and WGSL backends in active development. All backends are written in Sounio itself.

**Formal verification.** A Lean 4 formalization in [`formal/`](formal/) proves properties of the epistemic type system — uncertainty non-negativity, confidence bounds, propagation correctness — with zero `sorry` statements across 27+ theorems.

**96+ stdlib modules.** 219,000+ lines of Sounio across 678 files covering scientific computing: epistemic types, PK/PD modeling, fMRI pipelines, causal inference, Bayesian statistics, signal processing, graph metrics, and more.

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

## The 168 Theorem

While developing Sounio's octonion multiplication backend (for the AVX-512 Fano-plane kernel), we discovered and proved a combinatorial fact that appears not to have been explicitly stated in the literature:

> *The number of ordered triples (i, j, k) ∈ {1,…,7}³ for which the octonion basis associator [eᵢ, eⱼ, eₖ] is nonzero is exactly 168 = |PSL(2,7)|.*

The decomposition is 343 = 133 (repeated indices) + 42 (Fano-line triples) + **168** (non-collinear triples). We also report that sedenion nonzero associator counts are multiples of 168, and that the primitive zero-divisor pair count 336 = 2 × 168.

The result was verified computationally in Sounio and independently reproduced in Python/NumPy.

**Paper:** "The 168 Theorem: PSL(2,7) Governs Non-Associativity and Zero-Divisor Structure in the Cayley–Dickson Tower" — Agourakis & Gerenutti (2026). Submitted to *Advances in Applied Clifford Algebras*. <!-- TODO: Add DOI link when available -->

---

## Get started

The repo ships pre-built Linux x86_64 compiler binaries. No build step required.

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio

export SOUC="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-jit"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

$SOUC --version                              # souc 1.0.0-beta.4
$SOUC check examples/hello.sio              # type-check
$SOUC run examples/epistemic_bmi.sio        # run with JIT
$SOUC repl                                   # interactive REPL
```

Native compilation via the self-hosted lean driver:

```bash
$SOUC run self-hosted/compiler/render_native_compile_driver_lean.sio -- input.sio output.elf
```

GPU compilation (requires the GPU profile binary):

```bash
export SOUC_GPU="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
$SOUC_GPU build examples/kernel_matmul.sio --backend gpu -o /tmp/kernel_matmul.ptx
```

For detailed setup: [INSTALL.md](INSTALL.md) · [docs/guide/MINIMUM_VIABLE_SOUNIO.md](docs/guide/MINIMUM_VIABLE_SOUNIO.md)

---

## Standard Library

96+ modules, 219K+ lines across 678 `.sio` files (including ~40 stub modules and 111 disabled files representing roadmap inventory). Gate status: 81 pass / 0 fail / 5 skip ([artifacts/stdlib/stdlib_reliability_status.v1.json](artifacts/stdlib/stdlib_reliability_status.v1.json)).

| Module | What it does |
|---|---|
| `epistemic/` | Core `Knowledge[T]` types, GUM uncertainty propagation, provenance, autodiff |
| `units/` | Compile-time dimensional analysis (GUM/ISO 17025, BIPM SI) |
| `causal/` | Pearl's do-calculus, d-separation, PC algorithm, causal discovery |
| `fmri/` | NIfTI parsing, motion correction, atlas parcellation (AAL, Desikan-Killiany) |
| `medlang/` | PK/PD domain-specific language with PBPK modeling |
| `hypercomplex_graph/` | Octonion-labeled graphs, Ollivier-Ricci curvature, connectomics |
| `gpu/` | GPU kernel support (PTX, SPIR-V, Metal) |
| `bayes/` | Bayesian inference, MCMC, variational methods |
| `connectivity/` | Graph metrics, network neuroscience |
| `signal/` | Signal processing, spectral analysis |
| `optimize/` | Optimization: BFGS, Nelder-Mead, differential evolution |
| `linalg/` | Linear algebra with uncertainty tracking |
| `quantum/` | Quantum computing primitives, VQE |
| `cybernetic/` | Second-order cybernetics (distinction, eigenform, autopoiesis) |
| `data/` | DataFrames and tabular manipulation |
| `ode/` | ODE solvers with error propagation |

---

## GPU Codegen

Sounio compiles GPU kernels from `.sio` source via dedicated codegen drivers in `self-hosted/gpu/`.

```bash
# Compile a kernel to PTX
$SOUC_GPU build kernel.sio --backend gpu -o kernel.ptx
```

| Backend | Status | Implementation |
|---|---|---|
| PTX (NVIDIA CUDA) | Active | `self-hosted/gpu/ptx.sio`, `lower_to_ptx.sio`, `epistemic_ptx.sio` |
| SPIR-V (Vulkan) | In development | `self-hosted/gpu/spirv.sio`, `spirv_render.sio` |
| Metal (Apple) | In development | `self-hosted/gpu/metal.sio`, `metal_render.sio` |
| WGSL (WebGPU) | In development | `self-hosted/gpu/wgsl_render.sio` |

---

## Formal Verification

The [`formal/`](formal/) directory contains Lean 4 formalizations of Sounio's type system properties:

- **[Epistemic.lean](formal/Epistemic.lean)** — 27+ theorems proving uncertainty propagation correctness, confidence bounds, and subtyping relations. Zero `sorry` statements.
- **[LinearTypes.lean](formal/LinearTypes.lean)** — Formal linear type system with usage multiplicity proofs.
- **[OctonionGraph.lean](formal/OctonionGraph.lean)** — Path product norm invariance theorem for octonion-labeled graphs.

The epistemic type invariants (uncertainty non-negativity, `ε ∈ [0,1]`, GUM propagation rules) are expressed as Lean 4 theorems and verified mechanically.

---

## Current State

This is an active research repository, not a "everything is production-ready" release. The safest public summary comes from the committed gate artifacts:

- `souc check` works on canonical fixtures including `examples/hello.sio` and the vancomycin PK/PD tests
- Stdlib gate: **81 pass / 0 fail / 5 skip** across 86 test lanes
- Science pipeline: `fmri` and `darwin_pbpk` lanes pass
- GPU profile: PTX emission via `--backend gpu`
- Self-hosted compiler: `--check`, `--ir-dump`, `--ir-roundtrip`, `--native-compile`
- 4 soft runtime regression probes remain in `soft` mode
- The JIT binary reports `souc 1.0.0-beta.4`; source version is `1.0.0-beta.6`

For the conservative contract: [docs/guide/MINIMUM_VIABLE_SOUNIO.md](docs/guide/MINIMUM_VIABLE_SOUNIO.md)

---

## Architecture

**Pipeline:** Source → Lexer → Parser → AST → Check → HIR → SIR → HLIR (SSA) → Codegen

| Directory | Purpose |
|---|---|
| `self-hosted/lexer/`, `parser/` | Frontend (tokenizer, recursive descent) |
| `self-hosted/check/`, `types/` | Bidirectional type inference + algebraic effects |
| `self-hosted/ir/` | IR lowering, optimization, e-graph equality saturation |
| `self-hosted/native/` | x86-64 ELF emission |
| `self-hosted/gpu/` | PTX, SPIR-V, Metal, WGSL codegen |
| `self-hosted/compiler/` | Codegen drivers (lean, IR, GPU) |
| `stdlib/epistemic/` | `Knowledge[T]`, uncertainty (GUM), provenance |
| `stdlib/units/` | Dimensional analysis |
| `bootstrap/` | stage0 (C) → boot2g → boot1 → self-hosted chain |
| `formal/` | Lean 4 proofs |
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

**Platform.** Pre-built binaries are Linux x86_64 only. macOS Mach-O backend exists but is not regularly tested. Windows is not yet supported.

**JIT memory.** The Cranelift JIT compiling the self-hosted compiler itself grows to 14–35 GB RSS and is OOM-killed on most machines. `$SOUC run` works fine for normal programs; self-compilation uses the native bootstrap chain instead.

**Native backend.** `native-v2` is a preview lane: x86-64 emits scalar-core ELFs with GC metadata; AArch64 is compile-only. The stable CLI exposes `--backend=native`.

**WASM.** The emitter exists (`self-hosted/wasm/`) but is not yet wired into the normal CLI flow.

**No closure literals.** Named function references work (`let f = square`), but `|x| x + 1` lambda syntax is not supported. See [docs/compiler/KNOWN_LIMITATIONS.md](docs/compiler/KNOWN_LIMITATIONS.md).

**`&!` array mutation.** Bare `&![T; N]` mutable references don't propagate mutations in the interpreter. Workaround: wrap in a struct. Struct `&!` refs work correctly.

**FFI.** `extern "C"` is limited to math functions (`sqrt`, `sin`, `pow`, etc.). Integer FFI (`malloc`, `getpid`) silently terminates. Use fixed-size struct arrays instead of dynamic allocation.

**Optional dependencies.** LLVM 15+ for `--features llvm`, Z3 for SMT-backed refinement types, CUDA toolkit for GPU execution (codegen works without it).

Full list: [docs/compiler/KNOWN_LIMITATIONS.md](docs/compiler/KNOWN_LIMITATIONS.md)

---

## Roadmap

### Done
- [x] Self-hosted compiler (true fixed-point)
- [x] Epistemic type system (`Knowledge[T]`, confidence gating, provenance)
- [x] PTX GPU codegen
- [x] Lean 4 formal verification (27+ theorems, zero `sorry`)
- [x] MedLang PK/PD DSL
- [x] fMRI preprocessing pipeline
- [x] Language Server Protocol (LSP)
- [x] Package manager (manifest + resolver, no public registry yet)
- [x] Interactive REPL
- [x] Source formatter
- [x] 96+ stdlib modules

### Next
- [ ] SPIR-V / Metal / WGSL backend completion
- [ ] LLVM backend (alternative to Cranelift JIT)
- [ ] WASM CLI integration
- [ ] Distributed uncertainty propagation
- [ ] Package registry (`siopkg publish`)
- [ ] macOS / AArch64 regular testing

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
  note      = {Self-hosted compiler with epistemic types, GPU codegen, and Lean 4 verification}
}
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).

---

<p align="center"><em>At the horizon of certainty, where ancient columns meet the endless sea.</em></p>
<p align="center">SOUNIO</p>

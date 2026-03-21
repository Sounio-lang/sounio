<!-- docs:meta
topic_id: repo.paper.oopsla2027.outline
authority: repo_only
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.paper.oopsla2027.outline
-->

# Sounio OOPSLA 2027 Systems Paper Outline (R5 Draft)

## Positioning

- Venue: OOPSLA 2027 (PACMPL systems-heavy framing)
- Paper role: engineering and systems design paper, not primary type-theory paper
- Central narrative: Sounio operationalizes scientific semantics in a self-hosted systems compiler/runtime with explicit reproducibility artifacts

## One-Paragraph Pitch

Sounio is a self-hosted systems language and compiler stack for verifiable scientific computing. The key systems contribution is a Scientific IR (SIR) and gate-driven engineering workflow that keep uncertainty semantics, causal operations, and reproducibility constraints visible across parsing, typing, optimization, and backend lowering. The paper should argue that Sounio is not only expressive at the language level, but also operationally disciplined: bootstrap closure, artifactized gate status, and script-first evidence linking claims to machine-readable outputs.

## Draft Abstract (Systems-Facing)

Scientific software frequently combines numerically delicate models, uncertain measurements, and production constraints, yet mainstream systems toolchains lack first-class support for these concerns. We present Sounio, a self-hosted systems language and compiler pipeline designed for verifiable scientific workloads. Sounio centers a Scientific IR (SIR) with explicit scientific operators and preserves uncertainty/causal semantics through typed lowering and backend code generation. The implementation includes a three-stage self-host workflow, fail-closed policy gates, and artifact-first status reporting for reproducibility. We describe compiler architecture, bootstrap constraints, and backend strategy (native plus accelerator lanes), then evaluate on scientific benchmark lanes including pharmacokinetic and uncertainty workloads. The result is a systems approach where language semantics and operational evidence co-evolve: claims are tied to scripts and generated artifacts rather than prose-only assertions.

## Section Blueprint

### 1. Introduction

Objective:
- Frame the engineering gap: systems languages optimize control/performance but usually externalize scientific semantics and reproducibility contracts.

Claims to make:
- Sounio treats scientific correctness constraints as compiler/runtime concerns, not optional library policy.
- Evidence for the system exists as versioned scripts plus JSON artifacts.

Evidence pointers:
- `docs/compiler/ARCHITECTURE.md`
- `docs/compiler/SCIENTIFIC_FEATURES_ARCHITECTURE.md`
- `artifacts/omega/paper_repro_gate_status.v1.json`

Deliverables in section:
- Problem statement, contributions list, and evaluation preview.

### 2. Language Surface for Scientific Systems

Objective:
- Show the user-facing scientific primitives that motivate backend/compiler choices.

Subtopics:
- Epistemic values and uncertainty-aware arithmetic.
- Causal operations and intervention-centric APIs.
- Effects and units as systems guardrails.

Code anchor suggestions:
- `examples/real_world/05_pkpd_data_analysis.sio`
- `stdlib/causal/core.sio`

Figure/table candidates:
- Table: language constructs -> compiler/runtime handling path.

### 3. Compiler Architecture

Objective:
- Describe end-to-end pipeline and where scientific semantics are preserved.

Subtopics:
- Front-end stages: source, parser, AST/HIR checks.
- Middle-end: SIR/HLIR semantics-aware transforms.
- Backend boundary: native and accelerator emission strategy.

Evidence pointers:
- `docs/compiler/ARCHITECTURE.md`
- `docs/compiler/TYPE_SYSTEM_ARCHITECTURE.md`
- `docs/compiler/CODE_GENERATION_ARCHITECTURE.md`

Figure/table candidates:
- Figure: pipeline map with data structures and invariants.

### 4. Scientific IR (SIR) as Systems Core

Objective:
- Make SIR the unique systems contribution: explain why direct lowering is insufficient for scientific invariants.

Subtopics:
- SIR design goals and instruction families.
- Optimization strategy for uncertainty-heavy compute paths.
- Semantics preservation boundaries from typed source to executable lanes.

Evidence pointers:
- `docs/compiler/SIR_PASSES.md`
- `docs/compiler/SCIENTIFIC_FEATURES_ARCHITECTURE.md`

Figure/table candidates:
- Figure: representative SIR snippet with annotations.
- Table: pass goals, preconditions, and expected effect.

### 5. Self-Hosted Bootstrap and Policy Closure

Objective:
- Show operational rigor: self-host/closure is tested with fail-closed contracts, not ad hoc scripts.

Subtopics:
- Bootstrap workflow and closure expectations.
- No-Rust policy boundaries and explicit failure classes.
- Artifactized gate outcomes as engineering telemetry.

Evidence pointers:
- `docs/implementation/RUSTLESS_CUTOVER.md`
- `docs/implementation/RUSTLESS_COMPLETE.md`
- `artifacts/omega/strict_no_rust_closure_gate.v1.json`

Figure/table candidates:
- Table: bootstrap stages and expected pass/fail/not_run surfaces.

### 6. Backend Strategy and Runtime Integration

Objective:
- Explain production tradeoffs across native and accelerator backends.

Subtopics:
- Native targets and code emission contracts.
- GPU/runtime attest lanes and capability gating.
- Runtime bridge patterns for scientific kernels/ODE-like workloads.

Evidence pointers:
- `docs/compiler/CODE_GENERATION_ARCHITECTURE.md`
- `artifacts/omega/gpu_runtime_attest_gate.v1.json`
- `artifacts/omega/gpu_codegen_parity.v1.json`

Figure/table candidates:
- Figure: backend matrix and dispatch policy.

### 7. Evaluation

Objective:
- Report systems outcomes with explicit scope and reproducibility path.

Subsections:
- 7.1 Scientific/uncertainty microbenchmarks.
- 7.2 Domain lane examples (PK/PBPK and related workloads).
- 7.3 Build/runtime overhead and policy-gate status.
- 7.4 Reproducibility pipeline results.

Evidence pointers:
- `benchmarks/README.md`
- `benchmarks/results/NVIDIA_L4_BENCHMARKS.md`
- `benchmarks/results/l4_raw_data.json`
- `paper/reproduce.sh`
- `artifacts/omega/paper_repro_gate_status.v1.json`

Output discipline:
- Report pass/fail/not_run where applicable.
- Preserve blocked status as first-class result instead of collapsing into narrative success.

### 8. Related Work

Objective:
- Situate Sounio among systems languages, scientific stacks, and compiler infrastructures.

Comparison axes:
- Systems implementation depth.
- Scientific semantic integration depth.
- Reproducibility contract strength.

Potential clusters:
- Systems languages and self-hosting compilers.
- Scientific computing toolchains.
- IR-centric systems (general-purpose and domain-specific).

### 9. Conclusion

Objective:
- Re-state systems thesis: scientific semantics and operational reproducibility can be compiler/runtime responsibilities.

Close with:
- What the current system demonstrates.
- What remains open as engineering debt and research direction.

## Claim -> Artifact Matrix (Fail-Closed Writing Contract)

Use this matrix during drafting/revision. A claim without an artifact link should be labeled as design intent, not empirical result.

| Claim class | Required evidence | Location |
| --- | --- | --- |
| Reproducibility workflow exists and runs | Script + status artifact | `paper/reproduce.sh`, `artifacts/omega/paper_repro_gate_status.v1.json` |
| Bootstrap/policy closure behavior | Gate artifact with explicit blockers | `artifacts/omega/strict_no_rust_closure_gate.v1.json` |
| GPU/runtime readiness statement | Runtime attest/parity artifacts | `artifacts/omega/gpu_runtime_attest_gate.v1.json`, `artifacts/omega/gpu_codegen_parity.v1.json` |
| Benchmark claim | Raw data or benchmark summary file | `benchmarks/results/l4_raw_data.json`, `benchmarks/results/NVIDIA_L4_BENCHMARKS.md` |
| Architecture claim | Source docs + code path | `docs/compiler/*.md`, `self-hosted/` compiler modules |

## Figures and Tables Backlog

Figures:
- F1: end-to-end compiler pipeline with scientific invariants.
- F2: SIR-centered optimization and lowering flow.
- F3: policy gate surface (pass/fail/not_run + blockers).

Tables:
- T1: contributions and where they land in implementation.
- T2: evaluation matrix with script/artifact provenance.
- T3: failure taxonomy used by gate artifacts.

## Writing Guardrails

- Keep empirical scope narrow and artifact-backed.
- Avoid absolute language unless backed by measurable evidence in repo artifacts.
- Separate present-tense implementation facts from future roadmap.
- Preserve uncertainty language where benchmark coverage is partial.

## Build and Review Loop

Recommended drafting loop for this paper lane:

1. Draft prose section with explicit claim labels.
2. Attach artifact/file pointer for each claim.
3. Run reproducibility scripts for paper lane:
   - `paper/reproduce.sh`
   - `scripts/paper_repro_gate.sh`
4. Update any claim whose supporting artifact changed.
5. Export final claim-artifact checklist into submission package notes.

## H7: Epistemic Session Types

### Motivation

Scientific web data pipelines combine external service calls with epistemic uncertainty tracking. Existing session type frameworks (Honda 1993, Gay and Hole 2005, Wadler 2012 GV) model protocol structure but carry no epistemic metadata. H7 fuses session types with Sounio's `Knowledge<T>` type, producing `EpistSession<P, ε>` where ε is a compile-time uncertainty bound derived from source provenance.

### Novel Contributions

1. **`EpistSession<P, ε>`**: Protocol type that carries an uncertainty bound alongside the communication structure. Protocol variables α are row-polymorphic (H6.2), enabling open protocols extensible at call sites.
2. **Rule E-Session-Recv**: Receive operations on epistemic sessions automatically lift the payload into `Knowledge<T>` with ε = 1 - trust(source). No explicit annotation needed.
3. **Rule E-Causal-Post**: HTTP POST is typed as a causal intervention (H6.4 CausalEffect) — sending a request to modify server state is a do(X←x) operation whose downstream effect is tracked.
4. **Zero-cost erasure**: All epistemic session annotations erase at compile time. Machine code is identical to untyped send/recv. Verified by inspection of selfhosted native codegen.

### Related Type Systems Superseded or Extended

- Linear/affine session types (Caires/Pfenning 2010): H7 adds epistemic grading (H6.5) on top of the linear discipline.
- Graded session types (Orchard/Yoshida 2016): H7 uses the H6.3 aleatoric/epistemic decomposition rather than a single resource grade.
- Session types + effects (Lindley/Morris 2016 LINKS): H7 integrates with algebraic effects (H6.2 row-poly) rather than treating effects separately.

### Stdlib Files

- `stdlib/web/http.sio` — typed HTTP request/response structs with quality scoring
- `stdlib/web/epistemic_http.sio` — EpistemicResponse, DataSource, provenance_quality
- `stdlib/web/websocket.sio` — WsMessage session types
- `stdlib/science/ncbi.sio` — NCBI genomics typed session (`GeneQuery` → `Knowledge<GeneExpression>`)
- `stdlib/science/clintrials.sio` — ClinicalTrials.gov typed session with evidence quality scoring

### Formal Rules

See `paper/epistemic-types/session_types.tex` for the full grammar, duality, E-Session-Recv, and E-Causal-Post rules.

### Gate

`scripts/sprint19_epistemic_session_gate.sh` → `artifacts/sprint19/epistemic_session_gate.v1.json`

## Immediate Next Tasks

1. Create `paper/oopsla2027/main.tex` using this blueprint and existing PACMPL style from `paper/popl2027/main.tex`.
2. Populate Sections 1-4 first (story + architecture + SIR), then lock figures.
3. Fill Evaluation only after rerunning the paper repro gate so numbers and statuses are current.
4. Integrate H7 formal rules from `paper/epistemic-types/session_types.tex` into Section 2 (language surface) once H7 checker is wired.

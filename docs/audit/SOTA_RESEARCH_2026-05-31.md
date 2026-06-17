# Sounio SOTA positioning — deep-research findings + chosen direction (2026-05-31)

Deep-research run `wf_17f1dbf1-add`: 106 agents, 3.7M tokens, ~17min, 3-vote adversarial verification.
Full machine result: /tmp/.../tasks/wqe9389i2.output. This doc = the actionable synthesis.

## The landscape (HIGH confidence, all 3-0 verified)

| Thread | SOTA system | What it actually proves | Gap for Sounio |
|---|---|---|---|
| Verified self-hosting bootstrap | **CakeML** (HOL4, in-logic) | Semantics preservation reaching the compiler's OWN x86-64 machine code ("proof-grounded bootstrapping"); REPL in machine code provably prints only semantics-permitted results; smallest TCB; in some configs down to verified Silver ISA on FPGA | No uncertainty/epistemic types; ML not gradual; multi-YEAR multi-person effort |
| Verified (not self-hosted) | CompCert | Coq-verified C compiler, but extracted to OCaml + built by unverified toolchain | Not self-hosted |
| Lightweight correctness | **Alive2** (PLDI'21) | Bounded SMT translation validation for LLVM; sound-but-incomplete; finds real miscompiles | Per-pass, not whole-compiler |
| Trusting-trust defense | **Diverse Double-Compiling** (Wheeler PhD'09) | Machine-checked (Prover9/Mace4/Ivy) countermeasure to Thompson's attack: compile source twice, once with a diverse compiler, compare | **THIS is reachable for Sounio** |
| Bootstrap trust (ecosystem) | Guix full-source bootstrap / GNU Mes | 357-byte hex0 seed → ~22,000-node graph → toolchain; + reproducible bit-identical builds | rustc's own trusting-trust mitigation is STILL OPEN (issue #48707) |

## The mundane-but-correct codegen fix (HIGH, 3-0) — System V AMD64 psABI §3.2.3
- Large aggregates = MEMORY class → passed ON THE STACK by the caller.
- Large returns → caller allocates, passes address in **%rdi** (hidden first arg / sret), callee
  returns that address in %rax. This is exactly what rustc/GCC/LLVM do; the single mutable context
  (TyCtxt / LLVMContext) is passed BY POINTER. lean_single's hand-rolled by-value 164KB Checker +
  `rep movsq` slot-count copy is the anti-pattern. (The "hand-rolled rep movsq is buggy" framing is
  engineering judgment from the spec, LOW confidence as literature — but the psABI fix itself is HIGH.)

## The novelty verdict (MEDIUM, 2-1 — appropriately hedged)
**No surveyed system combines a compiler's own epistemic/uncertainty type system (compile-time
uncertainty/confidence as first-class types, GUM/JCGM-100 propagation) with bootstrap-certified
self-hosting.** GUM is a metrology METHODOLOGY, not a type system. Granule (ICFP'19) does graded/
coeffect types for RESOURCE tracking — closest neighbor — but NOT metrological uncertainty, NOT
confidence gates, NOT self-hosted, NOT bootstrap-certified.
⚠️ ABSENCE-OF-EVIDENCE caveat (the dissent, which is correct): not finding prior art ≠ proof none
exists. Must still check grey lit / "probabilistic types" / "information-flow" / "gradual verification"
before any publication claim. → This is exactly the 2nd targeted search to run.

## CHOSEN DIRECTION (user, 2026-05-31): the BRIDGE — DDC + epistemic triad
Max ambition that is actually FEASIBLE (not the multi-year CakeML throne):
1. **Trust layer (reachable):** turn the reproducible bit-identical fixed point `bded845` into a
   genuine TRUST story via Diverse Double-Compiling (Wheeler): compile lean_single.sio with a DIVERSE
   trusted compiler path and compare to the self-built fixed point → machine-checkable anti-trusting-
   trust guarantee. Sounio already HAS the bit-identical fp (stage2==stage3==bded845); DDC is the
   formal cherry on top. This is concrete, citable (Wheeler, Guix, reproducible-builds.org), and
   nobody has applied DDC to an epistemically-typed self-hosted compiler.
2. **Novelty layer (the unclaimed territory):** GUM/JCGM-100 uncertainty as a first-class compile-time
   type (Knowledge<T> + confidence gates) in a SELF-HOSTED compiler, with effect-system soundness
   mechanized in Lean4 (EpistemicEffects.lean). This is the defensible "SOTA+++" the research could not
   find prior art for.
3. **The honest north star (say it, don't promise it):** full CakeML-style machine-code-up verification
   of the epistemic semantics is a multi-year program. The BRIDGE delivers a publishable artifact NOW
   (self-hosted + epistemic + DDC-trusted + Lean4-soundness) while naming machine-code-up as the horizon.

## Immediate engineering gate (unchanged, blocks the demo)
mc.elf must RUN hello. Prerequisite bug: `(*p).field[idx]=v` deref-field-array-store miscompiles in
lean_single (no handler; pre-existing, baseline-reproduced). Then the *mut Checker refactor (psABI-aligned
single-context-by-pointer = literally the SOTA fix above). The codegen fix and the paper's architecture
are THE SAME MOVE: pass the context by pointer like every real compiler.

## 2nd deep-research (wf_366fc261-afd, 103 agents, 3.6M tok, adversarial "kill the novelty") — VERDICT
Goal was to KILL the novelty claim by finding prior art under differently-named framings. It could NOT.
Per-near-neighbor exclusions (all verified; confidence as noted):

| Near-neighbor | What it is | Verdict | Conf |
|---|---|---|---|
| **Uncertain<T>** (Bornholt/McKinley ASPLOS'14) | runtime library, distribution wrapper, Monte-Carlo sampling over lazy Bayesian net; conditionals via runtime SPRT hypothesis test | ADJACENT — value-level not compile-time; probabilistic not GUM (zero refs to GUM/JCGM/ISO); no compiler dimension | high (3-0) |
| **Granule** (graded modal types) | tracks resource usage / information flow via grades | ADJACENT — closest TYPE-SYSTEM relative, but no GUM uncertainty, no confidence gate | high (3-0) |
| **Fuzz/DFuzz** (sensitivity types) | function sensitivity for differential privacy via linear types | ADJACENT — error-propagation-related but privacy-motivated, not GUM standard uncertainty | high (3-0) |
| **Puffin** (Gray&Ferson 2023 "uncertainty compiler") | source-to-source transpiler injecting runtime UQ objects | ADJACENT — runtime/intrusive, not type-level, not self-hosted | high (3-0) |
| **"epistemic types"** in PL | epistemic-logic / knowledge modalities (agents, security) | DISTINCT — not measurement uncertainty; does not preempt Sounio's usage | high (3-0) |
| Units-of-measure (F#/Kennedy) | compile-time dimensional consistency | ADJACENT — units WITHOUT uncertainty; combining dims+uncertainty propagation unattested | medium (2-1) |
| NIST Uncertainty Machine / metrology libs | GUM uncertainty WITHOUT types (value-level) | ADJACENT — uncertainty without types | medium (2-1) |
| CakeML/CompCert/F*/Idris2/Lean4 | conventional type systems | DISTINCT — none carry uncertainty types; reflexive use unattested | medium (2-1) |
| DDC (Wheeler) applications | only gcc/tcc/Mes-style conventional toolchains | DISTINCT — never applied to a rich dependent/effect/uncertainty-typed self-hosted compiler | medium (2-1) |

**VERDICT (the adversarial run's own words):** "metrological-uncertainty-as-a-compile-time-type in a
self-hosted certified compiler appears genuinely UNCLAIMED — high confidence on the per-source
exclusions; MEDIUM-HIGH on the global absence claim (bounded by search coverage)." Finding [11]
(the full combination unclaimed) is conf=LOW/inference — appropriately hedged.

**Residual risk reviewers WILL raise (the open questions to pre-empt in any paper):**
1. "Compile-time uncertainty propagation collapses to known abstract-interpretation / interval
   analysis" — must differentiate GUM first-order law-of-propagation + coverage factors + the
   confidence GATE as a type-checker admit/block, vs interval AI.
2. Grey-lit / unpublished crates: a Rust/Idris "uncertain units" dependent-types package could exist
   off-web. Low risk but acknowledge.
3. Certified-self-hosting overlap: CakeML + a bolted-on numeric-error pass. Distinguish: Sounio's
   uncertainty is a TYPE that gates compilation, not an analysis pass.

**Net positioning (defensible):** The three INGREDIENTS each have prior art (graded types; verified
bootstrap; GUM metrology). The COMBINATION — GUM-metrological uncertainty as a compile-time type with
confidence gates, in a self-hosted compiler, with DDC trust + Lean4 effect soundness — is unclaimed.
Frame the contribution as the SYNTHESIS + the working artifact, not as inventing any single ingredient.

## Next actions
A. 2nd targeted deep-research to harden the novelty claim (metrological types / GUM-in-type-systems /
   coeffects-for-uncertainty / gradual certified compilation / information-flow-as-uncertainty).
B. Fix deref-field-array-store in lean_single (bootstrap-safe: 0 such sites in lean_single itself).
C. *mut Checker refactor (= adopt the psABI single-context-by-pointer pattern) → mc.elf runs hello.
D. DDC trust story over fixed point bded845 (Wheeler-style), + write the positioning paper stub.

Citations: CakeML POPL'14 + JFP backend + Kumar diss; CompCert; Alive2 PLDI'21 + Necula PLDI'00;
Wheeler DDC diss'09; Guix full-source-bootstrap 2023 + GNU Mes; System V AMD64 psABI §3.2.3;
Granule ICFP'19; JCGM-100/GUM (BIPM); rust-lang/rust#48707. (URLs in the machine result.)

<!-- docs:meta
topic_id: repo.docs.proposals.certified-interval-primitive-2026-10-22
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.proposals.certified-interval-primitive-2026-10-22
-->

# Proposal: certified interval primitive and the separation of `u` and `e` in `Knowledge`

- **Status:** proposal. Not implemented. Five decisions are owed by the maintainer (marked **[AUTHOR]**, §4). [`docs/decisions/README.md`](../decisions/README.md) admits only decisions already taken, so this lives here until they are; it becomes the next ADR when they are, and the open ones are indexed in [`PENDING.md`](../decisions/PENDING.md) §D-34.
- **Date:** 2026-10-22
- **Replaces a direction, not code:** a 2026-09-23 design decision, recorded outside this repository, that kernel rounding error should enter `Knowledge<…>` as a GUM Type B component added in quadrature. It was never implemented (see §1), so nothing on `main` changes behaviour when this proposal is accepted.
- **Related:** [ADR-009](../decisions/adr-009-verified-foreign-reference.md) (oracle languages: the test harness is C++23), [ADR-011](../decisions/adr-011-mathlib-scope-in-formal-verification.md) (Mathlib scope; *not* affected, see D9), [ADR-012](../decisions/adr-012-hyper-native-v2-bridge.md) (AVX-512 lowering precedent), [ADR-004](../decisions/adr-004-capacity-guards-over-silent-corruption.md) (fail closed). D7 needs an IR-level effect barrier for global floating-point state. **The IR has none on `main`**: effects are checked at function-signature level (`with IO, Mut, …`, `self-hosted/check/effects.sio`), and `self-hosted/ir/` has no effects module (measured 2026-10-22). D7 is therefore also a requirement on that missing piece.

---

## 1. Context

On `main`, `Knowledge<T>` propagation carries **no rounding term at all**. Measured 2026-10-22: neither `self-hosted/check/epistemic.sio` nor `self-hosted/ir/lower.sio` adds a rounding component to a propagated uncertainty, and `stdlib/epistemic/knowledge.sio` uses ulp-sized tolerances only to classify a slightly negative propagated variance as zero (`ep_cov_output_verdict`, verdicts 0–3). The variance a program prints is the variance of the inputs pushed through the model, as if the arithmetic were exact.

The direction recorded on 2026-09-23 was to close that gap with a GUM Type B estimate added in quadrature. That is defensible metrology, but it is an *estimate*, not a *guarantee*, and every target proposed for Marco 0 exposes directed rounding — per instruction or as global state:

| Target | Directed rounding mechanism | Scope |
|---|---|---|
| x86-64 AVX-512 | EVEX embedded rounding `{rd-sae}` / `{ru-sae}` | Per instruction, ZMM register–register and scalar forms only |
| RISC-V F/D (scalar) | 3-bit `rm` field (`010` RDN, `011` RUP) | Per instruction |
| RISC-V V (RVV) | Dynamic `frm` CSR only | **Global state** |
| AArch64 (emitted today by the seed, `--target aarch64-linux`) | `FPCR.RMode`, plus `FPCR.FZ` flush-to-zero | **Global state** — the D7 problem, as on RVV; not in Marco 0 |
| NVIDIA PTX | `.rm` / `.rp` on `add/sub/mul/fma/div/sqrt` (f32, f64) | Per instruction |

This makes a **certified enclosure** of every elementary floating-point operation cheap, and turns the rounding term of `Knowledge` from an estimate into a bound. Because Madaros owns the path from type to emitted instruction, the guarantee can survive compilation. Sound interval reasoning over neural networks exists (interval bound propagation, Gowal et al. 2018; floating-point-sound verification, Jia & Rinard 2021); what no mainstream stack does is carry directed rounding from the source type to the emitted instruction.

An independent adversarial review of the first draft of this proposal (a separate reviewer given the draft, not its author's reasoning, and instructed to refute it) found five errors in it. They are corrected here and listed in §9 so the record is honest about what changed.

## 2. Decision drivers

1. Sounio's change criterion: the change must make dishonest numerical code harder to write.
2. The existing determinism floor: bit-exact results across architectures without implicit FMA contraction ([`docs/compiler/numerical_determinism.md`](../compiler/numerical_determinism.md)).
3. No third-party component in the build path. Third-party tools (MPFR, `objdump`, XED, `nvdisasm`, Spike) are admitted **only as test oracles**.
4. Lean verification without widening the Mathlib scope fixed in ADR-011.

## 3. Decision summary

Introduce an `Interval<F>` primitive (inf-sup, IEEE 1788-2015 set-based semantics, NaN-free ABI) whose four arithmetic operations carry the rounding direction **in the opcode**. Lower it to x86-64 EVEX, RISC-V F/D and RVV, and PTX. Split `Knowledge` into a rigorous enclosure `e` (primary) and a statistical uncertainty `u` (derived), with conversion from `e` to `u` permitted only explicitly and recorded in provenance.

## 4. Decisions

### D1. Reference semantics: inf-sup
`Interval<F> = [lo, hi]`, `lo ≤ hi`, IEEE 1788-2015 set-based flavour, is the semantics that is specified, proved and tested.

Midpoint-radius is admitted later as a **sound lowering, not an equivalent one**: for linear algebra (Rump-style GEMM) it over-encloses, with an error term that depends on dimension. The obligation is containment, `inf-sup ⊆ mid-rad`, proved once. Bit identity is required *within* a lowering, never *across* lowerings.

Conversion inf-sup → mid-rad uses `mid = RN(lo/2 + hi/2)` (not `(lo+hi)/2`, which can overflow) and `rad = RU(max(mid − lo, hi − mid))`.

### D2. `u` and `e` are separate components of `Knowledge` **[AUTHOR]**
`Knowledge<T>` carries:
- `e : Interval<T>` — rigorous enclosure **of the value** (not a symmetric error radius, which loses asymmetry under non-linear maps). Primary.
- `u` — GUM standard uncertainty, statistical. Derived.

Folding `e` into `u` is **not a category error** (GUM §4.3.7 converts a rigorous bound `a` into `u = a/√3` under a rectangular distribution); it is a *lossy* operation. It is therefore allowed only as an explicit operation, `e_to_typeB(dist = rect | tri | …)`, recorded in provenance. The implicit quadrature proposed on 2026-09-23 is withdrawn before it lands.

- *Recommendation:* adopt.
- *Consequence:* no site on `main` adds rounding into `u` today (§1), so the rule costs nothing now and forbids the shortcut before anyone takes it; the type checker refuses an implicit fold.
- *Prior art:* the pair `(e, u)` is a degenerate case of a probability box (p-box; Ferson, Kreinovich et al.), the established framework that joins interval and probabilistic uncertainty. The representation must not preclude a later generalisation of `Knowledge` to p-boxes.

### D3. Propagation of `u` under a non-degenerate `e` **[AUTHOR]**
The GUM law of propagation evaluates a Jacobian at a point. When `e` says the true value may lie anywhere in an interval, a point Jacobian silently assumes a location the program does not know. This is a lie inside the type.

Options:
- (a) Interval Jacobian: `u` is bounded over `e`. Sound; may be pessimistic.
- (b) Refuse propagation unless `e` is degenerate (`lo == hi`).
- (c) Point Jacobian permitted only through an explicit `assume_point(at = mid)` recorded in provenance.

- *Recommendation:* (a) as default, (c) as the explicit escape hatch; (b) is too restrictive for Marco 0.

### D4. Interval ABI: NaN-free, sentinel empty, canonical zero
1. **Empty** is encoded `[+∞, −∞]`, not 1788's `[NaN, NaN]`. NaN never appears in an interval value; this is what makes D11's bit-identity claim possible.
2. **Entire** is `[−∞, +∞]`.
3. **Zero sign** is canonicalised: a lower endpoint of zero is `−0`, an upper endpoint is `+0`. Canonicalisation is performed by the operation, not by the consumer.
4. **Decorations** (IEEE 1788): at minimum `ill`, `trv`, `def`, `dac`, `com`, carried alongside the interval. Required for sound branching and division.
5. **Two-piece results** of division by an interval containing zero (`[−∞, a] ∪ [b, +∞]`) return the hull, with decoration `trv`. Documented, not hidden.

### D5. Special cases follow IEEE 1788, not hardware
Hardware produces NaN where 1788 defines a value. These are special-cased in the operation:

| Expression | Hardware | 1788 result |
|---|---|---|
| `0 · ∞` (endpoint product) | NaN | `0` |
| `∞ − ∞` (endpoint difference) | NaN | the appropriate infinite bound |
| `x / [0, 0]` | NaN/∞ | empty, decoration `trv` |

Arithmetic is non-trapping. Endpoints may be `±∞`.

### D6. Explicit interval FMA **[AUTHOR]**
`ifma.rd` / `ifma.ru` are IEEE-exact and available on every target (EVEX `VFMADD… {er}`, RISC-V `FMADD` with `rm`, PTX `fma.rm/.rp`). The Madaros no-FMA floor exists to forbid **implicit contraction**, which changes results silently. An explicit opcode whose rounding is part of its identity does not violate that intent.

- *Recommendation:* allow explicit `ifma.rd/.ru`; keep implicit contraction forbidden everywhere.

### D7. Rounding direction is opcode identity; global FP state is an effect
1. `add.rd`, `add.ru`, `mul.rd`, … are distinct opcodes. CSE keys on (opcode, rounding, operand order). Reassociation is forbidden on interval opcodes.
2. Rewriting `add.rd(a, b)` ↔ `neg(add.ru(neg a, neg b))` is a **legal canonicalisation** (IEEE 754 directed rounding is sign-symmetric, including overflow and signed zero), not a miscompilation.
3. **Global FP state is not eliminated by per-instruction rounding.** On x86, `{er}` overrides only `MXCSR.RC`; **DAZ and FTZ still apply.** Writes to `MXCSR` (`ldmxcsr`) and to RISC-V `frm` (`fsrm`) are therefore effects. They must be registered as barriers with whatever effect authority the IR adopts, or confined to opaque kernel regions. Until that authority exists, interval kernels are opaque regions.
4. Every interval kernel asserts on entry that `MXCSR == 0x1F80` (default: all exceptions masked, RN, DAZ=0, FTZ=0) and that `frm` holds the value the kernel expects; on violation it fails closed.

### D8. Constant folding uses Sounio's own softfloat
The compiler host may have FMA enabled or DAZ/FTZ set. Folding interval opcodes with host arithmetic would make results host-dependent. Interval constants are folded with a **host-independent directed-rounding softfloat written in Sounio**, or not folded at all. MPFR is not linked into `souc`; it is an oracle in the test harness only.

### D9. Lean model: ℚ ∪ {±∞}, axiomatic rounding. ADR-011 unchanged
Reals are unnecessary for Marco 0. Finite binary floats are dyadic rationals, and exact `+ − × ÷` of rationals are rationals. The model:

- Carrier: `Rat` (Lean core / Batteries) extended with `±∞`.
- `RD`, `RU` axiomatised: monotone; `RD x ≤ x ≤ RU x`; idempotent on representable values; `RD x = −RU(−x)`.
- Theorems: enclosure for `+ − × ÷` over this model, including the D5 special cases.

No Mathlib, no density argument. Reals enter only with `sqrt` and transcendentals, in a later ADR. The link between the axioms and real hardware is established by **differential testing, not by proof**, and the ADR states so explicitly. The axioms appear in the generated `formal/AXIOM_INVENTORY.md` (`scripts/dev/gen_axiom_inventory.sh`, file and line only); the NOT ESTABLISHED status until each is discharged against a bit-level IEEE model must be stated beside each axiom in the Lean source, since the inventory has no status field.

### D10. Comparisons and branching **[AUTHOR]**
Interval comparison is tri-state: *certainly*, *possibly*, *never*. The natural home is the existing `Contest<T, Family, Policy>` machinery (`self-hosted/check/check.sio`, which today requires a declared `models` family); how a tri-state comparison fits that signature is part of this decision.

An `if` whose condition is only *possibly* true is the main source of lies in interval code. Options:
- (a) Reject at compile time unless the condition is certain (error code), with an explicit `hull_branches { … }` construct that evaluates both branches and returns the hull.
- (b) Widen silently to the hull of both branches.

- *Recommendation:* (a). Silent widening hides a decision the programmer should make.

### D11. Targets of Marco 0 **[AUTHOR]**
| Target | Status in Marco 0 | Notes |
|---|---|---|
| x86-64 EVEX ZMM + scalar | **In** | Runs on the Xeon 6148 now. YMM embedded rounding is **not** available (added then removed from AVX10.2); never plan a YMM path. |
| RISC-V F/D scalar | **In** (Spike) | Static `rm`; no negation trick needed. |
| RISC-V V | **In** (Spike) | `frm = RUP` once per kernel plus the negation trick; `fsrm` is an effect (D7). |
| NVIDIA PTX | **In** | No `.ftz`, no `*.approx`; `souc` emits PTX directly, so no `--use_fast_math` path exists. |
| AMD RDNA | **Out, Marco 0b** | Rounding mode lives in the wave `MODE` register (global state, as RVV); to be verified against the RDNA4 ISA manual before scheduling. Strategic priority for sovereign GPU. |
| Apple Metal | **Out** | No directed rounding exposed, no fp64, possible fp32 subnormal flush. |
| Intel AMX (TMUL) | **Out, permanently for certified paths** | `TDPBF16PS` ignores MXCSR, uses fixed RNE and forces DAZ = FTZ = 1. Neither directed nor random rounding is possible. AMX may serve only as an uncertified round-to-nearest baseline, and its results may not populate `e`. |
| U250 FPGA | **Out** | No RISC-V core with RVV is selected yet; Spike is a simulator check, not a hardware check. |

- *Recommendation:* as tabled.

### D12. min/max are not lowered to ISA min/max
ISA min/max disagree across targets: x86 `MINPD/MAXPD` return the *second operand* when both inputs are zero or one is NaN; RISC-V `FMIN`/`vfmin` implement `minimumNumber` with `−0 < +0`; PTX `min` uses `min_num`. Endpoint selection in multiplication is therefore lowered to **compare-and-blend**, which is ISA-independent and SIMD-friendly. This, together with D4, is what makes bit identity achievable.

## 5. Arithmetic (normative)

With `↓ = RD`, `↑ = RU`:

1. `[a] + [b] = [↓(a.lo + b.lo), ↑(a.hi + b.hi)]`
2. `[a] − [b] = [↓(a.lo − b.hi), ↑(a.hi − b.lo)]`
3. `[a] × [b] = [min↓(P), max↑(P)]`, `P` the four endpoint products, each computed in both directions, special cases per D5, min/max per D12. The 9-case sign analysis is a permitted scalar specialisation.
4. `[a] ÷ [b]`: 1788 set-based semantics; denominator containing zero per D4.5.

**Negation trick.** `↓x = −↑(−x)`. Storing `(−lo, hi)` lets one round-up operation produce both endpoints of a sum. Exact for all inputs, including overflow and signed zero.

## 6. Literals and I/O
1. Decimal literals convert with directed rounding: `0.1` → `[↓0.1, ↑0.1]`. Requires correctly rounded decimal-to-binary conversion in both directions, implemented in Sounio.
2. Output of certified values is in hexadecimal floating point by default; decimal output rounds outward.

## 7. Acceptance criteria for Marco 0
1. **Enclosure:** zero violations against MPFR over N random cases plus the edge-case corpus (subnormals, `±∞`, signed zero, D5 cases, empty, entire).
2. **Tightness:** for every case, `lo == RD(exact)` and `hi == RU(exact)` as computed by MPFR. When the exact result is representable, `lo == hi`. *An implementation that returns `[−∞, +∞]` passes enclosure and must fail here.*
3. **Bit identity:** endpoints and decorations bit-identical across x86 EVEX, RISC-V scalar, RVV (Spike) and PTX, under the D4 ABI.
4. **Global state:** a test that sets DAZ/FTZ before calling an interval kernel must fail closed, not return a wrong enclosure.
5. **Lean:** enclosure theorems for `+ − × ÷` in the D9 model, `sorry`-free; axioms listed in `AXIOM_INVENTORY.md`.
6. **Type system:** implicit `e`→`u` folding is a compile error (D2); a `possibly`-conditioned `if` is a compile error under D10(a).
7. **Toolchain:** no third-party component in the build path of any of the above; oracles appear only in the test harness, written in C++23 per ADR-009.

## 8. Consequences and risks

**Positive.** Rounding error becomes a certified quantity; Sounio's "refuses to compute lies" extends from the type to the emitted instruction, across three ISAs, with cross-target differential verification.

**Risks, ranked.**
1. **Global FP state leaking under stateless-looking opcodes** (DAZ/FTZ surviving `{er}`, `frm` on RVV): breaks enclosure and bit identity silently. Mitigated by D7.3–7.4 and acceptance criterion 4.
2. **Undefined `u`/`e` composition:** `u` propagated at a point that `e` says may be wrong. Mitigated by D3.
3. **Tests that a trivially sound implementation passes:** mitigated by the tightness criterion and the pinned ABI.

**Negative.** Intervals suffer the dependency problem (`x − x ≠ [0, 0]`); across long computations, such as a full transformer, they become uselessly wide. Marco 0 certifies **individual operations**, not whole models. The structural remedy is affine arithmetic (Stolfi and de Figueiredo), whose noise symbols are first-order correlation structure and map naturally onto Sounio's affine-support Q and `Correlated`. The Marco 0 IR must admit an `AffineInterval` lowering without redesign.

**Q interaction.** A `Knowledge` whose `e` is `Entire` or empty must not report high Q. Either Q collapses or a flag is raised; otherwise "value known with high quality, enclosure unknown" remains expressible.

## 9. Corrections to the first draft
1. *"Adding interval width in quadrature with `u` is a category error."* Overstated; GUM §4.3.7 allows the conversion. Corrected to "lossy, therefore explicit" (D2).
2. *"The effect system need not model anything, since rounding is in the opcode."* Wrong: DAZ/FTZ and `frm` remain global state (D7.3).
3. *"Endpoints are bit-identical across targets because IEEE 754 specifies directed rounding."* True only for non-NaN results with canonical zeros and ISA-independent min/max (D4, D12).
4. *"Lean enclosure proofs need reals, in tension with ADR-011."* Wrong: ℚ ∪ {±∞} suffices for `+ − × ÷` (D9).
5. *"inf-sup and mid-rad equivalence proved once."* Wrong: containment, not equivalence (D1).

## 10. Out of scope (future ADRs)
Random-direction rounding (CESTAC, p = ½ up/down per operation) as a third lowering mode beside RD and RU, for discrete stochastic arithmetic; its cost on AVX-512 is at least 3.5× per FMA (two rounded FMAs plus a masked blend plus mask generation), not 2×. Affine arithmetic (Marco 1); `sqrt` and transcendentals (correctly rounded libm or ulp-bound widening; reals in Lean); mid-rad GEMM; AMD RDNA lowering; RISC-V core on the U250; Metal.

## 11. References
- IEEE Std 1788-2015, *Interval Arithmetic*; IEEE Std 1788.1-2017 (simplified).
- IEEE Std 754-2019, *Floating-Point Arithmetic*.
- JCGM 100:2008 (GUM), §4.3.7 and §5.1.
- Intel® 64 and IA-32 SDM Vol. 2, §2.7 (EVEX, embedded rounding); `VADDPD`, `MINPD` reference pages.
- Intel AVX10.2 Architecture Specification, document 361050, revision 4.0 and later (change log: removal of YMM embedded rounding).
- RISC-V Unprivileged ISA, F and D extensions; RISC-V V extension specification (§ floating-point rounding).
- NVIDIA, *Parallel Thread Execution ISA*: `add`, `mul`, `fma`, `div`, `sqrt`, `min`, `.ftz`.
- Gowal, S., et al. (2018). On the effectiveness of interval bound propagation for training verifiably robust models. arXiv:1810.12715.
- Jia, K., & Rinard, M. (2021). Exploiting verified neural networks via floating point numerical error. *SAS 2021*. arXiv:2003.03021.
- Rump, S. M. (1999). Fast and parallel interval arithmetic. *BIT Numerical Mathematics*, 39(3), 534–554.
- Stolfi, J., & de Figueiredo, L. H. (1997). *Self-validated numerical methods and applications*. IMPA.
- Ferson, S., Kreinovich, V., et al. (2003). *Constructing probability boxes and Dempster–Shafer structures*. Sandia National Laboratories, SAND2002-4015.
- Intel® 64 and IA-32 SDM, `TDPBF16PS` (AMX: MXCSR not consulted; RNE; DAZ/FTZ forced).
- Vignes, J. (2004). Discrete stochastic arithmetic for validating results of numerical software. *Numerical Algorithms*, 37, 377–390.

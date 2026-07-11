<!-- docs:meta
topic_id: repo.docs.research.sedenion-cd-qbig
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-cd-qbig
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The exact Cayley–Dickson product over unbounded ℚ (all 16 components)

**One line.** Closes the Paper 1 open item: the full 16-component sedenion product over ℚ with
**arbitrary-precision** rational coefficients — the i64 version (`sedenion_cd_full16_q.sio`) capped at
coefficients ~1e8; this reaches magnitudes ~10^80 exactly, via a minimal self-contained signed BigInt.

## What is certified
Common-denominator reduction: a rational sedenion is 16 integer numerators `A[16]` over one common denom
`Ad`; then `(A·B)[k] = (Σ_{i⊕j=k} σ(i,j) A[i] B[j]) / (Ad·Bd)` — an **integer** 16-component product of
numerators. Two cases:
- **Case 1 — exact annihilation at scale 10^80.** The zero divisor `(e₃+e₁₀)(e₆−e₁₅)`, each coefficient
  scaled by `10^40`, produces **exactly 0 in all 16 components** at magnitude `10^80`. An i64 product
  overflows into garbage here; exactness holds.
- **Case 2 — general unbounded product.** A 16-component product with rational coefficients whose output
  components reach ~`10^70`, cross-verified component-by-component.

## Why a local BigInt (and the honest boundary on #651)
The stdlib `math::bignat` full `big_mul(a,b)` **SIGSEGVs** under souc v0.80.0 — a single call, no arrays,
no loop: a clean minimal `#651`/`#637` struct-copy repro. `[BigInt;16]` also SIGSEGVs (the
array-of-structs wall). This brick **circumvents — it does not fix — `#651`**, exactly as the i64
common-denominator circumvented `#637`:
1. a minimal in-file **signed BigInt** (base-1e9 limbs, `[i64;12]`) built only from primitives that
   empirically compile — schoolbook `big×big` = `mul_small` + limb-shift + `add`-in-loop (a spike
   confirmed this exact accumulate-in-loop pattern runs, which is what the stdlib `big_mul` trips on);
2. the 16 numerators are held in **flat scalar arrays** `[i64;N]`, never `[BigInt;16]`.

Reproducing the full stdlib BigInt path is compiler-team work (`#651`); the math goal is met today.

## Cross-verification (3 independent legs)
- **souc** (bin/souc AND stage2): `tests/run-pass/sedenion_cd_qbig.sio` → `CDQBIG OK`. Emits each
  component's **residue mod 1e9+7** (a per-limb *decimal* printer is miscompiled *differently* by the two
  souc builds — bin/souc breaks print-in-loop, stage2 adds `print_int` newlines — but
  `print("RES ") print_int(x)` is byte-identical on both), plus the structural annihilation flag.
- **Python oracle** (exact ℚ): `scripts/research/sedenion_cd_qbig_oracle.py` — emits the exact decimals
  and the residues; gate `scripts/ci/sedenion_cd_qbig_gate.sh` compares residues in order.
- **Lean `native_decide`** (native arbitrary-precision `Int`, the digit-exact witness):
  `formal/lean4/SounioCDqbig.lean` — `case1_annihilates` (all 16 comps 0 at 10^80), `case2_exact`
  (the 16 exact component values).

## Reproduce
```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_cd_qbig.sio
python3 scripts/research/sedenion_cd_qbig_oracle.py
bash scripts/ci/sedenion_cd_qbig_gate.sh
(cd formal/lean4 && lake build SounioCDqbig)
```

## Relation to the arc
This is the *infrastructure* frontier opened in Paper 1 (exact algebra over unbounded ℚ), distinct from
the physics/bridge arc. It makes the exact zero-divisor detection unconditional in coefficient size:
annihilation is now certifiable at any scale, not just within i64.

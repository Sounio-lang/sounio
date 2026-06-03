# Effect/ident validation re-sweep — net-NEGATIVE, not shippable (2026-06-02)

With the codegen fall-through bug FIXED + bin/souc swapped (`5082bf67e`), the banked
effect-validation patch (`fn_sigs_e008_env_ontology_reports.patch` = fn_sigs + int-int + env +
ontology + reports) finally RUNS without the 364-crash flood. Re-swept and oracle-vetted.

## Result: parity got WORSE (485 → 390 / 845)

Both states vs the SAME oracle (fixed bin/souc full-compile rc; rc=0 ⟹ type-check passed):

| state | AGREE | false-pass (oracle reject, mc pass) | false-positive (oracle pass, mc reject) | crash |
|---|---|---|---|---|
| +7 baseline | **485** | 80 | 280 | 0 |
| mc_eff (effect patch) | **390** | 256 | 192 | 7 |

The patch is a **net −95 oracle-parity → DO NOT SHIP.** check.sio reverted to +7. The codegen fix
stays live (it is independent and a strict win).

## Why it diverges in BOTH directions

- **+192 → 256 false-passes (under-rejection):** the fn_sigs fix turns return-type checking on,
  flipping many lenient-rejects to passes — but the effect/ident checks catch only SOME of what
  should then be rejected. Sampled oracle reasons on the false-passes: `effect not declared`
  (the callee-effects check fires for minimal direct calls but MISSES many real cases —
  method-calls / transitive / builtin effect paths), and `unknown identifier`/`unknown field`
  (ident-resolution is NOT implemented — the lenient unresolved-callee branch never errors).
- **280 → 192 false-positives (over-rejection):** the now-exercised arg-checking is STRICTER than
  the lenient oracle — `E004` (literal width) and `E009` (call-arg type mismatch) dominate
  (algebra_demo alone: 36×E004 + 38×E009). The oracle coerces int literals / widths the modular
  checker rejects.

## The +38/+99 projection was wrong

It assumed effect-validation cleanly converts the 120 effect false-passes to correct rejects with
nothing else changing. Reality: (a) the effect check is PARTIAL (misses many effect cases);
(b) ident-resolution is absent (61+ unknown-ident false-passes remain); (c) the arg-checker
over-rejects (E004/E009) — a SEPARATE pre-existing strictness now exercised; (d) 7 residual
crashes from a different codegen bug (rip in an mmap region, not the fixed fall-through).

To actually improve parity the modular checker needs, as real work (NOT a quick patch):
1. Complete effect-validation (method/transitive/builtin effect paths), not just direct calls.
2. Implement identifier/method resolution + error on genuinely-unresolved names.
3. Relax the arg-checker to the oracle's int-literal/width coercion (kill the E004/E009 false-positives).
4. Diagnose the 7 rip-in-mmap residual crashes (a distinct codegen bug from the fall-through one).
Each is substantial; together they are a multi-session front-half effort, not +38.

## What IS solid
The codegen fall-through fix (live in bin/souc) is the real deliverable of this arc: it removes
the layout-sensitive return-addr-smash class (364→7 crashes when the arg path is exercised),
validated by fixed-point + 507/507 run-pass + 847/847 examples + reproducer + payoff.

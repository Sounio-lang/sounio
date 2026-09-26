# LLM Offload Log

_This file was deleted from `main` by 3944ff825 (PR #2646, 2026-09-23; 4128
lines removed). This branch restarts it with its own entry only; the earlier
history is recoverable with `git show 3944ff825^:.claude/llm_offload_log.md`._

## 2026-09-26T21:10Z — Claude — Rodgers–Rowland Kpu→Kp conversion (D5), partial audit fix

| 2026-09-26 | xai/grok-4.6 [OK] | math-review | stdlib/darwin_pbpk/core/rodgers_rowland.sio (rr_neutral Kp = Kpu·fu_p, ∂Kp/∂logP, ∂Kp/∂fu; tests T5, T6); stdlib/darwin_pbpk/core/tissue_composition.sio (T1 consistency test) | PASS | All five claims [OK], no other error. Raw: workspace `/tmp/llm-offload-7U4gGQ/`. |

**Trigger**: PK math change. `rr_neutral` computed Kp = Kpu / fu_p; with
Kpu = C_t / C_u,p and C_u,p = fu_p·C_p,total, the tissue : total-plasma
coefficient is Kp = Kpu·fu_p. The old form is off by 1/fu_p² (156.25× at
fu_p = 0.08), and ∂Kp/∂logP, ∂Kp/∂fu inherited it (∂Kp/∂fu also had the wrong
sign). Citation corrected to J Pharm Sci 95(6):1238–1257, doi:10.1002/jps.20502
(PMID 16639716; checked on PubMed).

**Claims reviewed**: (1) Kp = Kpu·fu_p; (2) ∂Kp/∂logP = P·ln10·(fn_L + 0.3 fn_P)·fu_p
and ∂Kp/∂fu = Kpu > 0; (3) the scaled-uncertainty fu term |∂Kp/∂fu|·u_fu =
(Kp/fu)·u_fu keeps its magnitude and holds only while Kpu has no fu-dependent
term; (4) T5 ratio Kp(0.08)/Kp(1) = 0.08 (old form 12.5), T6 central
difference in fu at logP = 0 positive and within 1e-8 relative of the analytic
value; (5) composition test: fractions ≥ 0, water + lipid ≤ 1, residual
f_protein = 1 − water − lipid within 1e-9.

**Outcome**: PASS. Both engines (Madaros md5 ce11a247 via `/workspace/sounio/bin/souc`,
and `SOUNIO_SOUC_ENGINE=lean_single`): both modules PASS, `check: OK`.
Sabotage controls, both engines: old Kpu/fu conversion restored → T5 and T6
FAIL in 14/14 tissues, T1–T4 still PASS (they are scale-invariant, which is why
D5 went undetected); liver f_ew 0.161 → 0.171 without adjusting f_protein → T1 FAIL.

**Not covered (open)**: the composition values and the fn_L + 0.3 fn_P form are
unverified against the primary paper, which was not accessible; the 2006
albumin term is absent (D4); ionisation terms for weak bases (D6). Two further
defects found and left unfixed (out of dispatch scope): `rr_pow10(7.0)`
returns 8614669.5 (13.9 % low; 20-term Taylor exp at x = 16.1), and `rr_sqrt`
(12 Newton steps from y = x) returns 25764 for √1e8, so `kp_unc_raw` is wrong
wherever the variance is large, including the rapamycin case.

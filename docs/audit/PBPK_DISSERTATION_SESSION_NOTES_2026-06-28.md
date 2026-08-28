<!-- docs:meta
topic_id: repo.docs.audit.pbpk-dissertation-session-notes-2026-06-28
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pbpk-dissertation-session-notes-2026-06-28
-->

# PBPK Dissertation — Session Audit Notes (2026-06-28)

Forensic notes from the dissertation reconciliation session. Empirical-first: every
claim is anchored to a path + command + observed marker. EN-UK orthography.

Audited HEAD: `c25ccdc6f` (branch `fix/madaros-tuple-let-desugar`); dissertation
science tree `stdlib/darwin_pbpk/` was byte-identical to `fix/dissertation-gates-green-20260625`.

---

## 1. Two compiler surfaces — the #1 calibration trap

When auditing the PBPK dissertation (`stdlib/darwin_pbpk/`, gates
`scripts/ci/dissertation_*.sh`), distinguish the compiler surface before reading any
gate result (operating principle R5: intrinsic property vs defect):

- **Default Madaros** (`bin/souc`, v0.80.0) on a compiler-WIP branch **fails** the
  dissertation gates — it segfaults on the modular path
  (`imported_compile → lower_array: seed_begin → SIGSEGV`) and false-rejects valid
  multi-module code via the in-development imported-typecheck (E137 / E175 / E015 /
  E008). These are **compiler-state artifacts, not science regressions.**
- **lean_single fixed-point engine** (`SOUNIO_SOUC_ENGINE=lean_single`, ELF
  `bin/souc-lean-single-x86_64`) compiles and runs the same sources **clean**:

| Gate | Result | Markers |
|---|---|---|
| `dissertation_pbpk28_parity_gate` | PASS | `PBPK28_PARITY_PASS 14/14`, `MASS_CONSERVATION_PASS 12/12`, `TMDD_PARITY_PASS 3/3`, `PD_PARITY_PASS 1/1`, `SEMAGLUTIDE_PARITY_PASS 14/14`, `SEMA_TMDD/SEMA_PD` |
| `dissertation_frontend_parity_gate` | PASS | `PARITY_PASS 14/14` |
| `dissertation_confidence_gate_gate` | PASS | C4: honest-ceiling compiles, over-claim + non-literal ε rejected |
| `dissertation_pbpk_suite_gate` | PASS | 51/53 active; 2 PENDING (clinical, awaiting observed data) |
| `dissertation_pbpk_hessian_gate` | science PASS | `HESSIAN_PBPK28_DUAL_RHO_PASS`; CSV byte-identical to golden |

**Gotchas**
- lean_single does **not** `chmod +x` its emitted ELF, so gates guarding with
  `[[ -x ELF ]]` (hessian, dossier) report FAIL even though the compile succeeded —
  chmod manually before running.
- "CSV differs from golden" failures were downstream of blocked compiles, not numeric
  drift: `benchmarks/pbpk/hessian_budget.csv` matched byte-for-byte under lean_single.

**To get honest dissertation evidence on a WIP branch:** force
`SOUNIO_SOUC_ENGINE=lean_single`, or audit from `fix/dissertation-gates-green-20260625`.

---

## 2. §4.9 second-order Hessian GUM numbers refreshed to the M6 prior

The M6 hepatic-prior update (`d052806ef`) lowered CL_hepatic CV 58 % → 38 %
(var 51.85 → 22.20, μ = 12.4 L/h; modern pop-PK: Jiao 2009 / Sabo 2021,
√(0.238² + 0.299²) ≈ 0.382). This softened every CL_hepatic-derived figure. The
briefing's expected v1.1 values (ratio ≈ 1.644, Jensen bias ≈ 1.373) are **stale**.

Canonical values at HEAD (from `epistemic_pbpk28_hessian.sio` under lean_single,
`HESSIAN_PBPK28_DUAL_RHO_PASS`), recorded in
`docs/dissertation/results/pbpk28_epistemic_v1.md`:

| Quantity | v1 (stale) | HEAD (M6) |
|---|---|---|
| CL_hepatic ρ_literal / ρ̃ | 0.581 / 0.169 | **0.380 / 0.072** |
| §4.10.3 CL_hepatic σ² | 51.85 | **22.20** |
| CV(CL) | 58 % | **38 %** |
| AUC_ref / mean-corrected | — | **0.611694 / 0.729467 mg·h/L** (Jensen +19.25 %) |
| var₂/var₁ | — | **1.284** (SD +13 %, variance +28 %); u₁ = 0.2605, u₂ = 0.2952 |

All other parameters unchanged. Verified by `bin/llm-offload -t math-review -p xai`
("NO MATHEMATICAL ERRORS"; log marker `HESSIAN-PBPK28-S4.9-NUMERIC-REFRESH-MATH-OK`).
Landed via PR #485 (feature branch) and PR #488 (`main`). Narrative shift: from
"CL_hepatic strongly dominates the second-order correction (~17 %)" to "weakly
nonlinear throughout; CL_hepatic the largest single contributor at ~7 %, only
marginally ahead of Kp_brain (ρ̃ = 0.061)".

The committed file is the in-repo results-of-record; downstream chapter prose must be
regenerated from these numbers.

---

## 3. Honest stubs / pending (declare as limitations)

- LLL mapping α still a stub identity — `pd/coronary_smc_prolif.sio:100`:
  `alpha = 1.0  // mm per unit-N — placeholder; calibrate vs RAVEL`.
- `coronary_smc` sub-compartment authored but parity gate exercises the heart-proxy
  ("pending JS engine adoption", `compartments/coronary_smc.sio:16-18`).
- Clinical validation `PENDING` — `validation/pbpk28_{rapamycin,semaglutide}_clinical.sio`
  carry `OBSERVED DATA IS PENDING — DO NOT FABRICATE`; these are the 2 PENDING suite items.
- Hill EC50 ε = 0.40 (`pd/hill_mtor.sio:90`) — the declared Value-of-Information target.

---

## 4. Published gists (for retrieval)

Account `agourakis82`:
- PBPK dissertation audit (HEAD `c25ccdc6f`, public):
  `https://gist.github.com/agourakis82/97f6c52535bfd112b04794f20bccb861`
- Full dissertation data (all 54 `docs/dissertation/**` files + `00_INDEX.md`, secret):
  `https://gist.github.com/agourakis82/d1f222edf00aa8c58abd7a8c1d8f495a`

---

## Appendix: reproducibility

```bash
# compiler surfaces
./bin/souc check tests/run-pass/dissertation_pbpk14_hessian.sio          # Madaros: E175/E137/E008/E009 (exit 1)
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check  …                       # lean_single: 0 errors (exit 0)
./bin/souc compile tests/stdlib/darwin_pbpk/hessian_correction_test.sio -o /tmp/hc.elf   # Madaros: SEGFAULT
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile … -o /tmp/hc.elf       # lean_single: ELF emitted (exit 0)

# gates (force the fixed-point engine)
SOUNIO_SOUC_ENGINE=lean_single bash scripts/ci/dissertation_pbpk28_parity_gate.sh   # PASS

# §4.9 numbers
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile stdlib/darwin_pbpk/epistemic_pbpk28_hessian.sio -o /tmp/h28.elf
chmod +x /tmp/h28.elf && /tmp/h28.elf      # AUC_ref=0.611694 mean_corr=0.729467 var1=0.067863 var2=0.087119
```

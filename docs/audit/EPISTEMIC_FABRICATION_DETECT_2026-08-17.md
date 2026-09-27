<!-- docs:meta
topic_id: repo.docs.audit.epistemic-fabrication-detect-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-fabrication-detect-2026-08-17
-->

# Epistemic fabrication detection — zero GUM variance & bit-pattern confidence

**Date:** 2026-08-17  
**Lane:** grok-cli4 / epistemic-fabricate-fix  
**Witnesses:** `docs/audit/DISSERTATION_PBPK_SUITE_TRIAGE_2026-08-16.md` (§ Cluster C)  
**Thesis risk:** a language that claims *refusal over fabrication* cannot ship
silent `var=0.000` or a confidence that is the integer reading of IEEE bits.

---

## Vacuous-sweep push (this session)

Pushed on `lane/grok-cli4/ws-f-visibility-20260817`:

- `93dbad3780` — taxonomy U0–U13, `gate_contract_probe`, mechanical vacuous
  fixture sweep + planted positive control  
- Branch: `origin/lane/grok-cli4/ws-f-visibility-20260817` (also PR #1765 stack)

---

## Measured witnesses (this worktree, default Madaros vs lean_single)

### F1 — `tests/run-pass/rapamycin_epistemic_adaptive.sio`

| Engine | `var(blood)` | epist_active | exit (before fix) |
|---|---|---:|---|
| lean_single | ~9e-6 | 17 | 0 PASS |
| Madaros | **0.000000** | **0** | **0** with `FAIL: … mech=no` printed |

Madaros **fabricates zero variance** (concentrations still match lean). The test
previously printed FAIL but **returned no status** (`fn main()` without `-> i64`),
so the process still exited 0 — silent to any gate that only reads rc.

### F2 — `stdlib/darwin_pbpk/epistemic_pbpk28.sio` TEST 6

| Engine | printed AUC confidence | TEST 6 |
|---|---|---|
| lean_single | **0.671038** | PASS |
| Madaros | **4604219396932172800.000000** | FAIL |

Decode: integer `4604219396932172800` = `0x3fe57925b61afc00` = IEEE bits of
**≈0.671038** (the lean value). Madaros **holds / compares** a magnitude
`~4.6e18` (TEST 6 range check fails), and `print_f64` prints that huge float
as `4604219396932172800.000000`.

### Print-path check (fable-1 prior — do not mis-attribute)

fable-1 / known Madaros print defects:

| Defect | Symptom |
|---|---|
| Unclassified scalar → **char\*** printer | **SIGSEGV** (strlen on small integer as pointer) — `MADAROS_PRINT_INT_DISPATCH` |
| `println` kind 0 on index/computed locals | same char\* class or wrong printer |
| `print_f64` of a **live f64** | decimal float text |

**F2 is not the char\* family.** A char\* misroute on this bit pattern would
fault, not print a tidy `….000000` decimal. F2 matches **f64 bits reinterpreted
as integer then converted back to f64** (bitcast/sitofp), so both the
**comparison** and **print_f64** see ~4.6e18. Adjacent: historical f64→i64
param bitcast (GUM k95). Plain `print_f64(0.4)` and a minimal `ConfBox{conf}`
return probe print `0.671038` correctly — corruption is on the
**EpResult28 / multi-module return path** for `auc_blood_conf`, not global
`print_f64` death.

---

## Mechanism class (do not rediscover)

| Adjacent known | Relation |
|---|---|
| **`KCONF-BITCAST-SITOFP`** — Knowledge.confidence f64 payload / i64 layout | **F2 closed as named mechanism** — see `MADAROS_F64_BITCAST_SITOFP_BOUNDARY_2026-08-17.md` |
| f64→i64 param cast bitcast (GUM k95 stuck 1.960) | Same **family** (D1); F2 is the dual sitofp edge on Knowledge.confidence |
| Unclassified scalar → char\* (fable-1) | **Ruled out for F2** — would SEGV, not print 4.6e18 |
| `print_int` garbled after f64 print | Related kind confusion; not the F2 print shape |
| `variance_of` overflow 2^63 on deep chains | Variance slot / depth; F1 is **collapse to 0** under Madaros adaptive |
| E170 / E035 Epistemic effect | Separate surface |

**F2 root (measured 2026-08-17):** `ir_register_knowledge_layout` marks
`confidence` as `is_float=3` (i64) while constructors store f64 epsilon bits;
`pk.confidence * sensitivity` emits sitofp → ~4e18. Minimal witness:
`tests/run-pass/f64_bitcast_boundary_knowledge_conf.sio` (R21/R25). Gate:
`scripts/ci/f64_bitcast_sitofp_boundary_gate.sh`.

**Root fix** for F1/F2 proper is in **Madaros variance slots + Knowledge
confidence field kind** (`self-hosted/`), not dissertation math. This lane does
**not** rebuild Madaros (fleet CPU lock). It makes fabrication **detectable and
fail-closed** at the science surface so dissertation CI cannot green-wash it.

---

## Changes shipped

1. **`rapamycin_epistemic_adaptive.sio`**
   - `main() -> i64`
   - Require `variance_of(c_blood) > 1e-18`
   - On failure: print `FABRICATED_ZERO` / `EPISTEMIC_FABRICATION` and **`return 1`**

2. **`epistemic_pbpk28.sio` TEST 6**
   - Detect `conf > 1.0` and `conf > 1e15` as `EPISTEMIC_FABRICATION`
   - Keep range check; refuse silent “looks like a number”

3. **`scripts/ci/epistemic_fabrication_detect_gate.sh`**
   - Runs both programs under Madaros (+ lean reference for F1)
   - Fails if zero variance PASSes or exits 0
   - Fails if huge confidence prints without fabrication marker / all-pass
   - lean_single must keep non-zero variance (refutes “physics is zero”)

4. This audit document.

---

## Re-measure

```bash
export MADAROS_STACK_KB=524288
./bin/souc run tests/run-pass/rapamycin_epistemic_adaptive.sio; echo rc=$?
# expect rc!=0 and FABRICATED_ZERO on current Madaros

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/run-pass/rapamycin_epistemic_adaptive.sio
# expect PASS rc=0 and var(blood) > 0

./bin/souc run stdlib/darwin_pbpk/epistemic_pbpk28.sio | rg 'TEST 6|FABRICATION|confidence'
# expect EPISTEMIC_FABRICATION on Madaros today

bash scripts/ci/epistemic_fabrication_detect_gate.sh
```

---

## Residual compiler work (handoff)

| ID | Owner suggestion | Acceptance |
|---|---|---|
| F1 variance slots / adaptive GUM under Madaros | compiler lane | adaptive PASS on Madaros with var≈lean |
| F2 `KCONF-BITCAST-SITOFP` (Knowledge.confidence kind) | compiler lane | R21/R25 `KCONF_ALL_OK`; ep28 TEST 6 probability-scale; preserve user i64 `confidence` (#1496) |

Until then, dissertation CI must treat Madaros F1/F2 as **red toolchain**, not
science fail — which this detection gate enforces.

---

*No self-hosted/ edits. No fabricated green.*

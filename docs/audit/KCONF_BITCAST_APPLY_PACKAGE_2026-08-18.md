<!-- docs:meta
topic_id: repo.docs.audit.kconf-bitcast-apply-package-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.kconf-bitcast-apply-package-2026-08-18
-->

# KCONF-BITCAST-SITOFP — ready-to-apply package (2026-08-18)

**Lane:** grok-cli2 / `e170-halo-pgx`  
**Status:** patch prepared **without** writing `self-hosted/ir/lower.sio` (claimed by
`claude--session-b708b85e` / fable-1 CEI). When that lease frees — or the holder
lands the hunk — apply, rebuild Madaros **from source**, then run the gates below.  
**Parent audits:** `MADAROS_F64_BITCAST_SITOFP_BOUNDARY_2026-08-17.md`,
`EPISTEMIC_FABRICATION_DETECT_2026-08-17.md`, `#1792` fail-closed detectors.  
**Sibling:** PR **#1848** (halo E170 test-source) — orthogonal; already green locally.

---

## 0. Mechanism (one sentence)

Constructors store an f64 probability in `Knowledge.confidence`, but
`ir_register_knowledge_layout` marks the field `is_float: 3` (integer). Arithmetic
and compare emit `sitofp` on the IEEE bit pattern → magnitude ~4×10¹⁸.

---

## 1. Patch (do not invent a second fix)

**File:** `docs/audit/repro/kconf_bitcast_sitofp_layout.patch`  
**Target:** `self-hosted/ir/lower.sio` only (`check.sio` not required).

```bash
# From a worktree that HOLDS the lower.sio claim (or after release):
git apply --check docs/audit/repro/kconf_bitcast_sitofp_layout.patch
git apply docs/audit/repro/kconf_bitcast_sitofp_layout.patch
```

**Substance of the hunk:**

| Line class | Before | After |
|---|---|---|
| `fields[2]` Knowledge.confidence | `is_float: 3` | `is_float: 1` |
| Comment above `ir_register_knowledge_layout` | “confidence is i64…” (#1496 freeze) | Honest f64 + why #1496 stays closed |

**Why this does not reopen #1496:**

- `field_is_float_by_name_simple("confidence")` is **unanimous** — mixed float+int
  name → `false`.
- FieldGet already does
  `field_is_float_for_base_ref(...) || field_is_float_by_name_simple(...)`.
- MiniI / Epistemic `confidence: i64` keep `is_float: 3` and resolve via typed base
  (`field_is_int_for_base_ref`). Gate rows **R23** (user f64) and **R24** (user i64)
  must both stay OK.

**Do not:** flip `field_is_float_by_name_simple` alone; do not change MiniEp layout;
do not weaken E170; do not treat lean_single as the claim oracle after the fix
(Madaros source-built is the clock — ADR-008).

**Dry-run receipt (this session):** patch applied cleanly to a **copy** under
`/tmp/kconf-apply-test/`; reserved `self-hosted/ir/lower.sio` in the worktree
left byte-identical (sha256 prefix `af3079cafccfb7d3` at prep time).

---

## 2. Rebuild (instrument trust)

Prebuilt `./bin/souc` does **not** track `self-hosted/` edits. After apply:

```bash
# Prefer Slurm when available. On-pod only through the modular builder
# (self-locks — do NOT wrap in souc-build-lock.sh; flock is not reentrant):
bash scripts/dev/build_modular_madaros.sh
# or: make build-madaros
./bin/souc --version   # must reflect the just-built Madaros
```

Refute instrument: if `git diff` shows the hunk but behaviour is unchanged, you
are still on a stale ELF.

---

## 3. Gate — `f64_bitcast_sitofp_boundary`

```bash
export MADAROS_STACK_KB=524288
bash scripts/ci/f64_bitcast_sitofp_boundary_gate.sh
```

| Arm | Before fix (main today) | After fix (accept) |
|---|---|---|
| CONTROLS R01–R13 | `CONTROLS_ALL_OK` | unchanged |
| USER R23 (f64 conf) + R24 (i64 conf) | `USER_CONF_ALL_OK` | **must remain** (`R23_BITCAST_SITOFP` = fail) |
| KCONF Madaros | `R25_BITCAST_SITOFP` + `EPISTEMIC_FABRICATION` + **rc≠0** | **`KCONF_ALL_OK` + rc=0** |
| KCONF lean_single | `KCONF_ALL_OK`, R25 ~0.66 | unchanged (reference) |

Companion (optional, already fail-closed via #1792):

```bash
bash scripts/ci/epistemic_fabrication_detect_gate.sh
```

---

## 4. Acceptance — `epistemic_pbpk28` TEST 6

### 4.1 What must **not** appear

```text
AUC confidence: 4604219396932172800.000000
EPISTEMIC_FABRICATION: confidence > 1.0 (not a probability)
EPISTEMIC_FABRICATION: confidence looks like f64 bit-pattern-as-integer
FAIL: expected confidence in [0.20, 0.90]
```

That integer is `sitofp` of the IEEE-754 bit pattern of a healthy ~0.671038
probability (measured; see boundary audit §4).

### 4.2 What must appear (oracle measured 2026-08-18)

**Engine oracle for the number:** lean_single on `origin/main` (and this worktree)
today prints:

```text
TEST 6: Confidence in [0.20, 0.90]
  AUC confidence: 0.671038
  [PASS]
```

Full suite under lean_single: `ALL 9 TESTS PASSED`.

**Madaros after fix must match that physics**, not invent a new band.

| Criterion | Value | Notes |
|---|---|---|
| Printed `AUC confidence` | **`0.671038`** | Exact `print_f64` match to lean_single on rapamycin priors |
| Absolute tolerance | **`|conf − 0.671038| < 1e-5`** | Wider than ulp; tight enough to refuse any sitofp-class residue |
| Band (already in source) | `(0.20, 0.90)` | `in_band` in TEST 6 |
| Fabrication flags | **absent** | `conf_gt_one` and `conf_bitpat` both false |
| Process | **rc=0**, `ALL 9 TESTS PASSED` | Do not accept 8/9 with TEST 6 skipped |

### 4.3 Derivation (why 0.671038, not a free parameter)

```text
conf_auc = Σ_c  priors.kn[c].confidence * sensitivity[c]
```

Rapamycin prior ε vector (`ep28_rapamycin_priors`):

| c | param | ε (`c[i]`) |
|---|---|---|
| 0 | cl_hepatic | 0.65 |
| 1 | cl_renal | 0.50 |
| 2 | fu_plasma | 0.72 |
| 3 | kp_brain | 0.63 |
| 4 | kp_liver | 0.60 |
| 5 | kp_kidney | 0.55 |
| 6 | kp_adipose | 0.40 |

`sensitivity[c]` is the normalised fractional contribution of each prior to
`AUC_blood` variance (finite-difference GUM, same run). The weighted sum under
lean_single is **0.671038**. That is the acceptance pin — not an arbitrary
mid-band pick. Micro-probe R25 uses a fixed toy weight
`0.80·0.5 + 0.60·0.3 + 0.40·0.2 = 0.66` (same family, different numbers).

### 4.4 Commands

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
export MADAROS_STACK_KB=524288

# Positive control BEFORE claiming closure (must still fabricate on stale binary):
./bin/souc run stdlib/darwin_pbpk/epistemic_pbpk28.sio | rg 'TEST 6|AUC confidence|FABRICATION'

# After source-built Madaros with patch:
./bin/souc run stdlib/darwin_pbpk/epistemic_pbpk28.sio | rg 'TEST 6|AUC confidence|FABRICATION|ALL 9'
# Expect: AUC confidence: 0.671038  then [PASS]  then ALL 9 TESTS PASSED
```

---

## 5. Time budget (for the lease holder / next writer)

| Step | Wall estimate |
|---|---|
| `git apply` hunk | &lt; 1 min |
| Madaros rebuild from source | 15–40 min (dominates; use Slurm if live) |
| `f64_bitcast_sitofp_boundary_gate.sh` | &lt; 2 min |
| `epistemic_pbpk28` run | &lt; 2 min |
| **Total exclusive lower.sio** | **~30–45 min** after rebuild slot available |

If fable-1 already rebuilds for CEI: landing the hunk in that rebuild is ~5 min
extra edit cost.

---

## 6. Out of scope

- E170 / `halo_pgx_gate_pass` — PR #1848 (test `with Epistemic`).
- rc=182 handle wall — separate dispatch.
- R22 `.epsilon == 0.0` under Madaros — noted in boundary audit; not this flip.
- Weakening Madaros E170 to match lean_single Confidence laxity — forbidden.

---

## 7. AI disclosure

Package authored by AI agent (grok-cli2) under human direction. Numbers:
lean_single `AUC confidence: 0.671038` and Madaros fabrication
`4604219396932172800` re-measured 2026-08-18 on worktree `/workspace/.wt/e170-halo`
at `origin/main` tip. GAIDeT-ICMJE 2025. No fabricated green: patch not applied
to the reserved file in this session.

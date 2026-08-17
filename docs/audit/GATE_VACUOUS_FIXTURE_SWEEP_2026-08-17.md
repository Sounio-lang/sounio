<!-- docs:meta
topic_id: repo.docs.audit.gate-vacuous-fixture-sweep-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.gate-vacuous-fixture-sweep-2026-08-17
-->

# Mechanical vacuous-fixture sweep (2026-08-17)

**Lane:** grok-cli4 / gate-vacuity-sweep  
**Constraints:** `FLEET_CONSTRAINTS.md` — green gate ≠ measurement; positive control required.  
**Taxonomy:** `docs/audit/GATE_UNMEASURE_TAXONOMY_2026-08-17.md` (U3 empty/missing fixtures, U12 positive control, U13 check-count fallacy).  
**Instrument:** `scripts/ci/gate_vacuous_fixture_sweep.{sh,py}`  
**Artifacts:** `artifacts/audit/vacuous_fixture_sweep/`

---

## Positive control (instrument validation)

Planted deliberately vacuous gate:

```text
scripts/ci/fixtures/vacuity_positive_control/vacuous_seed_gate.sh
# nullglob over …/vacuity_positive_control/*.sio  (directory has zero .sio files)
# prints VACUOUS_SEED_GATE_OK cases=0 and exits 0
```

```bash
bash scripts/ci/fixtures/vacuity_positive_control/vacuous_seed_gate.sh
# VACUOUS_SEED_GATE_OK cases=0

export ROOT_DIR=$PWD OUT_DIR=artifacts/audit/vacuous_fixture_sweep
python3 scripts/ci/gate_vacuous_fixture_sweep.py
# positive_control_fired=yes
# VACUOUS …/vacuous_seed_gate.sh  empty_globs=1

bash scripts/ci/gate_vacuous_fixture_sweep.sh
# GATE_VACUOUS_FIXTURE_SWEEP_OK scanned=455 vacuous=6 seed_fired=1
# Exit 2 if seed not flagged (instrument defect).
```

**Control fired.** A census returning zero while this seed is present would be rejected.

---

## Concrete seed: `madaros_visibility_context_gate.sh` (#1702)

| Fact | Measurement |
|---|---|
| Gate on tree / main? | **Yes** (`scripts/ci/madaros_visibility_context_gate.sh`) |
| `ambiguous_public_*.sio` in `tests/compiler/madaros_visibility_context/`? | **ABSENT** (only `duplicate_private_*` + README) |
| Does **main’s** gate source name `ambiguous_public`? | **No** (0 refs; only `duplicate_private_*` + multimodule true-private paths) |
| Sweep classification of this gate | **`non_vacuous`** — `files=5` existing paths |

Existing enumerated inputs (all present):

```text
tests/compiler/madaros_visibility_context/duplicate_private_single_main.sio
tests/compiler/madaros_visibility_context/duplicate_private_18_main.sio
tests/multimodule/visibility_fn_private_main.sio
tests/multimodule/visibility_struct_private_main.sio
tests/multimodule/visibility_enum_private_main.sio
(+ corresponding leaf/lib files for the reducers)
```

**Bounded conclusion:** Issue #1702’s “gate present, ambiguous fixtures absent” is **true as a historical control gap** (`cfdf1b7e0b` never reached main). It is **false** that the **main gate body** currently enumerates those missing files and greens over an empty set. Main’s gate is non-vacuous for the **duplicate_private / true-private** set it actually runs. The missing ambiguous control is taxonomy **U5/U6/U12** (never wired), not U3 empty-glob on a wired path list.

#1702’s caveat that defects were never re-verified on a current-source build remains: this sweep only proves **fixture existence / enumeration**, not checker semantics. Prior positive control of ambiguous-public (when restored) **did fire** as silent bind (`check: OK`, `run_rc=101`) — separate measurement.

---

## Sweep method

Static analysis of every `scripts/ci/*_gate.sh` plus the planted seed:

1. Extract quoted repo-relative paths (`tests/…/*.sio`, etc.).
2. Extract `$FIXTURES/…` / `$ROOT_DIR/…` joins when `FIXTURES=` is assigned.
3. Extract `for … in …*.ext; do` globs (no `**`, no absolute escapes).
4. Classify:
   - `non_vacuous` — ≥1 extracted path exists as a file  
   - `zero_files_empty_or_missing` — patterns found, **zero** existing files  
   - `no_fixture_patterns` — no static fixture-like paths (dynamic-only gates)

**Limits (honest):** does not execute gates; does not expand all bash arrays; may false-positive on odd string shapes (`examples/render/../../…`, generated `$WORK` paths); may false-negative dynamic discovery. Positive control proves the empty-glob arm.

---

## Census (this worktree, 2026-08-17)

```text
scanned_gates=455
vacuous_gates=6   (includes 1 planted seed)
non_vacuous_with_files≈213
no_fixture_patterns≈236
positive_control_fired=yes
```

### Vacuous table (mechanical + human triage)

| Gate | Claims (header) | Enumerates today | Count | Verdict |
|---|---|---|---:|---|
| **vacuous_seed_gate.sh** (planted) | Deliberate empty glob | `…/vacuity_positive_control/*.sio` | **0** | **VACUOUS (control)** |
| **kretikos_kaxi_iso_budget_gate.sh** | KAXI ISO budget | `benchmarks/pbpk/gum_budget.csv` | **0** (file missing; dir has `hessian_budget.csv` etc.) | **VACUOUS / stale path** |
| **kretikos_kaxi_phase_z_assoc_gate.sh** | Phase Z assoc GUM PTX | `tests/golden/kaxi_ptx/f32_assoc_gum/pbpk_8comp_assoc.ptx` | **0** missing | **VACUOUS / missing golden** |
| madaros_f128_f256_format_identity_gate.sh | f128/f256 format identity | mis-parsed `examples/render/../../tests/…` | 0 as written; **real** paths under `tests/compile-fail/f128_*` **exist** | **FALSE POSITIVE** (path normalise gap) |
| madaros_wide_int_gate.sh | i128/i256/u128/u256 | generates `$WORK/*.sio` at runtime | 0 static | **FALSE POSITIVE** (synthetic fixtures) |
| stdlib_evolution_gate.sh | stdlib evolution docs+tests | bare `SOUNIO_STYLE_GUIDE.md`; real file is `docs/guide/SOUNIO_STYLE_GUIDE.md`; many `tests/stdlib/…` paths exist in body | mixed | **FALSE POSITIVE** on bare name; **re-check** if those stdlib tests resolve |

**High-confidence real holes after triage: 2 gates + 1 planted control.**  
**Instrument health: PASS (seed fired).**

Full machine table:  
`artifacts/audit/vacuous_fixture_sweep/all_gates_fixture_status.tsv`  
`artifacts/audit/vacuous_fixture_sweep/vacuous_gates.tsv`

---

## How this maps to the four fleet unmeasure modes

| Mode | This sweep |
|---|---|
| Empty fixture / missing input (U3) | Primary target — flags zero-match globs/lists |
| Suite ABSENT (U5) | Visibility ambiguous fixtures never on main |
| Abort-before-evaluate (U4) | Out of scope (needs runtime) |
| Defective completeness observer (U13) | Out of scope (Impact/CI Decision) — see taxonomy doc |

---

## Commands to re-run

```bash
# Planted control alone
bash scripts/ci/fixtures/vacuity_positive_control/vacuous_seed_gate.sh   # cases=0

# Full mechanical sweep (fails if control not caught)
bash scripts/ci/gate_vacuous_fixture_sweep.sh

# Persist
GATE_VACUOUS_SWEEP_DIR=artifacts/audit/vacuous_fixture_sweep \
  bash scripts/ci/gate_vacuous_fixture_sweep.sh
```

---

## Non-claims

- Does not prove any gate’s semantic correctness under Madaros.
- Does not re-litigate #1702 checker defects without a current-source build.
- Does not wire the sweep into CI yet (opt-in instrument).
- False positives remain possible; triage column is required before filing bugs.

---

## Deliverables (`ls -la`)

```text
scripts/ci/gate_vacuous_fixture_sweep.sh
scripts/ci/gate_vacuous_fixture_sweep.py
scripts/ci/fixtures/vacuity_positive_control/vacuous_seed_gate.sh
scripts/ci/fixtures/vacuity_positive_control/README.md
docs/audit/GATE_VACUOUS_FIXTURE_SWEEP_2026-08-17.md
artifacts/audit/vacuous_fixture_sweep/{all_gates_fixture_status.tsv,vacuous_gates.tsv,status.env}
```

*Halt boundary: mechanical existence/enumeration only. Next step if desired: fail-closed CI wiring for the two high-confidence missing goldens, and normalise `..` path segments in the sweeper.*

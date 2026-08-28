# Lane 8c — Regulatory Dossier Generator (dissertation contribution #3 wrapper)

**Owner:** Kimi 2.5 (offload). Reviewer: Claude B.
**Branch:** `coord/lane-8c-dossier` off `origin/main` (91d48adb).
**Worktree:** `/workspace/sounio-lane-8c-dossier`.
**Companion contract:** `.claude/PARALLEL_BLOCKER_CONTRACT.md`.

## Why

`benchmarks/pbpk/gum_budget.csv` (Phase Y, ISO 17025 layout) is a machine artifact. Submission to a regulator (or to a thesis examiner) wants a **narrative wrapper**: who/what/when, the model card, the parameter-priors table, the validation evidence, and the final ISO uncertainty budget rendered in human-readable form. This lane delivers a Sounio-driven generator that:

- ingests `benchmarks/pbpk/gum_budget.csv` (Phase Y) + optional `benchmarks/pbpk/hessian_budget.csv` (Lane 8a, if present)
- ingests Phase J pass/fail markers from `scripts/ci/kretikos_kaxi_phase_j_gate.sh` log
- ingests clinical reference rows from `stdlib/darwin_pbpk/validation/rapamycin_clinical.sio` outputs (read-only)
- renders one Markdown dossier `artifacts/dissertation/dossier_rapamycin.md` matching `docs/dissertation/dossier_template.md`

This is mostly text-glue. Numerics are read, not re-computed. PDF rendering deferred (template stays Markdown so a `pandoc` pass is one user command later).

## CLAIM (announce in `artifacts/omega/agent_handoff.log.md`)

NEW files only:

- `scripts/dissertation/dossier_generator.sio` — main generator.
- `docs/dissertation/dossier_template.md` — section skeleton with `{{placeholder}}` slots.
- `scripts/ci/dissertation_dossier_gate.sh` — runs generator + diffs against golden.
- `tests/run-pass/dossier_smoke.sio` — smoke test that exercises the generator's CSV-read + template-fill paths against in-tree minimal fixtures.
- `tests/golden/dissertation/dossier_rapamycin_snapshot.md` — golden snapshot for diff.

**Disjoint check:**
- Lane 1, 2, 3, 4, 5, 7, 8a, 8b file sets do not intersect any of the above.
- `docs/dissertation/` and `scripts/dissertation/` and `tests/golden/dissertation/` are new directories owned exclusively by this lane.

## Acceptance gate

`bash scripts/ci/dissertation_dossier_gate.sh` rc=0. Must check:

1. `bin/souc check scripts/dissertation/dossier_generator.sio` rc=0.
2. `bin/souc compile tests/run-pass/dossier_smoke.sio -o /tmp/dossier_smoke && /tmp/dossier_smoke` final line `PASS dossier_smoke`.
3. Generator runs against committed minimal fixtures (encode them inline in the smoke test) and produces output identical to `tests/golden/dissertation/dossier_rapamycin_snapshot.md` modulo a single `generated_at_utc` line which the gate strips before diff.

## Template required sections (§ headings exact)

```
# PBPK Dossier — Rapamycin (Sirolimus)

§1. Subject of submission
§2. Model card (PBPK14 + DES coupling)
§3. Parameter priors (CL, Vd, fu, Kp_brain, ...)
§4. Numerical method (Tsit5, tolerance, step bounds)
§5. ISO 17025 GUM budget (1st order)             ← from gum_budget.csv
§6. ISO 17025 GUM budget (2nd order, if present)  ← from hessian_budget.csv
§7. Confidence gate evidence (Phase J)            ← from phase_j log line
§8. Clinical validation                            ← from validation/rapamycin_clinical.sio outputs
§9. Audit trail (commit SHAs, generated_at_utc, sounio version)
```

## Sounio dialect pins

Read first: `docs/guide/SOUNIO_QUICK_START.md`, `docs/guide/SOUNIO_GOTCHAS.md`, `sounio_llm_training.md`.

This lane is heavy on **string formatting** and **CSV parse**. Two Sounio gotchas to respect:

- For string concat use the `+` overload only on `String`, not on string literals — convert with `to_string` first.
- CSV parse: prefer the existing `stdlib/csv/lib.sio` if present; otherwise hand-roll a split-on-comma loop. Do NOT pull `regex` for this — overkill.

`effect io` is required on the file-read and file-write paths.

## Reuse (read-only)

- `benchmarks/pbpk/gum_budget.csv` — input; format is row-per-source with columns `(source, type_a_b, u_i, c_i, contribution_pct)`.
- `stdlib/darwin_pbpk/validation/rapamycin_clinical.sio` — read clinical reference values.
- `scripts/ci/kretikos_kaxi_phase_j_gate.sh` log — parse last line for pass/fail.
- `bin/souc` — for `souc check` on the generator.

## Procedure

1. CLAIM record (lane=8c) in `artifacts/omega/agent_handoff.log.md`.
2. Land template + smoke test first (smallest churn). Capture golden snapshot.
3. Implement generator (~300-500 LoC: 9 section renderers, CSV parser, template filler).
4. Wire gate.
5. Commit prefix `[lane-8c]`. PR title `[lane-8c] dissertation regulatory dossier generator (Markdown)`.

## Blocker shape

`BLK-20260510-lane8c-<slug>` per contract. Most likely class: `evidence-gap` (CSV input drift) or `compiler-semantics` (string-handling edge case in Sounio).

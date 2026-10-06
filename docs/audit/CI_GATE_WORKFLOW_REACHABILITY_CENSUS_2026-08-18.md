<!-- docs:meta
topic_id: repo.docs.audit.ci-gate-workflow-reachability-census-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ci-gate-workflow-reachability-census-2026-08-18
-->

# CI gate workflow reachability census

**6/6 never was a CI signal (measured at `64924d371a`; at `12ebda238d` two of the six are already in Contracts).**

**Date:** 2026-08-18  
**SHA measured (pass 1):** `465008a76b`  
**SHA measured (pass 2 leftover class):** `64924d371a`  
**SHA that expired the present-tense 6/6 leftover claim:** `12ebda238d` (#1880)  
**Live leftover after #1880 + #1893 (this merge / `origin/main`):** reachable **91** / leftover **380** (population 471). The umbrella-closure **25** is unchanged — that walk starts at the umbrella, not at `sigpipe_hygiene_gate.sh`.  
**Instrument:** `python3 scripts/dev/ci_gate_workflow_reachability.py`  
**Table:** [`CI_GATE_WORKFLOW_REACHABILITY_CENSUS_2026-08-18.tsv`](CI_GATE_WORKFLOW_REACHABILITY_CENSUS_2026-08-18.tsv)

This is a **census**, not a cleanup. Nothing was wired. Wiring the leftover set in one pull request would break CI and would repeat the #1778 failure mode in reverse: a green tick that does not mean the intended surface ran.

## Why 390 is not a fact

A basename grep of `.github/` finds **78** of **468** `scripts/ci/*_gate.sh` files and therefore **390** unmentioned names. That 390 is an **upper bound on direct orphans**, not a count of gates the CI never runs:

- a workflow can invoke a gate only through another script (umbrella, `madaros_full_gate.sh`, `package_pbpk_gum_gate.sh`);
- `scripts/ci/gate_vacuity_gate.sh` **lists** every `*_gate.sh` via `git ls-files` and does **not** execute them — treating that list as reachability would zero the leftover set and invent a green;
- a name can appear in a comment (`madaros_corpus_regression_gate.sh` is named in `ci.yml` as *deliberately NOT wired*).

Saying “390 orphans” as a fact is the same class of error the day spent closing: a plausible number where an absence should have been declared.

## Instrument (validated before the count)

| Control | What would refute the instrument | Result |
|---|---|---|
| Reachable set non-empty | `workflow_reachable_transitive = 0` | **85** |
| Named direct orphans stay unreachable | any of the three #1780-family gates appears in the invoke graph | all three **unreachable** |
| Vacuity lister is not an invoker | leftover collapses toward 0 because `gate_vacuity_gate.sh` is in `ci.yml` | leftover stays **383** |

Regenerate:

```text
python3 scripts/dev/ci_gate_workflow_reachability.py
```

Roots are `.github/workflows/*.yml` only. Edges are real invocations (`bash`/`sh`/`source`, YAML `run:`, `for g in scripts/ci/…_gate.sh` continuation lines, and `make <target>` recipes when a workflow actually runs that target). Comments and `git ls-files` / `find` inventories are not edges.

The only `make` a workflow runs today is `make build-madaros` (`madaros-prebuilt-refresh.yml`). That target does not pull the `make check` gate forest. Makefile-only gates are therefore **not** workflow-reachable.

## Measured totals

| Quantity | N | What it is |
|---:|---:|---|
| `scripts/ci/*_gate.sh` | 468 | population |
| Named in `.github/` (any file, including comments) | 78 | the grep that produced 390 |
| Mention-only leftover upper bound (468−78) | 390 | **not** the leftover |
| Workflow-reachable after transitive invoke | **85** | includes 8 gates never named in `.github/` |
| Named in `.github/` but not invoked | 1 | `madaros_corpus_regression_gate.sh` (comment only) |
| Leftover (468−85) | **383** | not reachable from any workflow at any depth |

Eight gates are reachable **only** transitively (not named in `.github/`):

| Gate | Via |
|---|---|
| `generated_ontology_gate.sh` | `run_ontology_validation.sh` |
| `madaros_blocker_contract_gate.sh` | `claude_operational_contract_gate.sh` |
| `madaros_global_capacity_gate.sh` | `madaros_current_source_f64_lowering_gate.sh` |
| `madaros_global_f64_scratch_gate.sh` | same |
| `madaros_imported_call_arity_13_gate.sh` | same |
| `madaros_imported_capacity_gate.sh` | same |
| `madaros_imported_deref_f64_array_gate.sh` | f64-lowering + `madaros_full_gate.sh` |
| `package_import_science_gate.sh` | `package_pbpk_gum_gate.sh`, `sounio_package_support_gate.sh` |

## Leftover classification (383)

Buckets are **evidence-only**. A leftover without a header or an operator entry point stays `unclassified`. Forcing 317 names into “obsolete” would be another fabricated zero.

| Class | N | Rule |
|---|---:|---|
| `forgotten` | 10 | intended as a CI measurement (named orphan, or header `GATE_CONTRACT` / “positive control”) and no operator entry (Makefile / umbrella / dissertation / Slurm) |
| `manual-by-design` | 56 | dissertation_* , `bootstrap_chain_gate.sh`, or invoked from Makefile / `native_v2_cpu_compiler_umbrella_gate.sh` (operator / Slurm / `make check`) |
| `obsolete` | 0 | header comment says obsolete / superseded / do not use — **none matched** after excluding code-token hits |
| `unclassified` | 317 | leftover, no evidence for the three buckets above |

### Forgotten (10) — #1778 shape

A gate that can fail its own positive control, and that no workflow runs, is indistinguishable from a gate that passes. That is the #1778 contract.

| Gate | Why forgotten, not manual |
|---|---|
| `epistemic_measure_correspondence_gate.sh` | #1780 Lean↔checker correspondence; positive controls in-file; **not** in any workflow |
| `epistemic_fabrication_detect_gate.sh` | silent zero-variance / out-of-range confidence; `GATE_CONTRACT`; **not** in any workflow |
| `madaros_ontology_enforcement_gate.sh` | inverse-role witnesses + valid control; **not** in any workflow |
| `f64_bitcast_sitofp_boundary_gate.sh` | F2 confidence fabrication; `GATE_CONTRACT` |
| `madaros_f128_f256_ladder_gate.sh` | V0-B must FAIL under V0-A; positive control named |
| `madaros_f128_f256_v0c_wire_gate.sh` | same ladder family, `GATE_CONTRACT` in header |
| `madaros_f128_f256_v0d_softfloat_gate.sh` | same |
| `madaros_print_f64_negative_gate.sh` | residual #890; positive control `+2.0` |
| `mli_s3_bit_identity_gate.sh` | S3 O5 bit-identity; positive control on IEEE 1.0 / `ret` |
| `stdlib_source_byte_ceiling_gate.sh` | E229 wall; must FAIL if a file exceeds CAP |

Adjacent, not auto-bucketed: `ci.yml` line 515 says `madaros_corpus_regression_gate.sh` is **deliberately NOT wired**. That is a declared absence, not a silent one. It remains leftover.

### Manual-by-design (56) — includes the six dissertation gates and bootstrap

6/6 never was a CI signal (measured at `64924d371a`; at `12ebda238d` two of the six are already in Contracts).

All six dissertation gates are leftover **on the measurement SHA** (`64924d371a` / this census). After `12ebda238d` (#1880) `dissertation_confidence_gate_gate.sh` and `dissertation_frontend_parity_gate.sh` are in Contracts. The other four stay leftover and stay red:

- `dissertation_confidence_gate_gate.sh` — leftover on `64924d371a`; Contracts as of `12ebda238d`
- `dissertation_dossier_gate.sh` — leftover on both SHAs (red)
- `dissertation_frontend_parity_gate.sh` — leftover on `64924d371a`; Contracts as of `12ebda238d`
- `dissertation_pbpk28_parity_gate.sh` — leftover on both SHAs (red)
- `dissertation_pbpk_hessian_gate.sh` — leftover on both SHAs (red)
- `dissertation_pbpk_suite_gate.sh` — leftover on both SHAs (red; umbrella only)

`dissertation_pbpk_suite_gate.sh` **is** called by `native_v2_cpu_compiler_umbrella_gate.sh`. The umbrella is **not** in any workflow (handoff / Slurm / operator). So the suite is operator-reachable and workflow-unreachable. That is manual-by-design, not forgotten.

`bootstrap_chain_gate.sh` is the same shape: local bootstrap proof, no workflow edge.

The rest of the 56 are Makefile `check` / ontology / knowledge-runtime / suffering-aware / umbrella children. They have an operator entry. They do not have a GitHub Actions entry.

### Unclassified (317)

Largest name-prefixes: `madaros_*` (43), `native_*` (34), `fo_*` (29), `kretikos_*` (24), then semantic / mercyful / particle / moonshot / proof-carrying. This census does **not** call them obsolete. A second pass can split them once each header is read; doing that from the filename would fabricate a class.

## What this does not authorise

- Wiring the 383 leftovers.
- Wiring even the 10 forgotten gates in this change-set — each needs its own cost, skip rules, and a demonstrated positive control in CI, not a bulk `for g in scripts/ci/*_gate.sh`.
- Treating `make check` as CI. It is not, until a workflow runs that target.

## Next-Command (if a later lane wires)

Start with the three already shown to fire their positive controls and to be absent from every workflow:

1. `epistemic_measure_correspondence_gate.sh`
2. `epistemic_fabrication_detect_gate.sh`
3. `madaros_ontology_enforcement_gate.sh`

One gate per pull request. Re-run this instrument after each wire; the forgotten count must fall by exactly one, and the reachable count must rise by exactly one.

---

## Pass 2 — leftover class (measured 2026-08-18, SHA `64924d371a`)

**Instrument:** `python3 scripts/dev/ci_gate_leftover_class_pass2.py`  
**Table:** [`CI_GATE_LEFTOVER_CLASS_PASS2_2026-08-18.tsv`](CI_GATE_LEFTOVER_CLASS_PASS2_2026-08-18.tsv)

Pass 1 counted leftovers. This pass classifies them. Filename prefixes are not a class. A quoted historical phrase is not a class.

Population on this SHA: **470** `*_gate.sh`, **88** workflow-reachable, **382** leftover. (Pass 1 was 468 / 85 / 383 on `465008a76b`. Wires since then, plus two new gates, plus a scan-list false-positive removed — see below.)

### Instrument corrections before the class count

| Control | What would refute it | Result |
|---|---|---|
| Leftover non-empty | leftover = 0 | **382** |
| Six dissertation gates leftover | any of the six in the invoke graph | all six **unreachable** on `64924d371a`. After `12ebda238d` (#1880) confidence + frontend_parity are in Contracts; the other four stay leftover |
| Six unmentioned in `.github/` | any dissertation `*_gate.sh` string under `.github/` | **zero** mentions; `git log -S dissertation_pbpk_suite_gate -- .github/` is empty for the whole history |
| Scan-list is not an invoke | `mli_s3_bit_identity_gate.sh` reachable because `sigpipe_hygiene_gate.sh` lists it | leftover (listed for `grep -q` hygiene, never executed) |
| Quoted SUPERSEDED is not obsolete | `lean_single_fixed_point_gate.sh` class = obsolete | **not** obsolete — the header quotes a *wrong* old line |

`BARE_GATE_RE` now applies only to workflow YAML. Applying it to `.sh` files counted scan-lists as execution. Pass 1 did not hit that on `465008a76b` because `mli_s3` was not yet in the sigpipe list.

### Leftover classes (382)

| Class | N | Rule |
|---|---:|---|
| `forgotten` | **36** | Header is a measurement contract (`GATE_CONTRACT`, “positive control”, “evidence gate”, “Acceptance:”, “HARD GATE/PATH”) and there is no live operator entry |
| `manual-by-design` | **29** | Makefile recipe, `native_v2_cpu_compiler_umbrella_gate.sh` child, `bootstrap_chain_gate.sh`, or header says hand/Slurm/operator |
| `obsolete` | **0** | Current header *asserts* the gate is obsolete / do-not-use. Quoted “old header said SUPERSEDED” does not count |
| `unclassified` | **317** | Leftover, no evidence for the three buckets. Same size as pass 1 — the extra forgotten came out of the old filename-`dissertation_*` / coarse manual bucket, not from inventing a class for 317 names |

Do not call the 317 obsolete. A third pass that reads every body (not just the header) can split them; doing it from `madaros_*` / `fo_*` would fabricate a class.

Pass 1 put all six dissertation gates in `manual-by-design` because the name started with `dissertation_`. That was a filename rule. Pass 2 drops it.

### The six dissertation gates — CI never ran them (on `64924d371a`)

6/6 never was a CI signal (measured at `64924d371a`; at `12ebda238d` two of the six are already in Contracts).

| Gate | Pass-2 class on `64924d371a` | Workflow-reachable on `64924d371a` | After `12ebda238d` |
|---|---|---|---|
| `dissertation_confidence_gate_gate.sh` | forgotten | no | Contracts (#1880) |
| `dissertation_dossier_gate.sh` | forgotten | no | leftover, red |
| `dissertation_frontend_parity_gate.sh` | forgotten | no | Contracts (#1880) |
| `dissertation_pbpk28_parity_gate.sh` | forgotten | no | leftover, red |
| `dissertation_pbpk_hessian_gate.sh` | forgotten | no | leftover, red |
| `dissertation_pbpk_suite_gate.sh` | manual-by-design | no | leftover, red; still only via leftover umbrella |

The umbrella is itself leftover (not in any workflow). So the suite is operator-reachable if someone runs the umbrella by hand, and still **not** a GitHub Actions signal.

`git grep` on `64924d371a:.github` finds no `dissertation_*_gate`. A `-S` history walk of `.github/` for `dissertation_pbpk_suite_gate` is empty through that SHA: the suite was never added to a workflow and later removed. It was never there. After `12ebda238d` the confidence and frontend_parity names are in Contracts; the other four still are not.

June qualification prose called these “6/6 dissertation CI gates green”. The 2026-08-16 remasure (`docs/audit/DISSERTATION_DOSSIER_RESOLUTION_DISPATCH_2026-08-16.md`) on `6f2c4e2461` via **Slurm** (job 9908) was 1 pass / 3 fail / 2 unmeasured — not Actions. The 2026-08-17 suite remasure is 19 fail / 53 (`docs/audit/DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-17.md`): twelve toolchain, seven rc=182, zero science. Actions never saw either number.

Two further vacuous-green paths, unused by CI only because CI never calls them:

- `SOUNIO_DPS_GATE_SKIP=1` makes `dissertation_pbpk_suite_gate.sh` exit 0
- `dissertation_confidence_gate_gate.sh` exits 0 on non-Linux / non-x86_64

**Verdict:** yes, **on `64924d371a`**. 6/6 never was a CI signal (measured at `64924d371a`; at `12ebda238d` two of the six are already in Contracts — confidence and frontend_parity, both green, `SOUNIO_GATE_SOUC` pinned). The other four (dossier, hessian, pbpk28, suite) stay leftover and stay red. A numeral without a SHA expires the moment a later wire lands; this one will not.

Live leftover after that wire and the #1893 scan-list fix: **91** reachable / **380** leftover. The umbrella-closure **25** does not move — that walk does not start at sigpipe.

This pass does **not** authorise wiring the remaining four, the 36 forgotten, or the 317 unclassified. The suite is a 53-entry compile/run forest; putting it in Contracts while red would redden every PR the way a V0-A-red f128 ladder would.

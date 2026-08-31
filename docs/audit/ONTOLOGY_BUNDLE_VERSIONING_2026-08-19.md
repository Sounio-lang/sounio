<!-- docs:meta
topic_id: repo.docs.audit.ontology-bundle-versioning-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-27
validated_by: lane/minimax-cli2/ontology-versioning-2026-08-19 (re-measured on rebase by claude-2)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ontology-bundle-versioning-2026-08-19
-->

# Ontology bundle versioning - registers the source-of-truth release per bundle

**Filed:** 2026-08-19 - **Lane:** minimax-cli2 (relocated to /workspace/.wt/minimax-cli2/ontology-versioning-2026-08-19/) - **Status:** red-before / green-after proven; ci.yml proposal deferred per fleet coordination.

## Semantic declaration (required before self-hosted changes)

The 6-dimension declaration this work rests on:

- **Concept-IDs.** SOUNIO-ONTOLOGICAL-VALIDATION (#1967, **merged
  2026-08-19**; `docs/internal/concepts/ontological-validation.md`) argues that
  types are ontological terms - `CHEBI:9168` references the entity "rapamycin",
  a subclass of macrolide, and `disjoint` refuses invalid sums. If that is so,
  the version of the bundle is part of the meaning of any program that names
  one. The existing manifest at `stdlib/ontology/generated/MANIFEST.tsv` already
  counts what the bundle HAS (class_count, const_count, disjoint_count); this
  work adds what the bundle CAME FROM (upstream_release, fetched_at, source_uri,
  source_sha256). The same concept from PR #1967 is what motivates the
  provenance columns; the work here is the registry it needs to be
  registrable.

- **Intent-Preserved.** The founder's 2026-08-19 decision is that types are
  ontological terms and the version of the bundle is part of the meaning of
  the program. The intent is unchanged: a `CHEBI:9168` today and a `CHEBI:9168`
  tomorrow should be referentially traceable to the same release. This work
  preserves that intent by making the release registrable; it does not yet
  decide what to DO when a program is found to disagree with the recorded
  release (refuse, warn, reverify - deferred to the founder).

- **Transformation.** The existing MANIFEST.tsv is extended in-place with four
  columns at the end of the header:
  `upstream_release`, `fetched_at`, `source_uri`, `source_sha256`. The
  gate (`scripts/ci/generated_ontology_manifest_gate.sh`) is extended to
  enforce the new columns and to print, on every passing run, the list of
  bundles with any UNKNOWN column. No new mechanism - per the founder's
  directive (no second mechanism: extend the existing manifest).

- **Claims-Introduced.** (i) The 9 currently shipped bundles (alg, chebi, go,
  hpo, loinc, part, phys, qm, snomed) all declare UNKNOWN - none is sourced
  from a known upstream community release today; the in-tree
  `source/<bundle>_slice.json` files all carry the placeholder
  `version: "fixture-2026.03"`, which is a local label, not a release
  identifier. (ii) The `source_sha256` of each bundle matches the live file
  at gate time (drift detection); the hash is recorded and re-verified. (iii)
  UNKNOWN is a legitimate value and the gate passes with all 9 marked
  UNKNOWN, but every passing run prints the list - same visibility pattern
  as Reserved-Since and Evidence-Does-Not-Count.

- **Claims-Forbidden.** This PR does NOT introduce pinning semantics. Programs
  do not yet declare "this build uses bundle X at version Y"; the manifest
  records the world's version, not the program's pin. Deciding whether the
  program should refuse / warn / reverify on bundle drift is OUT OF SCOPE for
  this PR; it remains a founder decision. This PR also does NOT enforce
  plausibility: the gate does not classify "ChEBI release 227 (claimed)" as
  distinguishable from "ChEBI release 227 (real)" - author discipline plus
  visibility is the corrective mechanism, not the gate. This is documented
  so a future PR does not assume plausibility is enforced.

- **Authoritative-Only-If.** **Hypothesis-grade** until all 9 bundles are
  either sourced from a real upstream release OR an act is recorded that
  commits to treating fixtures as authoritative for the program's meaning.
  Today every bundle is UNKNOWN, so the registry is consistent with
  reality - but that means the language is, today, *not* strongly typed in
  the sense #1967 argues for. A future PR that imports a real upstream
  ChEBI release will lift this hypothesis to theorem-grade.

## What changed

| File | Change |
|---|---|
| `stdlib/ontology/generated/MANIFEST.tsv` | Header extended with 4 columns; all 9 bundle rows filled with UNKNOWN on the first three columns and a real SHA-256 on `source_sha256`. (10 lines, 17 cols.) |
| `scripts/ci/generated_ontology_manifest_gate.sh` | `EXPECTED_HEADER` extended; new `assert_sha256` and `assert_optional_unknown_or_nonempty` helpers; per-row check for upstream_release / fetched_at / source_uri / source_sha256; drift detection against the live source file; visibility list of unknown bundles printed on every passing run. |

## Column design (justification per column)

| Column | Type | Allowed values | Why this name |
|---|---|---|---|
| `upstream_release` | string | `UNKNOWN` or a release identifier (`ChEBI release 227`, `GO 2026-03-01`, `HPO v2026-01-16`, `LOINC 2.77`, ...) | The release the bundle corresponds to. Not the bundle's own version, not the import date. |
| `fetched_at` | YYYY-MM-DD or `UNKNOWN` | ISO 8601 date, UTC, no time zone | When the source was obtained from upstream. Required format rejected at the gate. |
| `source_uri` | http(s) URL or `UNKNOWN` | Canonical upstream URI | The URL the source was obtained from. The in-tree path is reconstructed from `bundle`; the URI is the upstream record. |
| `source_sha256` | 64-char lowercase hex | Real hash only - `UNKNOWN` is rejected | SHA-256 of `source/<bundle>_slice.json` - drift is caught at gate time. |

The expected examples (`ChEBI release 227`, `GO 2026-03-01`, `HPO v2026-01-16`,
`LOINC 2.77`) are quoted verbatim in the column header comment inside the
gate so future authors see the expected shape when filling in real bundles.

## UNKNOWN policy

The founder's prime directive is: do not invent a plausible number - that
is exactly the error this work is meant to surface. `UNKNOWN` is therefore
the honest answer when no upstream record exists, and the gate accepts it
as legitimate.

Visibility pattern (per the founder): on every passing run the gate prints
the list of bundles with any UNKNOWN column, regardless of whether the
gate passes or fails. No expiration, no age-based blocking - the visibility
*is* the constraint. This is consistent with Reserved-Since and
Evidence-Does-Not-Count, which the founder referenced by name.

`source_sha256` is the one column where UNKNOWN is rejected outright: the
source file is always in the tree, so a real SHA-256 is always computable.
An unknown hash there is a different defect class (drift or missing
source).

## Phase D proof: red-before / green-after

Evidence captured 2026-08-19 on this lane. Three scenarios verified live,
each followed by a revert to green.

**RED-A - column-count envelope refused.**
Tampered row 3 (`chebi`) column 14 to empty (not UNKNOWN). Gate output:
```
FAIL  manifest row has 16 columns, expected 17:
        stdlib/data/data/ontology/bundles/chebi.dontology
        stdlib/ontology/generated/chebi.sio
        CHEBIGenerated
        64
        64
        64
        0
        64
        classes+subclass+numeric-constants+no-disjointness
        tests/run-pass/ontology_generated_chebi_types.sio
        tests/compile-fail/ontology_generated_chebi_reject.sio
        stdlib/ontology/typed/chebi.sio
        tests/run-pass/ontology_typed_bridge_chebi.sio
        UNKNOWN         # 14 - dropped because empty
        UNKNOWN         # 15
        5988a467338254244853649b49e7d53ee1489255e994e2d47ab169f68630e1b0
exit=1
```
This is the column-count envelope: when a provenance value is empty, the
row collapses to 16 columns and the gate refuses. Reverted to green.

**RED-B - drift detection refused.**
Tampered row 3 column 17 (`source_sha256`) to a fictional hash
(`deadbeef...beef`). Gate output:
```
FAIL  source_sha256 drift for chebi: manifest=deadbeef0000000000000000000000000000000000000000000000000000beef actual=5988a467338254244853649b49e7d53ee1489255e994e2d47ab169f68630e1b0
Why: the in-tree source file changed; the manifest must be refreshed
     before any release-grade claim is made about this bundle.
exit=1
```
This is the drift detector: when the recorded hash disagrees with the live
file, the gate refuses with the actual hash so the author knows what to
update to. Reverted to green.

**GREEN - UNKNOWN across all 9 bundles accepted.**
Restored manifest with all 9 bundles UNKNOWN on `upstream_release`,
`fetched_at`, `source_uri` and a real SHA-256 on `source_sha256`. Gate:
```
Generated ontology manifest gate passed.
This proves the stable public manifest covers the nine generated bundles,
their stubs, positive/negative witnesses, typed bridges, declared
class/const/disjoint limits, and upstream-source provenance, without
making Python part of PL reasoning.

Bundles with UNKNOWN upstream provenance (visibility, not failure):
  - alg
  - chebi
  - go
  - hpo
  - loinc
  - part
  - phys
  - qm
  - snomed
These bundles have no recorded upstream community release. UNKNOWN is
legitimate per founder directive (no plausible-number invention), but the
list above is always printed - same visibility pattern as Reserved-Since
and Evidence-Does-Not-Count. No expiration, no age blocking.
exit=0
```

## ci.yml re-link - PROPOSAL ONLY, NOT EDITED

The dispatch identifies this gate as one of fourteen that no workflow
names. PR #1967, as filed on 2026-08-19, documented the catalog as
17 ontology-related gates, 3 named by any workflow, 14 never run
including `generated_ontology_manifest_gate.sh`. The dispatch instructs
that this PR coordinate with the lane measuring the 14 (currently
grok-cli3) and that no other gate be linked in this PR.

**Re-measured 2026-08-27 on `origin/main` @ `055825a3f9`** (rebase of this
PR). Two corrections to the sentence above, neither of which changes what
this PR does:

1. **The totals moved.** There are now **19** `scripts/ci/*ontolog*.sh`
   gate scripts, of which **5** are named by a file under
   `.github/workflows/` (`madaros_ontology_enforcement_gate.sh`,
   `ontology_cli_smoke_gate.sh`, `ontology_frontiers_gate.sh`,
   `ontology_multi_ontology_gate.sh`, `run_ontology_validation.sh`). The
   17/3 pair is stale; the **14 unnamed is unchanged**, and
   `generated_ontology_manifest_gate.sh` is still one of them —
   `git grep -c generated_ontology_manifest_gate -- .github/workflows/`
   returns nothing. The proposal-only framing of this section therefore
   still holds.

2. **"Never run" is withdrawn upstream and is withdrawn here.** The merged
   #1967 concept document
   (`docs/internal/concepts/ontological-validation.md`) states that the
   count measures **direct invocation** — whether a workflow names the
   script — not **coverage**, and that the wording "fourteen never run"
   "claimed more than the instrument supported and is withdrawn". The
   correct statement is that fourteen gates, this one among them, are
   **named by no workflow**; whether a running parent covers any of them
   is **unmeasured**.

Per the same pattern as lane/minimax-cli1 (#1961, exactly analogous),
this PR proposes the ci.yml diff but does not edit `ci.yml`. The
proposal:

```diff
       - name: Docs registry
         run: bash scripts/dev/check_docs_registry.sh
+      # Records upstream provenance for every ontology bundle. UNKNOWN
+      # is a legitimate value (per founder directive, no plausible-number
+      # invention) and does NOT cause failure; the gate prints the list of
+      # bundles with unknown provenance on every passing run. Drift between
+      # the recorded and live source_sha256 IS a hard failure.
+      - name: Generated ontology manifest (with provenance)
+        run: bash scripts/ci/generated_ontology_manifest_gate.sh
       # Concept contracts must declare Status and not drift past/behind
       # bindings.tsv evidence (both directions). No skip escape.
       - name: Concept status ladder (bindings evidence)
         run: bash scripts/ci/concept_status_gate.sh
```

The gate is independent of the workflow edit; either party can land the
diff once the lane curating ci.yml acknowledges it. No other gate is
touched.

## What was decided here and what stays open

**Decided here:**
- The 4 columns and their semantics (registered, validated, in place).
- `UNKNOWN` is legitimate on three of the four; `source_sha256` is real.
- Drift between recorded and live `source_sha256` is a hard gate failure.
- Visibility list of UNKNOWN bundles on every passing run.

**Open by design (founder's call):**
- Pinning: what a program does when the bundle updates (refuse / warn /
  reverify). Out of scope this round.
- Plausibility enforcement: the gate does not classify invented from real
  identifiers. Author discipline + visibility is the corrective mechanism
  per the founder's "do not invent a plausible number" directive. A future
  PR that imports a real upstream release is the natural time to add a
  check that asserts the recorded `upstream_release` against the
  upstream's manifest or version file.

## Founder rule - no revert acknowledged

Per the founder's standing rule (no revert without notice, effective
across all lanes until lifted - see
`founder-no-revert-rule-2026-08-19.md`), no agent is to revert this PR
or its commits without saying so and waiting. The four columns and the
visibility pattern are the deliverable; pulling them out re-introduces
the silence this work is meant to surface.

## Language rule (EN-UK in spec) acknowledged

This audit doc, the gate script, the manifest column values, and the
commit message are written in EN-UK per the founder's standing language
rule. The literal value `UNKNOWN` is used in user-visible text and in
column values in place of any non-English token. Dispatches remain in
the founder's language and are not transcribed into this document.

## FLEET_CONSTRAINTS check

- [x] Work off `origin/main`, on my own branch, my own PR. Worktree at
      `/workspace/.wt/minimax-cli2/ontology-versioning-clean-2026-08-19/`,
      branch `lane/minimax-cli2/ontology-versioning-2026-08-19-clean`,
      rooted at `origin/main` (`f9b3147364`). No writes to
      `/workspace/sounio/` - verified `git status` clean on the shared
      checkout at every step.
- [x] No `git add -A`, no `checkout`, no `stash`, no `clean` in
      `/workspace/sounio/`.
- [x] No full self-compile / `make build` / `lake build` / test suite on
      this pod. The gate is a bash script that reads a TSV and
      greps/counts.
- [x] `./bin/souc` not invoked.
- [x] No Slurm launch.
- [x] PR is DRAFT; I do not merge.
- [x] Atomic commits, one logical change each.
- [x] No `Co-Authored-By` trailer. No AI attribution in commit message.
- [x] EN-UK orthography throughout, including the literal value
      `UNKNOWN` and all prose, comments, and column names.
- [x] Docs registry: `topic-registry.v1.json` synced after the doc
      commit, never before.
- [x] No revert of anyone else's work. This PR only extends an existing
      manifest + an existing gate.

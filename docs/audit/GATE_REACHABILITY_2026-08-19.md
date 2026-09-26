<!-- docs:meta
topic_id: repo.docs.audit.gate-reachability-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.gate-reachability-2026-08-19
-->

# Gate reachability — the gate that checks whether the gates run

This declaration precedes the edit. Authorized 2026-08-19: a silent
gate is indistinguishable from a passing gate (S+G+R,
SOUNIO-EFFORT-LOCATION / PR #1972). Build
`scripts/ci/gate_reachability_gate.sh` plus a per-gate manifest.
Wire it to `ci.yml` the same way `concept_status_gate.sh` is wired.
Do not wire anything else. Do not revert.

```text
Semantic-Lane-ID: gate-reachability-20260819
Owner: grok-cli2
Concept-IDs: none
  (dispatch cites SOUNIO-EFFORT-LOCATION / PR #1972; that id is not
   a row in docs/internal/concepts/registry.tsv on origin/main
   f9b3147364. Not invented here.)
Intent-Preserved: a gate that does not run is not a pass; declaring
  a gate off-CI without owner and reason is not cheaper than
  declaring it well; the off-CI list stays visible when the gate
  is green
Transformation: every top-level scripts/ci/*.sh must have a
  manifest row (admitted | not-admitted | obsolete). admitted must
  be named under .github/. not-admitted and obsolete require a
  non-empty owner and reason. Missing row is red. Malformed is
  redder than absent.
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: of the 547 top-level scripts/ci/*.sh on
  origin/main f9b3147364, 104 are named under .github/ and 443
  are not (19.0%), by git grep -F <basename> -- .github/; those
  104 may be declared admitted and the 443 may be declared
  not-admitted with bootstrap debt
Claims-Forbidden: 443 isolated accidents; a green contracts job
  means the unrun gates passed; the bootstrap reason is a
  per-gate review; this gate being green means the 443 have
  owners; Madaros is fixed-point-verified; wiring this gate
  made main green (main is red for another reason)
Assumptions: the unit is scripts/ci/*.sh (non-recursive), matching
  the founder count; scripts/ci/fixtures/*.sh are not gates;
  "named" means git grep -F <basename> on .github/ (comments
  count, as the founder instrument); the two known cases hold:
  concept_status_gate.sh is origin/main ci.yml:68;
  exact_bitwise_rebracket_authority_gate.sh is unnamed
Write-Set: scripts/ci/gate_reachability_gate.sh,
  scripts/ci/gate_reachability.manifest.tsv, this file,
  .github/workflows/ci.yml (one step, after coord)
Read-Set: origin/main scripts/ci/*.sh, .github/,
  scripts/ci/concept_status_gate.sh, ci.yml:68,
  grok-cli3 ontology-14, glm-cli1 rebracket-10
Positive-Witness: no manifest → red; bootstrap manifest → green;
  --self-test fixture colours; dce_reach-style: admitted but
  unnamed → red, then that edit is undone before merge
Negative-Witness: a scripts/ci/*.sh with no row that still
  exits 0; not-admitted with empty owner that still exits 0;
  this gate itself unnamed under .github/ after merge
Acceptance-Gate: scripts/ci/gate_reachability_gate.sh exit 0
  on this branch with the committed manifest; --self-test
  exit 0; the negative admitted-unwired case exit 1
Integration-Target: origin/main at the branch point (f9b3147364)
Authoritative-Only-If: the 547/104/443 count is re-derived in
  this file with a command, and the two known cases match
```

## Instrument, validated before the gate existed

SHA: `origin/main` = `f9b314736421f6cff0ca02ffe02c6cb7def71a0a`

```text
git ls-tree --name-only origin/main -- scripts/ci/ | grep '\.sh$' | wc -l
# 547   (non-recursive; recursive **/*.sh is 556, of which 9 are
#        scripts/ci/fixtures/self_falsifying_*.sh and are not gates)

for f in concept_status_gate.sh exact_bitwise_rebracket_authority_gate.sh; do
  git grep -c --fixed-string "$f" origin/main -- .github/
done
# concept_status_gate.sh → origin/main:.github/workflows/ci.yml:68 (1)
# exact_bitwise_rebracket_authority_gate.sh → no match (0)

# named = basename hits at least one .github/ file: 104
# unnamed: 443
# reachable: 104/547 = 19.0%
```

Both known cases match. The founder number is the top-level glob,
not the recursive tree. This gate enumerates that glob.

## Vocabulary

The founder dispatch named the three states in Portuguese (operational
instruction). The product vocabulary is EN-UK: `admitted`,
`not-admitted`, `obsolete`. Owner `unassigned`. Bootstrap reason
`inherited at 2026-08-19 bootstrap, state pending review`.

## Bootstrap (not a review)

The 104 named rows enter as `admitted`.
The 443 unnamed rows enter as `not-admitted` with
`owner=unassigned` and
`reason=inherited at 2026-08-19 bootstrap, state pending review`.

No individual reasons. The printed not-admitted list is the debt.

The new gate is a 548th `scripts/ci/*.sh`. After it is named in
`ci.yml` it is `admitted` and the unnamed count stays 443.

## Refutation

This gate is the wrong thing if:

- **R1** it exits 0 when the manifest is absent.
- **R2** `not-admitted` with empty owner or reason exits 0.
- **R3** an `admitted` row whose basename is not under `.github/`
  exits 0.
- **R4** wiring it requires touching any workflow step other than
  its own, or reverting someone else's commit.

## Receipts

Instrument on `origin/main` `f9b3147364`, then this branch:

| | no manifest | bootstrap, unwired | after `ci.yml` step |
|---|---|---|---|
| scripts | 548 | 548 | 548 |
| named | 104 | 104 | **105** |
| unnamed | 444 | 444 | **443** |
| rc | **1** (`missing-manifest` + 548 `unlisted=`) | 0 | **0** |

Negative phase (undone before commit):
`exact_bitwise_rebracket_authority_gate.sh` flipped to `admitted`
without a `.github/` mention → rc=1
`admitted-unwired=exact_bitwise_rebracket_authority_gate.sh`.
Restored. Not a revert of anyone else's commit.

`--self-test` (fixture dir, six colours) → `GATE_REACHABILITY_SELFTEST_OK`.

On green the gate still prints all 443 `not-admitted` rows with
`owner=unassigned`. That list is the debt.

`ci.yml` delta is seven lines: one contracts step immediately after
`concept_status_gate.sh`. Nothing else wired. grok-cli3's ontology
14 stay off-CI (#1970). glm-cli1's rebracket 10 stay off-CI.

```text
Semantic-Outcome: the 443 unrun gates are no longer silent. They
  are a printed, counted debt. A new scripts/ci/*.sh without a
  row is red. An admitted row that CI does not name is red.
  Declaring not-admitted without owner and reason is red.
  This gate names itself under .github/. The 443 are not owned.
```

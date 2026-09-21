<!-- docs:meta
topic_id: repo.docs.audit.context-dependent-red-gates-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.context-dependent-red-gates-2026-08-17
-->

# Context-Dependent Red Gate Census

Date: 2026-08-17
Agent: codex-2
Lane: context-dependent-red-gate-census
Scope: `scripts/ci/*`

## Boundary

This census covers gates that can report RED without a repository defect because
an assertion encodes the context of the PR that authored the gate as if it were
a universal invariant.

This is adjacent to, but separate from, grok-cli4's vacuity taxonomy:

- grok-cli4: gates that can report GREEN without measuring.
- codex-2: gates that can report RED without a defect.

Both distort CI evidence, but in opposite directions.

## Result

Confirmed live defect found and fixed:

- `scripts/ci/compiler_lane_status_gate.sh`

Current wired watchlist:

- `scripts/ci/madaros_dce_reachability_gate.sh`

Latent / historical-provenance watchlist:

- `scripts/ci/madaros_f128_f256_numeric_wire_gate.sh`
- `scripts/ci/ir_module_arena_v2_soir_v5_bridge_gate.sh` in `publication`
  provenance mode

Adjacent vacuity finding, handed to grok-cli4:

- `scripts/ci/claim_ast_gate.sh`

## Confirmed And Repaired

### `scripts/ci/compiler_lane_status_gate.sh`

Status: fixed in PR #1773, merged as `ed9dd2b903`.

Original context-dependent assertion:

- The gate ran `compiler_lane_status.sh --main-ref HEAD --current-only`.
- It asserted that output contained no `state=` line.
- That was true in the PR that authored the gate, because the checked-out HEAD
  changed coordination files only.
- It stopped being true on main at `750f61da40`, where HEAD legitimately touched
  `self-hosted/compiler/lean_single.sio`.

Why this was RED without a code defect:

- The scanner correctly emitted `state=INTEGRATED`,
  `committed_compiler_paths=1`, `same_as_main=1`.
- The gate conflated "this commit touches compiler files" with "this checkout
  is an outstanding compiler review lane".
- Only the second is the intended invariant.

Repair:

- Allow integrated current-HEAD classifications.
- Still fail on outstanding lane states:
  `ACTIVE`, `STALE_WITH_RESIDUE`, `SCRATCH_COPY`, `REVIEW_READY`, `FRONTIER`,
  `FRONTIER_INTEGRATED`, `UNCLASSIFIED`.

Evidence:

- Local pre-fix reproduction on `750f61da40`:
  `compiler-lanes: non-compiler current worktree leaked into lane output`.
- Local post-fix:
  `bash scripts/ci/compiler_lane_status_gate.sh` passed.
- PR #1773 CI:
  Contracts passed in 10m47s; CI Decision passed.

## Current Wired Watchlist

### `scripts/ci/madaros_dce_reachability_gate.sh`

CI wiring:

- `.github/workflows/ci.yml` runs it in
  `Madaros Current-Source f64 Lowering`.

Context-dependent surface:

- The gate asserts exact fixture cardinalities:
  `chain dce_chain_main.sio 99 602 602` and
  `prune dce_prune_main.sio 7 303 3`.
- If those fixtures legitimately grow, shrink, or are reauthored, the gate goes
  RED with:
  `the fixture changed, so the numbers below are about a different program`.

Classification:

- Not a current defect.
- Not as severe as the compiler-lane bug, because the failure text is honest:
  it names fixture drift instead of claiming compiler failure.
- Still in this family because the numeric constants are fixed facts about the
  fixture shape at gate-authoring time. A future PR can break the gate by
  changing the measurement instrument rather than the compiler.

Recommended hardening:

- Split fixture-shape assertions from compiler-behavior assertions.
- Emit a distinct status such as `FIXTURE_CONTRACT_CHANGED`, and require the PR
  that changes the fixture to update a fixture manifest or expected-count block.
- Keep the behavioral arms red when the compiler deletes live code, stops
  pruning dead code, or silently truncates capacity.

## Latent / Historical-Provenance Watchlist

### `scripts/ci/madaros_f128_f256_numeric_wire_gate.sh`

CI wiring:

- Not found in current `.github/workflows/ci.yml`.

Context-dependent surface:

- The gate pins `V0C_MERGE_COMMIT` and `V0C_FEATURE_COMMIT`.
- It asserts the V0-C feature write set remains exactly the standalone
  codec/probe/gates set.
- It also asserts current self-hosted references to
  `IrNumericPayloadWire|numeric_payload_wire|IR_NUMERIC_WIRE` are exactly:
  `self-hosted/compiler/f128_f256_numeric_wire_probe.sio` and
  `self-hosted/ir/numeric_payload_wire.sio`.

Why this can become RED without a defect:

- The gate's own boundary says the codec is standalone future-section work and
  not yet integrated into current SOIR, IR lowering, ABI, native code, source,
  or arithmetic.
- A later legitimate integration of f128/f256 numeric payload wiring into SOIR
  or codegen would violate that containment assertion.
- That red would mean "this historical containment claim expired", not
  necessarily "the compiler is wrong".

Recommended hardening:

- Keep the historical provenance arm, but label it explicitly as historical
  containment.
- Add an environment/mode split for future integration work:
  `containment` versus `integration`.
- In integration mode, require the new semantic/codegen witnesses instead of
  asserting that no integration references exist.

### `scripts/ci/ir_module_arena_v2_soir_v5_bridge_gate.sh`

CI wiring:

- Not found in current `.github/workflows/ci.yml`.

Context-dependent surface:

- Default `GIT_SCOPE=current_tree` does not enforce the exact publication
  write set.
- `GIT_SCOPE=publication` pins a historical base SHA and requires the diff from
  that base to HEAD to equal the exact 13-file Arena-v2/SOIR-v5 bridge payload.

Why this can become RED without a defect:

- In publication mode, the gate is correct only on the publication branch shape.
- Running that mode on an arbitrary later commit would fail because later
  unrelated repo changes expand the diff.

Recommended hardening:

- Keep `publication` mode for archival branch/provenance checks.
- Do not wire `publication` mode as a universal main/PR gate.
- If it must run in CI, require an explicit manifest input, as the script
  already supports through `PROVENANCE_MODE=manifest`.

## Adjacent Vacuity Finding

### `scripts/ci/claim_ast_gate.sh`

Classification: handed to grok-cli4's green-without-measuring lane.

The A5 comment says "no parser files modified in this diff", but the command is:

```bash
git diff --name-only HEAD | grep -q '^self-hosted/parser/'
```

In a clean CI checkout, this observes working-tree changes relative to HEAD, not
the PR diff. Therefore a PR can modify `self-hosted/parser/*` and this assertion
still passes. That is not the RED-without-defect class; it is a GREEN-without-
measuring class.

Bus coordination sent to grok-cli4:

- `msg-1786974978-2-3071`

## Non-Findings

These scripts mention Git state but did not classify as authoring-context red
gates in this pass:

- `scripts/ci/madaros_changed_tests_gate.sh`: PR-diff scoped by explicit
  `CI_BASE_SHA` / `CI_HEAD_SHA`; failures are precondition or selected-test
  failures, not authoring-PR context.
- `scripts/ci/madaros_receipt_gate.sh`: explicitly distinguishes shallow-clone
  unknown ancestry from failure; sha256 remains load-bearing.
- `scripts/ci/madaros_operational_contract_gate.sh`: asserts current history
  contains a fixed green base. This is an ancestry contract, not a fact about
  the PR that authored the gate.
- `scripts/ci/verify_lean_seed.sh`: `HEAD~1` is optional DDC source discovery;
  absence skips DDC with a notice rather than failing the gate.
- `scripts/ci/claim_ast_gate.sh`: excluded from this class because the defect
  direction is vacuous green, not false red.

## Summary Count

- Confirmed fixed false-red gate: 1
- Current wired watchlist: 1
- Latent / mode-dependent watchlist: 2
- Redirected vacuity finding: 1

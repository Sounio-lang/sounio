# lean_single named-function `pub use` contract

This fixture set pins GitHub issue #842 without claiming that the compiler bug
is fixed. The supported claim boundary is deliberately narrow:

```sio
pub use public_leaf::{public_route_value}
use public_facade::{public_route_value}
```

The acceptance contract covers a named function in a braced `pub use` list.
It does not establish semantics for types, structs, constants, aliases, globs,
renames, or chained re-exports.

## State matrix

`scripts/ci/lean_single_pub_use_reexport_gate.sh` emits these independent
receipt fields:

| State | `facade_forwarding` | `selective_reexport` | Classification |
|---|---:|---:|---|
| Current seed | 0 | 1 | public forwarding absent |
| Rejected load-whole-leaf patch | 1 | 0 | omitted leaf symbol overexposed |
| Issue #842 acceptance | 1 | 1 | named-function re-export is selective |

`private_import_isolated` is telemetry for a separate residual. It does not
block issue #842 acceptance.

Every semantic rejection must exit exactly `1`, contain an anchored
`unknown identifier` diagnostic, and contain the exact line
`typecheck: failed`. Signal exits, fatal diagnostics, and `rc >= 128` are not
semantic success. The gate self-test accepts a valid `rc=1` receipt, rejects
the same expected substring with `rc=139`, and rejects `rc=1` contaminated by
a fatal diagnostic:

```bash
SOUNIO_LEAN_SINGLE_PUB_USE_GATE_SELF_TEST_ONLY=1 \
  bash scripts/ci/lean_single_pub_use_reexport_gate.sh
```

## Blocker: named function forwarding

```text
Blocker-ID: BLK-20260713-lean-single-import-visibility
Status: owned
Severity: B1
Class: compiler-semantics
Owner: Codex compiler coordination (/root)
Lane: future dedicated lean_single module bindings/exports implementation
Worktree: /tmp/sounio-issue-842-pub-use-20260713
Branch: codex/issue-842-pub-use-20260713
Files-Owned: scripts/ci/lean_single_pub_use_reexport_gate.sh; tests/compiler/pub_use_reexport/*; tests/run-pass/pub_use_reexport_direct_control.sio; tests/run-pass/fixtures/pub_use_reexport_leaf.sio
Files-Read-Only: self-hosted/compiler/lean_single.sio
Do-Not-Touch: self-hosted/compiler/main.sio; self-hosted/ir/lower.sio; self-hosted/check/check.sio
Repro: SOUNIO_LEAN_SINGLE_PUB_USE_GATE_BIN=bin/souc-lean-single-x86_64 bash scripts/ci/lean_single_pub_use_reexport_gate.sh
Observed: direct import passes, but the named function is unknown through its public facade
Expected: the listed function resolves through the facade while an omitted public leaf function remains unknown
Acceptance-Gate: scripts/ci/lean_single_pub_use_reexport_gate.sh exits 0 with facade_forwarding=1 selective_reexport=1
Evidence-Level: E3
Evidence: reproducible gate receipt; latest local baseline under /tmp/sounio-842-baseline-*
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: design module-owned binding/export records and contextual lookup before editing compiler semantics
```

This B1 is explicitly transferred out of the diagnostic evidence lane. It
blocks the future implementation lane, not a PR that adds only the executable
contract and receipts in this directory.

## Residual: private import visibility

```text
Blocker-ID: BLK-20260713-lean-single-private-import-visibility
Status: classified
Severity: B4
Class: compiler-semantics
Owner: Codex compiler coordination (/root)
Lane: lean_single private import visibility follow-up
Worktree: /tmp/sounio-issue-842-pub-use-20260713
Branch: codex/issue-842-pub-use-20260713
Files-Owned: tests/compiler/pub_use_reexport/private_*.sio
Files-Read-Only: self-hosted/compiler/lean_single.sio
Do-Not-Touch: issue #842 named-function acceptance semantics
Repro: run the issue #842 gate and inspect private_import_isolated
Observed: a public leaf function reached through private use is callable by the consumer
Expected: private use does not add the leaf function to the facade public surface
Acceptance-Gate: gate receipt reports private_import_isolated=1
Evidence-Level: E3
Evidence: private.compile.log in the gate receipt directory
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: schedule a separate visibility lane; this residual does not block issue #842
```

## Handoff and PR scope

The diagnostic branch owns only the gate and fixtures named above. It contains
no compiler-semantic patch and must not be presented as fixing #842. A PR from
this branch should say:

```text
Adds an executable named-function re-export contract for issue #842.
Pins the current missing-forwarding behavior, rejects whole-leaf overexposure,
and records private-import globalization under a separate non-blocking receipt.
No compiler implementation or public language claim changes in this PR.
```

The next implementation lane should start from current `origin/main`, retain
these witnesses unchanged, and make the acceptance gate green without a
facade-only byte scanner or load-whole-leaf shortcut.

### Formal handoff

```text
Current-SHA: branch HEAD; resolve immutably with git rev-parse HEAD and use the SHA from the final agent receipt
Current-Branch: codex/issue-842-pub-use-20260713
Current-Worktree: /tmp/sounio-issue-842-pub-use-20260713
Dirty-Status: clean after the diagnostic commit; verify with git status --short
Owned-Files: scripts/ci/lean_single_pub_use_reexport_gate.sh; tests/compiler/pub_use_reexport/*; tests/run-pass/pub_use_reexport_direct_control.sio; tests/run-pass/fixtures/pub_use_reexport_leaf.sio
Do-Not-Touch: self-hosted/compiler/lean_single.sio; self-hosted/compiler/main.sio; self-hosted/ir/lower.sio; self-hosted/check/check.sio
Last-Green-Gates: gate self-test rc=0; canonical direct control PASS 1/1; canonical pub_use inventory contains only the direct control; bash -n; git diff --check
Failing-Gates: baseline classifier rc=1 with facade_forwarding=0; rejected naive candidate rc=1 with selective_reexport=0
Open-Blockers: BLK-20260713-lean-single-import-visibility transferred to the future bindings/exports implementation lane; BLK-20260713-lean-single-private-import-visibility is B4 telemetry
Artifacts: /tmp/sounio-842-baseline-d2dd32b0f; /tmp/sounio-842-naive-d2dd32b0f; reproducible with the gate
Next-Command: git fetch origin main
```

Diagnostic-Lane-Status: `locally-green`. The red classifier receipts above are
the expected evidence produced by the diagnostic contract, not failed required
gates for this evidence-only lane. Before publishing a PR, re-evaluate the
branch against current `origin/main`; the original evidence base was
`81a36104478fa79ca685c9b2ef87cee1c39dfa5d`.

# CI layers

`.github/workflows/ci.yml` runs in three layers. Each layer is a job that
requires every job in it to have succeeded, and each is checked by the same
evaluator, `scripts/ci/evaluate_ci_decision.py`, through `CI_LAYER`.

| Layer | Job | Contains | Purpose |
| --- | --- | --- | --- |
| fast | `fast-pr-gate` | `contracts`, `gate-wave-0`, `sounio-lint`, `website` | Feedback that must not wait on a full compiler build |
| deep | `deep-compiler-gate` | `native-selfhost-linux-x86_64`, `source-bootstrap-selfhost-linux-x86_64`, `madaros-current-source-deref-f64`, `native-selfhost-macos-arm64`, `full-test-suite`, `madaros-witness-gate`, `lean-proofs` | Compiler qualification |
| complete | `ci-decision` | everything selected by impact | The single verdict |

`FAST` and `DEEP` in `scripts/ci/evaluate_ci_decision.py` are the source of
truth for membership. The `needs` lists in `ci.yml` must match them exactly.

## Why a layer cannot pass by checking less

`evaluate` fails when the aggregator's `needs` set does not equal the layer
exactly:

```python
if layer in ("fast", "deep"):
    expected = set(required) | {"impact"}
    if set(needs) != expected:
        failures.append(f"needs mismatch: ...")
```

So removing a job from a layer's `needs` is a failure, not a smaller check. That
is the property that makes layering safe in a repository where absence of
evidence must never read as success.

The complete decision keeps its previous behaviour and its `needs` set is pinned
statically by `scripts/ci/ci_decision_contract.mjs`, which is where that
invariant already lived.

## What decides whether a run is exhaustive

The `layers` step on `impact` derives an `exhaustive` output: true on every
event except `pull_request`, and on a pull request only when the impact
classifier reports `full`. The evaluator recomputes the same value and fails if
the two disagree, so a workflow that stops producing it fails closed rather than
silently narrowing the run.

## Adding or moving a job

1. Add the job to `ci.yml` with its `if:` condition.
2. Add the same condition to `selected_jobs()` in
   `scripts/ci/evaluate_ci_decision.py`. The two are a pair.
3. If the job belongs in a layer, add it to `FAST` or `DEEP` and to that
   aggregator's `needs` list.
4. Run `python3 scripts/ci/test_ci_layers.py`. It parses `ci.yml` and fails if
   the aggregators' `needs` and the evaluator disagree.
5. Run `bash scripts/ci/impact_ci_selftest.sh` and
   `node scripts/ci/ci_decision_contract.mjs . --selftest`.

## Verifying that no gate was dropped

This is already enforced, and the enforcement is worth knowing about before
touching `ci.yml`.

`scripts/ci/gate_workflow_reference_ratchet.sh` freezes the number of gate
scripts that no workflow names. All 300 gate scripts `ci.yml` invokes are named
by `ci.yml` alone — none of them appears in another workflow — so removing one
from `ci.yml` makes it unnamed, the frozen count rises, and the ratchet fails.
A restructure that drops a gate cannot pass quietly.

`scripts/dev/check_workflow_script_refs.sh` covers the other direction: every
`scripts/...` path a workflow names must exist and be executable.

As a cross-check while restructuring, the set comparison is still worth running,
because it names the specific gates rather than only counting them:

```bash
git show origin/main:.github/workflows/ci.yml | grep -oE 'scripts/[a-z0-9_/.-]+\.(sh|py)' | sort -u > /tmp/before
grep -oE 'scripts/[a-z0-9_/.-]+\.(sh|py)' .github/workflows/ci.yml | sort -u > /tmp/after
comm -23 /tmp/before /tmp/after   # must be empty
```

That is how the reconciliation that introduced these layers was checked: main
invoked 303 gate scripts and the layered workflow still invokes 303, with an
empty difference.

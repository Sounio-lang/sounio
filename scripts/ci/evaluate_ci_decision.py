#!/usr/bin/env python3
"""Fail closed on missing evidence in Fast, Deep, or complete CI qualification.

The pipeline is layered. `impact` classifies the change and decides whether this
run is exhaustive; each aggregator job requires every job in its layer to have
succeeded, and `ci-decision` requires the whole selected set.

Every layer is checked by this same evaluator through CI_LAYER.

For the fast and deep layers `evaluate` also fails when the aggregator's `needs`
set does not match the layer exactly. That is what makes a layer unable to pass
by silently omitting a job: dropping one from `needs` is a failure, not a
smaller check. The complete decision keeps its previous behaviour -- its `needs`
set is pinned statically by scripts/ci/ci_decision_contract.mjs instead, which
is where that invariant already lived.

The `required` map below mirrors the `if:` conditions in
.github/workflows/ci.yml. If a job's `if:` changes, change it here too -- the
two are a pair, and scripts/ci/ci_decision_contract.mjs checks the needs lists.
"""

from __future__ import annotations

import json
import os
import sys

IMPACT_KEYS = (
    "docs",
    "website",
    "compiler",
    "runtime",
    "stdlib",
    "tests",
    "lean",
    "math",
    "ontology",
    "clinical",
    "sio",
    "full",
)

# Fast layer: cheap feedback that must not wait on a full compiler build.
FAST = {"contracts", "gate-wave-0", "sounio-lint", "website"}

# Deep layer: compiler qualification. Expensive, and the layer a reviewer waits
# on before trusting a compiler change.
DEEP = {
    "native-selfhost-linux-x86_64",
    "source-bootstrap-selfhost-linux-x86_64",
    "madaros-current-source-deref-f64",
    "native-selfhost-macos-arm64",
    "full-test-suite",
    "madaros-witness-gate",
    "lean-proofs",
}


def truthy(value: str | None) -> bool:
    return value == "true"


def is_nightly(event: str, nightly: bool) -> bool:
    """True when this run must actually execute the nightly jobs.

    A scheduled run always is. A workflow_dispatch is only when the operator
    ticked the `nightly` input, matching the `if:` on r6-corpus-sweep and
    slow-lane.
    """
    if event == "schedule":
        return True
    return nightly


def selected_jobs(impact, event, nightly=False):
    if event not in {"pull_request", "push", "merge_group", "schedule", "workflow_dispatch"}:
        raise ValueError(f"unsupported event: {event}")
    if any(impact.get(k) not in {"true", "false"} for k in IMPACT_KEYS):
        raise ValueError("missing or malformed impact classification")

    has = lambda *keys: any(impact[k] == "true" for k in keys)

    # The `layers` step on `impact` derives this from the event and the
    # classifier. Recomputing it here and comparing is deliberate: a mismatch
    # means the workflow and the evaluator disagree about what this run is.
    exhaustive = event != "pull_request" or has("full")
    if impact.get("exhaustive") != str(exhaustive).lower():
        raise ValueError("exhaustive selection disagrees with event/impact")

    compiler = has("compiler", "runtime", "stdlib", "tests", "full")
    required = {
        "contracts": True,
        "native-selfhost-linux-x86_64": compiler,
        "source-bootstrap-selfhost-linux-x86_64": has("compiler", "full"),
        "madaros-current-source-deref-f64": has("compiler", "stdlib", "tests", "full"),
        "native-selfhost-macos-arm64": has("compiler", "full"),
        "full-test-suite": compiler,
        "madaros-witness-gate": compiler,
        "gate-wave-0": compiler,
        "sounio-lint": has("compiler", "stdlib", "tests", "sio", "full"),
        "lean-proofs": has("lean", "full"),
        "website": has("website", "full"),
        # R6 and slow-lane are selected by the EVENT, not by impact
        # classification: they are nightly-only, so `skipped` is the correct
        # outcome on a pull request and a non-evaluation on a scheduled run.
        "r6-corpus-sweep": is_nightly(event, nightly),
        "slow-lane": is_nightly(event, nightly),
    }
    return required


def evaluate(needs, event, nightly=False, layer="all"):
    if needs.get("impact", {}).get("result") != "success":
        return ["impact did not succeed"]
    try:
        required = selected_jobs(needs["impact"].get("outputs", {}), event, nightly)
    except ValueError as exc:
        return [str(exc)]

    failures = []
    if layer == "fast":
        required = {key: required[key] for key in FAST}
    elif layer == "deep":
        required = {key: required[key] for key in DEEP}
    elif layer != "all":
        return [f"unknown CI layer: {layer}"]

    if layer in ("fast", "deep"):
        # Exactness is the safety property of a layer: an aggregator that stops
        # listing a job must fail, not quietly check less.
        expected = set(required) | {"impact"}
        if set(needs) != expected:
            failures.append(
                f"needs mismatch: missing={sorted(expected - set(needs))} "
                f"extra={sorted(set(needs) - expected)}"
            )

    for job, selected in required.items():
        result = needs.get(job, {}).get("result", "missing")
        if selected and result != "success":
            failures.append(f"selected job {job} ended as {result}")
        elif not selected and result not in {"success", "skipped"}:
            failures.append(f"unselected job {job} ended as {result}")
    return failures


def main() -> int:
    try:
        failures = evaluate(
            json.loads(os.environ.get("NEEDS_JSON", "{}")),
            os.environ.get("GITHUB_EVENT_NAME", ""),
            os.environ.get("NIGHTLY_INPUT", "").lower() == "true",
            os.environ.get("CI_LAYER", "all"),
        )
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        failures = [f"invalid evidence: {exc}"]

    for failure in failures:
        print(f"CI_DECISION_FAIL: {failure}", file=sys.stderr)
    if not failures:
        print(f"CI_DECISION_PASS layer={os.environ.get('CI_LAYER', 'all')}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

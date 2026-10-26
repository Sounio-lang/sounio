#!/usr/bin/env python3
"""Layer selection, fail-closed negative controls, and workflow/evaluator agreement.

Stdlib only, no YAML dependency: the workflow is parsed textually, the same way
scripts/ci/ci_decision_contract.mjs does it, so this cannot silently skip on a
runner without pyyaml.

Run: python3 scripts/ci/test_ci_layers.py

The last test is the important one: it reads .github/workflows/ci.yml and
asserts that each aggregator's `needs` set is exactly the layer the evaluator
expects. That is what keeps the workflow and the evaluator from drifting apart,
which is the failure this whole mechanism exists to prevent.
"""

import copy
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_ci_decision import (  # noqa: E402
    DEEP,
    FAST,
    IMPACT_KEYS,
    evaluate,
    selected_jobs,
)

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# ci-decision's needs are pinned statically by ci_decision_contract.mjs; these
# are the jobs it lists beyond the evaluator's own map.
DECISION_EXTRAS = {"stdlib-deep-science", "fast-pr-gate", "deep-compiler-gate"}

EVENTS = ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
KEYSETS = (
    (),
    ("compiler",),
    ("runtime",),
    ("stdlib",),
    ("tests",),
    ("lean",),
    ("website",),
    ("sio",),
    ("full",),
)


def job_needs(text, job):
    """The `needs` list of `job`, parsed textually."""
    lines = text.split("\n")
    try:
        start = lines.index("  %s:" % job)
    except ValueError:
        raise AssertionError("%s missing from the workflow" % job)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^  [A-Za-z0-9_-]+:$", lines[i]):
            end = i
            break
    block = lines[start:end]
    try:
        n = block.index("    needs:")
    except ValueError:
        raise AssertionError("%s has no needs:" % job)
    needs = []
    for line in block[n + 1:]:
        if not line.strip() or line.strip().startswith("#"):
            continue
        m = re.match(r"^      - ([A-Za-z0-9_-]+)$", line)
        if not m:
            break
        needs.append(m.group(1))
    return needs


def impact_for(keys, event):
    impact = {
        k: str(k in keys or event != "pull_request" or "full" in keys).lower()
        for k in IMPACT_KEYS
    }
    impact["exhaustive"] = str(event != "pull_request" or "full" in keys).lower()
    return impact


def fixture(keys=(), event="pull_request", nightly=False, layer="all"):
    impact = impact_for(keys, event)
    required = selected_jobs(impact, event, nightly)
    if layer == "fast":
        scope = FAST
    elif layer == "deep":
        scope = DEEP
    else:
        scope = set(required)
    needs = {k: {"result": "success" if required[k] else "skipped"} for k in scope}
    needs["impact"] = {"result": "success", "outputs": impact}
    return needs


class Layers(unittest.TestCase):
    def test_every_selected_job_must_answer(self):
        for event in EVENTS:
            for keys in KEYSETS:
                for nightly in (False, True):
                    for layer in ("fast", "deep", "all"):
                        needs = fixture(keys, event, nightly, layer)
                        self.assertEqual(
                            evaluate(needs, event, nightly, layer), [],
                            f"clean fixture rejected: {event} {keys} {nightly} {layer}",
                        )
                        for job in list(needs):
                            for result in ("failure", "cancelled", "missing", "skipped"):
                                if result == "skipped" and needs[job]["result"] == "skipped":
                                    continue
                                bad = copy.deepcopy(needs)
                                if result == "missing":
                                    del bad[job]
                                else:
                                    bad[job]["result"] = result
                                with self.subTest(event=event, keys=keys, layer=layer,
                                                  job=job, result=result):
                                    self.assertTrue(
                                        evaluate(bad, event, nightly, layer),
                                        f"a {result} {job} passed the {layer} layer",
                                    )

    def test_layer_needs_must_match_exactly(self):
        """Dropping a job from a layer's `needs` must fail, not shrink the check."""
        for layer in ("fast", "deep"):
            needs = fixture(("compiler",), "push", False, layer)
            for job in [j for j in needs if j != "impact"]:
                bad = copy.deepcopy(needs)
                del bad[job]
                with self.subTest(layer=layer, dropped=job):
                    self.assertTrue(evaluate(bad, "push", False, layer))

    def test_classification_is_evidence(self):
        needs = fixture(("compiler",), "pull_request")
        for key in (*IMPACT_KEYS, "exhaustive"):
            bad = copy.deepcopy(needs)
            del bad["impact"]["outputs"][key]
            with self.subTest(missing=key):
                self.assertTrue(evaluate(bad, "pull_request"))

        # The workflow's `layers` step and the evaluator must agree about what
        # this run is; a disagreement is a failure, not a default.
        bad = copy.deepcopy(needs)
        bad["impact"]["outputs"]["exhaustive"] = "true"
        self.assertTrue(evaluate(bad, "pull_request"))

        # A failed classifier must not be able to pass a layer.
        bad = copy.deepcopy(needs)
        bad["impact"]["result"] = "failure"
        self.assertTrue(evaluate(bad, "pull_request"))

    def test_layers_are_disjoint_and_inside_the_selected_set(self):
        self.assertTrue(FAST and DEEP)
        self.assertEqual(FAST & DEEP, set())
        selected = set(selected_jobs(impact_for((), "push"), "push"))
        self.assertTrue(FAST <= selected, f"FAST outside selected: {FAST - selected}")
        self.assertTrue(DEEP <= selected, f"DEEP outside selected: {DEEP - selected}")

    def test_intended_policy(self):
        # Fast layer is cheap: it must not contain a full compiler self-host.
        self.assertNotIn("native-selfhost-linux-x86_64", FAST)
        self.assertNotIn("full-test-suite", FAST)
        # Deep layer is where compiler qualification lives.
        for job in ("native-selfhost-linux-x86_64", "full-test-suite", "madaros-witness-gate"):
            self.assertIn(job, DEEP)
        # Nightly jobs are event-selected, so they are in no layer and are
        # checked only by the complete decision.
        for job in ("r6-corpus-sweep", "slow-lane"):
            self.assertNotIn(job, FAST | DEEP)
        # A PR that does not touch the compiler must not require the deep jobs.
        needs = fixture((), "pull_request", False, "deep")
        self.assertEqual(needs["full-test-suite"]["result"], "skipped")
        self.assertEqual(evaluate(needs, "pull_request", False, "deep"), [])

    def test_workflow_needs_match_the_evaluator(self):
        """The workflow and the evaluator must not drift apart."""
        text = WORKFLOW.read_text(encoding="utf-8")
        impact = impact_for((), "push")
        required = selected_jobs(impact, "push")

        for job, layer in (("fast-pr-gate", "fast"), ("deep-compiler-gate", "deep")):
            needs = set(job_needs(text, job))
            scope = FAST if layer == "fast" else DEEP
            self.assertEqual(
                needs, scope | {"impact"},
                f"{job} needs {sorted(needs)}, evaluator expects {sorted(scope | {'impact'})}",
            )

        decision = set(job_needs(text, "ci-decision"))
        self.assertEqual(
            decision, set(required) | {"impact"} | DECISION_EXTRAS,
            "ci-decision needs disagree with selected_jobs",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

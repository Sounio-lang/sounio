#!/usr/bin/env python3
"""Synthetic adversarial v3 screening tests; no model or hardware submission."""
import copy
import json
import unittest
import evaluate_external_overhead_v3 as v3
import test_evaluate_external_overhead as prior

class V3Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = prior.EvaluationTests()
        self.fixture.setUp()
        self.spec = v3.specification()
        self.baseline, self.observed = self.fixture.arms()
        for rows in self.observed["external"]:
            for row in rows:
                row.update(schema="pireus-external-memory-observation-v2",
                           observer_profile="external-observer-deadline-v3")
            rows[0].update(scheduling="actual-start-deadline-no-catchup",
                           interval_seconds=0.2, observer_resource_scope="observer-process-only")
            samples = rows[1:-1]
            for i,s in enumerate(samples):
                s.update(sample_gap_ns=None if i == 0 else s["monotonic_ns"]-samples[i-1]["monotonic_ns"],
                         duration_ns=100)
                s["observer_resources"] = dict(observer_pid=s["observer_pid"],
                    scope="observer-process-only", process_cpu_ns=1000+i*100,
                    status=dict(error=None, value=dict(VmRSS=10000,VmHWM=12000),
                        format="kB-fields", monotonic_ns=s["monotonic_ns"],duration_ns=50))

    def test_complete_synthetic_screen_requires_separate_custody(self):
        r = v3.evaluate(self.spec,self.baseline,self.observed)
        self.assertTrue(r["numerical_semantic_screen_pass"])
        self.assertFalse(r["loaded_model_overhead_qualified"])
        self.assertFalse(r["pilot_acceptance"])

    def test_v3_refusals(self):
        mutations = [
            ("version", lambda rows: rows[1].update(schema="old")),
            ("resource identity", lambda rows: rows[1]["observer_resources"].update(observer_pid=100)),
            ("CPU counter progression", lambda rows: rows[2]["observer_resources"].update(process_cpu_ns=1)),
            ("RSS/high-water", lambda rows: rows[1]["observer_resources"]["status"]["value"].pop("VmRSS")),
            ("status unavailable", lambda rows: rows[1]["observer_resources"]["status"].update(error="denied")),
            ("resource timestamp", lambda rows: rows[1]["observer_resources"]["status"].update(duration_ns=101)),
            ("overlaps successor", lambda rows: rows[1].update(duration_ns=500_000_001)),
            ("reported gap", lambda rows: rows[2].update(sample_gap_ns=1)),
            ("high-water decreased", lambda rows: rows[2]["observer_resources"]["status"]["value"].update(VmHWM=11000)),
            ("scheduling", lambda rows: rows[0].update(scheduling="sleep-after-read")),
            ("process identity", lambda rows: [r.update(observer_pid=r["target_pid"]) for r in rows]),
            ("timestamp type", lambda rows: rows[1].update(monotonic_ns=True)),
        ]
        for error,mutate in mutations:
            with self.subTest(error=error):
                rows=copy.deepcopy(self.observed["external"][0]);mutate(rows)
                with self.assertRaisesRegex(ValueError,error):v3.observer_resources(rows)

    def test_initial_boundary_gap_is_not_hidden_by_complete_tail(self):
        rows=copy.deepcopy(self.observed["external"][0])
        for r in rows[1:]:
            r["monotonic_ns"]+=600_000_000
            if r["stage"]=="SAMPLE":
                r["observer_resources"]["status"]["monotonic_ns"]+=600_000_000
        with self.assertRaisesRegex(ValueError,"boundary sampling gap"):v3.observer_resources(rows)

    def test_relaxed_acceptance_is_refused(self):
        self.spec["acceptance"]["maximum_loss_minimum_host_available_bytes"]*=2
        with self.assertRaisesRegex(ValueError,"specification mismatch"):
            v3.evaluate(self.spec,self.baseline,self.observed)

    def test_inherited_memory_bound_still_refuses(self):
        self.observed["guardian"][0]["minimum_bytes"]-=257*1024**2
        with self.assertRaisesRegex(ValueError,"memory overhead"):
            v3.evaluate(self.spec,self.baseline,self.observed)

    def test_inherited_response_equality_still_refuses(self):
        response=json.loads(self.observed["responses"][0][0])
        response["output_ids"][0]+=1
        raw=json.dumps(response).encode()
        self.observed["responses"][0][0]=self.observed["responses"][1][0]=raw
        with self.assertRaisesRegex(ValueError,"response content"):
            v3.evaluate(self.spec,self.baseline,self.observed)

    def test_real_cpu_resource_rows_are_readable_without_loaded_claim(self):
        root=v3.SPEC_PATH.parent.parent/"external-container-cpu-v3-20260909/attempt-11982"
        expected=json.loads((root/"qualification.json").read_bytes())
        for rank in (0,1):
            rows=[json.loads(s) for s in (root/f"collected/rank-{rank}/journal.jsonl").read_text().splitlines()]
            result=v3.observer_resources(rows)
            for key in ("samples","maximum_gap_including_boundaries_ns",
                        "observer_peak_rss_bytes","observer_high_water_bytes","observer_cpu_delta_ns"):
                self.assertEqual(result[key],expected["results"][rank]["oracle"][key])
            self.assertFalse(result["observer_overhead_budget_qualified"])

if __name__=="__main__":unittest.main()

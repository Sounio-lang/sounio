import copy
import json
import unittest
from verify_gate_custody import MANIFEST, verify


class GateCustody(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST.read_text())
        self.paths = [g["path"] for g in self.manifest["gates"]]

    def test_actual_git_inventory(self):
        report = verify(self.paths)
        self.assertEqual(report["named_gates"], 80)
        self.assertEqual(report["verified_original_snapshot_dependencies"], 456)
        self.assertFalse(report["runtime_replay"])
        self.assertFalse(report["current_hardware_acceptance"])

    def test_missing_duplicate_and_unknown_workflow_names_refuse(self):
        for paths in [[], self.paths[:-1], self.paths + self.paths[:1],
                      self.paths[:-1] + ["scripts/ci/not-a-gate.sh"]]:
            with self.subTest(paths=paths[-1:]):
                with self.assertRaises(ValueError):
                    verify(paths)

    def test_corrupt_source_or_promoted_claim_refuses(self):
        mutations = [
            lambda m: m["gates"][0].update(sha256="0" * 64),
            lambda m: m["gates"][0]["current_revision"].update(sha256="0" * 64),
            lambda m: m["gates"][0]["current_revision"].update(reason=""),
            lambda m: m.update(runtime_replay=True),
            lambda m: m["gates"][0].update(custody_verification_executes_gate=True),
            lambda m: m.update(baseline_commit="HEAD"),
            lambda m: m["snapshot_dependencies"].update(
                {next(iter(m["snapshot_dependencies"])): "0" * 64}),
        ]
        for mutate in mutations:
            manifest = copy.deepcopy(self.manifest)
            mutate(manifest)
            with self.assertRaises(ValueError):
                verify(self.paths, manifest)


if __name__ == "__main__":
    unittest.main()

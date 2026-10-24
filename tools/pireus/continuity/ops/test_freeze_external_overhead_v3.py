#!/usr/bin/env python3
"""V3 freeze and custody refusal checks; never launch a Slurm workload."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import freeze_external_overhead_v3 as f
import external_overhead_custody_v3 as c

class FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)/"frozen"
        f.create(cls.root)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def test_materialized_inventory_and_exact_helpers(self):
        spec=f.verify(self.root)
        self.assertEqual(spec["orchestration_sha256"],f.orchestration_hashes())
        self.assertEqual(spec["acceptance"],spec["screening_specification"]["acceptance"])
        self.assertEqual(spec["runtime_sha256"]["baseline"],spec["screening_specification"]["runtime_sha256"]["baseline"])
    def test_changed_cpu_prerequisite_refused(self):
        spec=f.evaluator.specification()
        spec["cpu_prerequisite"]["qualification_sha256"]="0"*64
        with patch.object(f.evaluator,"specification",return_value=spec):
            with self.assertRaisesRegex(ValueError,"CPU prerequisite changed"):f.specification()
    def test_second_creation_refused(self):
        with self.assertRaises(FileExistsError):f.create(self.root)
    def test_tampered_input_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"copy";shutil.copytree(self.root,root)
            target=root/"without-feedback/offline-bundle.json"
            target.chmod(0o644);target.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError,"artifact mismatch"):f.verify(root)
    def test_unknown_freeze_file_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"copy";shutil.copytree(self.root,root)
            (root/"extra.py").write_text("pass")
            with self.assertRaisesRegex(ValueError,"unexpected frozen"):f.verify(root)
    def test_helper_binding_tamper_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"copy";shutil.copytree(self.root,root)
            path=root/"execution-freeze.json";spec=json.loads(path.read_bytes())
            spec["orchestration_sha256"]["ops/launch_external_overhead_v3.py"]="0"*64
            path.write_bytes(f.encoded(spec))
            with self.assertRaisesRegex(ValueError,"helper identity"):f.verify(root)
    def test_actual_v3_cpu_identity_is_valid_without_model_claim(self):
        root=f.HERE/"validation/external-container-cpu-v3-20260909/attempt-11982/collected/rank-0"
        binding=json.loads((root/"binding.json").read_bytes())
        target=json.loads((root/"target.json").read_bytes())
        worker={"uid":binding["expected"]["worker_uid"],"boot_id":target["boot_id"]}
        runtime={"external_memory_observer.py":binding["observer_helper_sha256"],"offline_generate.py":target["entry_sha256"]}
        rows=c.external_identity(root,worker,"11982",0,[{"pid":target["pid"]}],runtime)
        self.assertEqual(rows[-1]["stage"],"TARGET_INVALIDATED")
        with tempfile.TemporaryDirectory() as tmp:
            mutated=Path(tmp)/"rank";shutil.copytree(root,mutated)
            binding["observed"]["job"]="wrong"
            (mutated/"binding.json").write_text(json.dumps(binding))
            ack=json.loads((mutated/"attached.json").read_bytes())
            ack["binding_sha256"]=f.digest((mutated/"binding.json").read_bytes())
            (mutated/"attached.json").write_text(json.dumps(ack))
            with self.assertRaisesRegex(ValueError,"observed binding"):
                c.external_identity(mutated,worker,"11982",0,[{"pid":target["pid"]}],runtime)
if __name__=="__main__":unittest.main()

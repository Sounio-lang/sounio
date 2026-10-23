import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import host_decode_journal_attempt as adapter

class JournalAttemptTests(unittest.TestCase):
    def test_protocol_preserves_workload_and_changes_only_runtime_transport(self):
        old=adapter.HERE/"validation/external-overhead-v3-preparation-20260910/host-decode-probe/source-v2/diagnostic-protocol.json"
        before=json.loads(old.read_bytes());after=adapter.context().protocol()
        for key in ("files_sha256","required_execution_profile","batch_size","max_new_tokens","operations"):
            self.assertEqual(before[key],after[key])
        changed=[n for n in before["runtime_sha256"] if before["runtime_sha256"][n]!=after["runtime_sha256"][n]]
        self.assertEqual(changed,["offline_generate.py"])
        self.assertFalse(after["claim_limits"]["pilot_acceptance"])
    def test_protocol_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"protocol.json";p.write_bytes(adapter.PROTOCOL.read_bytes()+b" ")
            with patch.object(adapter,"PROTOCOL",p),self.assertRaisesRegex(ValueError,"protocol changed"):adapter.context()
    def test_old_source_checks_refused(self):
        a=adapter.context();spec=a.specification()
        rows=[dict(name=name,head_sha="7c0c15d445ddcbc861568e26abebd60462338a17",status="completed",conclusion="success",id=i,started_at="2026-09-10T05:00:00Z") for i,name in enumerate(spec["required_source_checks"])]
        with self.assertRaises((ValueError,AssertionError)):a.source_checks(spec,rows)
    def test_materialization_preserves_pinned_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"freeze";a=adapter.context();spec=a.create(root)
            self.assertEqual(a.verify(root),spec)
            self.assertEqual(spec["runtime_sha256"]["baseline"]["offline_generate.py"],adapter.journals.RUNTIME_SHA)
            self.assertIn("ops/host_decode_journal_custody.py",spec["orchestration_sha256"])
    def test_composition_propagates_base_rejection_before_journal_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frozen=root/"freeze";frozen.mkdir();(frozen/"execution-freeze.json").write_bytes(b"freeze")
            spec={"source_commit":"source"}
            receipt=dict(schema="pireus-host-journal-composed-custody-v1",source_commit="source",
                freeze_sha256=hashlib.sha256(b"freeze").hexdigest(),base_collection_sha256="base",journal_collection_sha256="journal",job="11987")
            raw=json.dumps(receipt).encode();(root/"collection.json").write_bytes(raw)
            from types import SimpleNamespace
            def reject(*args):raise ValueError("source custody rejected")
            context=SimpleNamespace(verify=lambda p:spec,inspect_collection=reject)
            with patch.object(adapter,"context",return_value=context),patch.object(adapter.journals,"inspect") as journal:
                with self.assertRaisesRegex(ValueError,"source custody rejected"):adapter.inspect(frozen,root,hashlib.sha256(raw).hexdigest())
                journal.assert_not_called()

def load_tests(loader,tests,pattern):
    path=Path(__file__).with_name("test_host_decode_attempt.py")
    spec=importlib.util.spec_from_file_location("journal_attempt_inherited_tests",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.a=adapter.context()
    tests.addTests(loader.loadTestsFromModule(module))
    return tests
if __name__=="__main__":unittest.main()

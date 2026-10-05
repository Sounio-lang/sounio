import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
SOURCE=Path(__file__).resolve().parents[1]/"validation/external-overhead-v3-preparation-20260910/host-decode-probe/journal-v1/protocol/driver/driver.py"
class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        spec=importlib.util.spec_from_file_location("controlled_journal_driver",SOURCE)
        self.d=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.d)
        root=Path(self.tmp.name);self.d.ROOT=root;self.d.FROZEN=root/"freeze";self.d.FROZEN.mkdir()
        self.d.BASE=root;self.d.FROZEN.joinpath("execution-freeze.json").write_bytes(b"frozen")
        conf=root/"slurm.conf";conf.write_bytes(b"synthetic")
        self.contract=dict(driver_sha256=self.d.sha(SOURCE),slurm_conf=str(conf),slurm_conf_sha256=self.d.sha(conf),
            freeze_sha256=self.d.sha(self.d.FROZEN/"execution-freeze.json"),orchestration_sha256={},source_commit="source")
        (root/"contract.json").write_text(json.dumps(self.contract))
        self.attempt=SimpleNamespace(verify=Mock(return_value={"source_commit":"source"}),readiness=Mock(return_value={"ready":True}),launch=Mock())
        self.adapter=SimpleNamespace(context=lambda:self.attempt,collect=Mock(),inspect=Mock(return_value={"complete_probe_windows":True,"pilot_acceptance":False}))
    def invoke(self,mode="--execute",tmux=True):
        env={"TMUX":"synthetic"} if tmux else {}
        with patch.object(sys,"argv",["driver",mode]),patch.dict("os.environ",env,clear=True),patch.dict(sys.modules,{"host_decode_journal_attempt":self.adapter}):
            return self.d.main()
    def test_check_has_no_readiness_or_submission(self):
        self.assertEqual(self.invoke("--check"),0)
        self.attempt.readiness.assert_not_called();self.attempt.launch.assert_not_called()
    def test_no_tmux_and_existing_attempt_refuse_before_network(self):
        with self.assertRaisesRegex(ValueError,"tmux"):self.invoke(tmux=False)
        (self.d.ROOT/"attempt-entered.json").write_text("preserve")
        with self.assertRaisesRegex(ValueError,"already entered"):self.invoke()
        self.attempt.readiness.assert_not_called();self.attempt.launch.assert_not_called()
    def test_failed_source_checks_do_not_enter_attempt(self):
        self.attempt.readiness.side_effect=ValueError("CI not accepted")
        with self.assertRaisesRegex(ValueError,"CI not accepted"):self.invoke()
        self.assertFalse((self.d.ROOT/"attempt-entered.json").exists());self.attempt.launch.assert_not_called()
    def test_launch_refusal_is_saved_and_not_retried(self):
        self.attempt.launch.side_effect=ValueError("pair occupied")
        self.assertEqual(self.invoke(),1)
        self.attempt.launch.assert_called_once()
        self.assertTrue((self.d.ROOT/"launch-error.json").exists());self.adapter.collect.assert_not_called()
        with self.assertRaisesRegex(ValueError,"already entered"):self.invoke()
    def test_terminal_collection_failure_does_not_repeat_model(self):
        def launch(*args):
            stage=self.d.ROOT/"stage";stage.mkdir();(stage/"exit-code").write_text("0")
            (stage/"launch.log").write_text(json.dumps({"stage":"OVERHEAD_RUNTIME_VERIFIED","job":"123"})+"\n")
            return 0
        self.attempt.launch.side_effect=launch;self.adapter.collect.side_effect=ValueError("bad journal")
        self.assertEqual(self.invoke(),1)
        self.attempt.launch.assert_called_once();self.adapter.collect.assert_called_once()
        self.assertTrue((self.d.ROOT/"collection-or-inspection-error.json").exists())
    def test_driver_digest_failure_precedes_readiness(self):
        self.contract["driver_sha256"]="0"*64
        (self.d.ROOT/"contract.json").write_text(json.dumps(self.contract))
        with self.assertRaisesRegex(ValueError,"driver changed"):self.invoke()
        self.attempt.readiness.assert_not_called()
if __name__=="__main__":unittest.main()

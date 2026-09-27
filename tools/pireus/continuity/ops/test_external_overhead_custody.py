#!/usr/bin/env python3
import json,shutil,tempfile
from pathlib import Path
import unittest
from external_overhead_custody import accounting,load_packet,external_identity,digest

HERE=Path(__file__).resolve().parents[1]
class CustodyTests(unittest.TestCase):
    def row(self,state="COMPLETED",exitcode="0:0"):
        return ("11975|pireus-inkling-offline-generate|"+state+"|"+exitcode+"|gpuorangefs-multi-spark-3c59,gpuorangefs-multi-spark-8e54|2026-09-09T20:00:00|2026-09-09T20:01:00\n").encode()
    def test_terminal_success(self):
        self.assertEqual(accounting(self.row(),"11975",True)[2],"COMPLETED")
    def test_duplicate_accounting_refused(self):
        with self.assertRaises(ValueError):accounting(self.row()*2,"11975")
    def test_pending_refused(self):
        with self.assertRaises(ValueError):accounting(self.row("RUNNING"),"11975")
    def test_failed_preserved_not_qualified(self):
        self.assertEqual(accounting(self.row("FAILED","75:0"),"11975")[2],"FAILED")
        with self.assertRaises(ValueError):accounting(self.row("FAILED","75:0"),"11975",True)
    def packet(self,tmp,missing=None):
        p=Path(tmp);(p/"data").write_bytes(b"receipt")
        c=dict(schema="pireus-external-overhead-custody-v1",files_sha256={"data":digest(b"receipt")},missing=missing or {})
        raw=json.dumps(c).encode();(p/"collection.json").write_bytes(raw)
        return p,digest(raw)
    def test_changed_packet_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,pin=self.packet(tmp);load_packet(p,pin);(p/"data").write_bytes(b"altered")
            with self.assertRaises(ValueError):load_packet(p,pin)
    def test_missing_is_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,pin=self.packet(tmp,{"response":"/missing"})
            with self.assertRaises(ValueError):load_packet(p,pin)
    def test_collection_pin_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,pin=self.packet(tmp)
            with self.assertRaises(ValueError):load_packet(p,"0"*64)
    def cpu_identity(self,p):
        # Reuse real CPU handoff as an identity-only fixture, not model evidence.
        start=json.loads((p/"target.json").read_bytes())
        binding=json.loads((p/"binding.json").read_bytes())
        worker={"boot_id":start["boot_id"],"uid":binding["expected"]["worker_uid"]}
        runtime={"offline_generate.py":start["entry_sha256"],"external_memory_observer.py":binding["observer_helper_sha256"]}
        return worker,start,runtime
    def test_real_handoff_identity(self):
        p=HERE/"validation/external-container-cpu-v2-20260909/attempt/rank-0"
        w,t,r=self.cpu_identity(p)
        rows=external_identity(p,w,"11975",0,[{"pid":t["pid"]}],r)
        self.assertEqual(rows[-1]["stage"],"TARGET_INVALIDATED")
    def test_wrong_ack_target_refused(self):
        source=HERE/"validation/external-container-cpu-v2-20260909/attempt/rank-0"
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"rank";shutil.copytree(source,p)
            w,t,r=self.cpu_identity(p)
            a=json.loads((p/"attached.json").read_bytes());a["pid"]+=1
            (p/"attached.json").write_text(json.dumps(a))
            with self.assertRaises(ValueError):external_identity(p,w,"11975",0,[{"pid":t["pid"]}],r)
if __name__=="__main__":unittest.main()

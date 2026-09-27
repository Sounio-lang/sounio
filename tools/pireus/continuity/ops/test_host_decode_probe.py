#!/usr/bin/env python3
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
import prepare_host_decode_probe as probe

SOURCE=Path(__file__).resolve().parents[1]/"validation/external-overhead-v3-preparation-20260910/baseline-11983-negative/baseline-collected/rank-1/runtime/offline_generate.py"
class ProbeTests(unittest.TestCase):
    def test_reversible_only_declared_blocks(self):
        raw=SOURCE.read_bytes();text=probe.prepare(raw).decode()
        self.assertEqual(text.replace(probe.REPLACEMENT,probe.ANCHOR).replace(probe.HELPER+"\n","").encode(),raw)
        tree=ast.parse(text)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="host_decode_probe"]
        self.assertEqual(len(calls),2)
        self.assertEqual(text.count('if item["index"] == 0 and step < 15:'),2)
    def test_changed_source_refused(self):
        with self.assertRaisesRegex(ValueError,"source identity"):probe.prepare(SOURCE.read_bytes()+b"\n")
    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"probe.py";out.write_text("preserve")
            with self.assertRaises(FileExistsError):probe.write(SOURCE,out)
            self.assertEqual(out.read_text(),"preserve")
    def test_missing_counters_remain_unknown(self):
        class Unavailable:
            def __init__(self,path):pass
            def read_text(self):raise PermissionError("denied")
        class Cuda:
            def memory_allocated(self):return 123
            def memory_reserved(self):raise RuntimeError("unavailable")
        env=dict(Path=Unavailable,json=json,time=time,os=types.SimpleNamespace(
            environ={"SLURM_JOB_ID":"synthetic","PIREUS_RANK":"0"},getpid=lambda:7),
            torch=types.SimpleNamespace(cuda=Cuda()))
        exec(probe.HELPER,env)
        out=io.StringIO()
        with contextlib.redirect_stdout(out):env["host_decode_probe"]("HOST_DECODE_BEGIN",0,1)
        row=json.loads(out.getvalue())
        self.assertTrue(all(r["raw"] is None and r["error"]=="PermissionError" for r in row["files"].values()))
        self.assertIsNone(row["cuda_allocator"]["memory_reserved"]["value"])
        self.assertFalse(row["device_synchronized"])
    def test_capture_preserves_units_and_timestamp_containment(self):
        class Raw:
            def __init__(self,path):pass
            def read_text(self):return "MemFree: 123 kB\n"
        env=dict(Path=Raw,json=json,time=time,os=types.SimpleNamespace(
            environ={"SLURM_JOB_ID":"synthetic","PIREUS_RANK":"1"},getpid=lambda:8),
            torch=types.SimpleNamespace(cuda=types.SimpleNamespace(memory_allocated=lambda:1,memory_reserved=lambda:2)))
        exec(probe.HELPER,env);out=io.StringIO()
        with contextlib.redirect_stdout(out):env["host_decode_probe"]("HOST_DECODE_END",0,15)
        row=json.loads(out.getvalue())
        for metric in row["files"].values():
            self.assertEqual(metric["raw"],"MemFree: 123 kB\n")
            self.assertGreaterEqual(metric["monotonic_ns"],row["monotonic_ns"])
            self.assertLessEqual(metric["monotonic_ns"]+metric["duration_ns"],row["monotonic_ns"]+row["read_duration_ns"])
if __name__=="__main__":unittest.main()

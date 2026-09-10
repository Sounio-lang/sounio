#!/usr/bin/env python3
import ast
import contextlib
import io
import time
import types
import concurrent.futures
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import host_decode_journal as journal
import prepare_host_decode_journal as prepare

BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/"validation/external-overhead-v3-preparation-20260910/host-decode-probe/offline_generate.py"
def row(rank=0,n=0):
    return dict(schema="pireus-host-decode-probe-v1",job="11987",rank=str(rank),pid=42+rank,
                index=0,step=n//2+1,stage=("HOST_DECODE_BEGIN","HOST_DECODE_END")[n%2],
                raw="MemAvailable: 123 kB\n"*4000)
def read(path,rank=0):
    raw=path.read_bytes()
    return journal.read_journal(path,hashlib.sha256(raw).hexdigest(),"11987",rank,42+rank)
class JournalTests(unittest.TestCase):
    def test_concurrent_rank_journals_preserve_large_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            def run(rank):
                j=journal.HostDecodeJournal(tmp,"11987",rank,42+rank)
                for n in range(30):
                    receipt=j.write(row(rank,n))
                    self.assertLess(len(json.dumps(receipt)),512)
                j.close()
                self.assertEqual(read(j.path,rank),[row(rank,n) for n in range(30)])
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(run,range(2)))
    def test_existing_file_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"host-decode-11987-0.jsonl";p.write_text("preserve")
            with self.assertRaises(FileExistsError):journal.HostDecodeJournal(tmp,"11987",0,42)
            self.assertEqual(p.read_text(),"preserve")
            link=Path(tmp)/"host-decode-11987-1.jsonl";link.symlink_to(p)
            with self.assertRaises(FileExistsError):journal.HostDecodeJournal(tmp,"11987",1,43)
            self.assertEqual(p.read_text(),"preserve")
    def test_short_and_interrupted_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=journal.HostDecodeJournal(tmp,"11987",0,42);real=j.stream
            class Short:
                closed=False
                first=True
                def write(self,data):
                    if self.first:self.first=False;raise InterruptedError()
                    return real.write(data[:113])
                def close(self):real.close();self.closed=True
            j.stream=Short();j.write(row());j.close()
            self.assertEqual(read(j.path),[row()])
    def test_failed_write_latches_and_partial_line_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=journal.HostDecodeJournal(tmp,"11987",0,42);real=j.stream
            class Broken:
                closed=False
                first=True
                def write(self,data):
                    if self.first:self.first=False;return real.write(data[:127])
                    return 0
                def close(self):real.close();self.closed=True
            j.stream=Broken()
            with self.assertRaises(OSError):j.write(row())
            with self.assertRaisesRegex(ValueError,"unavailable"):j.write(row())
            with self.assertRaisesRegex(ValueError,"incomplete"):read(j.path)
    def test_rank_order_pid_and_limit(self):
        for key,value in [("rank","1"),("pid",99),("index",1),("step",2),("job","11988")]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                j=journal.HostDecodeJournal(tmp,"11987",0,42);bad=row();bad[key]=value
                with self.assertRaisesRegex(ValueError,"scope/order"):j.write(bad)
        with tempfile.TemporaryDirectory() as tmp:
            j=journal.HostDecodeJournal(tmp,"11987",0,42)
            for n in range(30):j.write(row(n=n))
            with self.assertRaisesRegex(ValueError,"scope/order"):j.write(row(n=30))
    def test_reader_rejects_digest_identity_and_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=journal.HostDecodeJournal(tmp,"11987",0,42);j.write(row());j.close()
            with self.assertRaisesRegex(ValueError,"digest"):journal.read_journal(j.path,"0"*64,"11987",0,42)
            with self.assertRaisesRegex(ValueError,"identity/order"):read(j.path,1)
            j.path.write_bytes(b'{"broken":\n')
            with self.assertRaises(ValueError):read(j.path)
    def test_empty_and_record_boundary_partial_remain_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=journal.HostDecodeJournal(tmp,"11987",0,42)
            self.assertEqual(read(j.path),[])
            j.write(row());j.close()
            self.assertEqual(len(read(j.path)),1)
    def test_generated_hook_writes_journal_and_small_stdout_receipt(self):
        tree=ast.parse(prepare.prepare(SOURCE.read_bytes()))
        hook=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="host_decode_probe")
        with tempfile.TemporaryDirectory() as tmp:
            env=dict(Path=Path,json=json,time=time,_host_decode_journal=None,
                     os=types.SimpleNamespace(environ={"SLURM_JOB_ID":"11987","PIREUS_RANK":"0"},getpid=lambda:42),
                     torch=types.SimpleNamespace(cuda=types.SimpleNamespace(memory_allocated=lambda:1,memory_reserved=lambda:2)),
                     HostDecodeJournal=lambda directory,job,rank,pid:journal.HostDecodeJournal(tmp,job,rank,pid))
            exec(compile(ast.Module(body=[hook],type_ignores=[]),"<generated-hook>","exec"),env)
            output=io.StringIO()
            with contextlib.redirect_stdout(output):env["host_decode_probe"]("HOST_DECODE_BEGIN",0,1)
            env["_host_decode_journal"].close()
            record=read(Path(tmp)/"host-decode-11987-0.jsonl")[0]
            self.assertEqual(record["schema"],"pireus-host-decode-probe-v1")
            self.assertEqual(set(record["files"]),{"host_meminfo","host_vmstat","process_status"})
            self.assertLess(len(output.getvalue()),512)
            self.assertEqual(json.loads(output.getvalue())["stage"],"HOST_DECODE_JOURNAL_RECORD")

    def test_preparation_changes_only_transport(self):
        raw=SOURCE.read_bytes();result=prepare.prepare(raw)
        text=result.decode();ast.parse(text)
        helper=Path(journal.__file__).read_text().split("\ndef read_journal(",1)[0]
        block=helper+"\n_host_decode_journal = None\n\n"
        original=text.replace(block+prepare.INSERT,prepare.INSERT).replace(prepare.REPLACEMENT,prepare.ANCHOR)
        self.assertEqual(original.encode(),raw)
        self.assertEqual(text.count('if item["index"] == 0 and step < 15:'),2)
        with self.assertRaisesRegex(ValueError,"source identity"):prepare.prepare(raw+b"\n")
if __name__=="__main__":unittest.main()

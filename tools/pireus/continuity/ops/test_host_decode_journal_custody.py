import base64,json,tempfile,unittest
from pathlib import Path
import host_decode_journal_custody as c
import test_host_decode_attempt as parent_tests

BASE=Path(__file__).resolve().parents[1]
def put(root,name,obj):
    raw=obj if isinstance(obj,bytes) else (json.dumps(obj)+"\n").encode()
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    return c.sha(raw)
class CustodyTests(unittest.TestCase):
    def fixture(self,tmp):
        root=Path(tmp)/"base";root.mkdir()
        runtime=(BASE/"validation/external-overhead-v3-preparation-20260910/host-decode-probe/journal-v1/offline_generate.py").read_bytes()
        workers=[dict(node=n,pod=f"worker-{i}",uid=f"uid-{i}",boot_id=f"boot-{i}") for i,n in enumerate(["spark-3c59","spark-8e54"])]
        start=dict(source_commit="synthetic",freeze_sha256="synthetic",workers=workers,
                   runtime_before_sha256=[{"offline_generate.py":c.sha(runtime)}]*2,input_sha256="synthetic")
        files={}
        files["start.json"]=put(root,"start.json",start)
        acct=(BASE/"validation/external-overhead-v3-preparation-20260910/host-decode-probe/source-v2/result-11987/collection/accounting.txt").read_bytes()
        files["accounting.txt"]=put(root,"accounting.txt",acct)
        self.rows={}
        for rank,w in enumerate(workers):
            values={
                f"rank-{rank}/runtime/offline_generate.py":runtime,
                f"rank-{rank}/boot-id.txt":(w["boot_id"]+"\n").encode(),
                f"rank-{rank}/runtime-before.json":dict(stage="OVERHEAD_RUNTIME_VERIFIED",job="11987",rank=str(rank),worker_uid=w["uid"],boot_id=w["boot_id"],runtime_sha256=start["runtime_before_sha256"][rank],input_sha256="synthetic",monotonic_ns=0)}
            for moment in ("before","after"):values[f"worker-{rank}-{moment}.json"]=dict(metadata=dict(uid=w["uid"]),spec=dict(nodeName=w["node"]))
            lifecycle=[dict(job="11987",rank=str(rank),pid=99+rank,monotonic_ns=t) for t in (0,4000)]
            values[f"worker-receipts/lifecycle-11987-{rank}.jsonl"]=b"".join((json.dumps(row)+"\n").encode() for row in lifecycle)
            for name,obj in values.items():files[name]=put(root,name,obj)
            rows=parent_tests.ProbeTests().rows()
            for row in rows:row.update(job="11987",rank=str(rank),pid=99+rank)
            self.rows[rank]=b"".join((json.dumps(row)+"\n").encode() for row in rows)
        manifest=dict(schema="pireus-host-decode-custody-v1",arm="baseline",job="11987",source_commit="synthetic",freeze_sha256="synthetic",terminal_state="COMPLETED",files_sha256=files,missing={})
        pin=put(root,"collection.json",manifest)
        return root,pin
    def remote(self,args):
        rank=int(args[args.index("pod")+1].split("-")[-1]) if "get" in args else int(args[args.index("exec")+1].split("-")[-1])
        if "get" in args:return json.dumps(dict(metadata=dict(uid=f"uid-{rank}"),spec=dict(nodeName=["spark-3c59","spark-8e54"][rank]))).encode()
        return json.dumps(dict(boot=f"boot-{rank}",data=base64.b64encode(self.rows[rank]).decode() if self.rows[rank] is not None else None)).encode()
    def collected(self,tmp):
        base,pin=self.fixture(tmp);out=Path(tmp)/"journal"
        c.collect(base,pin,out,run=self.remote)
        return base,pin,out,c.sha((out/"collection.json").read_bytes())
    def test_complete_collection_and_inspection_are_not_model_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.collected(tmp);result=c.inspect(*args)
            self.assertTrue(result["complete_probe_windows"])
            self.assertFalse(result["source_qualified"])
            self.assertFalse(result["full_diagnostic_custody_qualified"])
            self.assertFalse(result["pilot_acceptance"])
    def test_partial_and_missing_journals_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,pin=self.fixture(tmp);out=Path(tmp)/"journal"
            self.rows[0]=self.rows[0].splitlines(keepends=True)[0];self.rows[1]=None
            result=c.collect(base,pin,out,run=self.remote)
            self.assertIn("rank-1.jsonl",result["missing"])
            summary=c.inspect(base,pin,out,c.sha((out/"collection.json").read_bytes()))
            self.assertFalse(summary["complete_probe_windows"])
    def test_truncated_line_preserved_then_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,pin=self.fixture(tmp);out=Path(tmp)/"journal";self.rows[0]=self.rows[0][:-5]
            c.collect(base,pin,out,run=self.remote)
            self.assertEqual((out/"rank-0.jsonl").read_bytes(),self.rows[0])
            with self.assertRaisesRegex(ValueError,"incomplete"):c.inspect(base,pin,out,c.sha((out/"collection.json").read_bytes()))
    def test_corrupt_journal_and_wrong_pid_rejected(self):
        for resign in (False,True):
            with self.subTest(resign=resign),tempfile.TemporaryDirectory() as tmp:
                base,pin,out,jpin=self.collected(tmp)
                p=out/"rank-0.jsonl";p.write_bytes(p.read_bytes().replace(b'"pid": 99',b'"pid": 88'))
                if resign:
                    m=json.loads((out/"collection.json").read_bytes());m["files_sha256"]["rank-0.jsonl"]=c.sha(p.read_bytes());jpin=put(out,"collection.json",m)
                with self.assertRaises(ValueError):c.inspect(base,pin,out,jpin)
    def test_worker_change_refused_and_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,pin=self.fixture(tmp);out=Path(tmp)/"journal"
            def changed(args):
                value=json.loads(self.remote(args))
                if "metadata" in value:value["metadata"]["uid"]="replacement"
                return json.dumps(value).encode()
            with self.assertRaisesRegex(ValueError,"worker changed"):c.collect(base,pin,out,run=changed)
            with self.assertRaises(FileExistsError):c.collect(base,pin,out,run=self.remote)
    def test_boot_or_chronology_mutation_rejected_even_with_new_digest(self):
        for kind in ("boot","chronology"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                base,pin,out,jpin=self.collected(tmp)
                name="worker-0-boot.txt" if kind=="boot" else "rank-0.jsonl"
                p=out/name
                if kind=="boot":p.write_text("different-boot\n")
                else:
                    rows=[json.loads(line) for line in p.read_bytes().splitlines()]
                    rows[0]["monotonic_ns"]=5000
                    p.write_bytes(b"".join((json.dumps(row)+"\n").encode() for row in rows))
                m=json.loads((out/"collection.json").read_bytes())
                m["files_sha256"][name]=c.sha(p.read_bytes());jpin=put(out,"collection.json",m)
                with self.assertRaises(ValueError):c.inspect(base,pin,out,jpin)

    def test_old_runtime_cannot_supply_new_journal_custody(self):
        root=BASE/"validation/external-overhead-v3-preparation-20260910/host-decode-probe/source-v2/result-11987/collection"
        with self.assertRaisesRegex(ValueError,"journal runtime required"):
            c.base_collection(root,c.sha((root/"collection.json").read_bytes()))
if __name__=="__main__":unittest.main()

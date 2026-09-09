#!/usr/bin/env python3
"""Synthetic custody/scheduler fixtures only; no allocation or live collection."""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import cpu_v3_attempt as a

FROZEN=a.PACKET/"freeze"
OLD=a.HERE/"validation/external-container-cpu-v2-20260909/attempt"
class AttemptTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.p=a.packet(FROZEN);self.pin=a.digest((FROZEN/"protocol.json").read_bytes())
        self.ws=a.read(OLD/"manifest.json")["workers"]
    def record(self,state="COMPLETED",rc="0:0"):
        return "789|"+a.JOB_NAME+"|"+state+"|"+rc+"|"+",".join(a.SLURM_NODES)+"|2026-09-09T23:00:00|2026-09-09T23:00:10\n"
    def start(self):
        return dict(schema="pireus-cpu-v3-start",source_commit=self.p["source_commit"],protocol_sha256=self.pin,
            runtime_sha256=self.p["runtime_files_sha256"],workers=self.ws,command=a.command(self.ws,self.p,self.pin),
            launcher_sha256=a.digest(Path(a.__file__).read_bytes()),orchestration_sha256=a.orchestration_hashes())
    def test_command_cpu_only_guard_and_isolation(self):
        argv=a.command(self.ws,self.p,self.pin)
        for flag in ("--exclusive","--mem=512M","--time=5","-c2"):self.assertIn(flag,argv)
        self.assertIn("--reserve-gib 33",argv[-1])
        self.assertIn(a.remote_root(self.pin),argv[-1])
        self.assertIn("cpu-target.py",argv[-1])
        self.assertNotIn("qualify_model.py",argv[-1])
        self.assertNotIn("exec /scratch/pireus/runtime/serve_rank.sh",argv[-1])
        self.assertEqual(subprocess.run(["bash","-n"],input=argv[-1],text=True).returncode,0)
    def test_reversed_workers_refused(self):
        with self.assertRaisesRegex(ValueError,"order"):a.command(self.ws[::-1],self.p,self.pin)
    def test_no_tmux_no_network_or_allocation(self):
        with patch.dict(os.environ,{},clear=True),patch.object(a,"readiness") as ready:
            with self.assertRaisesRegex(ValueError,"tmux"):a.launch(FROZEN,self.root/"stage")
            ready.assert_not_called()
    def test_existing_attempt_not_retried(self):
        with patch.dict(os.environ,{"TMUX":"synthetic"}),patch.object(a,"readiness") as ready:
            with self.assertRaisesRegex(ValueError,"no retry"):a.launch(FROZEN,self.root)
            ready.assert_not_called()
    def test_ci_refusal_before_stage_or_transfer(self):
        stage=self.root/"new"
        with patch.dict(os.environ,{"TMUX":"synthetic"}),patch.object(a,"readiness",side_effect=ValueError("pending")),patch.object(a,"kube") as kube:
            with self.assertRaisesRegex(ValueError,"pending"):a.launch(FROZEN,stage)
            self.assertFalse(stage.exists());kube.assert_not_called()
    def test_accounting_requires_unique_successful_named_pair(self):
        self.assertEqual(a.accounting(self.record(),"789",True)[2],"COMPLETED")
        for raw in (self.record()*2,self.record("RUNNING"),self.record().replace(a.JOB_NAME,"wrong"),self.record("FAILED","75:0")):
            with self.assertRaises(ValueError):a.accounting(raw,"789",True)
    def test_partial_failed_collection_preserved(self):
        stage=self.root/"stage";stage.mkdir()
        a.write(stage/"start.json",self.start())
        for n in ("readiness.json","preflight.json"):a.write(stage/n,{})
        (stage/"launch.log").write_text("synthetic failure\n");(stage/"exit-code").write_text("75\n")
        def kube(*args):
            if "sacct" in args:return self.record("FAILED","75:0")
            code=args[-1];paths=ast.literal_eval(code.splitlines()[1].removeprefix("paths="))
            return json.dumps({k:None for k in paths})
        with patch.object(a,"kube",side_effect=kube),patch.object(a,"read_live",side_effect=lambda w:{"metadata":{"uid":w["uid"]}}):
            r=a.collect(FROZEN,stage,self.root/"collected","789")
        self.assertEqual(r["terminal_state"],"FAILED");self.assertTrue(r["missing"]);self.assertFalse(r["qualified"])
        with self.assertRaisesRegex(ValueError,"incomplete"):
            a.qualify(FROZEN,self.root/"collected",a.digest((self.root/"collected/collection.json").read_bytes()))
    def fixture(self):
        out=self.root/"packet";out.mkdir()
        start=self.start();a.write(out/"start.json",start)
        ready=dict(source_commit=self.p["source_commit"],protocol_sha256=self.pin,source_checks={
            name:dict(name=name,head_sha=self.p["source_commit"],id=i,status="completed",conclusion="success",started_at="2026-09-09T22:00:00Z")
            for i,name in enumerate(self.p["required_source_checks"])})
        a.write(out/"readiness.json",ready);a.write(out/"preflight.json",[])
        (out/"exit-code").write_text("0\n");(out/"accounting.txt").write_text(self.record())
        log=[]
        for rank,w in enumerate(self.ws):
            dest=out/f"rank-{rank}";shutil.copytree(OLD/f"rank-{rank}",dest)
            shutil.copytree(FROZEN/"source",dest/"source",dirs_exist_ok=True)
            for moment in ("before","after"):
                a.write(out/f"worker-{rank}-{moment}.json",{"metadata":{"uid":w["uid"]},"spec":{"nodeName":w["node"]}})
            (dest/"boot-id.txt").write_text(w["boot_id"]+"\n")
            phases=[json.loads(s) for s in (dest/"cpu-phases.jsonl").read_text().splitlines()]
            target=a.read(dest/"target.json");target.update(job="789",rank=str(rank),boot_id=w["boot_id"],
                entry_sha256=self.p["runtime_files_sha256"]["cpu-target.py"])
            (dest/"target.json").write_text(json.dumps(target))
            binding=a.read(dest/"binding.json")
            binding["expected"].update(job="789",rank=str(rank),worker_uid=w["uid"],boot_id=w["boot_id"])
            binding["observed"].update(binding["expected"])
            binding["observed"]["cgroup"]["membership"]=binding["observed"]["cgroup"]["membership"].replace("job_11975","job_789")
            binding["observer_helper_sha256"]=self.p["runtime_files_sha256"]["external_memory_observer.py"]
            (dest/"binding.json").write_text(json.dumps(binding))
            ack=a.read(dest/"attached.json");ack.update(job="789",rank=str(rank),
                handoff_sha256=a.digest((dest/"target.json").read_bytes()),binding_sha256=a.digest((dest/"binding.json").read_bytes()))
            (dest/"attached.json").write_text(json.dumps(ack))
            journal=[json.loads(s) for s in (dest/"journal.jsonl").read_text().splitlines()]
            bp=a.digest(json.dumps(binding,sort_keys=True).encode())
            for i,r in enumerate(journal):
                r.update(job="789",rank=str(rank),schema="pireus-external-memory-observation-v2",
                         observer_profile="external-observer-deadline-v3",binding_sha256=bp)
                if r["stage"]=="OBSERVER_START":
                    r.update(scheduling="actual-start-deadline-no-catchup",observer_resource_scope="observer-process-only")
                if r["stage"]=="SAMPLE":
                    r["observer_resources"]=dict(observer_pid=r["observer_pid"],scope="observer-process-only",process_cpu_ns=1000+i*100,
                        status=dict(value={"VmRSS":20*1024**2,"VmHWM":21*1024**2},error=None,format="kB-fields",
                                    monotonic_ns=r["monotonic_ns"],duration_ns=1))
            (dest/"journal.jsonl").write_text("".join(json.dumps(r)+"\n" for r in journal))
            result=a.read(dest/"result.json");result.update(job="789",rank=str(rank),entry_sha256=target["entry_sha256"])
            result["files_sha256"]={n:a.digest((dest/n).read_bytes()) for n in result["files_sha256"]}
            (dest/"result.json").write_text(json.dumps(result))
            barrier=dict(stage="CPU_V3_RUNTIME_VERIFIED",job="789",rank=str(rank),protocol_sha256=self.pin,
                worker_uid=w["uid"],boot_id=w["boot_id"],runtime_sha256=self.p["runtime_files_sha256"],monotonic_ns=phases[0]["monotonic_ns"]-1)
            a.write(dest/"runtime-before.json",barrier);log.append(barrier)
            log.append(dict(stage="MEMORY_GUARD_CHILD_EXIT",job="789",rank=str(rank),returncode=0,minimum_bytes=64*1024**3))
        (out/"launch.log").write_text("".join(json.dumps(r)+"\n" for r in log))
        self.repin(out)
        return out
    def repin(self,out):
        c=dict(schema="pireus-cpu-v3-collection",job="789",protocol_sha256=self.pin,missing=[],terminal_state="COMPLETED",
               files_sha256={str(f.relative_to(out)):a.digest(f.read_bytes()) for f in out.rglob("*") if f.is_file() and f!=out/"collection.json"})
        (out/"collection.json").write_text(json.dumps(c))
        return a.digest((out/"collection.json").read_bytes())
    def test_synthetic_full_custody_qualifies_cpu_only(self):
        out=self.fixture();r=a.qualify(FROZEN,out,a.digest((out/"collection.json").read_bytes()))
        self.assertTrue(r["cpu_container_control_qualified"])
        self.assertFalse(r["inference_executed"]);self.assertFalse(r["loaded_model_overhead_qualified"])
    def test_modified_artifact_without_repin_refused(self):
        out=self.fixture();pin=a.digest((out/"collection.json").read_bytes())
        (out/"rank-0/journal.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError,"collected file"):a.qualify(FROZEN,out,pin)
    def test_wrong_worker_even_with_collection_repin_refused(self):
        out=self.fixture();p=out/"worker-0-after.json";v=a.read(p);v["metadata"]["uid"]="wrong";p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,"worker custody"):a.qualify(FROZEN,out,self.repin(out))
    def test_weakened_guard_command_even_with_repin_refused(self):
        out=self.fixture();p=out/"start.json";v=a.read(p);v["command"][-1]=v["command"][-1].replace("--reserve-gib 33","--reserve-gib 32")
        p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,"command"):a.qualify(FROZEN,out,self.repin(out))
    def test_guard_stop_even_with_repin_refused(self):
        out=self.fixture()
        with (out/"launch.log").open("a") as f:f.write(json.dumps({"stage":"MEMORY_GUARD_STOP","job":"789","rank":"1"})+"\n")
        with self.assertRaisesRegex(ValueError,"guardian stopped"):a.qualify(FROZEN,out,self.repin(out))
    def test_observed_binding_cannot_disagree_with_expected(self):
        out=self.fixture();d=out/"rank-0"
        p=d/"binding.json";v=a.read(p);v["observed"]["job"]="790";p.write_text(json.dumps(v))
        p=d/"attached.json";v=a.read(p);v["binding_sha256"]=a.digest((d/"binding.json").read_bytes());p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,"observed binding"):a.qualify(FROZEN,out,self.repin(out))
    def test_failed_source_ci_even_with_repin_refused(self):
        out=self.fixture();p=out/"readiness.json";v=a.read(p)
        v["source_checks"]["CI Decision"]["conclusion"]="failure";p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,"not green"):a.qualify(FROZEN,out,self.repin(out))
if __name__=="__main__":unittest.main()

#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import external_memory_observer as observer

class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.proc=self.root/"proc"
        self.pid=123;self.uid="550f527c-456c-42d7-97a0-2be38321faf3"
        self.cg=self.root/"cgroup";self.target=self.cg/"slurm/job_713/step_0"
        self.target.mkdir(parents=True)
        def write(name,value):
            p=self.proc/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
        self.write=write
        write("sys/kernel/random/boot_id","boot-a")
        write("self/mountinfo",f"1 0 0:1 / {self.cg} rw - cgroup2 cgroup rw\n2 0 0:2 /var/lib/kubelet/pods/{self.uid}/etc-hosts /etc/hosts rw - ext4 disk rw\n")
        write("123/stat",self.stat(100))
        write("123/environ","SLURM_JOB_ID=713\0PIREUS_RANK=0\0DO_NOT_RECORD=secret\0")
        write("123/cgroup","0::/slurm/job_713/step_0\n")
        write("123/status","VmRSS: 64 kB\n")
        write("123/smaps_rollup","Rss: 64 kB\nPss: 60 kB\n")
        write("meminfo","MemAvailable: 2000 kB\n")
        write("vmstat","pgfault 100\n")
        write("pressure/memory","some avg10=0.00 total=0\n")
        (self.proc/"123/ns").mkdir()
        for key in ("pid","mnt","cgroup"):(self.proc/"123/ns"/key).symlink_to(key+":[42]")
        for name in ("memory.current","memory.peak"):(self.target/name).write_text("65536\n")
        for name in ("memory.stat","memory.events","memory.events.local","memory.pressure"):(self.target/name).write_text("counter 0\n")
        self.expected=dict(job="713",rank="0",pid=123,starttime_ticks=100,worker_uid=self.uid,boot_id="boot-a")

    def stat(self,start,state="S"):
        return "123 (name with ) parens) "+state+" "+" ".join(["0"]*18+[str(start)])+"\n"

    def binding(self):
        return observer.bind(self.expected,self.proc)

    def test_valid_identity_actual_cgroup_and_units(self):
        b=self.binding();r=observer.sample(b,self.proc)
        self.assertEqual(b["observed"]["cgroup"]["path"],str(self.target))
        self.assertEqual(r["metrics"]["process_smaps_rollup"]["value"]["Pss"],60*1024)
        self.assertEqual(r["metrics"]["cgroup_memory.current"]["value"],65536)
        self.assertNotIn("secret",json.dumps(b)+json.dumps(r))

    def test_wrong_job_rank_worker_boot_and_start_refused(self):
        for key,value in (("job","714"),("rank","1"),("worker_uid","wrong"),("boot_id","wrong"),("starttime_ticks",101)):
            with self.subTest(key=key),self.assertRaises(observer.IdentityError):
                observer.bind(self.expected|{key:value},self.proc)

    def test_pid_reuse_and_disappearance(self):
        b=self.binding()
        self.write("123/stat",self.stat(101))
        with self.assertRaises(observer.IdentityError):observer.sample(b,self.proc)
        (self.proc/"123/stat").unlink()
        with self.assertRaises(observer.IdentityError):observer.sample(b,self.proc)

    def test_cgroup_migration_refused(self):
        b=self.binding()
        self.write("123/cgroup","0::/slurm/job_713/step_1\n")
        (self.cg/"slurm/job_713/step_1").mkdir()
        with self.assertRaises(observer.IdentityError):observer.sample(b,self.proc)

    def test_worker_cgroup_never_substituted(self):
        self.write("123/cgroup","0::/kubepods/worker\n")
        (self.cg/"kubepods/worker").mkdir(parents=True)
        with self.assertRaisesRegex(observer.IdentityError,"declared Slurm job"):
            self.binding()

    def test_missing_and_malformed_metrics_are_null(self):
        b=self.binding();(self.proc/"123/smaps_rollup").unlink()
        (self.target/"memory.current").write_text("unsupported\n")
        r=observer.sample(b,self.proc)
        for key in ("process_smaps_rollup","cgroup_memory.current"):
            self.assertIsNone(r["metrics"][key]["value"])
            self.assertIsNotNone(r["metrics"][key]["error"])

    def test_identity_change_during_read_discards_sample(self):
        b=self.binding();original=observer.metric
        def changed(path,kind):
            result=original(path,kind)
            self.write("123/stat",self.stat(101))
            return result
        with patch.object(observer,"metric",side_effect=changed):
            with self.assertRaisesRegex(observer.IdentityError,"starttime"):
                observer.sample(b,self.proc)

    def test_journal_invalidates_without_samples_or_overwrite(self):
        b=self.binding();p=self.root/"journal.jsonl"
        with patch.object(observer,"sample",side_effect=observer.IdentityError("target exited")):
            self.assertEqual(observer.observe(b,p,0.1,1,self.proc),3)
        rows=[json.loads(x) for x in p.read_text().splitlines()]
        self.assertEqual([r["stage"] for r in rows],["OBSERVER_START","TARGET_INVALIDATED"])
        self.assertIsNone(rows[-1]["metrics"])
        self.assertNotEqual(rows[0]["observer_pid"],rows[0]["target_pid"])
        with self.assertRaises(FileExistsError):observer.observe(b,p,0.1,1,self.proc)

    def test_mount_root_mapping_and_ambiguous_refusal(self):
        mount=f"1 0 0:1 /slurm {self.cg} rw - cgroup2 cgroup rw\n"
        mapped=observer.resolve_cgroup("0::/slurm/job_713\n",mount)
        self.assertEqual(mapped["path"],str(self.cg/"job_713"))
        for member,table in (("0::/outside\n",mount),("0::/slurm/../other\n",mount),("0::/slurm/job_713\n",mount+mount)):
            with self.assertRaises(observer.IdentityError):observer.resolve_cgroup(member,table)

if __name__=="__main__":
    unittest.main()

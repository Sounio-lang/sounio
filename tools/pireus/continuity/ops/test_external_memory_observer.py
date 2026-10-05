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

    def test_cgroup_directory_replacement_refused(self):
        b=self.binding()
        self.target.rename(self.target.with_name("old-step"))
        self.target.mkdir()
        with self.assertRaisesRegex(observer.IdentityError,"identity or cgroup"):
            observer.sample(b,self.proc)

    def test_namespace_change_refused(self):
        b=self.binding();link=self.proc/"123/ns/cgroup"
        link.unlink();link.symlink_to("cgroup:[99]")
        with self.assertRaisesRegex(observer.IdentityError,"identity or cgroup"):
            observer.sample(b,self.proc)

    def test_permission_denial_is_missing_not_zero(self):
        b=self.binding();original=Path.read_text
        forbidden=self.proc/"123/smaps_rollup"
        def read(path,*args,**kwargs):
            if path==forbidden:raise PermissionError("control")
            return original(path,*args,**kwargs)
        with patch.object(Path,"read_text",read):
            r=observer.sample(b,self.proc)
        self.assertIsNone(r["metrics"]["process_smaps_rollup"]["value"])
        self.assertEqual(r["metrics"]["process_smaps_rollup"]["error"],"PermissionError")

    def timed_journal(self,durations,seconds=1.0,oversleep_ns=0):
        binding=self.binding();clock=[0];starts=[];sleeps=[]
        def sample(*args):
            began=clock[0];starts.append(began)
            duration=durations[min(len(starts)-1,len(durations)-1)]
            clock[0]+=duration
            return dict(stage="SAMPLE",monotonic_ns=began,duration_ns=duration,
                        identity_valid=True,metrics={})
        def sleep(seconds):
            ns=round(seconds*1e9);self.assertGreater(ns,0)
            sleeps.append(ns);clock[0]+=ns+oversleep_ns
        p=self.root/"timing.jsonl"
        with patch.object(observer,"sample",side_effect=sample), \
             patch.object(observer.time,"monotonic_ns",side_effect=lambda:clock[0]), \
             patch.object(observer.time,"sleep",side_effect=sleep):
            self.assertEqual(observer.observe(binding,p,0.2,seconds,self.proc),0)
        return starts,sleeps,[json.loads(s) for s in p.read_text().splitlines()]

    def test_slow_read_consumes_wait_without_catchup_burst(self):
        starts,sleeps,rows=self.timed_journal([410_000_000,64_000_000,50_000_000])
        self.assertEqual(starts,[0,410_000_000,610_000_000,810_000_000])
        self.assertEqual(sleeps,[136_000_000,150_000_000,140_000_000])
        self.assertEqual([r["sample_gap_ns"] for r in rows if r["stage"]=="SAMPLE"],
                         [None,410_000_000,200_000_000,200_000_000])
        self.assertEqual(rows[0]["observer_profile"],"external-observer-deadline-v3")
        self.assertEqual(rows[0]["schema"],"pireus-external-memory-observation-v2")

    def test_read_overrun_remains_visible(self):
        starts,_,rows=self.timed_journal([700_000_000,10_000_000])
        self.assertEqual(starts,[0,700_000_000,900_000_000])
        samples=[r for r in rows if r["stage"]=="SAMPLE"]
        self.assertEqual(samples[1]["sample_gap_ns"],700_000_000)
        self.assertGreater(samples[1]["sample_gap_ns"],500_000_000)

    def test_scheduler_oversleep_reanchors_actual_start(self):
        starts,_,_=self.timed_journal([50_000_000],oversleep_ns=80_000_000)
        self.assertEqual(starts,[0,280_000_000,560_000_000,840_000_000])

    def test_read_past_end_does_not_start_another_sample(self):
        starts,sleeps,rows=self.timed_journal([1_200_000_000])
        self.assertEqual(starts,[0]);self.assertEqual(sleeps,[])
        self.assertEqual(rows[-1]["stage"],"OBSERVER_END")
        self.assertEqual(rows[-1]["monotonic_ns"],1_200_000_000)

    def test_fast_reads_preserve_configured_cadence(self):
        starts,_,_=self.timed_journal([10_000_000])
        self.assertEqual(starts,[0,200_000_000,400_000_000,600_000_000,800_000_000])

    def test_observer_resources_are_separate_and_in_bytes(self):
        self.write("self/status","VmRSS: 23 kB\nVmHWM: 31 kB\n")
        with patch.object(observer.time,"process_time_ns",return_value=4321):
            row=observer.sample(self.binding(),self.proc)
        own=row["observer_resources"]
        self.assertEqual(own["observer_pid"],observer.os.getpid())
        self.assertEqual(own["process_cpu_ns"],4321)
        self.assertEqual(own["status"]["value"],{"VmRSS":23552,"VmHWM":31744})
        self.assertEqual(row["metrics"]["process_status"]["value"]["VmRSS"],65536)
        self.assertNotIn("observer_resources",row["metrics"])

    def test_missing_observer_rss_is_unknown_not_zero(self):
        for value in ("Name: observer\n","VmRSS: 12 kB\n"):
            self.write("self/status",value)
            own=observer.observer_resources(self.proc)
            self.assertIsNone(own["status"]["value"])
            self.assertEqual(own["status"]["error"],"MissingObserverRSSFields")
        (self.proc/"self/status").unlink()
        self.assertEqual(observer.observer_resources(self.proc)["status"]["error"],"FileNotFoundError")

if __name__=="__main__":
    unittest.main()

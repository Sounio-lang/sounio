#!/usr/bin/env python3
"""Controller fault controls with real atomic files and mocked Docker/PostgreSQL services."""
import json, pathlib, tempfile, unittest
from relocation_host_controller import Controller, atomic, sha, PROXY
TOKEN="a"*32
class Fake(Controller):
    def __init__(self,root):
        self.now=100.;self.is_running=True;self.proxy=False;self.identity_ok=True
        super().__init__(root,clock=lambda:self.now)
    def boot(self):return "test-boot"
    def identity(self,s):
        if not self.identity_ok:raise RuntimeError("Changed source")
    def running(self):return self.is_running
    def active(self):return self.proxy
    def docker(self,*args,timeout=30):
        if args[0]=="stop":self.is_running=False
        elif args[0]=="start":self.is_running=True
        else:raise AssertionError(args)
        return b""
    def unit(self,action,name):
        if name==PROXY and action in ["start","stop"]:self.proxy=action=="start"
        return b""
    def sql(self,q):
        if "SELECT current_user" in q:return b"memory|/var/lib/postgresql/data|on|off\ndefault\n0"
        if "pg_hba_file_rules" in q:return b"t\n0"
        if "pg_reload_conf" in q:return b"t"
        if "pg_terminate_backend" in q:return b"t"
        if "pg_stat_activity" in q:return b"0"
        if "cron.launch_active_jobs" in q:
            s=self.read();p=pathlib.Path(s["source_dir"])/"postgresql.conf"
            return b"off|on" if sha(p.read_bytes())==s["files"]["postgresql.conf"]["fenced_sha256"] else b"on|off"
        raise AssertionError(q)
    def tcp_rejected(self):return True
class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=pathlib.Path(self.tmp.name)
        self.root=self.base/"control";self.root.mkdir();self.source=self.base/"source";self.source.mkdir()
        self.public=self.base/"public";self.public.mkdir();self.c=Fake(self.root)
        files={}
        for name in ["pg_hba.conf","pg_ident.conf","postgresql.conf"]:
            original=("original "+name).encode();fenced=("fenced "+name).encode()
            (self.source/name).write_bytes(original);stat=(self.source/name).stat()
            (self.root/(name+".original")).write_bytes(original);(self.root/(name+".fenced")).write_bytes(fenced)
            files[name]={"original_sha256":sha(original),"fenced_sha256":sha(fenced),"mode":0o600,"uid":stat.st_uid,"gid":stat.st_gid}
        self.c.save({"state":"PREPARED","token":TOKEN,"source_dir":str(self.source),"files":files,
                     "activation_file":str(self.public/"active"),"deadline_monotonic":None,"boot_id":"test-boot"})
    def tearDown(self):self.tmp.cleanup()
    def originals(self):
        return all((self.source/n).read_bytes()==(self.root/(n+".original")).read_bytes() for n in self.c.read()["files"])
    def test_pause_and_rollback(self):
        self.assertEqual(self.c.pause(TOKEN)["state"],"FENCED")
        self.assertFalse(self.originals());self.assertTrue(self.c.is_running)
        self.assertEqual(self.c.rollback(TOKEN)["state"],"ROLLED_BACK")
        self.assertTrue(self.originals());self.assertTrue(self.c.is_running)
    def test_partial_fence_crash_restores_all_files(self):
        s=self.c.read();s.update(state="FENCING",deadline_monotonic=101,started_monotonic=100);self.c.save(s)
        n="pg_hba.conf";(self.source/n).write_bytes((self.root/(n+".fenced")).read_bytes())
        self.c.now=102;self.c.watch()
        self.assertTrue(self.originals());self.assertEqual(self.c.read()["state"],"ROLLED_BACK")
    def test_expiry_after_source_stop_returns_original(self):
        self.c.pause(TOKEN);self.c.prepare_endpoint(TOKEN);self.assertFalse(self.c.is_running)
        self.c.now=941
        with self.assertRaises(RuntimeError):self.c.commit(TOKEN)
        self.c.watch()
        self.assertTrue(self.c.is_running);self.assertFalse(self.c.proxy);self.assertTrue(self.originals())
    def test_committed_target_never_rolls_back(self):
        self.c.pause(TOKEN);self.c.prepare_endpoint(TOKEN);self.c.commit(TOKEN)
        self.assertEqual(self.c.read()["state"],"TARGET")
        self.assertEqual((self.public/"active").read_text().strip(),TOKEN)
        self.assertEqual((self.public/"active").stat().st_mode & 0o777,0o644)
        with self.assertRaises(RuntimeError):self.c.rollback(TOKEN)
        self.assertFalse(self.c.is_running);self.assertFalse(self.originals())
    def test_failure_after_durable_commit_keeps_target_and_reconciles(self):
        self.c.pause(TOKEN);self.c.prepare_endpoint(TOKEN)
        activate=self.c.activate
        self.c.activate=lambda s:(_ for _ in ()).throw(OSError("publication failure"))
        with self.assertRaises(OSError):self.c.commit(TOKEN)
        self.assertEqual(self.c.read()["state"],"TARGET")
        with self.assertRaises(RuntimeError):self.c.rollback(TOKEN)
        self.c.activate=activate;self.c.watch()
        self.assertEqual((self.public/"active").read_text().strip(),TOKEN);self.assertFalse(self.c.is_running)
    def test_unowned_configuration_is_preserved(self):
        self.c.pause(TOKEN);(self.source/"postgresql.conf").write_text("other operator")
        with self.assertRaises(RuntimeError):self.c.rollback(TOKEN)
        self.assertEqual((self.source/"postgresql.conf").read_text(),"other operator")
    def test_source_identity_change_refuses_pause(self):
        self.c.identity_ok=False
        with self.assertRaises(RuntimeError):self.c.pause(TOKEN)
        self.assertTrue(self.originals());self.assertEqual(self.c.read()["state"],"PREPARED")
    def test_wrong_token_and_short_commit_margin_refuse(self):
        with self.assertRaises(RuntimeError):self.c.pause("b"*32)
        self.c.pause(TOKEN);self.c.prepare_endpoint(TOKEN);self.c.now=900
        with self.assertRaises(RuntimeError):self.c.commit(TOKEN)
        self.assertNotEqual(self.c.read()["state"],"TARGET")
if __name__=="__main__":unittest.main()

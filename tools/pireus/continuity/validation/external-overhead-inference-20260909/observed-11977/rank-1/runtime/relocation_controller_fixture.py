#!/usr/bin/env python3
"""Exercise the production controller against a disposable real PostgreSQL cluster."""
import json, os, pathlib, pwd, subprocess, sys, time, uuid
from relocation_host_controller import Controller, sha, PROXY
def main():
    base=pathlib.Path(sys.argv[1])
    base.mkdir(mode=0o755,exist_ok=False);base.chmod(0o755)
    os.umask(0o077)
    user=pwd.getpwnam("postgres")
    pg=base/"pgdata";pg.mkdir(mode=0o700);os.chown(pg,user.pw_uid,user.pw_gid)
    sock=base/"socket";sock.mkdir(mode=0o700);os.chown(sock,user.pw_uid,user.pw_gid)
    ctl=base/"control";ctl.mkdir(mode=0o700)
    def run(args,check=True,timeout=40):
        with (base/"fixture-private.log").open("ab") as log:
            return subprocess.run(args,stdout=subprocess.PIPE,stderr=log,check=check,timeout=timeout)
    def pgcmd(*args,check=True):return run(["runuser","-u","postgres","--",*args],check=check)
    pgcmd("initdb","-D",str(pg),"-U","memory","--auth-local=trust","--auth-host=trust")
    with (pg/"postgresql.conf").open("a") as f:
        f.write("\nport=15432\nlisten_addresses='127.0.0.1'\nunix_socket_directories='"+str(sock)+"'\nshared_preload_libraries='pg_cron'\ncron.database_name='postgres'\nshared_buffers='32MB'\n")
    def start():pgcmd("pg_ctl","-D",str(pg),"-l",str(pg/"server-private.log"),"-w","-t","25","start")
    def stop():pgcmd("pg_ctl","-D",str(pg),"-m","fast","-w","-t","20","stop")
    def running():return pgcmd("pg_ctl","-D",str(pg),"status",check=False).returncode==0
    class RealFixture(Controller):
        proxy=False
        def identity(self,s):
            stat=pg.stat()
            if [stat.st_dev,stat.st_ino]!=s["directory_identity"]:raise RuntimeError("Fixture identity changed")
        def sql(self,q):
            return pgcmd("psql","-X","-qAt","-v","ON_ERROR_STOP=1","-U","memory","-d","postgres","-h",str(sock),"-p","15432","-c",q).stdout.strip()
        def running(self):return running()
        def active(self):return self.proxy
        def unit(self,action,name):
            if name==PROXY and action in ["start","stop"]:self.proxy=action=="start"
        def docker(self,*args,timeout=30):
            if args[0]=="stop":stop()
            elif args[0]=="start":start()
            else:raise RuntimeError("Unexpected fixture Docker operation")
            return b""
        def tcp_rejected(self):
            p=subprocess.run(["runuser","-u","postgres","--","psql","-X","-qAt","-U","memory","-d","postgres","-h","127.0.0.1","-p","15432","-c","SELECT 1"],capture_output=True,timeout=10)
            return p.returncode!=0 and b"pg_hba.conf rejects" in p.stderr
    token=uuid.uuid4().hex
    def prepare(root):
        root.mkdir(mode=0o700,exist_ok=True);c=RealFixture(root);files={}
        for name in ["pg_hba.conf","pg_ident.conf","postgresql.conf"]:
            p=pg/name;original=p.read_bytes();st=p.stat()
            if name=="pg_hba.conf":
                fenced=b"local all memory peer map=pireus_cutover\nlocal all all reject\nlocal replication all reject\nhost all all 0.0.0.0/0 reject\nhost all all ::/0 reject\nhost replication all 0.0.0.0/0 reject\nhost replication all ::/0 reject\n"
            elif name=="pg_ident.conf":fenced=b"pireus_cutover postgres memory\n"+original
            else:fenced=original+b"\ncron.launch_active_jobs=off\ndefault_transaction_read_only=on\n"
            (root/(name+".original")).write_bytes(original);(root/(name+".fenced")).write_bytes(fenced)
            files[name]={"original_sha256":sha(original),"fenced_sha256":sha(fenced),"mode":st.st_mode&0o777,"uid":st.st_uid,"gid":st.st_gid}
        stat=pg.stat()
        c.save({"state":"PREPARED","token":token,"source_dir":str(pg),"postgres_data_directory":str(pg),
                "directory_identity":[stat.st_dev,stat.st_ino],"files":files,
                "activation_file":str(base/(root.name+"-active")),"deadline_monotonic":None,"boot_id":c.boot()})
        return c
    began=time.monotonic();start();background=None
    try:
        c=prepare(ctl)
        c.sql("CREATE TABLE rollback_marker(v integer); INSERT INTO rollback_marker VALUES(1)")
        positive=pgcmd("psql","-X","-qAt","-U","memory","-d","postgres","-h","127.0.0.1","-p","15432","-c","SELECT 1")
        assert positive.stdout.strip()==b"1"
        with (base/"background-private.log").open("wb") as log:
            background=subprocess.Popen(["runuser","-u","postgres","--","psql","-X","-qAt","-U","memory","-d","postgres","-h",str(sock),"-p","15432","-c","BEGIN; INSERT INTO rollback_marker VALUES(2); SELECT pg_sleep(60)"],stdout=log,stderr=log)
        for _ in range(30):
            if c.sql("SELECT count(*) FROM pg_stat_activity WHERE pid<>pg_backend_pid() AND wait_event='PgSleep'")==b"1":break
            time.sleep(.1)
        else:raise RuntimeError("Fixture writer did not start")
        fenced=c.pause(token)
        assert fenced["state"]=="FENCED" and c.tcp_rejected()
        assert c.sql("SELECT current_setting('default_transaction_read_only'),current_setting('cron.launch_active_jobs')")==b"on|off"
        background.wait(timeout=10)
        assert c.sql("SELECT count(*) FROM rollback_marker")==b"1"
        c.prepare_endpoint(token);assert not running()
        state=c.read();state["deadline_monotonic"]=time.monotonic()+1;c.save(state)
        rollback_started=time.monotonic();c.watch();rollback_seconds=time.monotonic()-rollback_started
        assert c.read()["state"]=="ROLLED_BACK" and running()
        assert c.sql("SELECT count(*) FROM rollback_marker")==b"1"
        assert all((pg/n).read_bytes()==(ctl/(n+".original")).read_bytes() for n in state["files"])
        positive=pgcmd("psql","-X","-qAt","-U","memory","-d","postgres","-h","127.0.0.1","-p","15432","-c","SELECT 1")
        assert positive.stdout.strip()==b"1"
        second=prepare(base/"control-target")
        second.pause(token);second.prepare_endpoint(token);second.commit(token)
        assert not running() and second.read()["state"]=="TARGET"
        try:second.rollback(token)
        except RuntimeError:pass
        else:raise AssertionError("Stale source was allowed to return")
        assert not running()
        report={"real_postgresql_fixture":True,"production_source_touched":False,
                "production_endpoint_or_systemd_tested":False,"pre_fence_tcp_positive":True,
                "fenced_tcp_negative":True,"maintenance_peer_connection_pass":True,
                "read_only_and_cron_fence_pass":True,"uncommitted_writer_drained_without_committing":True,
                "deadline_after_database_stop_restores_original":True,"rollback_seconds":rollback_seconds,
                "exact_original_configurations_restored":True,"post_rollback_tcp_positive":True,
                "target_authority_refuses_stale_source_restart":True,"fixture_stopped":True,
                "elapsed_seconds":time.monotonic()-began,"production_cutover_accepted":False}
        (base/"summary.json").write_text(json.dumps(report,indent=2))
        print("REAL_POSTGRES_CONTROLLER_FIXTURE_PASS "+json.dumps(report),flush=True)
    finally:
        if running():stop()
        if background is not None and background.poll() is None:background.terminate();background.wait(timeout=10)
if __name__=="__main__":main()

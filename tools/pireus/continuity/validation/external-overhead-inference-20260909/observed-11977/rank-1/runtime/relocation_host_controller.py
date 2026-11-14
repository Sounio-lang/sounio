#!/usr/bin/env python3
"""Host-local PostgreSQL authority and deadline controller; source rollback before TARGET only."""
import argparse, contextlib, ipaddress, re, fcntl, hashlib, json, os, pathlib, shutil, subprocess, time
CONTAINER="beagle-memory-pg-pdb"
SOURCE=pathlib.Path("/home/demetrios/beagle-memory-pg-pdb/data")
ROOT=pathlib.Path("/var/lib/pireus/pg-relocation-20260907")
PUBLIC=pathlib.Path("/opt/pireus/pg-relocation-20260907")
PROXY="pireus-pg-forwarder.service"
WATCHDOG="pireus-pg-cutover-watchdog.service"
FENCED_STATES={"FENCING","FENCED","PROXY_MAINTENANCE"}
def sha(data):return hashlib.sha256(data).hexdigest()
def atomic(path,data,mode=0o600,uid=None,gid=None):
    temporary=path.with_name(path.name+".tmp-"+str(os.getpid()))
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode)
    try:
        os.fchmod(fd,mode)
        if uid is not None:os.fchown(fd,uid,gid)
        with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
        d=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(d)
        finally:os.close(d)
    finally:
        if temporary.exists():temporary.unlink()
def command(args,timeout=30):
    result=subprocess.run(args,capture_output=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError("Host operation failed: "+args[0])
    return result.stdout
class Controller:
    def __init__(self,root=ROOT,clock=time.monotonic):
        self.root=pathlib.Path(root);self.clock=clock
    @contextlib.contextmanager
    def lock(self):
        with (self.root/"lock").open("a") as f:
            deadline=time.monotonic()+5
            while True:
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()>=deadline:raise RuntimeError("Controller lock timeout")
                    time.sleep(.05)
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)
    def read(self):return json.loads((self.root/"state.json").read_text())
    def save(self,s):
        s["updated_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        atomic(self.root/"state.json",(json.dumps(s,indent=2)+"\n").encode())
    def docker(self,*args,timeout=30):return command(["docker",*args],timeout)
    def unit(self,action,name):
        return command(["systemctl",action,name],30)
    def sql(self,statement):
        return self.docker("exec","-u","postgres",CONTAINER,"sh","-c",
            'exec psql -X -qAt -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d postgres -c "$1"',
            "--",statement).strip()
    def boot(self):return pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    def identity(self,s):
        current=self.docker("inspect","--format","{{.Id}}",CONTAINER).decode().strip()
        d=pathlib.Path(s["source_dir"]);stat=d.stat()
        if current!=s["container_id"] or d.is_symlink() or [stat.st_dev,stat.st_ino]!=s["directory_identity"]:
            raise RuntimeError("Protected source identity changed")
    def running(self):
        return self.docker("inspect","--format","{{.State.Running}}",CONTAINER).strip()==b"true"
    def active(self):
        p=subprocess.run(["systemctl","is-active",PROXY],capture_output=True,timeout=15)
        return p.returncode==0 and p.stdout.strip()==b"active"
    def expired(self,s):
        return s.get("deadline_monotonic") is not None and (s["boot_id"]!=self.boot() or self.clock()>=s["deadline_monotonic"])
    def remaining(self,s):
        if s.get("deadline_monotonic") is None:return None
        return max(0,s["deadline_monotonic"]-self.clock()) if s["boot_id"]==self.boot() else 0
    def owned_files(self,s):
        for name,meta in s["files"].items():
            path=pathlib.Path(s["source_dir"])/name
            if path.is_symlink():raise RuntimeError("Source config path became symlink")
            if sha(path.read_bytes()) not in [meta["original_sha256"],meta["fenced_sha256"]]:
                raise RuntimeError("Source config has an unowned change; preserve it")
    def replace_files(self,s,which):
        self.owned_files(s)
        for name,meta in s["files"].items():
            data=(self.root/(name+"."+which)).read_bytes()
            if sha(data)!=meta[which+"_sha256"]:raise RuntimeError("Private configuration custody failed")
            atomic(pathlib.Path(s["source_dir"])/name,data,meta["mode"],meta["uid"],meta["gid"])
    def check_fence_files(self,s):
        self.identity(s)
        for name,meta in s["files"].items():
            if sha((pathlib.Path(s["source_dir"])/name).read_bytes())!=meta["fenced_sha256"]:
                raise RuntimeError("Source fence config changed")
    def pause(self,token):
        with self.lock():
            s=self.read()
            if s["token"]!=token or s["state"]!="PREPARED":raise RuntimeError("Pause state/token mismatch")
            self.identity(s);self.owned_files(s)
            if not self.running():raise RuntimeError("Source not running before pause")
            if self.sql("SELECT current_user,current_setting('data_directory'),current_setting('cron.launch_active_jobs'),current_setting('default_transaction_read_only'); SELECT source FROM pg_settings WHERE name='cron.launch_active_jobs'; SELECT count(*) FROM pg_prepared_xacts")!=("memory|"+s.get("postgres_data_directory","/var/lib/postgresql/data")+"|on|off\ndefault\n0").encode():
                raise RuntimeError("Source pre-pause settings changed")
            # Publish rollback intent before the first source configuration write.
            s.update(state="FENCING",started_monotonic=self.clock(),deadline_monotonic=self.clock()+840,
                     started_wall=time.time(),boot_id=self.boot())
            self.save(s)
            self.replace_files(s,"fenced")
            if self.sql("SELECT pg_reload_conf(); SELECT count(*) FROM pg_hba_file_rules WHERE error IS NOT NULL")!=b"t\n0":
                raise RuntimeError("Source fence reload failed")
            # Wait for SIGHUP to reach new backend sessions before draining.
            for _ in range(20):
                if self.sql("SELECT current_setting('cron.launch_active_jobs'),current_setting('default_transaction_read_only')")==b"off|on":break
                time.sleep(.1)
            else:raise RuntimeError("Source scheduled/read-write settings did not fence")
            self.sql("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE backend_type='client backend' AND pid<>pg_backend_pid()")
            if self.sql("SELECT count(*) FROM pg_stat_activity WHERE backend_type='client backend' AND pid<>pg_backend_pid()")!=b"0":
                raise RuntimeError("Source clients did not drain")
            if not self.tcp_rejected():
                raise RuntimeError("Source TCP rejection control failed")
            s.update(state="FENCED",client_drain_pass=True,tcp_rejection_pass=True)
            self.save(s)
            return self.public_status(s)
    def tcp_rejected(self):
        negative=subprocess.run(["docker","exec","-u","postgres",CONTAINER,"sh","-c",
            'PGCONNECT_TIMEOUT=3 PGPASSWORD="$POSTGRES_PASSWORD" exec psql -X -qAt -U "$POSTGRES_USER" -d postgres -h127.0.0.1 -c "SELECT 1"'],
            capture_output=True,timeout=15)
        return negative.returncode!=0 and b"pg_hba.conf rejects" in negative.stderr
    def prepare_endpoint(self,token):
        with self.lock():
            s=self.read()
            if s["token"]!=token or s["state"]!="FENCED" or self.remaining(s)<90:
                raise RuntimeError("Endpoint preparation state/deadline refused")
            self.check_fence_files(s)
            self.docker("stop","--time","10",CONTAINER,timeout=25)
            if self.running():raise RuntimeError("Source container did not stop")
            self.unit("enable",PROXY);self.unit("start",PROXY)
            if not self.active():raise RuntimeError("Maintenance proxy did not start")
            if pathlib.Path(s["activation_file"]).exists():raise RuntimeError("Proxy activation predates authority commit")
            s["state"]="PROXY_MAINTENANCE";self.save(s)
            return self.public_status(s)
    def commit(self,token):
        with self.lock():
            s=self.read()
            if s["token"]!=token or s["state"]!="PROXY_MAINTENANCE" or self.remaining(s)<60:
                raise RuntimeError("Authority commit state/deadline refused")
            self.check_fence_files(s)
            if self.running() or not self.active():raise RuntimeError("Source/proxy state changed before commit")
            # Once durable TARGET exists, stale-source rollback is forbidden, even
            # if publishing the public activation marker or the RPC reply fails.
            s.update(state="TARGET",committed_wall=time.time(),pause_seconds=self.clock()-s["started_monotonic"])
            self.save(s)
            self.activate(s)
            return self.public_status(s)
    def activate(self,s):
        if s["state"]!="TARGET":raise RuntimeError("Cannot activate before durable target authority")
        atomic(pathlib.Path(s["activation_file"]),(s["token"]+"\n").encode(),0o644)
    def rollback(self,token=None):
        with self.lock():
            s=self.read()
            if token is not None and s["token"]!=token:raise RuntimeError("Rollback token mismatch")
            if s["state"]=="TARGET":raise RuntimeError("TARGET authority forbids stale-source rollback")
            if s["state"]=="ROLLED_BACK":return self.public_status(s)
            self.identity(s)
            self.unit("stop",PROXY);self.unit("disable",PROXY)
            activation=pathlib.Path(s["activation_file"])
            if activation.exists():
                if activation.read_text().strip()!=s["token"]:raise RuntimeError("Unowned proxy activation marker")
                activation.unlink()
            self.replace_files(s,"original")
            if not self.running():self.docker("start",CONTAINER,timeout=30)
            deadline=self.clock()+25
            while True:
                try:
                    if self.sql("SELECT pg_reload_conf()")==b"t":
                        if self.sql("SELECT current_setting('cron.launch_active_jobs'),current_setting('default_transaction_read_only')")==b"on|off":break
                except RuntimeError:pass
                if self.clock()>=deadline:raise RuntimeError("Source rollback readiness deadline exceeded")
                time.sleep(.2)
            self.identity(s)
            s.update(state="ROLLED_BACK",rollback_wall=time.time(),rollback_pass=True)
            if s.get("started_monotonic") is not None:s["pause_seconds"]=self.clock()-s["started_monotonic"]
            self.save(s);return self.public_status(s)
    def public_status(self,s):
        return {k:s.get(k) for k in ["state","updated_utc","client_drain_pass","tcp_rejection_pass","pause_seconds","rollback_pass"]}|{"remaining_seconds":self.remaining(s)}
    def status(self,token):
        with self.lock():
            s=self.read()
            if token!=s["token"]:raise RuntimeError("Status token mismatch")
            if s["state"] in FENCED_STATES:
                self.check_fence_files(s)
                if self.expired(s):raise RuntimeError("Maintenance deadline expired")
            return self.public_status(s)
    def watch(self):
        while True:
            s=self.read()
            if s["state"]=="ROLLED_BACK":return
            if s["state"]=="TARGET":
                with self.lock():
                    s=self.read()
                    self.activate(s)
                    self.unit("start",PROXY)
                return
            if self.expired(s):
                try:
                    self.rollback()
                    print("SOURCE_ROLLBACK_COMPLETE",flush=True)
                    return
                except Exception as e:
                    print("SOURCE_ROLLBACK_RETRY "+type(e).__name__,flush=True)
            time.sleep(.5)
def stage(payload):
    if os.geteuid()!=0:raise RuntimeError("Host root required")
    if ROOT.exists() or PUBLIC.exists():raise RuntimeError("Existing cutover paths must be preserved")
    if not re.fullmatch("[0-9a-f]{32}",payload["token"]):raise RuntimeError("Invalid transaction token")
    for key in ["target_host","maintenance_client"]:
        if not isinstance(ipaddress.ip_address(payload[key]),ipaddress.IPv4Address):
            raise RuntimeError("Expected pinned IPv4 cluster endpoints")
    for unit in [PROXY,WATCHDOG]:
        if pathlib.Path("/etc/systemd/system",unit).exists():raise RuntimeError("Existing cutover service must be preserved")
    c=Controller()
    container=c.docker("inspect","--format","{{.Id}}",CONTAINER).decode().strip()
    mounts=json.loads(c.docker("inspect","--format","{{json .Mounts}}",CONTAINER))
    if not any(m.get("Type")=="bind" and m.get("Source")==str(SOURCE) and m.get("Destination")=="/var/lib/postgresql/data" and m.get("RW") for m in mounts):
        raise RuntimeError("Protected source bind mount differs")
    if c.docker("inspect","--format","{{.HostConfig.RestartPolicy.Name}}",CONTAINER).strip()!=b"unless-stopped":
        raise RuntimeError("Source restart contract changed")
    stat=SOURCE.stat()
    if SOURCE.is_symlink() or not c.running():raise RuntimeError("Source directory/running state invalid")
    ROOT.mkdir(parents=True,mode=0o700);PUBLIC.mkdir(parents=True,mode=0o755)
    PUBLIC.chmod(0o755)
    atomic(ROOT/"controller.py",pathlib.Path(__file__).read_bytes())
    atomic(PUBLIC/"forwarder.py",payload["proxy_code"].encode(),0o644)
    prefix="# pireus protected cutover "+payload["token"]+"\n"
    fenced_hba=(prefix+"local all memory peer map=pireus_cutover\nlocal all all reject\nlocal replication all reject\n"
                "host all all 0.0.0.0/0 reject\nhost all all ::/0 reject\n"
                "host replication all 0.0.0.0/0 reject\nhost replication all ::/0 reject\n").encode()
    files={}
    for name in ["pg_hba.conf","pg_ident.conf","postgresql.conf"]:
        path=SOURCE/name
        if path.is_symlink():raise RuntimeError("Source configuration symlink requires review")
        original=path.read_bytes();st=path.stat()
        if st.st_nlink!=1 or os.listxattr(path):
            raise RuntimeError("Source config has links or extended metadata requiring preservation")
        if name=="pg_hba.conf":fenced=fenced_hba
        elif name=="pg_ident.conf":fenced=(prefix+"pireus_cutover postgres memory\n").encode()+original
        else:fenced=original+("\n"+prefix+"cron.launch_active_jobs = off\ndefault_transaction_read_only = on\n").encode()
        atomic(ROOT/(name+".original"),original);atomic(ROOT/(name+".fenced"),fenced)
        files[name]={"original_sha256":sha(original),"fenced_sha256":sha(fenced),"uid":st.st_uid,"gid":st.st_gid,"mode":st.st_mode & 0o777}
    activation=PUBLIC/"target-active"
    args="/usr/bin/python3 "+str(PUBLIC/"forwarder.py")+" --listen-port 5433 --target-host "+payload["target_host"]+" --source-host 10.100.100.59 --activation-file "+str(activation)+" --activation-value "+payload["token"]
    for peer in ["127.0.0.1","::1",payload["maintenance_client"]]:args+=" --maintenance-client "+peer
    common="NoNewPrivileges=true\nPrivateTmp=true\nPrivateDevices=true\nProtectSystem=strict\nProtectHome=true\nProtectKernelTunables=true\nProtectKernelModules=true\nProtectControlGroups=true\nRestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX\nMemoryMax=128M\nTasksMax=32\nCPUQuota=100%\n"
    proxy="[Unit]\nDescription=Pireus PostgreSQL endpoint forwarder\nAfter=network-online.target\n\n[Service]\nType=simple\nDynamicUser=true\nExecStart="+args+"\nRestart=on-failure\nRestartSec=2\nTimeoutStopSec=20\n"+common+"\n[Install]\nWantedBy=multi-user.target\n"
    watcher="[Unit]\nDescription=Pireus PostgreSQL cutover deadline and authority recovery\nAfter=docker.service network-online.target\n\n[Service]\nType=simple\nExecStart=/usr/bin/python3 "+str(ROOT/"controller.py")+" watch\nRestart=on-failure\nRestartSec=2\nTimeoutStopSec=15\n\n[Install]\nWantedBy=multi-user.target\n"
    for name,body in [(PROXY,proxy),(WATCHDOG,watcher)]:
        atomic(pathlib.Path("/etc/systemd/system")/name,body.encode(),0o644)
    state={"state":"PREPARED","token":payload["token"],"container_id":container,
           "source_dir":str(SOURCE),"postgres_data_directory":"/var/lib/postgresql/data","directory_identity":[stat.st_dev,stat.st_ino],"files":files,
           "activation_file":str(activation),"deadline_monotonic":None,"boot_id":c.boot(),
           "target_host":payload["target_host"],"maintenance_client":payload["maintenance_client"],
           "proxy_sha256":sha(payload["proxy_code"].encode())}
    c.save(state)
    command(["systemctl","daemon-reload"])
    c.unit("enable",WATCHDOG);c.unit("start",WATCHDOG)
    return {"state":"PREPARED","source_configuration_changed":False,"source_container_stopped":False,"proxy_started":False,"deadline_watchdog_enabled":True}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["stage","pause","status","prepare-endpoint","commit","rollback","watch"])
    ap.add_argument("--token")
    args=ap.parse_args();os.umask(0o077)
    if args.action=="stage":
        import sys
        report=stage(json.load(sys.stdin))
    else:
        c=Controller()
        if args.action=="watch":c.watch();return
        if args.action=="pause":report=c.pause(args.token)
        elif args.action=="prepare-endpoint":report=c.prepare_endpoint(args.token)
        elif args.action=="commit":report=c.commit(args.token)
        elif args.action=="rollback":report=c.rollback(args.token)
        else:report=c.status(args.token)
    print(json.dumps(report),flush=True)
if __name__=="__main__":main()

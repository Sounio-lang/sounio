#!/usr/bin/env python3
"""Prepared, deadline-controlled logical cutover; only durable TARGET admits normal clients."""
import argparse, datetime, hashlib, json, os, pathlib, socket, struct, subprocess, sys, time, uuid
from relocation_rehearsal import kubectl, target, ADMIN_DB, POD
from relocation_probe_proxy import HOST, restrict_target_hba
from preflight import check_pair
HERE=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path("/workspace/.cache/pireus-continuity/protected-db-relocation-20260907")
CONTROLLER="/var/lib/pireus/pg-relocation-20260907/controller.py"
POLICY="pireus-pg-relocation-source-access"
def save(path,value):path.write_text(json.dumps(value,indent=2)+"\n")
def run(command,log,timeout=60,input=None):
    with log.open("ab") as errors:
        result=subprocess.run(command,input=input,stdout=subprocess.PIPE,stderr=errors,timeout=timeout)
    if result.returncode:raise RuntimeError("Cutover operation failed; private diagnostic retained")
    return result.stdout
def control(action,runtime,out):
    return json.loads(run(HOST+["python3",CONTROLLER,action,"--token",runtime["token"]],out/"controller-private.log",90))
def client_query(runtime,query,out,label):
    script='PGCONNECT_TIMEOUT=3 PGPASSWORD="$POSTGRES_PASSWORD" exec psql -X -qAt -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d memory -h10.100.100.59 -p5433 -c "$1"'
    return run(["kubectl","-n","beagle","exec",runtime["client_pod"],"--","sh","-c",script,"--",query],
               out/(label+"-private.log"),20).strip()
def target_sql(query,out):
    return run(target("psql",ADMIN_DB,["-X","-qAt","-v","ON_ERROR_STOP=1","-c",query]),
               out/"target-private.log",30).strip()
def run_script(name,params,out,label,timeout=900):
    raw=run([sys.executable,str(HERE/name),*params],out/(label+"-private.log"),timeout)
    (out/(label+"-stdout-private.log")).write_bytes(raw)
def current_client(runtime):
    p=json.loads(kubectl("get","pod",runtime["client_pod"],"-o","json"))
    if p["metadata"]["uid"]!=runtime["client_uid"] or p["status"].get("podIP")!=runtime["client_ip"] or not all(c["ready"] for c in p["status"]["containerStatuses"]):
        raise RuntimeError("Maintenance client identity/readiness changed")
def prepare(out):
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    proof={name:json.loads((BASE/name/"summary.json").read_text()) for name in ["rehearsal-v6","normal-v6","forwarder-v6","pool-v6","template0-audit"]}
    if not (proof["rehearsal-v6"]["core_within_780_seconds"] and proof["rehearsal-v6"]["schema_acl_sequence_acceptance"] and proof["rehearsal-v6"]["mismatches"]==0 and
            proof["normal-v6"]["runtime_mode_restored"] and proof["normal-v6"]["functional_pass"] and
            proof["forwarder-v6"]["forwarded_authentication_and_query_pass"] and
            proof["pool-v6"]["pool_io_observation_pass"] and proof["template0-audit"]["template0_content_audit_pass"]):
        raise RuntimeError("Prior complete rehearsal/audit gates missing")
    measured=sum(proof[n]["elapsed_seconds"] for n in ["rehearsal-v6","normal-v6","forwarder-v6"])
    if measured+120>900:raise RuntimeError("Measured rehearsal lacks controller/rollback margin")
    save(out/"host-preflight-private.json",check_pair())
    run_script("relocation_source_guard.py",["--output",str(out/"source-guard")],out,"source-guard",180)
    run_script("relocation_probe_proxy.py",["--output",str(out/"proxy-preflight")],out,"proxy-preflight",180)
    token=uuid.uuid4().hex;client="pireus-pg-cutover-client-"+uuid.uuid4().hex[:8]
    spec={"apiVersion":"v1","kind":"Pod","metadata":{"name":client,"namespace":"beagle","labels":{"app":"pireus-pg-cutover-client"}},
          "spec":{"restartPolicy":"Never","activeDeadlineSeconds":3600,"automountServiceAccountToken":False,
                  "nodeSelector":{"kubernetes.io/hostname":"r770-proxmox","kubernetes.io/arch":"amd64"},
                  "tolerations":[{"key":"sounio.dev/pool","operator":"Equal","value":"gpu-batch","effect":"NoSchedule"},
                                 {"key":"sounio.dev/compute","operator":"Equal","value":"heavy","effect":"NoSchedule"}],
                  "containers":[{"name":"client","image":"paradedb/paradedb@sha256:ea4e8267016c929924a15df03eaf2ad9bb2cda9d194169e03eca232303898cc7",
                    "command":["sleep","3600"],"resources":{"requests":{"cpu":"25m","memory":"32Mi"},"limits":{"cpu":"100m","memory":"128Mi"}},
                    "env":[{"name":"POSTGRES_USER","valueFrom":{"secretKeyRef":{"name":"pireus-pg-relocation-source-auth","key":"username"}}},
                           {"name":"POSTGRES_PASSWORD","valueFrom":{"secretKeyRef":{"name":"pireus-pg-relocation-source-auth","key":"password"}}}]}]}}
    run(["kubectl","create","-f","-"],out/"client-private.log",30,json.dumps(spec).encode())
    run(["kubectl","-n","beagle","wait","--for=condition=Ready","pod/"+client,"--timeout=120s"],out/"client-private.log",130)
    p=json.loads(kubectl("get","pod",client,"-o","json"))
    service=json.loads(kubectl("get","service","pireus-pg-relocation","-o","json"))
    runtime={"token":token,"client_pod":client,"client_uid":p["metadata"]["uid"],"client_ip":p["status"]["podIP"],"target_host":service["spec"]["clusterIP"]}
    save(out/"runtime-private.json",runtime)
    if client_query(runtime,"SELECT current_setting('data_directory'); SELECT inet_client_addr()",out,"original-endpoint")!=("/var/lib/postgresql/data\n"+runtime["client_ip"]).encode():
        raise RuntimeError("Maintenance client does not reach the original source endpoint")
    # All preparation precedes the source pause.
    run_script("relocation_prepare_bulk.py",["--source-output",str(BASE/"rehearsal-source-v6"),
        "--output",str(out/"bulk-prepared"),"--archive-prefix","rehearsal-v6"],out,"bulk-prepare",900)
    hba_dir=out/"target-hba";hba_dir.mkdir(mode=0o700);restrict_target_hba(hba_dir)
    host_script="/var/tmp/pireus-pg-cutover-controller-20260907.py"
    code=(HERE/"relocation_host_controller.py").read_bytes()
    uploader="import pathlib,sys,os;p=pathlib.Path(sys.argv[1]);b=sys.stdin.buffer.read();assert not p.exists(),'Existing host controller preserved';p.write_bytes(b);p.chmod(0o600)"
    run(HOST[:4]+["-i"]+HOST[4:]+["python3","-c",uploader,host_script],out/"stage-private.log",30,code)
    payload={"token":token,"target_host":runtime["target_host"],"maintenance_client":runtime["client_ip"],"proxy_code":(HERE/"relocation_proxy.py").read_text()}
    staged=json.loads(run(HOST[:4]+["-i"]+HOST[4:]+["python3",host_script,"stage"],out/"stage-private.log",90,json.dumps(payload).encode()))
    runtime["controller_sha256"]=hashlib.sha256(code).hexdigest()
    runtime["proxy_sha256"]=hashlib.sha256(payload["proxy_code"].encode()).hexdigest()
    runtime["prepared_at"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(out/"runtime-private.json",runtime)
    if control("status",runtime,out)["state"]!="PREPARED":raise RuntimeError("Host controller not prepared")
    save(out/"prepare-summary.json",{"prepared":True,"measured_rehearsal_seconds":measured,"source_write_pause":False,
         "target_empty_and_isolated":True,"host_controller":staged,"production_cutover_accepted":False})
    print("CUTOVER_PREPARED_SOURCE_ONLINE",flush=True)
def pool_recent(out):
    path=out/"pool-final"/"progress.jsonl"
    records=path.read_text().splitlines() if path.exists() else []
    if not records:raise RuntimeError("No fresh destination pool observation")
    p=json.loads(records[-1])
    age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(p["utc"])).total_seconds()
    if age>45 or not p["placement_pass"] or any(o["fresh_slow"] or o["fresh_stalled"] for o in p["osds"]):
        raise RuntimeError("Destination pool observation refused")
def execute(out):
    runtime=json.loads((out/"runtime-private.json").read_text())
    if not json.loads((out/"prepare-summary.json").read_text())["prepared"]:raise RuntimeError("Preparation missing")
    if control("status",runtime,out)["state"]!="PREPARED":raise RuntimeError("Source transaction already advanced")
    for operation,expected in [("is-active",b"active"),("is-enabled",b"enabled")]:
        if run(HOST+["systemctl",operation,"pireus-pg-cutover-watchdog.service"],out/"controller-private.log",20).strip()!=expected:
            raise RuntimeError("Durable source deadline watchdog is not ready")
    current_client(runtime)
    save(out/"execute-host-preflight-private.json",check_pair())
    run_script("relocation_source_guard.py",["--output",str(out/"execute-source-guard")],out,"execute-source-guard",180)
    actual=run(HOST+["sha256sum",CONTROLLER],out/"controller-private.log",30).decode().split()[0]
    if actual!=runtime["controller_sha256"]:raise RuntimeError("Installed controller custody changed")
    if client_query(runtime,"SELECT current_setting('data_directory')",out,"source-before-pause")!=b"/var/lib/postgresql/data":
        raise RuntimeError("Original source endpoint no longer matches")
    if target_sql("SELECT current_setting('wal_level'),current_setting('max_wal_senders'),current_setting('cron.launch_active_jobs')",out)!=b"minimal|0|off":
        raise RuntimeError("Destination bulk preparation changed")
    if json.loads(kubectl("get","ciliumnetworkpolicy","-o","json"))["items"]:
        raise RuntimeError("Unexpected destination ingress before pause")
    observer_log=(out/"pool-observer-private.log").open("xb")
    observer=subprocess.Popen([sys.executable,str(HERE/"relocation_pool_observer.py"),
        "--output",str(out/"pool-final"),"--stop-file",str(out/"stop-pool-final"),"--seconds","1200"],
        stdout=observer_log,stderr=observer_log)
    summary={"source_write_pause":False,"production_cutover_accepted":False,"authority":"SOURCE"}
    paused_start=None
    def guarded(name,params,label):
        with (out/(label+"-private.log")).open("xb") as log:
            p=subprocess.Popen([sys.executable,str(HERE/name),*params],stdout=log,stderr=log)
            try:
                while p.poll() is None:
                    state=control("status",runtime,out)
                    if state["state"]!="FENCED" or state["remaining_seconds"]<100:
                        raise RuntimeError("Source fence/window no longer supports final restoration")
                    if observer.poll() is not None:raise RuntimeError("Pool observer stopped during final restore")
                    pool_recent(out)
                    time.sleep(3)
                if p.returncode:raise RuntimeError("Final database phase failed; private diagnostic retained")
            finally:
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
    def remove_owned_policy():
        items=json.loads(kubectl("get","ciliumnetworkpolicy","-o","json"))["items"]
        for p in items:
            if p["metadata"]["name"]==POLICY and p["metadata"].get("labels",{}).get("pireus.sounio.dev/cutover-id")==runtime["token"]:
                run(["kubectl","-n","beagle","delete","ciliumnetworkpolicy",POLICY],out/"policy-private.log",30)
    try:
        deadline=time.monotonic()+60
        while not (out/"pool-final"/"progress.jsonl").exists():
            if observer.poll() is not None or time.monotonic()>deadline:raise RuntimeError("Pool observer did not start")
            time.sleep(.2)
        pool_recent(out)
        paused_start=time.monotonic()
        state=control("pause",runtime,out)
        if state["state"]!="FENCED":raise RuntimeError("Source did not fence")
        summary["source_write_pause"]=True
        print("SOURCE_FENCED_FINAL_SNAPSHOT_STARTED",flush=True)
        guarded("relocation_rehearsal.py",["--source-output",str(out/"final-source"),"--output",str(out/"final-restore"),
            "--globals-already-restored","--bulk-prepared","--final-cutover-token",runtime["token"]],"final-restore")
        restored=json.loads((out/"final-restore"/"summary.json").read_text())
        if not restored["source_write_pause"] or restored["mismatches"] or not restored["schema_acl_sequence_acceptance"]:
            raise RuntimeError("Final snapshot data/metadata acceptance failed")
        guarded("relocation_restore_runtime.py",["--prepared-output",str(out/"bulk-prepared"),"--source-output",str(out/"final-source"),
            "--output",str(out/"final-normal"),"--final-cutover-token",runtime["token"]],"final-normal")
        normal=json.loads((out/"final-normal"/"summary.json").read_text())
        if not normal["runtime_mode_restored"] or not normal["functional_pass"] or not normal["source_write_pause"]:
            raise RuntimeError("Final normal-runtime/function acceptance failed")
        current_client(runtime);pool_recent(out)
        target_pod=json.loads(kubectl("get","pod",POD,"-o","json"))
        if target_pod["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"]!="pireus-pg-relocation-ssd-data":
            raise RuntimeError("Final target volume changed")
        policy={"apiVersion":"cilium.io/v2","kind":"CiliumNetworkPolicy",
            "metadata":{"name":POLICY,"namespace":"beagle","labels":{"pireus.sounio.dev/cutover-id":runtime["token"]}},
            "spec":{"endpointSelector":{"matchLabels":{"app":"pireus-pg-relocation"}},
                    "ingress":[{"fromEntities":["remote-node"],"toPorts":[{"ports":[{"port":"5432","protocol":"TCP"}]}]}]}}
        run(["kubectl","create","-f","-"],out/"policy-private.log",30,json.dumps(policy).encode())
        state=control("prepare-endpoint",runtime,out)
        if state["state"]!="PROXY_MAINTENANCE":raise RuntimeError("Source endpoint did not enter maintenance-only forwarding")
        expected=("1\n"+target_pod["status"]["podIP"]+"\n/var/lib/postgresql/data/pgdata-source-bootstrap").encode()
        query="SELECT 1; SELECT inet_server_addr(); SELECT current_setting('data_directory')"
        for _ in range(8):
            try:
                if client_query(runtime,query,out,"maintenance-endpoint")==expected:break
            except RuntimeError:pass
            time.sleep(1)
        else:raise RuntimeError("Maintenance endpoint did not reach the verified target")
        # A non-maintenance workspace client must be refused before activation.
        try:
            with socket.create_connection(("10.100.100.59",5433),timeout=3) as s:
                s.sendall(struct.pack("!II",8,80877103));denied=s.recv(1)==b""
        except (ConnectionResetError,BrokenPipeError):denied=True
        if not denied:raise RuntimeError("Normal client bypassed pre-authority proxy gate")
        tls="import socket,struct,json;r={};\nfor h in ['127.0.0.1','::1']:\n with socket.create_connection((h,5433),timeout=3) as s:\n  s.sendall(struct.pack('!II',8,80877103));r[h]=s.recv(1).decode()\nprint(json.dumps(r))"
        ssl=json.loads(run(HOST+["python3","-c",tls],out/"tls-private.log",15))
        if set(ssl.values())!={"N"}:raise RuntimeError("Original IPv4/IPv6 SSL negotiation changed")
        save(out/"stopped-source-host-preflight-private.json",check_pair())
        pool_recent(out)
        committed=control("commit",runtime,out)
        if committed["state"]!="TARGET":raise RuntimeError("Target authority commit not observed")
        summary["authority"]="TARGET"
        print("TARGET_AUTHORITY_COMMITTED",flush=True)
        if client_query(runtime,query,out,"active-endpoint")!=expected:raise RuntimeError("Activated endpoint validation failed")
        # Check a formerly excluded client now receives the target SSL response.
        with socket.create_connection(("10.100.100.59",5433),timeout=3) as s:
            s.sendall(struct.pack("!II",8,80877103));ordinary=s.recv(1)==b"N"
        if not ordinary:raise RuntimeError("Normal clients were not activated")
        target_sql("ALTER SYSTEM SET cron.launch_active_jobs='on'",out)
        target_sql("SELECT pg_reload_conf()",out)
        for _ in range(20):
            if target_sql("SELECT current_setting('cron.launch_active_jobs')",out)==b"on":break
            time.sleep(.1)
        else:raise RuntimeError("Target scheduler activation failed")
        endpoint_seconds=time.monotonic()-paused_start
        run(["kubectl","-n","beagle","annotate","statefulset","pireus-pg-relocation","pireus.sounio.dev/database-authority=TARGET","--overwrite"],out/"authority-private.log",30)
        save(out/"post-cutover-host-preflight-private.json",check_pair())
        summary.update({"source_write_pause":True,"authority":"TARGET","pause_until_endpoint_and_scheduler_seconds":endpoint_seconds,
            "within_900_second_pause":endpoint_seconds<=900,"host_commit_pause_seconds":committed["pause_seconds"],
            "final_snapshot_source_fenced":True,"tables_compared":restored["tables_compared"],"mismatches":restored["mismatches"],
            "schema_acl_sequence_acceptance":True,"normal_runtime_restored":True,"functional_pass":True,
            "maintenance_endpoint_positive":True,"pre_commit_normal_client_negative":True,"normal_clients_activated":True,
            "ipv4_ipv6_ssl_preserved":True,"target_cron_enabled":True,"source_container_stopped":True,
            "original_source_directory_retained":True,"protected_host_preflight_pass":True,
            "production_cutover_accepted":endpoint_seconds<=900})
        (out/"stop-pool-final").touch(exist_ok=False)
        observer.wait(timeout=60);observer_log.close()
        observation=json.loads((out/"pool-final"/"summary.json").read_text())
        summary["destination_pool_io_pass"]=observation["pool_io_observation_pass"]
        summary["production_cutover_accepted"] &= observation["pool_io_observation_pass"]
        save(out/"summary.json",summary)
        print("PROTECTED_DATABASE_CUTOVER_RESULT "+json.dumps(summary),flush=True)
        if not summary["production_cutover_accepted"]:raise RuntimeError("Post-commit acceptance incomplete; target remains authoritative")
    except BaseException:
        try:
            state=control("status",runtime,out);summary["authority"]="TARGET" if state["state"]=="TARGET" else "SOURCE"
        except Exception:summary["authority"]="UNKNOWN"
        if summary["authority"]!="TARGET":
            try:
                state=control("rollback",runtime,out)
                if state["state"]=="ROLLED_BACK":
                    summary.update(authority="SOURCE",rollback_pass=True)
                    remove_owned_policy()
            except Exception:summary["rollback_pass"]=False
        summary["production_cutover_accepted"]=False
        save(out/"summary.json",summary)
        print("CUTOVER_STOP "+json.dumps(summary),flush=True)
        raise
    finally:
        if not (out/"stop-pool-final").exists():(out/"stop-pool-final").touch()
        if observer.poll() is None:
            try:observer.wait(timeout=60)
            except subprocess.TimeoutExpired:observer.terminate();observer.wait(timeout=10)
        observer_log.close()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["prepare","execute"])
    ap.add_argument("--output",type=pathlib.Path,required=True)
    args=ap.parse_args();os.umask(0o077)
    if not os.environ.get("TMUX"):raise RuntimeError("Use remote tmux for the maintenance controller")
    if args.action=="prepare":prepare(args.output)
    else:execute(args.output)
if __name__=="__main__":main()

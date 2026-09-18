#!/usr/bin/env python3
"""Switch only the isolated rehearsal target to the qualified retained SSD volume."""
import argparse, hashlib, json, os, pathlib, subprocess, time
from relocation_rehearsal import kubectl, target, sql, ADMIN_DB, POD
from relocation_isolated_storage import PVC
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-output",type=pathlib.Path,required=True)
    ap.add_argument("--volume-output",type=pathlib.Path,required=True)
    ap.add_argument("--output",type=pathlib.Path,required=True)
    args=ap.parse_args();os.umask(0o077)
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    qualified=json.loads((args.volume_output/"summary.json").read_text())
    if not all(qualified.get(k) for k in ["retain","synced_write_pass","fresh_pod_read_pass"]):
        raise RuntimeError("Fresh retained volume not qualified")
    sts=json.loads(kubectl("get","statefulset","pireus-pg-relocation","-o","json"))
    (args.output/"old-statefulset-private.json").write_text(json.dumps(sts))
    template=sts["spec"]["template"]["spec"]
    if template["volumes"][0]["persistentVolumeClaim"]["claimName"]!="pireus-pg-relocation-data":
        raise RuntimeError("Unexpected old rehearsal volume")
    pod=json.loads(kubectl("get","pod",POD,"-o","json"))
    if pod["spec"]["nodeName"]!="r770-proxmox":
        raise RuntimeError("Wrong target node")
    if json.loads(kubectl("get","ciliumnetworkpolicy","-o","json"))["items"]:
        raise RuntimeError("Target has active Cilium ingress")
    policy=json.loads(kubectl("get","networkpolicy","pireus-pg-relocation-rehearsal","-o","json"))
    if policy["spec"].get("ingress") or policy["spec"]["podSelector"]!={"matchLabels":{"app":"pireus-pg-relocation"}}:
        raise RuntimeError("Target isolation differs")
    old_args=template["containers"][0]["args"]
    if old_args.count("wal_level=replica")!=1 or old_args.count("cron.launch_active_jobs=off")!=1:
        raise RuntimeError("Prior rehearsal runtime not restored")
    current_claim=json.loads(kubectl("get","pvc",PVC,"-o","json"))
    if current_claim["spec"]["volumeName"]!=qualified["pv"]:
        raise RuntimeError("Qualified volume identity changed")
    new_args=list(old_args)
    for name,value in [("max_wal_size","8GB"),("min_wal_size","2GB")]:
        matches=[s for s in new_args if s.startswith(name+"=")]
        if matches and matches!=[name+"="+value]:raise RuntimeError("Unexpected WAL bound")
        if not matches:new_args+=["-c",name+"="+value]
    patch=[{"op":"test","path":"/metadata/resourceVersion","value":sts["metadata"]["resourceVersion"]},
           {"op":"test","path":"/spec/template/spec/volumes/0/persistentVolumeClaim/claimName","value":"pireus-pg-relocation-data"},
           {"op":"replace","path":"/spec/template/spec/volumes/0/persistentVolumeClaim/claimName","value":PVC},
           {"op":"replace","path":"/spec/template/spec/containers/0/args","value":new_args}]
    log_path=args.output/"switch-private.log"
    with log_path.open("xb") as log:
        subprocess.run(["kubectl","-n","beagle","patch","statefulset","pireus-pg-relocation","--type=json","--patch",json.dumps(patch)],stdout=log,stderr=log,check=True,timeout=30)
        subprocess.run(["kubectl","-n","beagle","rollout","status","statefulset/pireus-pg-relocation","--timeout=600s"],stdout=log,stderr=log,check=True,timeout=620)
    check=sql("SELECT oid FROM pg_roles WHERE rolname=current_user; SELECT count(*) FROM pg_database WHERE datname='memory'",log_path).strip()
    if check!=b"10\n0":
        raise RuntimeError("New volume is not a fresh source-aligned bootstrap")
    # Establish reloadable cron OFF before removing the bootstrap command-line override.
    sql("ALTER SYSTEM SET cron.launch_active_jobs='off';",log_path)
    at=new_args.index("cron.launch_active_jobs=off")
    if at==0 or new_args[at-1]!="-c":raise RuntimeError("Cron override syntax changed")
    del new_args[at-1:at+1]
    current=json.loads(kubectl("get","statefulset","pireus-pg-relocation","-o","json"))
    patch=[{"op":"test","path":"/metadata/resourceVersion","value":current["metadata"]["resourceVersion"]},
           {"op":"replace","path":"/spec/template/spec/containers/0/args","value":new_args}]
    with log_path.open("ab") as log:
        subprocess.run(["kubectl","-n","beagle","patch","statefulset","pireus-pg-relocation","--type=json","--patch",json.dumps(patch)],stdout=log,stderr=log,check=True,timeout=30)
        subprocess.run(["kubectl","-n","beagle","rollout","status","statefulset/pireus-pg-relocation","--timeout=600s"],stdout=log,stderr=log,check=True,timeout=620)
    if sql("SELECT setting,context,source FROM pg_settings WHERE name='cron.launch_active_jobs'",log_path).strip()!=b"off|sighup|configuration file":
        raise RuntimeError("Reloadable cron OFF not established")
    # There is no restored user database yet; no scheduled job can be duplicated.
    # Restore role attributes and passwords privately with the original bootstrap OID.
    globals_sql=(args.source_output/"globals-private.sql").read_text()
    if globals_sql.splitlines().count("CREATE ROLE memory;")!=1:
        raise RuntimeError("Unexpected bootstrap role dump")
    filtered="\n".join("-- Existing source bootstrap role" if l=="CREATE ROLE memory;" else l for l in globals_sql.splitlines())
    sql("BEGIN;\n"+filtered+"\nCOMMIT;\n",log_path)
    inventory=json.loads((args.source_output/"inventory.json").read_text())
    archives=[]
    for i,db in enumerate(inventory["databases"]):
        if not db["datallowconn"]:continue
        local=args.source_output/("db-"+str(i))/"database.dump"
        remote="/var/lib/postgresql/data/seed-v5-db-"+str(i)+".dump"
        with local.open("rb") as f:expected=hashlib.file_digest(f,"sha256").hexdigest()
        with local.open("rb") as f,log_path.open("ab") as log:
            subprocess.run(["kubectl","-n","beagle","exec","-i",POD,"--","sh","-c",
                'set -e; umask 077; test ! -e "$1"; cat > "$1"',"--",remote],
                stdin=f,stdout=log,stderr=log,check=True,timeout=600)
        if kubectl("exec",POD,"--","sha256sum",remote).decode().split()[0]!=expected:
            raise RuntimeError("Seed archive custody failed")
        archives.append({"index":i,"bytes":local.stat().st_size})
    report={"new_pvc":PVC,"old_pvc_retained":"pireus-pg-relocation-data",
            "source_write_pause":False,"production_endpoint_changed":False,
            "bootstrap_role_oid":10,"cron_launch_active_jobs":"off",
            "cron_control_source":"configuration file","cron_context":"sighup",
            "seed_archives_custody_pass":True,"seed_archives":archives,
            "production_cutover_accepted":False}
    (args.output/"summary.json").write_text(json.dumps(report,indent=2))
    print("ISOLATED_TARGET_VOLUME_SWITCH_PASS "+json.dumps(report),flush=True)
if __name__=="__main__":main()

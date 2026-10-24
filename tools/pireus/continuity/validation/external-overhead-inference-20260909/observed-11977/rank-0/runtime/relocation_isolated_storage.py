#!/usr/bin/env python3
"""Provision and test the OSD0-excluded rehearsal volume; private CSI credentials."""
import argparse, json, os, pathlib, subprocess, time
from relocation_rehearsal import kubectl
HOST=["kubectl","-n","beagle","exec","node-ephemeral-governance-kp96t","-c","governor","--","nsenter","-t","1","-m","-p","--"]
POOL_ROOT="/var/tmp/pireus-ceph-pg-pool-20260907"
PVC="pireus-pg-relocation-ssd-data"
SC="ceph-rbd-pireus-pg-ssd"
SECRET="pireus-pg-relocation-csi"
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=pathlib.Path,required=True)
    args=ap.parse_args();os.umask(0o077)
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    summary=json.loads(subprocess.check_output(HOST+["cat",POOL_ROOT+"/summary.json"],timeout=30))
    if not summary["all_new_pgs_active_clean"] or summary["excluded_osds"]!=[0]:
        raise RuntimeError("New pool placement not qualified")
    credentials=json.loads(subprocess.check_output(HOST+["cat",POOL_ROOT+"/csi-credentials-private.json"],timeout=30))
    if len(credentials)!=1 or credentials[0]["entity"]!="client.pireus-pg-relocation":
        raise RuntimeError("Wrong scoped CSI identity")
    entry=credentials[0]
    if entry["caps"]!={"mon":"profile rbd","osd":"profile rbd pool=pireus_pg_ssd","mgr":"profile rbd pool=pireus_pg_ssd"}:
        raise RuntimeError("Unexpected CSI capability scope")
    secret={"apiVersion":"v1","kind":"Secret","metadata":{"name":SECRET,"namespace":"ceph-csi-rbd"},
            "type":"Opaque","stringData":{"userID":"pireus-pg-relocation","userKey":entry["key"]}}
    (args.output/"csi-secret-private.json").write_text(json.dumps(secret))
    base=json.loads(kubectl("get","storageclass","ceph-rbd-ssd-checkpoints","-o","json"))
    sc={k:base[k] for k in ["apiVersion","kind","allowVolumeExpansion","parameters","provisioner","reclaimPolicy","volumeBindingMode"]}
    sc["metadata"]={"name":SC};sc["parameters"]["pool"]="pireus_pg_ssd"
    for k in sc["parameters"]:
        if k.endswith("-secret-name"):sc["parameters"][k]=SECRET
    if sc["reclaimPolicy"]!="Retain":raise RuntimeError("Retention changed")
    pvc={"apiVersion":"v1","kind":"PersistentVolumeClaim","metadata":{"name":PVC,"namespace":"beagle"},
         "spec":{"accessModes":["ReadWriteOnce"],"storageClassName":SC,"resources":{"requests":{"storage":"64Gi"}}}}
    with (args.output/"provision-private.log").open("xb") as log:
        for label,obj in [("secret",secret),("storageclass",sc),("pvc",pvc)]:
            subprocess.run(["kubectl","create","-f","-"],input=json.dumps(obj).encode(),stdout=log,stderr=log,check=True,timeout=40)
            if label!="secret":(args.output/(label+".json")).write_text(json.dumps(obj,indent=2))
        deadline=time.monotonic()+180
        while True:
            claim=json.loads(kubectl("get","pvc",PVC,"-o","json"))
            if claim.get("status",{}).get("phase")=="Bound":break
            if time.monotonic()>deadline:raise RuntimeError("New PVC binding timeout")
            time.sleep(2)
        volume=json.loads(kubectl("get","pv",claim["spec"]["volumeName"],"-o","json"))
        if volume["spec"]["persistentVolumeReclaimPolicy"]!="Retain" or volume["spec"]["csi"]["volumeAttributes"]["pool"]!="pireus_pg_ssd":
            raise RuntimeError("Wrong volume custody/placement")
        (args.output/"volume.json").write_text(json.dumps(volume,indent=2))
        for kind,filename in [("write","relocation_storage_job.json"),("read","relocation_storage_reader_job.json")]:
            job=json.loads(pathlib.Path(__file__).with_name(filename).read_text())
            name="pireus-pg-isolated-"+kind+"-20260907";job["metadata"]["name"]=name
            job["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"]=PVC
            (args.output/(kind+"-job.json")).write_text(json.dumps(job,indent=2))
            subprocess.run(["kubectl","create","-f","-"],input=json.dumps(job).encode(),stdout=log,stderr=log,check=True,timeout=30)
            subprocess.run(["kubectl","-n","beagle","wait","--for=condition=complete","job/"+name,"--timeout=620s"],stdout=log,stderr=log,check=True,timeout=640)
            output=kubectl("logs","job/"+name).decode()
            (args.output/(kind+"-result.log")).write_text(output)
            if "integrity.bin: OK" not in output or (kind=="read" and "FRESH_POD_INTEGRITY_PASS" not in output):
                raise RuntimeError("Volume integrity control failed")
    report={"pool":"pireus_pg_ssd","pvc":PVC,"pv":claim["spec"]["volumeName"],"retain":True,
            "volume_bytes":64*1024**3,"synced_write_pass":True,"fresh_pod_read_pass":True,
            "source_write_pause":False,"production_cutover_accepted":False}
    (args.output/"summary.json").write_text(json.dumps(report,indent=2))
    print("ISOLATED_VOLUME_CONTROL_PASS "+json.dumps(report),flush=True)
if __name__=="__main__":main()

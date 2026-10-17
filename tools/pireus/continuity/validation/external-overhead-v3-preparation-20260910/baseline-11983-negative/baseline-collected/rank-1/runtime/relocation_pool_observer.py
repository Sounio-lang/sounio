#!/usr/bin/env python3
"""Observe only OSDs serving the isolated pool, retaining global I/O limits separately."""
import argparse, concurrent.futures, datetime, json, os, pathlib, subprocess, time
from relocation_rehearsal import kubectl
from relocation_isolated_storage import HOST
def ceph(*args):
    return json.loads(subprocess.check_output(HOST+["ceph",*args,"--format","json"],timeout=35))
SCAN = """
import datetime,gzip,json,pathlib,sys
start=datetime.datetime.fromisoformat(sys.argv[1]);results=[]
for i in json.loads(sys.argv[2]):
 p=pathlib.Path('/var/log/ceph/ceph-osd.'+str(i)+'.log')
 if not p.exists():raise RuntimeError('OSD log unavailable')
 lines=p.read_text(errors='replace').splitlines()
 if pathlib.Path(str(p)+'.1.gz').exists():
  lines+=gzip.open(str(p)+'.1.gz','rt',errors='replace').read().splitlines()
 counts={'osd':i,'fresh_slow':0,'fresh_stalled':0}
 for line in lines:
  if 'slow operation observed' not in line and 'stalled read' not in line:continue
  stamp=datetime.datetime.fromisoformat(line.split()[0])
  if stamp>=start:
   if 'slow operation observed' in line:counts['fresh_slow']+=1
   if 'stalled read' in line:counts['fresh_stalled']+=1
 results.append(counts)
print(json.dumps(results))
"""
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=pathlib.Path,required=True)
    ap.add_argument("--stop-file",type=pathlib.Path,required=True)
    ap.add_argument("--seconds",type=int,default=1800)
    args=ap.parse_args();os.umask(0o077)
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    if args.stop_file.exists():raise RuntimeError("Stop marker predates observer")
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    deadline=time.monotonic()+args.seconds
    tree=ceph("osd","df","tree")
    hosts={o:n["name"] for n in tree["nodes"] if n["type"]=="host" for o in n["children"]}
    nodes={n["id"]:n for n in tree["nodes"] if n["type"]=="osd"}
    pgs=ceph("pg","ls-by-pool","pireus_pg_ssd")["pg_stats"]
    osds=sorted({o for pg in pgs for o in pg["acting"]})
    if 0 in osds or any(nodes[o]["device_class"]!="ssd" for o in osds):raise RuntimeError("Pool exclusion/class failed")
    pods=json.loads(kubectl("get","pods","-o","json"))["items"]
    groups={}
    for o in osds:groups.setdefault(hosts[o],[]).append(o)
    commands={}
    for host,ids in groups.items():
        candidates=[p for p in pods if p["metadata"]["name"].startswith("node-ephemeral-governance-") and p["spec"].get("nodeName")==host]
        if len(candidates)!=1:raise RuntimeError("Host observation surface ambiguous")
        p=candidates[0]
        commands[host]=["kubectl","-n","beagle","exec",p["metadata"]["name"],"-c","governor","--",
                        "nsenter","-t","1","-m","-p","--","python3","-c",SCAN,started,json.dumps(ids)]
    max_slow=max_stalled=0;samples=0;placement_pass=True;completed=False
    while time.monotonic()<deadline:
        current=ceph("pg","ls-by-pool","pireus_pg_ssd")["pg_stats"]
        placement= len(current)==8 and all(
            pg["up"]==pg["acting"] and len(pg["acting"])==3 and
            len({hosts[o] for o in pg["acting"]})==3 and
            all(o in osds for o in pg["acting"]) and
            set(pg["state"].split("+")) <= {"active","clean","scrubbing","deep"} and
            {"active","clean"} <= set(pg["state"].split("+")) for pg in current)
        placement_pass &= placement
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(commands)) as pool:
            futures=[pool.submit(subprocess.check_output,c,timeout=35) for c in commands.values()]
            observed=[o for f in futures for o in json.loads(f.result())]
        if sorted(o["osd"] for o in observed)!=osds:raise RuntimeError("Incomplete OSD log observation")
        slow=sum(o["fresh_slow"] for o in observed);stalled=sum(o["fresh_stalled"] for o in observed)
        max_slow=max(max_slow,slow);max_stalled=max(max_stalled,stalled);samples+=1
        record={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "placement_pass":placement,"osds":observed}
        with (args.output/"progress.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
        if args.stop_file.exists():completed=True;break
        time.sleep(20)
    report={"started_at":started,"ended_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pool":"pireus_pg_ssd","osds":osds,"excluded_osds":[0],"samples":samples,
            "fresh_slow_events":max_slow,"fresh_stalled_events":max_stalled,
            "placement_pass":placement_pass,"completion_marker_observed":completed,
            "pool_io_observation_pass":completed and samples>=2 and placement_pass and max_slow==0 and max_stalled==0,
            "global_cluster_io_repair_claimed":False,"production_cutover_accepted":False}
    (args.output/"summary.json").write_text(json.dumps(report,indent=2))
    print("POOL_IO_OBSERVATION "+json.dumps(report),flush=True)
    if not report["pool_io_observation_pass"]:raise SystemExit(2)
if __name__=="__main__":main()

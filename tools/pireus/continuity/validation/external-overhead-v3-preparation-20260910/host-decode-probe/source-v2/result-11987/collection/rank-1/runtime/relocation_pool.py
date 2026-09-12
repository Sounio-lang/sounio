#!/usr/bin/env python3
"""Create a bounded RBD pool excluding OSD0; run on the Ceph admin host."""
import json, os, pathlib, re, subprocess, time
POOL = "pireus_pg_ssd"
CLIENT = "client.pireus-pg-relocation"
ROOT = pathlib.Path("/var/tmp/pireus-ceph-pg-pool-20260907")
def run(*args):
    p = subprocess.run(list(args), capture_output=True, timeout=60)
    with (ROOT / "commands-private.log").open("ab") as f:
        f.write(p.stderr)
    if p.returncode:
        raise RuntimeError("Storage command failed; private diagnostic retained")
    return p.stdout
def ceph(*args):
    return run("ceph", *args)
def data(*args):
    return json.loads(ceph(*args, "--format", "json"))
def save(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2))
def mappings(name):
    path = str(ROOT / (name + ".osdmap"))
    ceph("osd", "getmap", "-o", path)
    output = run("osdmaptool", path, "--test-map-pgs-dump-all").decode()
    (ROOT / (name + "-mapping.txt")).write_text(output)
    result = {}
    for line in output.splitlines():
        m = re.match(r"([0-9]+\.[0-9a-f]+) raw .* up \(\[([^]]*)\]", line)
        if m:
            result[m[1]] = [int(x) for x in m[2].split(",") if x]
    return result
def main():
    os.umask(0o077)
    ROOT.mkdir(mode=0o700, exist_ok=False)
    if ceph("fsid").decode().strip() != "f591bae9-eec5-4ae0-abfe-466ed7528c9e":
        raise RuntimeError("Wrong cluster")
    pools = data("osd", "pool", "ls", "detail")
    if POOL in [p["pool_name"] for p in pools]:
        raise RuntimeError("Pool already exists; preserve it")
    auth = subprocess.run(["ceph", "auth", "get", CLIENT, "--format", "json"], capture_output=True, timeout=30)
    if auth.returncode == 0 or b"ENOENT" not in auth.stderr:
        raise RuntimeError("New CSI identity is not demonstrably absent")
    rule = data("osd", "crush", "rule", "dump", "replicated_ssd")
    if rule["type"] != 1 or rule["steps"] != [
        {"op":"take","item":-2,"item_name":"default~ssd"},
        {"op":"chooseleaf_firstn","num":0,"type":"host"},{"op":"emit"}]:
        raise RuntimeError("Unexpected SSD host-failure-domain rule")
    health = data("health", "detail")
    if set(health.get("checks", {})) - {"BLUESTORE_SLOW_OP_ALERT", "DB_DEVICE_STALLED_READ_ALERT"}:
        raise RuntimeError("New cluster health blocker")
    tree = data("osd", "df", "tree")
    nodes = {n["id"]:n for n in tree["nodes"] if n["type"]=="osd"}
    hosts = {c:n["name"] for n in tree["nodes"] if n["type"]=="host" for c in n["children"]}
    balancer = data("balancer", "status")
    if balancer["mode"] != "upmap":
        raise RuntimeError("Balancer mode changed")
    raw_allow = ceph("balancer", "pool", "ls").strip()
    allow = json.loads(raw_allow) if raw_allow else []
    save("before.json", {"pools":pools,"rule":rule,"health":health,"tree":tree,
                         "balancer":balancer,"balancer_pool_allowlist":allow})
    ceph("balancer", "off")
    try:
        before_maps = mappings("before")
        crush_before = data("osd", "crush", "dump")
        # Upmap optimization does not use per-pool weight sets in this release.
        # Preserve automatic balancing for every currently eligible existing pool.
        eligible = allow or [p["pool_name"] for p in pools]
        ceph("balancer", "pool", "add", *eligible)
        ceph("osd", "pool", "create", POOL, "8", "8", "replicated", "replicated_ssd")
        ceph("osd", "pool", "set", POOL, "size", "3")
        ceph("osd", "pool", "set", POOL, "min_size", "2")
        ceph("osd", "pool", "set", POOL, "pg_autoscale_mode", "off")
        ceph("osd", "pool", "set-quota", POOL, "max_bytes", str(64*1024**3))
        ceph("osd", "pool", "application", "enable", POOL, "rbd")
        ceph("osd", "crush", "weight-set", "create", POOL, "flat")
        ceph("osd", "crush", "weight-set", "reweight", POOL, "osd.0", "0")
        crush_after = data("osd", "crush", "dump")
        if {k:v for k,v in crush_before.items() if k!="choose_args"} != {k:v for k,v in crush_after.items() if k!="choose_args"}:
            raise RuntimeError("Existing CRUSH topology changed")
        after_maps = mappings("after")
        if any(after_maps.get(pg)!=v for pg,v in before_maps.items()):
            raise RuntimeError("Existing pool placement changed; do not provision")
        new_maps = {pg:v for pg,v in after_maps.items() if pg not in before_maps}
        if len(new_maps)!=8:
            raise RuntimeError("Unexpected new pool PG count")
        for pg, replicas in new_maps.items():
            if len(replicas)!=3 or 0 in replicas or len({hosts[o] for o in replicas})!=3:
                raise RuntimeError("Excluded OSD or host-failure-domain violation")
            for o in replicas:
                n=nodes[o]
                if n["device_class"]!="ssd" or n["kb_avail"]*1024 < 2*64*1024**3:
                    raise RuntimeError("Insufficient SSD capacity margin")
                if (n["kb_used"]*1024+64*1024**3)/(n["kb"]*1024) >= 0.85:
                    raise RuntimeError("Worst-case pool quota would approach nearfull")
        actual_allow = json.loads(ceph("balancer", "pool", "ls"))
        if set(actual_allow)!=set(eligible) or POOL in actual_allow:
            raise RuntimeError("Balancer exclusion did not persist")
        creds=data("auth","get-or-create",CLIENT,"mon","profile rbd",
                   "osd","profile rbd pool="+POOL,"mgr","profile rbd pool="+POOL)
        save("csi-credentials-private.json",creds)
        deadline=time.monotonic()+180
        while True:
            pgs=data("pg","ls-by-pool",POOL)["pg_stats"]
            if len(pgs)==8 and all(p["state"]=="active+clean" and p["up"]==p["acting"] and 0 not in p["acting"] for p in pgs):
                break
            if time.monotonic()>deadline:
                raise RuntimeError("New pool did not reach clean placement")
            time.sleep(3)
        save("after.json", {"pool":data("osd","pool","get",POOL,"all"),
             "crush":crush_after,"pgs":pgs,"balancer_pool_allowlist":actual_allow})
        report={"pool":POOL,"size":3,"min_size":2,"quota_bytes":64*1024**3,
                "excluded_osds":[0],"pg_count":8,"pgs":new_maps,
                "existing_pool_placements_unchanged":True,"existing_crush_topology_unchanged":True,
                "existing_pool_balancing_preserved":True,"new_pool_automatic_upmap_excluded":True,
                "future_pool_balancer_enrollment_requires_explicit_add":True,
                "all_new_pgs_active_clean":True,"source_write_pause":False,
                "io_qualification_pass":False,"production_cutover_accepted":False}
        save("summary.json",report)
        print(json.dumps(report),flush=True)
    finally:
        if balancer["active"]:
            ceph("balancer","on")
        save("balancer-final.json",data("balancer","status"))
if __name__=="__main__":
    main()

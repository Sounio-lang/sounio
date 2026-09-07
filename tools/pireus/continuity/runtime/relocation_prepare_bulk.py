#!/usr/bin/env python3
"""Prepare EMPTY isolated databases before a source maintenance window."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from relocation_rehearsal import kubectl, target, sql, ident, literal, ADMIN_DB, POD

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive-prefix", required=True)
    p.add_argument("--normal-statefile", type=Path)
    args = p.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    policy = json.loads(kubectl("get", "networkpolicy", "pireus-pg-relocation-rehearsal", "-o", "json"))
    if policy["spec"].get("ingress") or policy["spec"]["podSelector"] != {"matchLabels": {"app": "pireus-pg-relocation"}}:
        raise RuntimeError("Destination is not isolated")
    pod = json.loads(kubectl("get", "pod", POD, "-o", "json"))
    if pod["spec"]["nodeName"] != "r770-proxmox":
        raise RuntimeError("Wrong destination node")
    sts = json.loads(kubectl("get", "statefulset", "pireus-pg-relocation", "-o", "json"))
    if args.normal_statefile:
        sts = json.loads(args.normal_statefile.read_text())
    (args.output / "normal-statefulset-private.json").write_text(json.dumps(sts))
    old_args = sts["spec"]["template"]["spec"]["containers"][0]["args"]
    if old_args.count("wal_level=replica") != 1:
        raise RuntimeError("Expected normal replica configuration")
    new_args = ["wal_level=minimal" if a == "wal_level=replica" else a for a in old_args]
    if any(a.startswith("max_wal_senders=") for a in new_args):
        raise RuntimeError("Unexpected sender override")
    new_args += ["-c", "max_wal_senders=0"]
    patch = {"spec": {"template": {"spec": {"containers": [{"name": "postgres", "args": new_args}]}}}}
    with (args.output / "prepare-private.log").open("ab") as errors:
        subprocess.run(["kubectl", "-n", "beagle", "patch", "statefulset", "pireus-pg-relocation",
                        "--type=strategic", "--patch", json.dumps(patch)], stdout=errors, stderr=errors,
                       check=True, timeout=30)
        subprocess.run(["kubectl", "-n", "beagle", "rollout", "status", "statefulset/pireus-pg-relocation",
                        "--timeout=600s"], stdout=errors, stderr=errors, check=True, timeout=620)
    inventory = json.loads((args.source_output / "inventory.json").read_text())
    total = 0
    for i, db in enumerate(inventory["databases"]):
        if not db["datallowconn"]:
            continue
        archive = "/var/lib/postgresql/data/" + args.archive_prefix + "-db-" + str(i) + ".dump"
        listing = kubectl("exec", POD, "--", "pg_restore", "--create", "--list", archive)
        selected = []
        for line in listing.splitlines():
            if line.startswith(b";") or not line.strip():
                continue
            fields = line.split(maxsplit=3)
            if len(fields) == 4 and re.match(rb"^(DATABASE(?: PROPERTIES)? -|(?:ACL|COMMENT|SECURITY LABEL) - DATABASE )", fields[3]):
                selected.append(line)
        if not any(b" DATABASE - " in line for line in selected):
            raise RuntimeError("Archive database creation metadata missing")
        log_path = args.output / ("db-" + str(i) + "-prepare-private.log")
        exists = sql("SELECT count(*) FROM pg_database WHERE datname=" + literal(db["datname"]), log_path).strip()
        if exists == b"1":
            sql("ALTER DATABASE " + ident(db["datname"]) + " ALLOW_CONNECTIONS false; "
                "ALTER DATABASE " + ident(db["datname"]) + " IS_TEMPLATE false; "
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=" +
                literal(db["datname"]) + " AND pid<>pg_backend_pid();", log_path)
        with log_path.open("ab") as errors:
            subprocess.run(target("pg_restore", ADMIN_DB,
                ["--create", "--clean", "--if-exists", "--exit-on-error", "--use-list=/dev/stdin", archive], True),
                input=b"\n".join(selected)+b"\n", stdout=errors, stderr=errors, check=True, timeout=600)
        total += 1
        print(json.dumps({"database_index": i, "empty_destination_prepared": True}), flush=True)
    receipt = {"prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "databases_prepared": total, "source_write_pause": False,
               "target_wal_level": "minimal", "target_max_wal_senders": 0,
               "production_cutover_accepted": False, "must_restore_runtime_mode": True}
    (args.output / "summary.json").write_text(json.dumps(receipt, indent=2))
    print("BULK_TARGET_PREPARED " + json.dumps(receipt), flush=True)

if __name__ == "__main__":
    main()

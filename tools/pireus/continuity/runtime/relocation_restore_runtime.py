#!/usr/bin/env python3
"""Restore isolated target runtime settings and verify post-restart behavior."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from relocation_rehearsal import target, kubectl, ADMIN_DB

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-output", type=Path, required=True)
    p.add_argument("--source-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    original = json.loads((args.prepared_output / "normal-statefulset-private.json").read_text())
    normal_args = original["spec"]["template"]["spec"]["containers"][0]["args"]
    if normal_args.count("wal_level=replica") != 1:
        raise RuntimeError("Captured normal runtime configuration invalid")
    current = json.loads(kubectl("get", "statefulset", "pireus-pg-relocation", "-o", "json"))
    current_args = current["spec"]["template"]["spec"]["containers"][0]["args"]
    if current_args.count("wal_level=minimal") != 1 or current_args.count("max_wal_senders=0") != 1:
        raise RuntimeError("Target is not in the expected isolated bulk mode")
    started = time.monotonic()
    patch = {"spec": {"template": {"spec": {"containers": [{"name": "postgres", "args": normal_args}]}}}}
    with (args.output / "runtime-private.log").open("xb") as log:
        subprocess.run(["kubectl", "-n", "beagle", "patch", "statefulset", "pireus-pg-relocation",
                        "--type=strategic", "--patch", json.dumps(patch)], stdout=log, stderr=log, check=True, timeout=30)
        subprocess.run(["kubectl", "-n", "beagle", "rollout", "status", "statefulset/pireus-pg-relocation",
                        "--timeout=600s"], stdout=log, stderr=log, check=True, timeout=620)
        settings = subprocess.run(target("psql", ADMIN_DB, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c",
            "SELECT current_setting('wal_level'),current_setting('max_wal_senders'),current_setting('archive_mode'),current_setting('fsync'),current_setting('full_page_writes'),current_setting('synchronous_commit'),current_setting('cron.launch_active_jobs')"]),
            stdout=subprocess.PIPE, stderr=log, timeout=30)
        if settings.returncode or settings.stdout.strip() != b"replica|10|off|on|on|on|off":
            raise RuntimeError("Normal runtime settings failed verification")
        restart_seconds = time.monotonic() - started
        inventory = json.loads((args.source_output / "inventory.json").read_text())
        analyze_started = time.monotonic()
        analyzed = 0
        for db in inventory["databases"]:
            if not db["datallowconn"]:
                continue
            subprocess.run(target("psql", db["datname"], ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", "ANALYZE"]),
                           stdout=log, stderr=log, check=True, timeout=180)
            analyzed += 1
        analyze_seconds = time.monotonic() - analyze_started
        functional_started = time.monotonic()
        subprocess.run(target("psql", "memory", ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-f", "-"], True),
                       input=Path(__file__).with_name("relocation_functional.sql").read_bytes(),
                       stdout=log, stderr=log, check=True, timeout=180)
        functional_seconds = time.monotonic() - functional_started
    report = {"runtime_mode_restored": True, "settings": "replica|10|off|on|on|on|off",
              "restart_seconds": restart_seconds, "analyzed_databases": analyzed,
              "analyze_seconds": analyze_seconds, "functional_seconds": functional_seconds,
              "elapsed_seconds": time.monotonic() - started, "functional_pass": True,
              "source_write_pause": False, "production_cutover_accepted": False}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print("NORMAL_RUNTIME_AND_FUNCTIONAL_PASS " + json.dumps(report), flush=True)

if __name__ == "__main__":
    main()

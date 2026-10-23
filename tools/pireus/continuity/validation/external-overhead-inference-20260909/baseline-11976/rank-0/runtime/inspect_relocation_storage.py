#!/usr/bin/env python3
"""Read-only Ceph evidence for the approved relocation; never prints credentials."""
import argparse
import base64
import datetime
import json
import subprocess

def inspect():
    ns = "ceph-csi-rbd"
    secret = json.loads(subprocess.check_output(
        ["kubectl", "-n", ns, "get", "secret", "csi-rbd-secret", "-o", "json"],
        timeout=30))
    uid = base64.b64decode(secret["data"]["userID"]).decode()
    key = base64.b64decode(secret["data"]["userKey"]).decode()
    keyring = ("[client." + uid + "]\n key = " + key + "\n").encode()
    pods = json.loads(subprocess.check_output(
        ["kubectl", "-n", ns, "get", "pods", "-o", "json"], timeout=30))
    candidates = [
        p["metadata"]["name"] for p in pods["items"]
        if p["spec"].get("nodeName") == "r770-proxmox"
        and any(c["name"] == "csi-rbdplugin" for c in p["spec"]["containers"])
        and any(c["name"] == "csi-rbdplugin" and c.get("ready")
                for c in p.get("status", {}).get("containerStatuses", []))
    ]
    if len(candidates) != 1:
        raise RuntimeError("Expected one ready R770 CSI plugin")
    results = {}
    for label, command in (("health", "health detail"), ("capacity", "df detail")):
        # Key travels over stdin only. A private transient file is removed on exit.
        shell = (
            'set -eu; umask 077; keyfile=$(mktemp); '
            'trap \'rm -f "$keyfile"\' EXIT; cat >"$keyfile"; '
            'ceph --name "$1" --keyring "$keyfile" '
            '-m 10.100.100.2:3300,10.100.100.4:3300,10.100.100.3:3300 '
            + command + ' --format json'
        )
        result = subprocess.run(
            ["kubectl", "-n", ns, "exec", "-i", candidates[0],
             "-c", "csi-rbdplugin", "--", "sh", "-c", shell,
             "--", "client." + uid],
            input=keyring, capture_output=True, timeout=45)
        if result.returncode:
            raise RuntimeError("Ceph " + label + " failed, rc=" + str(result.returncode))
        results[label] = json.loads(result.stdout)
    pool = next(p for p in results["capacity"]["pools"] if p["name"] == "rbd_ssd")
    checks = results["health"].get("checks", {})
    # This records a conservative stop, not an assertion about which PG holds a future PVC.
    reasons = []
    nearfull = checks.get("POOL_NEARFULL", {}).get("detail", [])
    if any("pool 'rbd_ssd'" in x.get("message", "") for x in nearfull):
        reasons.append("target_pool_nearfull")
    for check in ("BLUESTORE_SLOW_OP_ALERT", "DB_DEVICE_STALLED_READ_ALERT",
                  "OSD_FULL", "POOL_FULL", "PG_AVAILABILITY", "PG_DEGRADED"):
        if check in checks:
            reasons.append(check)
    if pool["stats"]["max_avail"] < 64 * 1024**3:
        reasons.append("less_than_requested_64GiB_available")
    return {
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_pool": pool, "health": results["health"],
        "storage_precheck": "STOP" if reasons else "NEEDS_PROVISIONING_AND_IO_VALIDATION",
        "stop_reasons": reasons,
        "provisioning_verified": False, "io_acceptance_verified": False,
        "source_mutated": False, "target_created": False,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="New sanitized evidence JSON file")
    args = parser.parse_args()
    report = inspect()
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        with open(args.output, "x") as stream:
            stream.write(text)
    print(text, end="")
    raise SystemExit(2 if report["storage_precheck"] == "STOP" else 0)

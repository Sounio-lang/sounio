#!/usr/bin/env python3
"""Bounded forwarding probe on Spark port15433; production port5433 is untouched."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from relocation_dump import SOURCE_POD, SOURCE_CONTAINER
from relocation_rehearsal import kubectl, target, ADMIN_DB, POD

HOST = ["kubectl", "-n", "beagle", "exec", SOURCE_POD, "--", "nsenter", "-t", "1", "-m", "-p", "-n", "--"]
ROOT = "/opt/pireus/pg-relocation-probe3-20260907"
UNIT = "pireus-pg-forwarder-probe3.service"

def restrict_target_hba(output):
    result = subprocess.run(target("psql", ADMIN_DB, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", "SHOW hba_file"]),
                            capture_output=True, check=True, timeout=30)
    path = result.stdout.decode().strip()
    if path != "/var/lib/postgresql/data/pgdata-source-bootstrap/pg_hba.conf":
        raise RuntimeError("Unexpected destination HBA path")
    original = kubectl("exec", POD, "--", "cat", path)
    (output / "target-hba-before-private.conf").write_bytes(original)
    prefix = b"# pireus-source-proxy-only-v1\nhost all all 10.100.100.59/32 scram-sha-256\nhost all all 0.0.0.0/0 reject\nhost all all ::/0 reject\n"
    if not original.startswith(prefix):
        if b"# pireus-source-proxy-only-" in original:
            raise RuntimeError("Existing destination HBA guard differs")
        shell = 'set -e; umask 077; test ! -e "$2"; cat > "$2"; chown --reference="$1" "$2"; chmod --reference="$1" "$2"; mv -- "$2" "$1"'
        with (output / "hba-private.log").open("ab") as log:
            subprocess.run(["kubectl", "-n", "beagle", "exec", "-i", POD, "--", "sh", "-c", shell, "--",
                            path, path + ".pireus-" + uuid.uuid4().hex], input=prefix + original,
                           stdout=log, stderr=log, check=True, timeout=30)
    reload = subprocess.run(target("psql", ADMIN_DB, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c",
        "SELECT pg_reload_conf(); SELECT count(*) FROM pg_hba_file_rules WHERE error IS NOT NULL"]),
        capture_output=True, check=True, timeout=30)
    if reload.stdout.strip() != b"t\n0":
        raise RuntimeError("Destination HBA validation failed")
    script = 'PGCONNECT_TIMEOUT=3 PGPASSWORD="$POSTGRES_PASSWORD" exec psql -X -qAt -U "$POSTGRES_USER" -d memory -h127.0.0.1 -c "SELECT 1"'
    denied = subprocess.run(["kubectl", "-n", "beagle", "exec", POD, "--", "sh", "-c", script],
                            capture_output=True, timeout=20)
    (output / "hba-denied-private.log").write_bytes(denied.stderr)
    if denied.returncode == 0 or b"pg_hba.conf rejects" not in denied.stderr:
        raise RuntimeError("Destination HBA negative control failed")
    return {"target_hba_source_restricted": True, "target_hba_reject_control_pass": True,
            "target_hba_restriction_retained": True}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.monotonic()
    service = json.loads(kubectl("get", "service", "pireus-pg-relocation", "-o", "json"))
    target_ip = service["spec"]["clusterIP"]
    policies = json.loads(kubectl("get", "networkpolicy", "-o", "json"))["items"]
    if len(policies) != 1 or policies[0]["metadata"]["name"] != "pireus-pg-relocation-rehearsal" or policies[0]["spec"].get("ingress"):
        raise RuntimeError("Unexpected rehearsal policy set")
    code = Path(__file__).with_name("relocation_proxy.py").read_text()
    unit = "[Unit]\nDescription=Pireus PostgreSQL forwarding probe\nAfter=network-online.target\n\n[Service]\nType=simple\nDynamicUser=true\nExecStart=/usr/bin/python3 " + ROOT + "/forwarder.py --listen-port 15433 --target-host " + target_ip + " --source-host 10.100.100.59\nRestart=no\nRestartSec=2\nTimeoutStopSec=20\nNoNewPrivileges=true\nPrivateTmp=true\nPrivateDevices=true\nProtectSystem=strict\nProtectHome=true\nProtectKernelTunables=true\nProtectKernelModules=true\nProtectControlGroups=true\nRestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX\nMemoryMax=128M\nTasksMax=32\nCPUQuota=100%\n"
    policy = {"apiVersion": "cilium.io/v2", "kind": "CiliumNetworkPolicy",
              "metadata": {"name": "pireus-pg-relocation-probe-access", "namespace": "beagle"},
              "spec": {"endpointSelector": {"matchLabels": {"app": "pireus-pg-relocation"}},
                       "ingress": [{"fromEntities": ["remote-node"],
                                    "toPorts": [{"ports": [{"port": "5432", "protocol": "TCP"}]}]}]}}
    if json.loads(kubectl("get", "ciliumnetworkpolicy", "-o", "json"))["items"]:
        raise RuntimeError("Unexpected preexisting Cilium policy")
    cidrs = ["10.100.100.59/32"]
    hba = restrict_target_hba(args.output)
    ready = subprocess.run(target("psql", ADMIN_DB, ["-X", "-qAt", "-c", "SELECT 1"]), capture_output=True, timeout=30)
    if ready.returncode or ready.stdout.strip() != b"1":
        raise RuntimeError("Destination not ready for network controls")
    def login(host, port, label):
        script = 'PGCONNECT_TIMEOUT=3 PGPASSWORD="$POSTGRES_PASSWORD" exec psql -X -qAt -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d memory -h "$1" -p "$2" -c "SELECT 1"'
        cmd = HOST + ["docker", "exec", "-u", "postgres", SOURCE_CONTAINER, "sh", "-c", script, "--", host, str(port)]
        with (args.output / (label + "-private.log")).open("wb") as errors:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=errors, timeout=20)
        return result
    negative = login(target_ip, 5432, "denied")
    if negative.returncode == 0:
        raise RuntimeError("Network isolation negative control unexpectedly connected")
    token = uuid.uuid4().hex
    policy["metadata"]["labels"] = {"pireus.sounio.dev/probe-id": token}
    deployed = False
    allowed = False
    report = {"denied_connection_rc": negative.returncode, "source_database_stopped": False,
              "production_endpoint_changed": False, "production_cutover_accepted": False,
              "forwarder_sha256": hashlib.sha256(code.encode()).hexdigest(), "postgres_allowed_source_cidrs": cidrs,
              "cilium_ingress_entities": ["remote-node"], **hba}
    try:
        deployment = """import json,pathlib,sys,subprocess
j=json.load(sys.stdin)
root=pathlib.Path(j['root'])
unit=pathlib.Path('/etc/systemd/system')/j['unit']
if root.exists() or unit.exists():
 if not root.exists() or not unit.exists() or (root/'forwarder.py').read_text()!=j['code'] or unit.read_text()!=j['unit_text']:
  raise RuntimeError('Existing probe files differ; preserve them')
 state=subprocess.run(['systemctl','is-active',j['unit']],capture_output=True,text=True).stdout.strip()
 if state not in ['inactive','failed']:raise RuntimeError('Existing probe is active')
else:
 root.mkdir(parents=True,mode=0o755)
 (root/'forwarder.py').write_text(j['code'])
 (root/'forwarder.py').chmod(0o644)
 unit.write_text(j['unit_text'])
(root/'owner-token').write_text(j['token'])
subprocess.run(['systemctl','daemon-reload'],check=True)
subprocess.run(['systemctl','start',j['unit']],check=True)
"""
        deploy_cmd = HOST[:4] + ["-i"] + HOST[4:] + ["python3", "-c", deployment]
        with (args.output / "deploy-private.log").open("wb") as log:
            subprocess.run(deploy_cmd, input=json.dumps({"root": ROOT, "unit": UNIT, "code": code, "unit_text": unit, "token": token}).encode(),
                           stdout=log, stderr=log, check=True, timeout=40)
        deployed = True
        with (args.output / "policy-private.log").open("wb") as log:
            subprocess.run(["kubectl", "create", "-f", "-"], input=json.dumps(policy).encode(),
                           stdout=log, stderr=log, check=True, timeout=30)
        allowed = True
        time.sleep(2)
        for attempt in range(6):
            positive = login("172.17.0.1", 15433, "forwarded-" + str(attempt))
            if positive.returncode == 0 and positive.stdout.strip() == b"1":
                break
            time.sleep(2)
        else:
            raise RuntimeError("Forwarded authentication/query failed")
        report["positive_attempts"] = attempt + 1
        network_script = 'PGCONNECT_TIMEOUT=3 PGPASSWORD="$POSTGRES_PASSWORD" exec psql -X -qAt -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d memory -h "$1" -p15433 -c "SELECT 1; SELECT inet_client_addr()"'
        with (args.output / "kubernetes-client-private.log").open("wb") as errors:
            k8s = subprocess.run(["kubectl", "-n", "beagle", "exec", POD, "--", "sh", "-c", network_script,
                                  "--", "10.100.100.59"], stdout=subprocess.PIPE, stderr=errors, timeout=20)
        if k8s.returncode or not k8s.stdout.startswith(b"1\n"):
            raise RuntimeError("Kubernetes client forwarding failed")
        report["kubernetes_client_authentication_pass"] = True
        report["target_observed_client_address"] = k8s.stdout.decode().splitlines()[1]
        tls_probe = """import socket,struct,json
results={}
for host in ['127.0.0.1','::1']:
 with socket.create_connection((host,15433),timeout=5) as s:
  s.sendall(struct.pack('!II',8,80877103))
  results[host]=s.recv(1).decode()
assert set(results.values())=={'N'},'SSL negotiation changed'
print(json.dumps(results))
"""
        ssl = json.loads(subprocess.check_output(HOST + ["python3", "-c", tls_probe], timeout=20))
        report.update({"forwarded_authentication_and_query_pass": True, "ssl_negotiation": ssl,
                       "forwarder_sha256": hashlib.sha256(code.encode()).hexdigest(),
                       "ipv4_ipv6_transport_pass": True})
    finally:
        try:
            remaining = json.loads(kubectl("get", "ciliumnetworkpolicy", "-o", "json"))["items"]
            own = [item for item in remaining if item["metadata"].get("labels", {}).get("pireus.sounio.dev/probe-id") == token]
            for item in own:
                subprocess.run(["kubectl", "-n", "beagle", "delete", "ciliumnetworkpolicy", item["metadata"]["name"]],
                               stdout=subprocess.DEVNULL, check=True, timeout=30)
            report["temporary_policy_removed"] = True
        finally:
            cleanup = """import pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1])
marker=root/'owner-token'
if marker.exists() and marker.read_text()==sys.argv[2]:
 subprocess.run(['systemctl','stop',sys.argv[3]],check=True)
 print('OWNED_PROBE_STOPPED')
else:
 print('NO_OWNED_PROBE_SERVICE')
"""
            try:
                with (args.output / "cleanup-private.log").open("wb") as log:
                    result = subprocess.run(HOST + ["python3", "-c", cleanup, ROOT, token, UNIT],
                                            stdout=subprocess.PIPE, stderr=log, check=True, timeout=40)
                report["probe_service_stopped"] = result.stdout.strip() in [b"OWNED_PROBE_STOPPED", b"NO_OWNED_PROBE_SERVICE"]
            finally:
                report["elapsed_seconds"] = time.monotonic() - started
                (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print("FORWARDER_TRANSPORT_PROBE_PASS " + json.dumps(report), flush=True)

if __name__ == "__main__":
    main()

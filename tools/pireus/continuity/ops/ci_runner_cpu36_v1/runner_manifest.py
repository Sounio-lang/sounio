"""Generate an isolated one-shot CPU runner or its registration-free preflight."""
import json
from pathlib import Path

NAME = "pireus-ci-cpu36-v1"
NAMESPACE = "pireus-ci"
IMAGE = "docker.io/library/ubuntu@sha256:a61567bd31828687156d735ea8eb01ba4e37636e225dd6a48ba94136a70d9d61"
EXCEPT = ["0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8",
          "169.254.0.0/16", "172.16.0.0/12", "192.168.0.0/16",
          "198.18.0.0/15", "224.0.0.0/4", "240.0.0.0/4"]

def resources(mode="preflight"):
    if mode not in ("preflight", "execute"):
        raise ValueError("unknown runner mode")
    label = {"app": NAME}
    namespace = dict(apiVersion="v1", kind="Namespace", metadata={"name": NAMESPACE})
    policy = dict(apiVersion="networking.k8s.io/v1", kind="NetworkPolicy",
        metadata={"name": NAME, "namespace": NAMESPACE},
        spec={"podSelector": {"matchLabels": label}, "policyTypes": ["Ingress", "Egress"],
              "ingress": [], "egress": [
                  {"to": [{"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "kube-system"}},
                           "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}}}],
                   "ports": [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}]},
                  {"to": [{"ipBlock": {"cidr": "0.0.0.0/0", "except": EXCEPT}}],
                   "ports": [{"protocol": "TCP", "port": 443}, {"protocol": "TCP", "port": 80}]}]})
    config = dict(apiVersion="v1", kind="ConfigMap",
        metadata={"name": NAME, "namespace": NAMESPACE},
        data={"bootstrap.sh": Path(__file__).with_name("bootstrap.sh").read_text()})
    volumes = [{"name": "bootstrap", "configMap": {"name": NAME}},
               {"name": "work", "emptyDir": {"sizeLimit": "20Gi"}}]
    mounts = [{"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
              {"name": "work", "mountPath": "/runner"}]
    if mode == "execute":
        volumes.append({"name": "jit", "secret": {"secretName": NAME+"-jit", "defaultMode": 256}})
        mounts.append({"name": "jit", "mountPath": "/jit", "readOnly": True})
    container = {"name": "runner", "image": IMAGE, "command": ["bash", "/bootstrap/bootstrap.sh"],
        "env": [{"name": "PIREUS_RUNNER_MODE", "value": mode}],
        "resources": {"requests": {"cpu": "4", "memory": "36Gi", "ephemeral-storage": "24Gi"},
                      "limits": {"cpu": "4", "memory": "36Gi", "ephemeral-storage": "28Gi"}},
        "securityContext": {"allowPrivilegeEscalation": False, "privileged": False,
                            "capabilities": {"drop": ["ALL"], "add": ["CHOWN", "SETUID", "SETGID", "DAC_OVERRIDE", "FOWNER", "SETPCAP"]}},
        "volumeMounts": mounts}
    job = dict(apiVersion="batch/v1", kind="Job",
        metadata={"name": NAME+"-"+mode, "namespace": NAMESPACE},
        spec={"backoffLimit": 0, "activeDeadlineSeconds": 1200 if mode=="preflight" else 10800,
              "template": {"metadata": {"labels": label}, "spec": {
                  "automountServiceAccountToken": False, "restartPolicy": "Never",
                  "hostNetwork": False, "hostPID": False, "hostIPC": False,
                  "nodeSelector": {"kubernetes.io/hostname": "dl380-proxmox"},
                  "securityContext": {"seccompProfile": {"type": "RuntimeDefault"}},
                  "containers": [container], "volumes": volumes}}})
    return [namespace, policy, config, job]

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--mode", choices=("preflight","execute"), default="preflight")
    args=p.parse_args()
    print(json.dumps({"apiVersion":"v1","kind":"List","items":resources(args.mode)},indent=2))

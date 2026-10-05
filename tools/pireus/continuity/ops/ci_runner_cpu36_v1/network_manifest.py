"""Produce two bounded, credential-free controls for the existing runner policy."""
import copy,json
from pathlib import Path
from runner_manifest import resources,NAME,NAMESPACE
PROBE_NAME=NAME+"-network-v1"

def generate():
    config=dict(apiVersion="v1",kind="ConfigMap",
        metadata={"name":PROBE_NAME,"namespace":NAMESPACE},immutable=True,
        data={"probe.py":Path(__file__).with_name("network_probe.py").read_text()})
    result=[config]
    for role in ("control","isolated"):
        job=copy.deepcopy(resources()[3])
        job["metadata"]["name"]=PROBE_NAME+"-"+role
        job["spec"]["activeDeadlineSeconds"]=600
        pod=job["spec"]["template"]
        pod["metadata"]["labels"]={"app":NAME if role=="isolated" else PROBE_NAME+"-control"}
        s=pod["spec"]
        s["volumes"]=[{"name":"probe","configMap":{"name":PROBE_NAME}}]
        c=s["containers"][0]
        c["name"]="probe"
        c["volumeMounts"]=[{"name":"probe","mountPath":"/probe","readOnly":True}]
        c["env"]=[]
        c["resources"]={"requests":{"cpu":"200m","memory":"256Mi","ephemeral-storage":"1Gi"},
                        "limits":{"cpu":"1","memory":"512Mi","ephemeral-storage":"2Gi"}}
        c["command"]=["bash","-c",
            "set -euo pipefail; export DEBIAN_FRONTEND=noninteractive; "
            "apt-get update >/tmp/setup.log 2>&1; "
            "apt-get install -y --no-install-recommends python3 ca-certificates >>/tmp/setup.log 2>&1; "
            "exec setpriv --reuid=1000 --regid=1000 --clear-groups --bounding-set=-all --no-new-privs "
            "python3 /probe/probe.py --role "+role]
        result.append(job)
    return {"apiVersion":"v1","kind":"List","items":result}

if __name__=="__main__":
    print(json.dumps(generate(),indent=2))

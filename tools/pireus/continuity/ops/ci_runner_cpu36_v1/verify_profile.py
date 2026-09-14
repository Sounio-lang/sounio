"""Reject a dispatch whose actual runner differs from the declared CPU36 profile."""
import json, os, pathlib, re, subprocess

def verify(env, memory, cpu, uid, token_present):
    if env.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise ValueError("CPU36 profile requires an explicit workflow dispatch")
    if not env.get("PIREUS_RUNNER_LABEL"):
        raise ValueError("CPU36 runner label is not configured")
    if env.get("RUNNER_ENVIRONMENT") != "self-hosted":
        raise ValueError("CPU36 profile did not reach a self-hosted runner")
    base=env.get("PIREUS_BASE_SHA","")
    if not re.fullmatch("[0-9a-f]{40}",base):
        raise ValueError("CPU36 profile requires a frozen comparison base SHA")
    if memory.strip()!=str(36*1024**3):
        raise ValueError("CPU36 cgroup memory limit mismatch")
    if cpu.strip()!="400000 100000":
        raise ValueError("CPU36 cgroup CPU quota mismatch")
    if uid!=1000 or token_present:
        raise ValueError("CPU36 process identity or service-account isolation mismatch")
    return dict(profile="pireus-cpu36-v1",base_sha=base,source_sha=env["GITHUB_SHA"],
                runner_name=env.get("RUNNER_NAME"),memory_max_bytes=int(memory),
                cpu_max=cpu.strip(),job_timeout_minutes=150,qualified=False)

if __name__=="__main__":
    root=pathlib.Path("/sys/fs/cgroup")
    result=verify(os.environ,(root/"memory.max").read_text(),(root/"cpu.max").read_text(),
                  os.getuid(),pathlib.Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists())
    subprocess.run(["git","cat-file","-e",result["base_sha"]+"^{commit}"],check=True)
    print(json.dumps(result))

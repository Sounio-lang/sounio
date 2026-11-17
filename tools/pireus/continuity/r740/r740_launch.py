#!/usr/bin/env python3
"""Separate r740 launcher. No Spark host-fence, lease, or check_pair.

The process that submits work is srun. The sbatch text is rendered from the
same fields so it can be reviewed; this module never executes sbatch.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from gpu_device_key import marker_name
from phase_a import bundle as build_phase_a_bundle
from proposals import phase_a, phase_b
from run_inputs import require_run_inputs, sha256_file, stage_verified_inputs

HERE = Path(__file__).resolve().parent
PARTITION = "gpu-orangefs"
NODE = "gpuorangefs-multi-r740-proxmox"
DEFAULT_SCRATCH = "/scratch/pireus/r740"
DEFAULT_MEM = "16G"
DEFAULT_MINUTES = 30
MEM_MAX_GIB = 64
SIF = "/scratch/pireus/images/pireus-r740-amd64.sif"
STAGED = (
    "r740_observe.py", "gpu_device_key.py", "paired_timing.py", "run_inputs.py",
    "run_in_container_r740.sh",
)
GPUS = (
    {"gres": "gpu:a5000:1", "sm": 86, "arch": "sm_86", "name": "rtx-a5000"},
    {"gres": "gpu:quadro_rtx_8000:1", "sm": 75, "arch": "sm_75", "name": "quadro-rtx-8000"},
)
FORBIDDEN = (
    "check_pair", "pireus-spark-pair", "gpu:gb10", "spark-3c59", "spark-8e54",
    "--exclusive", "host-fence", "kubectl", "SLURM_OWNED", "inkling-spark",
    "ld-linux-aarch64",
)


def parse_mem(text):
    if not isinstance(text, str) or len(text) < 2 or not text.endswith("G"):
        raise ValueError("mem must look like 16G")
    number = text[:-1]
    if not number.isdigit() or (len(number) > 1 and number.startswith("0")):
        raise ValueError("mem must look like 16G")
    gib = int(number)
    if gib < 1 or gib > MEM_MAX_GIB:
        raise ValueError("mem must be from 1G to 64G")
    return text


def parse_minutes(minutes):
    if not isinstance(minutes, int) or minutes < 1 or minutes > 120:
        raise ValueError("minutes must be from 1 to 120")
    return minutes


def new_run_name(clock, pid, entropy):
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", clock):
        raise ValueError("clock")
    if not isinstance(pid, int) or pid < 1:
        raise ValueError("pid")
    if not re.fullmatch(r"[0-9a-f]{16}", entropy):
        raise ValueError("entropy")
    return clock + "-" + str(pid) + "-" + entropy


def remote_run(scratch, name):
    if not name or "/" in name or name.startswith("."):
        raise ValueError("run name")
    return str(Path(scratch) / "runs" / name)


def load_json(path):
    return json.loads(Path(path).read_text())


def image_token(lock):
    digest = lock.get("derived_sif_sha256")
    if (lock.get("status") in ("PINNED", "DECLARED") and isinstance(digest, str) and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest)):
        return digest
    return "UNPINNED"


def verify_sif_before_submit(lock):
    """Hash the SIF on this host before any srun. A declared digest is not a measurement."""
    token = image_token(lock)
    if token == "UNPINNED":
        raise RuntimeError("r740 amd64 image is not pinned; refusing to submit")
    definition = HERE / "pireus-r740-amd64.def"
    if sha256_file(definition) != lock.get("definition_sha256"):
        raise RuntimeError("r740 image definition does not match the lock; refusing to submit")
    path = Path(lock.get("sif_path") or SIF)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("r740 SIF is missing; refusing to submit")
    expected_bytes = lock.get("sif_bytes")
    if isinstance(expected_bytes, int) and path.stat().st_size != expected_bytes:
        raise RuntimeError("r740 SIF size does not match the lock; refusing to submit")
    if sha256_file(path) != token:
        raise RuntimeError("r740 SIF sha256 does not match the lock; refusing to submit")
    return token


def rule_sha256(source=None):
    path = Path(source) if source else HERE / "arch_gain_decision.sio"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_rule_frozen(lock, source=None):
    if lock.get("schema") != "pireus-arch-gain-rule-v1":
        raise ValueError("gain rule schema")
    if lock.get("source_sha256") != rule_sha256(source):
        raise ValueError("per-architecture gain rule hash drifted")
    if lock.get("r740_observation_files") != 0 or lock.get("frozen_before_r740_observations") is not True:
        raise ValueError("gain rule was not frozen before r740 observations")
    thresholds = lock.get("thresholds") or {}
    if thresholds.get("median_gain_ppm") != 50000 or thresholds.get("ci95_low_ppm_exclusive") != 0:
        raise ValueError("gain thresholds drifted")
    boot = lock.get("bootstrap") or {}
    if (boot.get("samples"), boot.get("seed"), boot.get("multiplier"), boot.get("modulus"),
            boot.get("low_index"), boot.get("high_index")) != (4000, 20260906, 48271, 2147483647, 99, 3899):
        raise ValueError("bootstrap drifted")


def bash_script(remote, phase, sm, token):
    container = remote + "/bin/run_in_container_r740.sh"
    observe = remote + "/bin/r740_observe.py"
    return "\n".join([
        "set -euo pipefail",
        "export PIREUS_R740_SIF_SHA256=" + token,
        'echo "' + token + "  " + SIF + '" | sha256sum -c -',
        "exec " + container + " python3 " + observe + " --phase " + phase
        + " --run " + remote + " --expect-sm " + str(sm),
        "",
    ])


def srun_argv(step):
    argv = [
        "srun", "-p", PARTITION, "-w", NODE,
        "--nodes=1", "--ntasks=1", "--ntasks-per-node=1", "-c8",
        "--gres=" + step["gres"], "--mem=" + step["mem"],
        "--propagate=NONE", "--kill-on-bad-exit=1",
        "--time=" + str(step["minutes"]), "--job-name=" + step["job_name"],
        "bash", "-c", step["script"],
    ]
    blob = "\n".join(argv)
    for token in FORBIDDEN:
        if token in blob:
            raise ValueError("forbidden launcher token: " + token)
    if blob.count("--gres=") != 1:
        raise ValueError("a step must request exactly one gres")
    if "--mem=" not in blob:
        raise ValueError("mem is required")
    return argv


def sbatch_text(step):
    return "\n".join([
        "#!/bin/bash",
        "#SBATCH -p " + PARTITION,
        "#SBATCH -w " + NODE,
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH -c 8",
        "#SBATCH --gres=" + step["gres"],
        "#SBATCH --mem=" + step["mem"],
        "#SBATCH --propagate=NONE",
        "#SBATCH --kill-on-bad-exit=1",
        "#SBATCH --time=" + str(step["minutes"]),
        "#SBATCH --job-name=" + step["job_name"],
        "#SBATCH --no-requeue",
        "# submitted: false. The launcher executes srun, not sbatch.",
        step["script"].rstrip("\n"),
        "",
    ])


def build_plan(scratch, name, mem=DEFAULT_MEM, minutes=DEFAULT_MINUTES, image_token_value="UNPINNED"):
    mem = parse_mem(mem)
    minutes = parse_minutes(minutes)
    remote = remote_run(scratch, name)
    steps = []
    for phase in ("probe", "timing"):
        for gpu in GPUS:
            step = {
                "name": phase + "-sm" + str(gpu["sm"]),
                "phase": phase,
                "gres": gpu["gres"],
                "sm": gpu["sm"],
                "arch": gpu["arch"],
                "mem": mem,
                "minutes": minutes,
                "job_name": "pireus-r740-" + phase + "-sm" + str(gpu["sm"]),
                "marker": remote + "/" + marker_name(phase, gpu["sm"]),
                "script": bash_script(remote, phase, gpu["sm"], image_token_value),
            }
            step["argv"] = srun_argv(step)
            step["sbatch_review"] = sbatch_text(step)
            step["sbatch_submitted"] = False
            steps.append(step)
    return {
        "schema": 1,
        "partition": PARTITION,
        "node": NODE,
        "scratch": str(scratch),
        "remote_run": remote,
        "image_token": image_token_value,
        "submits": "srun",
        "sbatch_submitted": False,
        "spark_lease": False,
        "check_pair": False,
        "host_fence": False,
        "phase_a_proposals": len(phase_a()),
        "phase_b_proposals": len(phase_b()),
        "steps": steps,
    }


def stage_into(root):
    bindir = Path(root) / "bin"
    bindir.mkdir(exist_ok=False)
    for name in STAGED:
        target = bindir / name
        target.write_bytes((HERE / name).read_bytes())
        if name.endswith(".sh"):
            target.chmod(0o755)
    probe = HERE.parent / "gpu_probe.py"
    (bindir / "gpu_probe.py").write_bytes(probe.read_bytes())
    return bindir


def allocate_run(scratch, name):
    if Path(scratch).is_symlink():
        raise ValueError("scratch root is a symlink")
    path = Path(remote_run(scratch, name))
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_steps(root, steps, runner):
    for step in steps:
        state = Path(root) / "stages" / (step["name"] + ".json")
        if state.exists():
            prior = json.loads(state.read_text())
            if prior.get("status") in ("failed", "started", "ok"):
                raise RuntimeError("refusing to resubmit stage " + step["name"])
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"status": "started", "name": step["name"]}) + "\n")
        try:
            rc = runner(step["argv"])
        except Exception:
            state.write_text(json.dumps({"status": "failed", "name": step["name"]}) + "\n")
            raise
        marker = Path(step["marker"])
        if rc != 0 or not marker.is_file():
            state.write_text(json.dumps({"status": "failed", "name": step["name"], "returncode": rc}) + "\n")
            raise RuntimeError("stage failed; not resubmitting " + step["name"])
        state.write_text(json.dumps({"status": "ok", "name": step["name"], "returncode": 0}) + "\n")
    return root


STEPS = ("probe-sm86", "probe-sm75", "timing-sm86", "timing-sm75")


def prepare_run(scratch, name, inputs):
    """Allocate a fresh run directory and stage both architectures. No srun."""
    inputs = Path(inputs)
    for arch in ("sm86", "sm75"):
        require_run_inputs(inputs / arch)
    root = allocate_run(scratch, name)
    stage_into(root)
    for arch in ("sm86", "sm75"):
        (root / arch).mkdir()
        stage_verified_inputs(root / arch, inputs / arch)
    origin = HERE / "phase-a" / "ORIGIN.json"
    if origin.is_file():
        (root / "phase-a-origin.json").write_bytes(origin.read_bytes())
    return root


def assert_probe_succeeded(root, step_name):
    if not step_name.startswith("timing-"):
        return
    probe = "probe-" + step_name.split("-", 1)[1]
    state = Path(root) / "stages" / (probe + ".json")
    if not state.is_file():
        raise RuntimeError("timing requires a successful probe of the same GPU in this run directory")
    prior = json.loads(state.read_text())
    if prior.get("status") != "ok":
        raise RuntimeError("timing refused because " + probe + " did not succeed")
    sm = int(step_name.rsplit("sm", 1)[1])
    marker = Path(root) / marker_name("probe", sm)
    if not marker.is_file():
        raise RuntimeError("timing requires the probe marker for the same GPU")


def run_step(scratch, name, step_name, runner, mem=DEFAULT_MEM, minutes=DEFAULT_MINUTES,
             image_lock=None, rule_lock=None):
    """One srun against an already prepared run directory. The SIF is hashed first."""
    if step_name not in STEPS:
        raise ValueError("unknown step")
    image_lock = image_lock if image_lock is not None else load_json(HERE / "r740-runtime-lock.json")
    rule_lock = rule_lock if rule_lock is not None else load_json(HERE / "arch_gain_rule.lock.json")
    assert_rule_frozen(rule_lock)
    token = verify_sif_before_submit(image_lock)
    root = Path(remote_run(scratch, name))
    if not root.is_dir():
        raise RuntimeError("run directory is not prepared")
    assert_probe_succeeded(root, step_name)
    plan = build_plan(scratch, name, mem, minutes, token)
    step = next(item for item in plan["steps"] if item["name"] == step_name)
    return run_steps(root, [step], runner)


def execute(scratch, name, runner, mem=DEFAULT_MEM, minutes=DEFAULT_MINUTES,
            image_lock=None, rule_lock=None, inputs=None):
    image_lock = image_lock if image_lock is not None else load_json(HERE / "r740-runtime-lock.json")
    rule_lock = rule_lock if rule_lock is not None else load_json(HERE / "arch_gain_rule.lock.json")
    assert_rule_frozen(rule_lock)
    token = verify_sif_before_submit(image_lock)
    if inputs is None:
        raise RuntimeError("r740 run inputs are missing; refusing to submit")
    require_run_inputs(inputs)
    plan = build_plan(scratch, name, mem, minutes, token)
    root = allocate_run(scratch, name)
    stage_into(root)
    stage_verified_inputs(root, inputs)
    return run_steps(root, plan["steps"], runner)


def materialize_argv(engine, context, proposal, sm):
    mode = {121: "ptx", 86: "ptx-sm86", 75: "ptx-sm75"}[sm]
    return [str(engine), str(context), str(proposal), mode]


def descriptive_report(per_architecture):
    if not per_architecture:
        raise ValueError("empty")
    sms = [row["sm"] for row in per_architecture]
    if len(sms) != len(set(sms)):
        raise ValueError("one decision per architecture")
    return {
        "schema": 1,
        "kind": "cross-architecture-descriptive",
        "gain_counts": False,
        "aggregate_gain_gate": None,
        "per_architecture": per_architecture,
    }


def subprocess_runner(argv):
    completed = subprocess.run(argv, check=False)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "bundle", "prepare", "run"))
    parser.add_argument("--mem", default=DEFAULT_MEM)
    parser.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    parser.add_argument("--scratch", default=DEFAULT_SCRATCH)
    parser.add_argument("--name")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--step", choices=STEPS)
    args = parser.parse_args()
    lock = load_json(HERE / "r740-runtime-lock.json")
    rule = load_json(HERE / "arch_gain_rule.lock.json")
    assert_rule_frozen(rule)
    if args.command == "bundle":
        if args.out is None:
            parser.error("bundle requires --out")
        build_phase_a_bundle(args.out, args.engine)
        print(json.dumps({"bundle": str(args.out), "engine": args.engine is not None}))
        return
    if args.name is None:
        parser.error(args.command + " requires --name")
    if args.command == "prepare":
        if args.inputs is None:
            parser.error("prepare requires --inputs")
        root = prepare_run(args.scratch, args.name, args.inputs)
        print(json.dumps({"prepared": str(root), "srun": False}))
        return
    if args.command == "run" and args.step:
        run_step(args.scratch, args.name, args.step, subprocess_runner, args.mem, args.minutes)
        return
    if args.command == "run":
        execute(args.scratch, args.name, subprocess_runner, args.mem, args.minutes, inputs=args.inputs)
        return
    plan = build_plan(args.scratch, args.name, args.mem, args.minutes, image_token(lock))
    plan["image_status"] = lock["status"]
    plan["will_submit"] = False
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()

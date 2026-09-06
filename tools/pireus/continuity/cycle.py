#!/usr/bin/env python3
"""Persisted Pireus proposal transport. All semantic decisions come from Sounio."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent
MODEL = "thinkingmachines/Inkling-Small-NVFP4"
REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

def atomic(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("immutable artifact differs: " + str(path))
        return
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)

def event(root, stage, artifact):
    data = artifact.read_bytes()
    row = dict(stage=stage, artifact=artifact.relative_to(root).as_posix(),
               sha256=digest(data))
    journal = root / "journal.jsonl"
    existing = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    if any(x == row for x in existing):
        return
    if any(x["artifact"] == row["artifact"] for x in existing):
        raise ValueError("journal artifact changed")
    with journal.open("ab") as out:
        out.write(encoded(row))
        out.flush()
        os.fsync(out.fileno())

def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["dependencies"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("frozen dependency changed: " + name)
    if digest((HERE / "admission.sio").read_bytes()) != manifest["admission_source_sha256"]:
        raise ValueError("admission source changed: prepare a new run")
    journal = root / "journal.jsonl"
    seen = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            if row["artifact"] in seen:
                raise ValueError("duplicate journal stage")
            seen.add(row["artifact"])
            if digest((root / row["artifact"]).read_bytes()) != row["sha256"]:
                raise ValueError("journal integrity failure")
    return manifest

def prepare(args):
    root = args.run
    root.mkdir(parents=True, exist_ok=False)
    dependencies = {}
    sources = [("context.json", args.context)] + [
        ("evidence-" + str(i) + ".bin", p) for i, p in enumerate(args.evidence)]
    for name, path in sources:
        data = path.read_bytes()
        atomic(root / name, data)
        dependencies[name] = digest(data)
    manifest = dict(schema=1, condition=args.condition, budget=args.budget,
                    round=args.round, model=MODEL, revision=REVISION,
                    dependencies=dependencies,
                    engine_sha256=args.engine_sha256,
                    admission_source_sha256=digest((HERE / "admission.sio").read_bytes()),
                    semantic_authority="Sounio", promotion_threshold_percent=5,
                    benchmark_blocks_per_node=30, formal_v13_v14="OPEN")
    atomic(root / "manifest.json", encoded(manifest))
    event(root, "prepare", root / "manifest.json")
    return manifest

def proposal_template(context_hash):
    return dict(schema=1, target=701202, dimension=16, precision=64, order=1,
                fma=0, kind=1, lane_stride=1, lane_offset=0, load=1,
                layout=0, unroll=1, context=context_hash)

def generate(args, manifest):
    root = args.run
    context = (root / "context.json").read_bytes()
    base = proposal_template(digest(context))
    for index in range(manifest["budget"]):
        prefix = root / ("%03d" % index)
        proposal = prefix.with_suffix(".proposal.json")
        request = prefix.with_suffix(".request.json")
        response = prefix.with_suffix(".response.json")
        if proposal.exists():
            event(root, "generate", proposal)
            continue
        if manifest["condition"] == "deterministic":
            p = base | dict(lane_stride=1 + 2 * (index % 8),
                            lane_offset=(index // 8 + manifest["round"]) % 16,
                            load=(index // 2) % 2, layout=index % 2,
                            unroll=(1, 2, 4, 8, 16)[index % 5])
            atomic(proposal, encoded(p))
        else:
            if not args.endpoint:
                raise ValueError("endpoint is required for Inkling")
            facts = context.decode() if manifest["condition"] == "inkling-ontology" else "withheld"
            prompt = (
                "Propose one untrusted lowering plan as a JSON object only. "
                "Do not add expected results, authority, claims, or new fields. "
                "Keep schema, target, dimension, precision, order, fma, kind and context unchanged. "
                "You may vary lane_stride in odd integers 1..15; lane_offset 0..15; "
                "load 0 (direct) or 1 (shuffle); layout 0 (AoS) or 1 (SoA); "
                "unroll in [1,2,4,8,16]. Output k accumulates ascending right operand j "
                "with left i=k XOR j, separate f64 multiply/add, no reassociation. "
                "Lane mapping k=(lane*lane_stride+lane_offset)%16. "
                "This is proposal %d in round %d. Frozen hardware facts: %s. Template: %s"
                % (index, manifest["round"], facts, json.dumps(base)))
            body = dict(model=args.served_model, messages=[dict(role="user", content=prompt)],
                        max_tokens=4096, temperature=0.7, seed=manifest["round"] * 1000 + index)
            if not response.exists():
                if request.exists():
                    raise RuntimeError("ambiguous interrupted generation; original request retained; "
                                       "do not automatically issue it twice: " + str(request))
                atomic(request, encoded(body))
                event(root, "request", request)
                req = urllib.request.Request(args.endpoint.rstrip("/") + "/v1/chat/completions",
                                             data=encoded(body), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=1800) as reply:
                    raw_response = reply.read()
                atomic(response, raw_response)
            event(root, "response", response)
            raw_content = json.loads(response.read_text())["choices"][0]["message"]["content"]
            if not isinstance(raw_content, str):
                raise ValueError("missing textual proposal; original response retained")
            # No repair, JSON normalization, code generation, or semantic filtering.
            atomic(proposal, raw_content.encode())
        event(root, "generate", proposal)

def validate(args, manifest):
    root = args.run
    if not args.engine:
        raise ValueError("actual native Sounio admission executable required")
    if digest(args.engine.read_bytes()) != manifest["engine_sha256"]:
        raise ValueError("admission executable identity changed")
    for index in range(manifest["budget"]):
        prefix = root / ("%03d" % index)
        proposal = prefix.with_suffix(".proposal.json")
        receipt = prefix.with_suffix(".receipt.json")
        if receipt.exists():
            event(root, "validate", receipt)
            continue
        if not proposal.exists():
            raise ValueError("generation incomplete")
        run = subprocess.run([str(args.engine.resolve()), str((root / "context.json").resolve()),
                              str(proposal.resolve())], capture_output=True, timeout=60)
        atomic(prefix.with_suffix(".admission.stdout"), run.stdout)
        atomic(prefix.with_suffix(".admission.stderr"), run.stderr)
        value = json.loads(run.stdout)
        if value.get("authority") != "Sounio" or run.returncode not in (0, 1):
            raise ValueError("invalid executable result; preserve diagnostics")
        if (run.returncode == 0) != (value.get("decision") == "ADMIT"):
            raise ValueError("exit code and decision disagree")
        atomic(receipt, run.stdout)
        event(root, "validate", receipt)

def report(root, manifest):
    receipts = [json.loads(p.read_text()) for p in sorted(root.glob("*.receipt.json"))]
    admitted = [x for x in receipts if x["decision"] == "ADMIT"]
    return dict(schema=1, condition=manifest["condition"], budget=manifest["budget"],
                generated=len(list(root.glob("*.proposal.json"))), validated=len(receipts),
                admitted=len(admitted), unique_plans=len({x["plan_id"] for x in admitted}),
                refused=len(receipts)-len(admitted), hardware_benchmarked=0,
                performance_gain="UNMEASURED", claim_ready=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "generate", "validate", "report", "resume"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--context", type=Path)
    ap.add_argument("--evidence", type=Path, nargs="*", default=[])
    ap.add_argument("--condition", choices=["deterministic", "inkling-no-ontology", "inkling-ontology"],
                    default="deterministic")
    ap.add_argument("--budget", type=int, choices=[8,32], default=8)
    ap.add_argument("--round", type=int, choices=[0,1,2], default=0)
    ap.add_argument("--engine-sha256")
    ap.add_argument("--engine", type=Path)
    ap.add_argument("--endpoint")
    ap.add_argument("--served-model", default=MODEL)
    args = ap.parse_args()
    if args.command == "prepare":
        if not args.context or not args.evidence or not args.engine_sha256:
            ap.error("prepare requires context, evidence and engine-sha256")
        if len(args.engine_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.engine_sha256):
            ap.error("invalid engine SHA256")
        prepare(args)
        return
    with (args.run / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify(args.run)
        if args.command == "generate":generate(args, manifest)
        elif args.command == "validate":validate(args, manifest)
        # resume verifies custody and reports remaining stages; it never silently repeats an HTTP call.
        print(json.dumps(report(args.run, manifest), indent=2))
if __name__ == "__main__":
    main()

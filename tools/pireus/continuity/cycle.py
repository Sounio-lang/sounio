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
import sys
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
    for name, expected in manifest["code_dependencies"].items():
        if digest((HERE / name).read_bytes()) != expected:
            raise ValueError("code dependency changed: " + name)
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
    if args.context_engine:
        generated = subprocess.run([str(args.context_engine.resolve())],capture_output=True,check=True,timeout=60)
        atomic(root / "context.json", generated.stdout)
        dependencies["context.json"] = digest(generated.stdout)
        sources = []
    else:
        sources = [("context.json", args.context)]
    sources += [
        ("evidence-" + str(i) + ".bin", p) for i, p in enumerate(args.evidence)]
    for name, path in sources:
        data = path.read_bytes()
        atomic(root / name, data)
        dependencies[name] = digest(data)
    manifest = dict(schema=1, condition=args.condition, kind=getattr(args, "kind", 1), budget=args.budget,
                    round=args.round, model=MODEL, revision=REVISION,
                    transport=getattr(args, "transport", "openai-chat"),
                    dependencies=dependencies,
                    engine_sha256=args.engine_sha256,
                    admission_source_sha256=digest((HERE / "admission.sio").read_bytes()),
                    code_dependencies={p.relative_to(HERE).as_posix():digest(p.read_bytes()) for p in [*sorted(HERE.glob("*.py")),*sorted(HERE.glob("*.sio")),*sorted((HERE/"runtime").glob("*.py")),*sorted((HERE/"runtime").glob("*.sio")),*sorted((HERE/"runtime").glob("*.sh")),*sorted((HERE/"runtime").glob("*.json"))]},
                    deduplicate_material=getattr(args,"deduplicate_material",False),
                    context_origin="Sounio ontology producer" if args.context_engine else "supplied frozen context",
                    context_engine_sha256=digest(args.context_engine.read_bytes()) if args.context_engine else None,
                    semantic_authority="Sounio", promotion_threshold_percent=5,
                    benchmark_blocks_per_node=30, formal_v13_v14="OPEN")
    atomic(root / "manifest.json", encoded(manifest))
    event(root, "prepare", root / "manifest.json")
    return manifest

def proposal_template(context_hash, kind=1):
    # kind=1 is a lowering of the fixed Cayley-Dickson product. kind=2 names a
    # new operator through a 16-bit bilinear phase code and still carries a
    # lane plan, because a proposed operator still has to be lowered. The only
    # added key is `phase`; every other field keeps its kind=1 meaning.
    base = dict(schema=1, target=701202, dimension=16, precision=64, order=1,
                fma=0, kind=kind, lane_stride=1, lane_offset=0, load=1,
                layout=0, unroll=1, context=context_hash)
    if kind == 2:
        base["phase"] = 0
    return base

# These are plain constants concatenated onto the format string below, so they
# are NOT consumed by the % operator: a literal percent stays a single percent
# here, unlike the inline string this replaced.
LOWERING_PROMPT = (
    "Propose one untrusted lowering plan as a JSON object only. "
    "Do not add expected results, authority, claims, or new fields. "
    "Keep schema, target, dimension, precision, order, fma, kind and context unchanged. "
    "You may vary lane_stride in odd integers 1..15; lane_offset 0..15; "
    "load 0 (direct) or 1 (shuffle); layout 0 (AoS) or 1 (SoA); "
    "unroll in [1,2,4,8,16]. Output k accumulates ascending right operand j "
    "with left i=k XOR j, separate f64 multiply/add, no reassociation. "
    "Lane mapping k=(lane*lane_stride+lane_offset)%16. ")

# The kind=2 prompt asks for an OPERATOR, not a lowering of one. It states the
# whole admissible space and the single number that selects a point in it,
# because the engine accepts nothing else from the proposer: no tensor, no
# digest, no claim. Everything a proposal asserts about its operator is
# recomputed by Sounio and by eisa_h.pireus_operator_admission.v1, and a
# proposal that misdescribes its own algebra is refused even when the algebra
# is sound.
OPERATOR_PROMPT = (
    "Propose one untrusted OPERATOR as a JSON object only. "
    "Do not add expected results, authority, claims, tensors, hashes, or new fields. "
    "Keep schema, target, dimension, precision, order, fma, kind and context unchanged. "
    "The operator is a product on 16 basis elements graded by XOR: "
    "e_i * e_j = sigma(i,j) e_(i XOR j). "
    "sigma(i,j) = cd_sigma(i,j) * (-1)^(i^T B j), where cd_sigma is the "
    "Cayley-Dickson sedenion sign and B is a 4x4 matrix over F2 that you choose. "
    "You choose B by giving `phase`, an integer 0..65535 whose bit (4*i + j) is B[i][j]. "
    "phase 0 reproduces the sedenions exactly and will be refused as not novel. "
    "Two phase codes that differ by a diagonal sign rescaling of the basis give "
    "the same operator, so only the symmetrised form matters: there are 1024 "
    "distinct classes, falling into 32 classes under the admitted change-of-basis "
    "actions, and the lowering grammar used so far has only ever reached 4 of them. "
    "Aim outside those 4. "
    "You may also vary lane_stride in odd integers 1..15; lane_offset 0..15; "
    "load 0 (direct) or 1 (shuffle); layout 0 (AoS) or 1 (SoA); unroll in [1,2,4,8,16], "
    "with the same evaluation order as a lowering proposal: output k accumulates "
    "ascending right operand j with left i=k XOR j, separate f64 multiply/add, "
    "no reassociation, lane mapping k=(lane*lane_stride+lane_offset)%16. ")

def request_body(manifest, context, index, served_model=MODEL):
    kind = manifest.get("kind", 1)
    base = proposal_template(digest(context), kind)
    facts = context.decode() if manifest["condition"] == "inkling-ontology" else "withheld"
    prompt = (
        (OPERATOR_PROMPT if kind == 2 else LOWERING_PROMPT) +
        "This is proposal %d in round %d. Frozen research context (declared ontology, not observed hardware facts): %s. Template: %s"
        % (index, manifest["round"], facts, json.dumps(base)))
    body = dict(model=served_model, messages=[dict(role="user", content=prompt)],
                max_tokens=4096, temperature=0.7, seed=manifest["round"] * 1000 + index)
    return body

def generate(args, manifest):
    if manifest.get("transport") in ("sglang-token-ids", "sglang-offline-token-ids"):
        raise ValueError("Use tokenized_cycle.py for the frozen token-ID transport")
    root = args.run
    context = (root / "context.json").read_bytes()
    base = proposal_template(digest(context), manifest.get("kind", 1))
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
            if manifest.get("kind", 1) == 2:
                # Fixed phase codes, not a search: enough to exercise transport
                # and admission without a Spark pair. 1128 is the operator the
                # upstream genesis run selected; the others are distinct codes
                # so one run reaches more than a single class.
                p["phase"] = (1128, 2, 8, 74, 198, 1129, 4, 16)[index % 8]
            atomic(proposal, encoded(p))
        else:
            if not args.endpoint:
                raise ValueError("endpoint is required for Inkling")
            body = request_body(manifest, context, index, args.served_model)
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

def materialize(args, manifest):
    root = args.run
    if not args.engine or digest(args.engine.read_bytes()) != manifest["engine_sha256"]:
        raise ValueError("matching Sounio admission/materialization executable required")
    for index in range(manifest["budget"]):
        prefix = root / ("%03d" % index)
        receipt = prefix.with_suffix(".receipt.json")
        if not receipt.exists():
            raise ValueError("admission incomplete")
        decision = json.loads(receipt.read_text())
        if decision["decision"] != "ADMIT":
            continue
        artifact = prefix.with_suffix(".ptx")
        if not artifact.exists():
            result = subprocess.run([str(args.engine.resolve()),str((root/"context.json").resolve()),
                                     str(prefix.with_suffix(".proposal.json").resolve()),"ptx"],
                                    capture_output=True,timeout=60)
            if result.returncode != 0:
                atomic(prefix.with_suffix(".materialization-error"),result.stdout+result.stderr)
                raise ValueError("Sounio materialization refused")
            atomic(artifact,result.stdout)
        event(root,"materialize",artifact)

def report(root, manifest):
    receipts = [json.loads(p.read_text()) for p in sorted(root.glob("*.receipt.json"))]
    admitted = [x for x in receipts if x["decision"] == "ADMIT"]
    benchmark=json.loads((root/"benchmark-report.json").read_text()) if (root/"benchmark-report.json").exists() else None
    return dict(schema=1, condition=manifest["condition"], kind=manifest.get("kind", 1), budget=manifest["budget"],
                generated=len(list(root.glob("*.proposal.json"))), validated=len(receipts),
                admitted=len(admitted), unique_plans=len({x["plan_id"] for x in admitted}),
                refused=len(receipts)-len(admitted), materialized=len(list(root.glob("*.ptx"))), hardware_benchmarked=len(benchmark["decisions"]) if benchmark else 0,
                gain_eligible=benchmark["gain_eligible"] if benchmark else 0,
                performance_gain="MEASURED" if benchmark else "UNMEASURED", claim_ready=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "generate", "validate", "materialize", "benchmark", "report", "resume"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--context", type=Path)
    ap.add_argument("--context-engine", type=Path)
    ap.add_argument("--evidence", type=Path, nargs="*", default=[])
    ap.add_argument("--condition", choices=["deterministic", "inkling-no-ontology", "inkling-ontology"],
                    default="deterministic")
    ap.add_argument("--kind", type=int, choices=[1, 2], default=1,
                    help="1 = lowering of the fixed product (default, unchanged behaviour); "
                         "2 = operator genesis, adds the 16-bit `phase` key")
    ap.add_argument("--budget", type=int, choices=[8,32], default=8)
    ap.add_argument("--deduplicate-material", action="store_true")
    ap.add_argument("--round", type=int, choices=[0,1,2], default=0)
    ap.add_argument("--engine-sha256")
    ap.add_argument("--engine", type=Path)
    ap.add_argument("--fixture-engine", type=Path)
    ap.add_argument("--parity-engine", type=Path)
    ap.add_argument("--gain-engine", type=Path)
    ap.add_argument("--endpoint")
    ap.add_argument("--served-model", default=MODEL)
    ap.add_argument("--transport", choices=["openai-chat", "sglang-token-ids", "sglang-offline-token-ids"], default="openai-chat")
    args = ap.parse_args()
    if args.command == "prepare":
        if bool(args.context) == bool(args.context_engine) or not args.evidence or not args.engine_sha256:
            ap.error("prepare requires exactly one context/context-engine, evidence and engine-sha256")
        if len(args.engine_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.engine_sha256):
            ap.error("invalid engine SHA256")
        prepare(args)
        return
    if args.command == "benchmark":
        if not all([args.engine,args.fixture_engine,args.parity_engine,args.gain_engine]):
            ap.error("benchmark requires admission, fixture, parity and gain executables")
        subprocess.run([sys.executable,str(HERE/"benchmark_pair.py"),"--run",str(args.run),
                        "--engine",str(args.engine),"--fixture-engine",str(args.fixture_engine),
                        "--parity-engine",str(args.parity_engine),"--gain-engine",str(args.gain_engine)],check=True)
        return
    with (args.run / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify(args.run)
        if args.command == "generate":generate(args, manifest)
        elif args.command == "validate":validate(args, manifest)
        elif args.command == "materialize":materialize(args, manifest)
        # resume verifies custody and reports remaining stages; it never silently repeats an HTTP call.
        print(json.dumps(report(args.run, manifest), indent=2))
if __name__ == "__main__":
    main()

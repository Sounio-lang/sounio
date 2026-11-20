"""Bundle a directory of *.proposal.json (for example ../inbox) for the r740 pilot.

Each proposal passes through the native Sounio admission executable against the
frozen Phase A context. ADMIT proposals are materialized to sm_86 and sm_75 PTX
by the same executable; REFUSE proposals are recorded and dropped. kind=1 and
kind=2 go through the same path; a kind=2 row also records its phase, because
its bit-exact reference is that operator's (operator_reference.py), not the
Cayley-Dickson fixture output. The two
controls stay the vendored Phase A bytes, and the engine must first reproduce
every vendored Phase A PTX (phase_a.check_engine). Nothing here calls Slurm.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

from phase_a import CONTROLS, PHASE, _entry, check_engine
from run_inputs import FIXTURES, require_run_inputs

ID = re.compile(r"[0-9A-Za-z][0-9A-Za-z_-]{0,63}")
ARCHES = (("sm86", "ptx-sm86"), ("sm75", "ptx-sm75"))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def proposal_files(directory):
    rows = []
    for path in sorted(Path(directory).glob("*.proposal.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("proposal is not a regular file: " + path.name)
        name = path.name[: -len(".proposal.json")]
        if not ID.fullmatch(name) or name in CONTROLS:
            raise ValueError("proposal id: " + name)
        rows.append((name, path))
    if not rows:
        raise ValueError("no *.proposal.json in " + str(directory))
    return rows


def admit(engine, context, proposal):
    """Run the admission executable as cycle.py validate does; return (decision, stdout, stderr, rc)."""
    run = subprocess.run([str(engine), str(context), str(proposal)], capture_output=True, timeout=60)
    value = json.loads(run.stdout)
    if value.get("authority") != "Sounio" or run.returncode not in (0, 1):
        raise ValueError("invalid admission result for " + Path(proposal).name)
    if (run.returncode == 0) != (value.get("decision") == "ADMIT"):
        raise ValueError("exit code and decision disagree for " + Path(proposal).name)
    return value["decision"], run.stdout, run.stderr, run.returncode


def materialize(engine, context, proposal, mode):
    run = subprocess.run([str(engine), str(context), str(proposal), mode], capture_output=True, timeout=60)
    if run.returncode != 0:
        raise RuntimeError(mode + " refused for an admitted proposal: " + Path(proposal).name)
    return run.stdout


def bundle(out, engine, proposals_dir):
    if engine is None:
        raise ValueError("--proposals-dir requires --engine (the native admission executable)")
    engine = Path(engine).resolve()
    origin = check_engine(engine)
    context = PHASE / "context.json"
    rows = proposal_files(proposals_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    receipts_dir = out / "admission"
    receipts_dir.mkdir()
    seen = {arch: {} for arch, _ in ARCHES}
    for arch, _ in ARCHES:
        for name in CONTROLS:
            seen[arch][origin["ptx"][arch][name]] = "control:" + name
        for name, digest in origin["ptx"][arch].items():
            seen[arch].setdefault(digest, "phase-a:" + name)
    admitted = []
    ledger = []
    for name, path in rows:
        payload = path.read_bytes()
        decision, stdout, stderr, rc = admit(engine, context, path)
        (receipts_dir / (name + ".receipt.json")).write_bytes(stdout)
        (receipts_dir / (name + ".admission.stderr")).write_bytes(stderr)
        body = json.loads(payload)
        row = {"id": name, "proposal_sha256": _sha(payload), "decision": decision, "returncode": rc,
               "kind": body.get("kind")}
        if body.get("kind") == 2:
            # kind=2 is a different operator; its parity reference is operator_reference.py at this phase.
            row["phase"] = body.get("phase")
        if "sign_mode" in body:
            # Same operator, different lowering; the parity reference does not change.
            row["sign_mode"] = body["sign_mode"]
        if "pack" in body:
            # Two vectors per 16-lane block; same operator, same parity reference.
            row["pack"] = body["pack"]
        if "vector_load" in body:
            # ld.global.v2.f64 staging under load=2; same operator, same parity reference.
            row["vector_load"] = body["vector_load"]
        if "block_threads" in body:
            # r740-only wide block (r740_wide_launch.py); same operator, same parity reference.
            row["block_threads"] = body["block_threads"]
        if decision != "ADMIT":
            row["reason"] = json.loads(stdout).get("reason")
        if decision == "ADMIT":
            ptx = {arch: materialize(engine, context, path, mode) for arch, mode in ARCHES}
            row["ptx_sha256"] = {arch: _sha(data) for arch, data in ptx.items()}
            dup = sorted({seen[arch].get(row["ptx_sha256"][arch]) for arch, _ in ARCHES} - {None})
            if dup:
                row["duplicate_of"] = dup
            else:
                for arch, _ in ARCHES:
                    seen[arch][row["ptx_sha256"][arch]] = name
                admitted.append((name, body["layout"], ptx, body.get("block_threads", 0)))
        ledger.append(row)
    if not admitted:
        raise RuntimeError("no admitted, materially new proposal to measure")
    fixtures = (PHASE / "numeric-fixtures.jsonl").read_bytes()
    fixture_sha = _sha(fixtures)
    for arch, _ in ARCHES:
        arch_dir = out / arch
        arch_dir.mkdir()
        (arch_dir / FIXTURES).write_bytes(fixtures)
        controls = []
        for name in CONTROLS:
            data = (PHASE / "ptx" / arch / (name + ".ptx")).read_bytes()
            rel = "controls/" + name + ".ptx"
            (arch_dir / "controls").mkdir(exist_ok=True)
            (arch_dir / rel).write_bytes(data)
            layout = json.loads((PHASE / "controls" / (name + ".proposal.json")).read_text())["layout"]
            controls.append(_entry(name, rel, data, layout))
        candidates = []
        for name, layout, ptx, threads in admitted:
            rel = name + ".ptx"
            (arch_dir / rel).write_bytes(ptx[arch])
            entry = _entry(name, rel, ptx[arch], layout)
            if threads not in (0, 16):
                # Absent for 16 threads, so a historical manifest keeps its bytes.
                entry["block_threads"] = threads
            candidates.append(entry)
        gpu = {"schema": 1, "fixtures_sha256": fixture_sha, "candidates": controls + candidates}
        bench = {
            "method": "paired-median-ppm-bootstrap4000-v1",
            "blocks": 30, "vectors": 16384, "replays": 32,
            "fixtures_sha256": fixture_sha,
            "controls": controls, "candidates": candidates,
        }
        (arch_dir / "gpu-manifest.json").write_text(json.dumps(gpu, sort_keys=True) + "\n")
        (arch_dir / "benchmark-manifest.json").write_text(json.dumps(bench, sort_keys=True) + "\n")
        require_run_inputs(arch_dir)
    summary = {
        "schema": 1,
        "kind": "pireus-r740-proposals-bundle",
        "proposals_dir": str(Path(proposals_dir).resolve()),
        "context_sha256": _sha(context.read_bytes()),
        "engine_sha256": _sha(engine.read_bytes()),
        "engine_reproduces_phase_a": True,
        "admitted": [name for name, _, _, _ in admitted],
        "proposals": ledger,
    }
    (out / "PROPOSALS.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    (out / "ORIGIN.json").write_bytes((PHASE / "ORIGIN.json").read_bytes())
    return out

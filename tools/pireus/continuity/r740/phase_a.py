"""Phase A bundle from the imported 2026-09-07 baseline. No new proposals."""
import hashlib
import json
import subprocess
from pathlib import Path

from run_inputs import FIXTURES, require_run_inputs

HERE = Path(__file__).resolve().parent
PHASE = HERE / "phase-a"
ORIGIN_COMMIT = "1a82c417e55c687748ee396d91457ec4b3fe15b4"
PROPOSALS = tuple(f"{i:03d}" for i in range(8))
CONTROLS = ("direct", "shuffle")


def load_origin():
    return json.loads((PHASE / "ORIGIN.json").read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_imported_sources(origin=None):
    origin = origin or load_origin()
    if origin.get("origin_commit") != ORIGIN_COMMIT or origin.get("budget") != 8 or origin.get("round") != 0:
        raise ValueError("phase A origin is not the 20260907 baseline")
    for rel, meta in origin["sources"].items():
        if _sha(PHASE / rel) != meta["sha256"]:
            raise ValueError("imported source drifted: " + rel)
    for arch, rows in origin["ptx"].items():
        for name, digest in rows.items():
            if _sha(PHASE / "ptx" / arch / f"{name}.ptx") != digest:
                raise ValueError("vendored ptx drifted: " + arch + "/" + name)
    return origin


def _items():
    rows = [(name, PHASE / "proposals" / f"{name}.proposal.json") for name in PROPOSALS]
    rows += [(name, PHASE / "controls" / f"{name}.proposal.json") for name in CONTROLS]
    return rows


def check_engine(engine):
    """Regenerate PTX and refuse if it differs from the imported baseline or the vendored files."""
    origin = verify_imported_sources()
    context = str(PHASE / "context.json")
    for name, proposal in _items():
        for mode, arch, pin in (
            ("ptx", None, origin["sm121_ptx_sha256"][name]),
            ("ptx-sm86", "sm86", origin["ptx"]["sm86"][name]),
            ("ptx-sm75", "sm75", origin["ptx"]["sm75"][name]),
        ):
            result = subprocess.run([str(engine), context, str(proposal), mode], capture_output=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(mode + " " + name + " refused")
            if hashlib.sha256(result.stdout).hexdigest() != pin:
                raise RuntimeError(mode + " " + name + " drifted")
            if arch is not None and result.stdout != (PHASE / "ptx" / arch / f"{name}.ptx").read_bytes():
                raise RuntimeError(mode + " " + name + " vendored bytes differ")
    return origin


def _entry(name, rel, payload, layout):
    return {"id": name, "layout": layout, "ptx": rel, "ptx_sha256": hashlib.sha256(payload).hexdigest()}


def _write_arch(out, arch):
    arch_dir = Path(out) / arch
    arch_dir.mkdir()
    fixtures = (PHASE / "numeric-fixtures.jsonl").read_bytes()
    (arch_dir / FIXTURES).write_bytes(fixtures)
    fixture_sha = hashlib.sha256(fixtures).hexdigest()
    controls = []
    for name in CONTROLS:
        payload = (PHASE / "ptx" / arch / f"{name}.ptx").read_bytes()
        rel = "controls/" + name + ".ptx"
        dest = arch_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        layout = json.loads((PHASE / "controls" / f"{name}.proposal.json").read_text())["layout"]
        controls.append(_entry(name, rel, payload, layout))
    candidates = []
    for name in PROPOSALS:
        payload = (PHASE / "ptx" / arch / f"{name}.ptx").read_bytes()
        rel = name + ".ptx"
        (arch_dir / rel).write_bytes(payload)
        layout = json.loads((PHASE / "proposals" / f"{name}.proposal.json").read_text())["layout"]
        candidates.append(_entry(name, rel, payload, layout))
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
    return fixture_sha


def bundle(out, engine=None):
    verify_imported_sources()
    if engine is not None:
        check_engine(engine)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for arch in ("sm86", "sm75"):
        _write_arch(out, arch)
    (out / "ORIGIN.json").write_bytes((PHASE / "ORIGIN.json").read_bytes())
    return out

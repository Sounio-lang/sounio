"""Manifest, fixture and PTX contract read by the r740 observer and the launcher.

The fresh run directory is empty until the launcher copies a bundle that has
already matched these digests. Probe and timing call the same checks before
they touch CUDA.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath

FIXTURES = "numeric-fixtures.jsonl"
GPU_MANIFEST = "gpu-manifest.json"
BENCH_MANIFEST = "benchmark-manifest.json"
RESERVED = {"bin", "stages", "markers"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("unsafe input path")
    rel = PurePosixPath(name)
    if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError("unsafe input path")
    if rel.parts[0] in RESERVED:
        raise ValueError("input path collides with the launcher")
    return rel


def _contained(root, path):
    resolved_root = Path(root).resolve()
    resolved = Path(path).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("input escapes the run directory")
    return resolved


def _hex_digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(label + " digest")
    return value


def _load_json(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing " + path.name)
    return json.loads(path.read_text())


def _rows(manifest, key, allow_empty):
    rows = manifest.get(key)
    if not isinstance(rows, list) or (not rows and not allow_empty):
        raise ValueError(key)
    return rows


def _check_ptx(root, rows, seen):
    verified = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("ptx row")
        rel = safe_relative(row.get("ptx"))
        digest = _hex_digest(row.get("ptx_sha256"), "ptx")
        path = _contained(root, Path(root).joinpath(*rel.parts))
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing ptx " + rel.as_posix())
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError("ptx digest " + rel.as_posix())
        previous = seen.get(rel.as_posix())
        if previous is not None and previous != digest:
            raise ValueError("ptx digest conflict " + rel.as_posix())
        seen[rel.as_posix()] = digest
        verified.append(rel.as_posix())
    return verified


def _fixtures(root, expected):
    path = Path(root) / FIXTURES
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing numeric-fixtures.jsonl")
    _contained(root, path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _hex_digest(expected, "fixtures"):
        raise ValueError("fixture digest")
    return raw, digest


def data_root(run, expect_sm):
    nested = Path(run) / ("sm%d" % expect_sm)
    if (nested / GPU_MANIFEST).is_file():
        return nested
    return Path(run)


def require_probe_inputs(root):
    root = Path(root)
    manifest = _load_json(root / GPU_MANIFEST)
    fixtures, digest = _fixtures(root, manifest.get("fixtures_sha256"))
    seen = {}
    _check_ptx(root, _rows(manifest, "candidates", allow_empty=False), seen)
    return manifest, fixtures, digest, seen


def require_timing_inputs(root):
    root = Path(root)
    manifest = _load_json(root / BENCH_MANIFEST)
    if manifest.get("method") != "paired-median-ppm-bootstrap4000-v1":
        raise ValueError("method")
    if manifest.get("blocks") != 30 or manifest.get("vectors") != 16384 or manifest.get("replays") != 32:
        raise ValueError("protocol")
    fixtures, digest = _fixtures(root, manifest.get("fixtures_sha256"))
    seen = {}
    _check_ptx(root, _rows(manifest, "controls", allow_empty=False), seen)
    _check_ptx(root, _rows(manifest, "candidates", allow_empty=True), seen)
    return manifest, fixtures, digest, seen


def require_run_inputs(root):
    probe_manifest, probe_fixtures, probe_digest, probe_ptx = require_probe_inputs(root)
    timing_manifest, timing_fixtures, timing_digest, timing_ptx = require_timing_inputs(root)
    if probe_digest != timing_digest or probe_fixtures != timing_fixtures:
        raise ValueError("fixture digest")
    ptx = dict(probe_ptx)
    ptx.update(timing_ptx)
    return {
        "fixtures_sha256": probe_digest,
        "ptx": ptx,
        "gpu_manifest": probe_manifest,
        "benchmark_manifest": timing_manifest,
    }


def stage_verified_inputs(root, source):
    """Copy a verified bundle into an empty run directory and read it back."""
    expected = require_run_inputs(source)
    source = Path(source)
    root = Path(root)
    names = [GPU_MANIFEST, BENCH_MANIFEST, FIXTURES, *expected["ptx"]]
    for name in names:
        rel = safe_relative(name)
        src = _contained(source, source.joinpath(*rel.parts))
        if not src.is_file() or src.is_symlink():
            raise ValueError("missing " + rel.as_posix())
        dest = root.joinpath(*rel.parts)
        if dest.exists() or dest.is_symlink():
            raise ValueError("staged input already exists")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
    staged = require_run_inputs(root)
    if staged["fixtures_sha256"] != expected["fixtures_sha256"] or staged["ptx"] != expected["ptx"]:
        raise ValueError("staged inputs drifted")
    receipt = {
        "schema": 1,
        "fixtures_sha256": staged["fixtures_sha256"],
        "ptx": staged["ptx"],
    }
    (root / "input-inventory.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return receipt

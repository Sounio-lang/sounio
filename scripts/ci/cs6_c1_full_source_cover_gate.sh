#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

aggregate="scripts/research/cs6_c1_full_source_cover_aggregate.py"
runner="scripts/research/cs6_c1_full_source_cover_run.py"
verifier="scripts/research/cs6_c1_full_source_cover_leaf_verify.py"
retained_verifier="scripts/research/cs6_c1_full_source_cover_retained_verify.py"
worker_source="scripts/research/cs6_c1_full_source_cover_probe.cpp"

python3 -m py_compile "$aggregate" "$runner" "$verifier" "$retained_verifier"

mutation_output="$(python3 "$aggregate" --self-test-mutations)"
grep -Fxq 'MUTATION_TESTS=8' <<<"$mutation_output"
grep -Fxq 'MUTATIONS_REJECTED=8' <<<"$mutation_output"

python3 "$retained_verifier"

python3 - "$aggregate" "$runner" <<'PY'
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import sys
import tempfile
from pathlib import Path


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


aggregate = load("cs6_cover_aggregate_gate", sys.argv[1])
runner = load("cs6_cover_runner_gate", sys.argv[2])

root = runner.Leaf(0, 0, 0, 0)
root_result = runner.LeafResult(
    root,
    "COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN",
    "NONE",
    False,
    True,
    "1" * 64,
    "2" * 64,
    hashlib.sha256(b"").hexdigest(),
    runner.ZERO_SHA256,
    runner.ZERO_SHA256,
    1,
    1,
)
results, nodes, waves = runner.build_adaptive_tree(
    lambda frontier: [root_result for _ in frontier], 1, 2
)
assert len(results) == len(nodes) == len(waves) == 1
assert nodes[root.identity].action == "UNRESOLVED"
row = runner.cover_node_fields(nodes[root.identity])
assert row[6:8] == ("UNRESOLVED", "NONE")
assert row[8:] == ("-",) * 8

certified = runner.LeafResult(
    root, "CERTIFIED", "AFFINE", True, False,
    "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64, 0, 1,
)
_, complete_nodes, complete_waves = runner.build_adaptive_tree(
    lambda frontier: [certified for _ in frontier], 7, 3
)
assert len(complete_nodes) == len(complete_waves) == 1
assert complete_nodes[root.identity].action == "CERTIFIED"

assert len(runner.scout_leaves(((2, 2),), 4, False)) == 16
for invalid in (
    lambda: runner.scout_leaves(((1, 2),), 4, False),
    lambda: runner.scout_leaves(((0, 0),), 1, True),
):
    try:
        invalid()
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid scout cardinality escaped")

with tempfile.TemporaryDirectory(prefix="cs6-c1-cover-gate-") as directory:
    root_path = Path(directory)
    tree = root_path / "nodes.tsv"
    tree.write_bytes(runner.nodes_tsv_bytes(nodes))
    parsed = aggregate.parse_tree(tree)
    terminals, accepted, unresolved = aggregate.verify_structure(parsed)
    assert len(terminals) == 1 and accepted == 0 and unresolved == 1
    tree_sha = aggregate.digest(tree)
    challenge = "1" * 64

    def invoke(output: Path, flag: str | None = None) -> int:
        arguments = [
            str(tree), "--bundle-root", str(root_path),
            "--root-challenge", challenge, "--output", str(output),
        ]
        if flag is not None:
            arguments.append(flag)
        with contextlib.redirect_stdout(io.StringIO()):
            return aggregate.main(arguments)

    certificate = root_path / "partial-certificate.txt"
    assert invoke(certificate) == 0
    certificate_text = certificate.read_text(encoding="ascii")
    assert "UNRESOLVED_AREA_NUMERATOR=1\n" in certificate_text
    assert "LOCAL_FULL_SOURCE_CERTIFICATE_COMPLETE=false\n" in certificate_text
    assert "FULL_SOURCE_CARRIER_PROVED=false\n" in certificate_text
    assert invoke(root_path / "require-local.txt", "--require-local-full-source") == 2
    assert invoke(root_path / "require-full.txt", "--require-full-source") == 3

    for forbidden in (certificate, root_path / "certificate-link.txt"):
        if forbidden.name.endswith("link.txt"):
            forbidden.symlink_to(tree.name)
        before = aggregate.digest(tree)
        try:
            invoke(forbidden)
        except aggregate.CoverError:
            pass
        else:
            raise AssertionError("nonexclusive certificate publication escaped")
        assert aggregate.digest(tree) == before == tree_sha

print("ADAPTIVE_STRUCTURE_TEST=PASS")
print("PARTIAL_AGGREGATE_FAIL_CLOSED_TEST=PASS")
print("EXCLUSIVE_PUBLICATION_TEST=PASS")
PY

if [[ "${CS6_C1_FULL_SOURCE_COVER_REPLAY:-0}" == "1" ]]; then
  capd_config="${CS6_CAPD_CONFIG:-/tmp/capd-build/bin/capd-config}"
  [[ -x "$capd_config" ]] || {
    echo "CAPD config is not executable: $capd_config" >&2
    exit 1
  }
  python3 - "$aggregate" "$worker_source" "$verifier" "$capd_config" <<'PY'
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path


spec = importlib.util.spec_from_file_location("cs6_cover_live_replay", Path(sys.argv[1]).resolve())
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load aggregate")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

source = Path(sys.argv[2]).resolve()
verifier = Path(sys.argv[3]).resolve()
capd_config = Path(sys.argv[4]).resolve()
identity = "U13-0000001024_S14-0000002048"
stored_root = "1" * 64
with tempfile.TemporaryDirectory(prefix="cs6-c1-cover-gate-replay-") as directory:
    root = Path(directory)
    build_root = Path(directory) / "build"
    build_root.mkdir()
    bundle = root / "bundle"
    for name in ("inputs", "receipts", "verifications"):
        (bundle / name).mkdir(parents=True, exist_ok=True)
    source_sha = module.digest(source)
    worker, metadata = module.compile_canonical_worker(
        source, source_sha, capd_config, "g++", build_root, 180
    )
    seed = module.Node(identity, "-", 13, 1024, 14, 2048, "CERTIFIED", "AFFINE")
    input_raw = module.leaf_input_bytes(seed)
    input_sha = module.digest_bytes(input_raw)
    stored_challenge = module.challenge(stored_root, identity, input_sha)
    input_path = bundle / "inputs" / f"{identity}.txt"
    receipt_path = bundle / "receipts" / f"{identity}.txt"
    verification_path = bundle / "verifications" / f"{identity}.txt"
    input_path.write_bytes(input_raw)
    fresh = subprocess.run(
        [worker, "13", "1024", "14", "2048", input_sha, stored_challenge],
        capture_output=True,
        timeout=180,
    )
    assert fresh.returncode == 0 and not fresh.stderr
    receipt_path.write_bytes(fresh.stdout)
    verified = subprocess.run(
        [
            sys.executable, verifier, receipt_path, "--source-sha", source_sha,
            "--input", input_path, "--challenge", stored_challenge,
            "--require-terminal",
        ],
        capture_output=True,
        timeout=180,
    )
    assert verified.returncode == 0 and not verified.stderr
    verification_path.write_bytes(verified.stdout)
    values = module.parse_verification(verified.stdout)
    assert values["CERTIFICATE_PASS"] == "true"
    node = module.Node(
        identity, "-", 13, 1024, 14, 2048, "CERTIFIED", values["LEAF_METHOD"],
        f"inputs/{identity}.txt", input_sha, stored_challenge,
        f"receipts/{identity}.txt", module.digest(receipt_path),
        f"verifications/{identity}.txt", module.digest(verification_path),
        values["PHYSICAL_SHA256"],
    )
    replay, tests, rejected = module.verify_terminal_artifacts(
        node, bundle, verifier, source_sha, stored_root,
        worker, "2" * 64, True, 180,
    )
    assert replay is not None and tests == rejected == 56
    assert metadata["CAPD_VERSION"] == "5.3.0"
    assert metadata["INTERVAL_BACKEND"] == "FILIB"
    module.verify_file_manifest(
        (build_root / "dependencies.sha256").read_bytes(), source
    )
    module.verify_file_manifest((build_root / "link-inputs.sha256").read_bytes())
    module.verify_file_manifest(
        (build_root / "runtime-libraries.sha256").read_bytes()
    )
    audit = root / "audit"
    audit_sha = module.publish_audit_bundle(build_root, audit, replay)
    assert audit_sha == module.digest(audit / "audit-manifest.txt")
    assert (audit / "replay-ledger.tsv").read_bytes() == replay
    for row in (audit / "files.sha256").read_text(encoding="ascii").splitlines():
        expected, name = row.split("  ", 1)
        assert module.digest(audit / name) == expected
    assert "REPLAY_LEDGER_RETAINED=true\n" in (
        audit / "audit-manifest.txt"
    ).read_text(encoding="ascii")
    collision = root / "audit-collision"
    collision.mkdir()
    sentinel = collision / "sentinel.txt"
    sentinel.write_text("preserve\n", encoding="ascii")
    link = root / "audit-link"
    link.symlink_to(build_root, target_is_directory=True)
    for forbidden in (collision, link):
        try:
            module.publish_audit_bundle(build_root, forbidden, replay)
        except module.CoverError:
            pass
        else:
            raise AssertionError("nonexclusive audit publication escaped")
    assert sentinel.read_text(encoding="ascii") == "preserve\n"
    assert link.is_symlink()
print("CANONICAL_CAPD_REPLAY_TEST=PASS")
print("DURABLE_REPLAY_AUDIT_TEST=PASS")
print("EXCLUSIVE_AUDIT_PUBLICATION_TEST=PASS")
print("AUDIT_MANIFEST_BINDING_TEST=PASS")
PY
else
  echo "CANONICAL_CAPD_REPLAY_TEST=SKIPPED (set CS6_C1_FULL_SOURCE_COVER_REPLAY=1)"
fi

echo "CS6_C1_FULL_SOURCE_COVER_GATE=PASS"

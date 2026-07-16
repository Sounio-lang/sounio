#!/usr/bin/env python3
"""Executable acceptance gate for the R0-R2 science boundary."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import tomllib
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_SPEC = importlib.util.spec_from_file_location(
    "sounio_registry_serve",
    ROOT / "scripts/dev/registry_serve.py",
)
if REGISTRY_SPEC is None or REGISTRY_SPEC.loader is None:
    raise RuntimeError("cannot load registry_serve.py")
registry_serve = importlib.util.module_from_spec(REGISTRY_SPEC)
REGISTRY_SPEC.loader.exec_module(registry_serve)

ATTESTOR = ROOT / "tools/science_boundary/attestor.py"
MADAROS = ROOT / "bin/madaros"
SOUC = ROOT / "bin/souc"
RAW_MADAROS = Path(
    os.environ.get("SOUNIO_SCIENCE_BOUNDARY_MADAROS_BIN")
    or os.environ.get("MADAROS_RAW_BIN")
    or (
        str(ROOT / "artifacts/self-hosted/madaros")
        if (ROOT / "artifacts/self-hosted/madaros").is_file()
        else str(ROOT / "bin/madaros-linux-x86_64")
    )
).resolve()
HEADER = (
    "path\tring\tevidence_status\tcontext_of_use\tvisibility\tenforcement\t"
    "next_gate\tallowed_claim_classes\tevidence_refs\tdeclared_by\tdeclared_at\t"
    "review_state\n"
)
BASE_ENV = {
    **os.environ,
    "MADAROS_RAW_BIN": str(RAW_MADAROS),
    "SOUNIO_MADAROS_BIN": str(RAW_MADAROS),
}
TESTS = 0


def check(condition: bool, message: str) -> None:
    global TESTS
    TESTS += 1
    if not condition:
        raise AssertionError(message)


def run(
    command: list[str],
    *,
    expected: int | set[int] = 0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env or BASE_ENV,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )
    expected_codes = {expected} if isinstance(expected, int) else expected
    if result.returncode not in expected_codes:
        rendered = " ".join(command)
        raise AssertionError(
            f"unexpected exit {result.returncode}, expected {sorted(expected_codes)}: {rendered}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def policy_row(
    path: str,
    ring: str,
    *,
    visibility: str = "public",
    claims: str = "compile|runtime",
    context: str = "boundary gate fixture",
) -> str:
    return "\t".join(
        (
            path,
            ring,
            "fixture",
            context,
            visibility,
            "strict",
            "package-boundary-receipt",
            claims,
            "gate:science_boundary_gate",
            "SOUNIO-SCIENCE-RESEARCH-BOUNDARY",
            "2026-07-15",
            "reviewed",
        )
    )


def write_policy(root: Path, rows: list[str], name: str = "science-rings.tsv") -> Path:
    return write(root / name, HEADER + "\n".join(rows) + "\n")


def source(root: Path, rel: str, imported: str = "") -> Path:
    use_line = f"use {imported}\n" if imported else ""
    return write(root / rel, use_line + "fn main() -> i64 { 0 }\n")


def dependency(root: Path, rel: str, imported: str = "") -> Path:
    use_line = f"use {imported}\n" if imported else ""
    return write(root / rel, use_line + "fn helper() -> i64 { 1 }\n")


def evaluate(
    src: Path,
    manifest: Path,
    receipt: Path,
    *,
    mode: str = "strict",
    claim: Path | None = None,
    expected: int | set[int] = 0,
    env: dict[str, str] | None = None,
    raw_report: bool = True,
) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    command = [
        sys.executable,
        str(ATTESTOR),
        "evaluate",
        "--source",
        str(src),
        "--mode",
        mode,
        "--manifest",
        str(manifest),
        "--receipt",
        str(receipt),
        "--compiler",
        str(RAW_MADAROS),
        "--engine-identity",
        "Madaros gate fixture",
    ]
    if raw_report and mode != "off":
        report = receipt.with_suffix(".closure.tsv")
        write_gate_closure_report(src, manifest.parent, report, env=env)
        command.extend(("--closure-report", str(report)))
    if claim is not None:
        command.extend(("--claim-contract", str(claim)))
    result = run(command, expected=expected, env=env)
    return json.loads(receipt.read_text(encoding="ascii")), result


def write_gate_closure_report(
    src: Path,
    root: Path,
    report: Path,
    *,
    env: dict[str, str] | None = None,
) -> Path:
    del root
    result = run(
        [str(RAW_MADAROS), "--science-boundary-closure", str(src.resolve())],
        env=env,
    )
    check("SOUNIO_BOUNDARY_CLOSURE_V1" in result.stdout, "raw fixture closure report is absent")
    return write(report, result.stdout)


def diagnostic_codes(receipt: dict[str, Any]) -> set[str]:
    return {item["code"] for item in receipt["diagnostics"]}


def assert_raw_ast_transitive_closure() -> None:
    main_path = "examples/projects/hello_pkg/src/main.sio"
    dependency_path = "examples/projects/hello_pkg/src/greet.sio"
    result = run([str(RAW_MADAROS), "--science-boundary-closure", main_path])
    report = result.stdout
    check("SOUNIO_BOUNDARY_CLOSURE_V1" in report, "raw AST closure header is absent")
    check("status\tcomplete" in report, "raw AST transitive closure is incomplete")
    check(f"node\t{main_path}" in report, "raw AST closure omitted the root module")
    check(f"node\t{dependency_path}" in report, "raw AST closure omitted the imported module")
    check(
        f"edge\t{main_path}\t{dependency_path}" in report,
        "raw AST closure omitted the import edge",
    )
    check("unresolved\t" not in report, "raw AST closure retained an unresolved import")


def matrix_fixture(root: Path) -> Path:
    dependency(root, "core/dep.sio")
    dependency(root, "package/dep.sio")
    dependency(root, "research/dep.sio")
    return write_policy(
        root,
        [
            policy_row("core", "pl-core"),
            policy_row("package", "scientific-package"),
            policy_row("research", "research"),
        ],
    )


def assert_matrix(root: Path) -> None:
    manifest = matrix_fixture(root)
    allowed = (
        ("core/main.sio", "core::dep"),
        ("package/main.sio", "package::dep"),
        ("package/main.sio", "core::dep"),
        ("research/main.sio", "research::dep"),
        ("research/main.sio", "package::dep"),
        ("research/main.sio", "core::dep"),
    )
    for index, (rel, imported) in enumerate(allowed):
        src = source(root, rel, imported)
        receipt, _ = evaluate(src, manifest, root / f"allowed-{index}.json")
        check(receipt["verdict"] == "OK", f"allowed dependency refused: {rel} -> {imported}")

    forbidden = (
        ("core/main.sio", "package::dep"),
        ("core/main.sio", "research::dep"),
        ("package/main.sio", "research::dep"),
    )
    for index, (rel, imported) in enumerate(forbidden):
        src = source(root, rel, imported)
        receipt, result = evaluate(
            src,
            manifest,
            root / f"forbidden-{index}.json",
            expected=20,
        )
        check(receipt["verdict"] == "REJECT", f"forbidden dependency accepted: {rel} -> {imported}")
        check("E-SRB-001" in diagnostic_codes(receipt), "ring inversion lacks E-SRB-001")
        check("error[E-SRB-001]" in result.stderr, "structured ring diagnostic was not printed")

    src = source(root, "core/main.sio", "package::dep")
    receipt, _ = evaluate(
        src,
        manifest,
        root / "host-forbidden.json",
        expected=21,
        raw_report=False,
    )
    check(receipt["verdict"] == "UNKNOWN", "host syntax closure decided a ring rejection")
    check("E-SRB-001" not in diagnostic_codes(receipt), "host syntax closure emitted authoritative E-SRB-001")

    visibility_manifest = write_policy(
        root,
        [
            policy_row("core", "pl-core", visibility="public"),
            policy_row("package", "pl-core", visibility="protected"),
            policy_row("research", "research"),
        ],
        "visibility.tsv",
    )
    src = source(root, "core/main.sio", "package::dep")
    receipt, _ = evaluate(src, visibility_manifest, root / "visibility.json", expected=20)
    check("E-SRB-002" in diagnostic_codes(receipt), "visibility leak lacks E-SRB-002")


def assert_unknowns(root: Path) -> None:
    dependency(root, "core/dep.sio")
    candidate_manifest = write_policy(
        root,
        [policy_row("core", "scientific-package-candidate")],
        "candidate.tsv",
    )
    src = source(root, "core/main.sio")
    receipt, _ = evaluate(src, candidate_manifest, root / "candidate.json", expected=21)
    check(receipt["verdict"] == "UNKNOWN", "candidate ring became conclusive")
    check("E-SRB-000" in diagnostic_codes(receipt), "candidate ring lacks E-SRB-000")

    unclassified_manifest = write_policy(
        root,
        [policy_row("core", "pl-core")],
        "unclassified.tsv",
    )
    dependency(root, "misc/dep.sio")
    src = source(root, "core/main.sio", "misc::dep")
    receipt, _ = evaluate(src, unclassified_manifest, root / "unclassified.json", expected=21)
    check(receipt["verdict"] == "UNKNOWN", "unclassified dependency became OK")

    src = source(root, "core/main.sio", "missing::module")
    receipt, _ = evaluate(src, unclassified_manifest, root / "unresolved.json", expected=21)
    check(receipt["graph"]["unresolved_imports"], "unresolved import missing from graph")

    saturated_root = root / "saturated-raw"
    for index in range(257):
        imported = f"chain::m{index + 1}" if index < 256 else ""
        rel = "main.sio" if index == 0 else f"chain/m{index}.sio"
        dependency(saturated_root, rel, imported)
    saturated_manifest = write_policy(
        saturated_root,
        [policy_row(".", "research")],
    )
    receipt, _ = evaluate(
        saturated_root / "main.sio",
        saturated_manifest,
        root / "saturated.json",
        expected=21,
    )
    check(receipt["graph"]["saturated"] is True, "saturated closure became complete")

    src = source(root, "core/main.sio", "core::dep")

    escape_manifest = write_policy(
        root,
        [policy_row("../outside", "research")],
        "escape.tsv",
    )
    receipt, _ = evaluate(src, escape_manifest, root / "escape.json", expected=21)
    check(receipt["verdict"] == "UNKNOWN", "escaping policy path became OK")

    duplicate_manifest = write_policy(
        root,
        [policy_row("core", "pl-core"), policy_row("core", "research")],
        "duplicate.tsv",
    )
    receipt, _ = evaluate(src, duplicate_manifest, root / "duplicate.json", expected=21)
    check(receipt["verdict"] == "UNKNOWN", "duplicate policy path became OK")

    missing_manifest = root / "missing.tsv"
    receipt, _ = evaluate(src, missing_manifest, root / "missing.json", expected=21)
    check(receipt["verdict"] == "UNKNOWN", "missing manifest became OK")

    malformed = write(root / "core/malformed.sio", "fn main( -> i64 { 0 }\n")
    receipt, _ = evaluate(malformed, unclassified_manifest, root / "malformed.json", expected=21)
    check(receipt["verdict"] == "UNKNOWN", "incomplete parser surface became OK")

    first = dependency(root, "cycle/first.sio", "cycle::second")
    dependency(root, "cycle/second.sio", "cycle::first")
    cycle_manifest = write_policy(
        root,
        [policy_row("cycle", "research")],
        "cycle.tsv",
    )
    receipt, _ = evaluate(first, cycle_manifest, root / "cycle.json")
    check(receipt["verdict"] == "OK", "finite cycle was not closed deterministically")

    src = source(root, "cycle/host-audit.sio")
    receipt, _ = evaluate(
        src,
        cycle_manifest,
        root / "host-audit.json",
        expected=21,
        raw_report=False,
    )
    check(receipt["verdict"] == "UNKNOWN", "host syntax audit became authoritative")
    check(receipt["engine"]["boundary_collector"] == "sounio-host-syntax-v1", "host collector was not disclosed")

    forged_report = write(
        root / "forged-missing-root.closure.tsv",
        "\n".join(
            (
                "SOUNIO_BOUNDARY_CLOSURE_V1",
                "status\tcomplete",
                "capacity\t256",
                "saturated\tfalse",
                "parse_failed\tfalse",
                "",
            )
        ),
    )
    forged_receipt = root / "forged-missing-root.json"
    run(
        [
            sys.executable,
            str(ATTESTOR),
            "evaluate",
            "--source",
            str(src),
            "--mode",
            "strict",
            "--manifest",
            str(cycle_manifest),
            "--receipt",
            str(forged_receipt),
            "--compiler",
            str(RAW_MADAROS),
            "--engine-identity",
            "Madaros forged-report fixture",
            "--closure-report",
            str(forged_report),
        ],
        expected=21,
    )
    forged_data = json.loads(forged_receipt.read_text(encoding="ascii"))
    check(forged_data["verdict"] == "UNKNOWN", "raw closure report without its root became authoritative")
    check("E-SRB-000" in diagnostic_codes(forged_data), "missing raw closure root lacks E-SRB-000")

    no_evidence_manifest = write_policy(
        root,
        [policy_row("core", "pl-core").replace("gate:science_boundary_gate", "")],
        "no-evidence.tsv",
    )
    src = source(root, "core/main.sio")
    receipt, _ = evaluate(src, no_evidence_manifest, root / "no-evidence.json", expected=20)
    check("E-SRB-003" in diagnostic_codes(receipt), "conclusive declaration without evidence lacks E-SRB-003")

    short_manifest = write(root / "short.tsv", HEADER + "core\tpl-core\n")
    receipt, _ = evaluate(src, short_manifest, root / "short.json", expected=21)
    check(receipt["verdict"] == "UNKNOWN", "short TSV row crashed or became authoritative")

    conflict_root = root / "conflict"
    conflict_src = source(conflict_root, "main.sio")
    write_policy(conflict_root, [policy_row(".", "research")])
    science_manifest(conflict_root)
    conflict_report = write_gate_closure_report(
        conflict_src,
        conflict_root,
        conflict_root / "conflict.closure.tsv",
    )
    conflict_receipt = conflict_root / "conflict.json"
    result = run(
        [
            sys.executable,
            str(ATTESTOR),
            "evaluate",
            "--source",
            str(conflict_src),
            "--mode",
            "strict",
            "--receipt",
            str(conflict_receipt),
            "--compiler",
            str(RAW_MADAROS),
            "--engine-identity",
            "Madaros gate fixture",
            "--closure-report",
            str(conflict_report),
        ],
        expected=21,
    )
    conflict_data = json.loads(conflict_receipt.read_text(encoding="ascii"))
    check(conflict_data["verdict"] == "UNKNOWN", "conflicting declarations became authoritative")
    check("E-SRB-000" in result.stderr, "conflicting declarations lack E-SRB-000")


def science_manifest(
    root: Path,
    *,
    claims: str = '["compile", "runtime"]',
    extra: str = "",
) -> Path:
    return write(
        root / "sounio.toml",
        "[science]\n"
        'schema = "sounio.science-manifest.v1"\n'
        'ring = "research"\n'
        'evidence-status = "fixture"\n'
        'context-of-use = "boundary gate fixture"\n'
        'visibility = "public"\n'
        f"allowed-claim-classes = {claims}\n"
        'evidence-refs = ["gate:science_boundary_gate"]\n'
        'next-gate = "package-boundary-receipt"\n'
        'declared-by = "SOUNIO-SCIENCE-RESEARCH-BOUNDARY"\n'
        'declared-at = "2026-07-15"\n'
        'review-state = "reviewed"\n'
        + extra,
    )


def write_claim(
    path: Path,
    requested: str,
    evidence: list[str],
    *,
    context: str = "boundary gate fixture",
    root_artifact: str = "main.sio",
) -> Path:
    root = path.parent
    body = (
        'schema = "sounio.claim-contract.v1"\n'
        'claim-id = "gate.claim"\n'
        f'requested-class = "{requested}"\n'
        f'context-of-use = "{context}"\n'
        f'root-artifact = "{root_artifact}"\n'
    )
    for index, evidence_type in enumerate(evidence):
        if evidence_type == "source":
            evidence_path = root / root_artifact
            evidence_ref = root_artifact
        elif evidence_type == "package":
            evidence_path = root / "sounio.toml"
            evidence_ref = "sounio.toml"
        elif evidence_type == "compiler":
            evidence_path = RAW_MADAROS
            evidence_ref = RAW_MADAROS.name
        else:
            evidence_ref = f"evidence/{index}-{evidence_type}.txt"
            evidence_path = write(root / evidence_ref, f"{evidence_type} fixture\n")
        digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        body += (
            "\n[[evidence]]\n"
            f'type = "{evidence_type}"\n'
            f'ref = "{evidence_ref}"\n'
            f'sha256 = "{digest}"\n'
        )
    return write(path, body)


def assert_manifest_and_claims(root: Path) -> None:
    src = source(root, "main.sio")
    legacy = write(
        root / "legacy.toml",
        '[epistemic]\nscore = 1.0\nregulatory-ready = true\ngum-compliant = true\n',
    )
    receipt, result = evaluate(src, legacy, root / "legacy.json", expected=21)
    check(receipt["verdict"] == "UNKNOWN", "legacy scalar metadata granted authority")
    check("W-SRB-LEGACY-001" in diagnostic_codes(receipt), "legacy warning is absent")
    check("W-SRB-LEGACY-001" in result.stderr, "legacy warning was not printed")

    incomplete_manifest = science_manifest(
        root,
        extra='\n[[example]]\nname = "missing-fields"\npath = "main.sio"\n',
    )
    receipt, _ = evaluate(src, incomplete_manifest, root / "example-missing.json", expected=21)
    check(receipt["verdict"] == "UNKNOWN", "scientific example without maturity became OK")

    calibrated_manifest = science_manifest(
        root,
        extra=(
            '\n[[example]]\nname = "calibrated"\npath = "main.sio"\n'
            'maturity = "calibrated"\ncontext-of-use = "boundary gate fixture"\n'
            'evidence-refs = ["dataset:fixture"]\n'
        ),
    )
    receipt, _ = evaluate(src, calibrated_manifest, root / "calibrated.json", expected=21)
    check(receipt["verdict"] == "UNKNOWN", "calibrated example without witnesses became OK")

    manifest = science_manifest(root, claims='["compile", "clinical"]')
    compile_only = write_claim(
        root / "compile-only.toml",
        "clinical",
        ["source", "package", "compiler", "compile", "runtime"],
    )
    receipt, _ = evaluate(src, manifest, root / "compile-only.json", claim=compile_only, expected=20)
    codes = diagnostic_codes(receipt)
    check("E-SRB-005" in codes, "clinical compile-only claim lacks E-SRB-005")
    check("E-SRB-006" in codes, "clinical provenance gap lacks E-SRB-006")

    unauthorized_manifest = science_manifest(root, claims='["compile"]')
    complete_clinical = write_claim(
        root / "unauthorized.toml",
        "clinical",
        [
            "source",
            "package",
            "compiler",
            "model",
            "data-manifest",
            "dataset",
            "split",
            "diagnostics",
            "gate",
            "review",
        ],
    )
    receipt, _ = evaluate(
        src,
        unauthorized_manifest,
        root / "unauthorized.json",
        claim=complete_clinical,
        expected=20,
    )
    check("E-SRB-007" in diagnostic_codes(receipt), "unauthorized claim lacks E-SRB-007")

    provenance = write_claim(root / "provenance.toml", "compile", ["source"])
    receipt, _ = evaluate(src, unauthorized_manifest, root / "provenance.json", claim=provenance, expected=20)
    check("E-SRB-006" in diagnostic_codes(receipt), "missing package/compiler lacks E-SRB-006")

    legacy_claim = write_claim(
        root / "legacy-claim.toml",
        "compile",
        ["source", "package", "compiler", "score"],
    )
    receipt, _ = evaluate(src, unauthorized_manifest, root / "legacy-claim.json", claim=legacy_claim, expected=20)
    check("E-SRB-004" in diagnostic_codes(receipt), "legacy score evidence lacks E-SRB-004")

    gum_manifest = science_manifest(root, claims='["gum-uncertainty"]')
    gum_claim = write_claim(
        root / "gum.toml",
        "gum-uncertainty",
        ["source", "package", "compiler"],
    )
    receipt, _ = evaluate(src, gum_manifest, root / "gum.json", claim=gum_claim, expected=20)
    check("E-SRB-004" in diagnostic_codes(receipt), "GUM claim without method/witness lacks E-SRB-004")

    gum_bound = write_claim(
        root / "gum-bound.toml",
        "gum-uncertainty",
        ["source", "package", "compiler", "method", "witness"],
    )
    receipt, _ = evaluate(src, gum_manifest, root / "gum-bound.json", claim=gum_bound)
    check(receipt["verdict"] == "OK", "GUM claim with method and witness did not pass")

    context_manifest = science_manifest(root, claims='["compile"]')
    wrong_context = write_claim(
        root / "wrong-context.toml",
        "compile",
        ["source", "package", "compiler"],
        context="different use",
    )
    receipt, _ = evaluate(
        src,
        context_manifest,
        root / "wrong-context.json",
        claim=wrong_context,
        expected=20,
    )
    check("E-SRB-004" in diagnostic_codes(receipt), "claim context mismatch lacks E-SRB-004")

    absolute_ref = write_claim(
        root / "absolute-ref.toml",
        "compile",
        ["source", "package", "compiler"],
    )
    absolute_ref.write_text(
        absolute_ref.read_text(encoding="utf-8").replace(
            f'ref = "{RAW_MADAROS.name}"',
            f'ref = "{RAW_MADAROS}"',
        ),
        encoding="utf-8",
    )
    absolute_receipt = root / "absolute-ref.json"
    receipt, _ = evaluate(
        src,
        context_manifest,
        absolute_receipt,
        claim=absolute_ref,
        expected=20,
    )
    check("E-SRB-006" in diagnostic_codes(receipt), "absolute evidence ref lacks E-SRB-006")
    check(str(ROOT) not in absolute_receipt.read_text(encoding="ascii"), "reject receipt leaked an absolute path")

    bound_manifest = science_manifest(root, claims='["compile"]')
    bound_claim = write_claim(
        root / "bound.toml",
        "compile",
        ["source", "package", "compiler", "gate"],
    )
    bound_receipt = root / "bound.json"
    receipt, _ = evaluate(src, bound_manifest, bound_receipt, claim=bound_claim)
    check(receipt["verdict"] == "OK", "content-bound compile claim did not pass")
    run(
        [
            sys.executable,
            str(ATTESTOR),
            "verify",
            "--receipt",
            str(bound_receipt),
            "--root",
            str(root),
            "--compiler",
            str(RAW_MADAROS),
        ]
    )

    forged_claim_receipt = root / "forged-claim-summary.json"
    forged_claim_data = json.loads(bound_receipt.read_text(encoding="ascii"))
    forged_claim_data["claim_contract"]["unexpected_authority"] = "hidden"
    identity_payload = json.loads(json.dumps(forged_claim_data))
    identity_payload["hashes"].pop("receipt_identity_sha256", None)
    forged_claim_data["hashes"]["receipt_identity_sha256"] = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    forged_claim_receipt.write_text(
        json.dumps(forged_claim_data, sort_keys=True, indent=2) + "\n",
        encoding="ascii",
    )
    forged_result = run(
        [
            sys.executable,
            str(ATTESTOR),
            "verify",
            "--receipt",
            str(forged_claim_receipt),
            "--root",
            str(root),
            "--compiler",
            str(RAW_MADAROS),
        ],
        expected=1,
    )
    check(
        "bad receipt claim contract" in forged_result.stderr,
        "receipt verifier accepted an undeclared claim summary field",
    )

    write(root / "evidence/3-gate.txt", "tampered\n")
    run(
        [
            sys.executable,
            str(ATTESTOR),
            "verify",
            "--receipt",
            str(bound_receipt),
            "--root",
            str(root),
            "--compiler",
            str(RAW_MADAROS),
        ],
        expected=1,
    )

    no_identity_receipt = root / "no-engine-identity.json"
    report = write_gate_closure_report(src, root, root / "no-engine.closure.tsv")
    run(
        [
            sys.executable,
            str(ATTESTOR),
            "evaluate",
            "--source",
            str(src),
            "--mode",
            "strict",
            "--manifest",
            str(context_manifest),
            "--receipt",
            str(no_identity_receipt),
            "--compiler",
            str(RAW_MADAROS),
            "--closure-report",
            str(report),
        ],
        expected=21,
    )
    no_identity = json.loads(no_identity_receipt.read_text(encoding="ascii"))
    check(no_identity["verdict"] == "UNKNOWN", "strict receipt without engine identity became OK")


def assert_receipts_and_elf(root: Path) -> None:
    hello = write(root / "hello.sio", (ROOT / "examples/hello.sio").read_text(encoding="utf-8"))
    manifest = write_policy(root, [policy_row(".", "research")])

    first, _ = evaluate(hello, manifest, root / "deterministic-a.json")
    second, _ = evaluate(hello, manifest, root / "deterministic-b.json")
    check(
        (root / "deterministic-a.json").read_bytes() == (root / "deterministic-b.json").read_bytes(),
        "repeated identity receipt is not deterministic",
    )
    original_digest = first["hashes"]["source_bundle_sha256"]
    hello.write_text(hello.read_text(encoding="utf-8") + "\n// digest change\n", encoding="utf-8")
    changed, _ = evaluate(hello, manifest, root / "changed.json")
    check(changed["hashes"]["source_bundle_sha256"] != original_digest, "source edit did not change digest")
    hello.write_text((ROOT / "examples/hello.sio").read_text(encoding="utf-8"), encoding="utf-8")

    madaros_elf = root / "madaros-hello"
    madaros_receipt = root / "madaros-hello.json"
    result = run(
        [
            str(MADAROS),
            "--science-boundary",
            "strict",
            "--science-manifest",
            str(manifest),
            "--emit-boundary-receipt",
            str(madaros_receipt),
            "build",
            str(hello),
            "-o",
            str(madaros_elf),
        ]
    )
    check(madaros_elf.is_file() and os.access(madaros_elf, os.X_OK), "Madaros strict build emitted no ELF")
    check(madaros_receipt.is_file(), "Madaros strict build emitted no receipt")
    check(run([str(madaros_elf)]).stdout.strip() == "Hello, Sounio", "Madaros strict ELF did not run")
    check("science-boundary: mode=strict verdict=OK" in result.stderr, "strict build did not report OK")

    verify = run(
        [
            sys.executable,
            str(ATTESTOR),
            "verify",
            "--receipt",
            str(madaros_receipt),
            "--root",
            str(root),
            "--compiler",
            str(RAW_MADAROS),
            "--elf",
            str(madaros_elf),
        ]
    )
    check("PACKAGE_BOUNDARY_RECEIPT_VERIFY_PASS" in verify.stdout, "valid receipt did not verify")
    run(
        [
            sys.executable,
            str(ATTESTOR),
            "verify",
            "--receipt",
            str(madaros_receipt),
            "--root",
            str(root),
            "--elf",
            str(madaros_elf),
        ],
        expected=1,
    )
    run(
        [
            sys.executable,
            str(ATTESTOR),
            "verify",
            "--receipt",
            str(madaros_receipt),
            "--root",
            str(root),
            "--compiler",
            str(RAW_MADAROS),
        ],
        expected=1,
    )

    public_elf = root / "public-hello"
    public_receipt = root / "public-hello.json"
    run(
        [
            str(SOUC),
            "--science-boundary",
            "strict",
            "--science-manifest",
            str(manifest),
            "--emit-boundary-receipt",
            str(public_receipt),
            "build",
            str(hello),
            "-o",
            str(public_elf),
        ]
    )
    check(public_elf.is_file() and public_receipt.is_file(), "public strict path did not emit ELF and receipt")
    check(run([str(public_elf)]).stdout.strip() == "Hello, Sounio", "public strict ELF did not run")

    auto_root = root / "auto-discovery"
    auto_hello = write(
        auto_root / "nested/hello.sio",
        (ROOT / "examples/hello.sio").read_text(encoding="utf-8"),
    )
    write_policy(auto_root, [policy_row(".", "research")])
    auto_elf = auto_root / "auto-hello"
    auto_receipt = auto_root / "auto-hello.json"
    auto_result = run(
        [
            str(MADAROS),
            "--emit-boundary-receipt",
            str(auto_receipt),
            "build",
            str(auto_hello),
            "-o",
            str(auto_elf),
        ]
    )
    auto_data = json.loads(auto_receipt.read_text(encoding="ascii"))
    check(auto_data["mode"] == "advisory", "discovered ancestor policy did not select advisory mode")
    check(auto_data["verdict"] == "OK", "auto-discovered policy did not use the raw AST closure")
    check("mode=advisory verdict=OK" in auto_result.stderr, "auto advisory verdict was not reported")

    receipt_data = json.loads(madaros_receipt.read_text(encoding="ascii"))
    serialized = madaros_receipt.read_text(encoding="ascii")
    check(receipt_data["schema"] == "sounio.package-boundary-receipt.v1", "bad receipt schema")
    check(receipt_data["assurance_level"] == "identity-only", "receipt assurance is not identity-only")
    check(receipt_data["engine"]["boundary_collector"] == "madaros-raw-ast-v1", "strict OK did not use raw AST closure")
    check(bool(receipt_data["hashes"]["closure_report_sha256"]), "raw AST report digest is absent")
    check("timestamp" not in serialized.lower(), "identity receipt contains a timestamp field")
    check(str(root) not in serialized, "identity receipt contains an absolute fixture path")

    tampered = root / "tampered.json"
    tampered_data = json.loads(serialized)
    tampered_data["verdict"] = "REJECT"
    tampered.write_text(json.dumps(tampered_data), encoding="ascii")
    run(
        [sys.executable, str(ATTESTOR), "verify", "--receipt", str(tampered), "--root", str(root)],
        expected=1,
    )

    run(
        [
            sys.executable,
            str(ATTESTOR),
            "verify",
            "--receipt",
            str(madaros_receipt),
            "--root",
            str(root),
            "--compiler",
            "/bin/true",
            "--elf",
            str(madaros_elf),
        ],
        expected=1,
    )

    reject_root = root / "reject"
    reject_manifest = matrix_fixture(reject_root)
    reject_src = source(reject_root, "core/main.sio", "package::dep")
    rejected_elf = reject_root / "stale-output"
    rejected_elf.write_text("stale", encoding="ascii")
    run(
        [
            str(MADAROS),
            "--science-boundary",
            "strict",
            "--science-manifest",
            str(reject_manifest),
            "build",
            str(reject_src),
            "-o",
            str(rejected_elf),
        ],
        expected=1,
    )
    check(not rejected_elf.exists(), "strict rejection left a final ELF")

    reject_preflight = reject_root / "reject-preflight.json"
    evaluate(
        reject_src,
        reject_manifest,
        reject_preflight,
        expected=20,
    )
    invalid_final_receipt = reject_root / "invalid-final.json"
    run(
        [
            sys.executable,
            str(ATTESTOR),
            "finalize",
            "--preflight-receipt",
            str(reject_preflight),
            "--source",
            str(reject_src),
            "--manifest",
            str(reject_manifest),
            "--compiler",
            str(RAW_MADAROS),
            "--elf",
            str(madaros_elf),
            "--artifact-label",
            "invalid-final",
            "--receipt",
            str(invalid_final_receipt),
        ],
        expected=2,
    )
    check(not invalid_final_receipt.exists(), "strict REJECT preflight was finalized")

    changed_preflight = root / "changed-preflight.json"
    evaluate(hello, manifest, changed_preflight)
    hello.write_text(hello.read_text(encoding="utf-8") + "\n// changed after preflight\n", encoding="utf-8")
    changed_final = root / "changed-final.json"
    run(
        [
            sys.executable,
            str(ATTESTOR),
            "finalize",
            "--preflight-receipt",
            str(changed_preflight),
            "--source",
            str(hello),
            "--manifest",
            str(manifest),
            "--compiler",
            str(RAW_MADAROS),
            "--elf",
            str(madaros_elf),
            "--artifact-label",
            "changed-final",
            "--receipt",
            str(changed_final),
        ],
        expected=2,
    )
    check(not changed_final.exists(), "source-hash failure emitted a finalized receipt")
    hello.write_text((ROOT / "examples/hello.sio").read_text(encoding="utf-8"), encoding="utf-8")

    unknown_elf = root / "unknown-output"
    run(
        [
            str(MADAROS),
            "--science-boundary",
            "strict",
            "--science-manifest",
            str(root / "absent.tsv"),
            "build",
            str(hello),
            "-o",
            str(unknown_elf),
        ],
        expected=1,
    )
    check(not unknown_elf.exists(), "strict UNKNOWN left a final ELF")

    raw_bypass_elf = root / "raw-bypass"
    raw_bypass = run(
        [
            str(MADAROS),
            "--science-boundary",
            "strict",
            "--science-manifest",
            str(manifest),
            "--native-v2-compile",
            str(hello),
            str(raw_bypass_elf),
        ],
        expected=1,
    )
    check(not raw_bypass_elf.exists(), "strict raw pass-through bypass emitted an ELF")
    check("E-SRB-000" in raw_bypass.stderr, "strict raw bypass lacks E-SRB-000")

    legacy_bypass_elf = root / "legacy-bypass"
    legacy_env = {**BASE_ENV, "SOUNIO_SOUC_ENGINE": "lean_single"}
    legacy_bypass = run(
        [
            str(SOUC),
            "--science-boundary",
            "strict",
            "build",
            str(hello),
            "-o",
            str(legacy_bypass_elf),
        ],
        expected=1,
        env=legacy_env,
    )
    check(not legacy_bypass_elf.exists(), "forced lean_single bypass emitted an ELF")
    check("E-SRB-000" in legacy_bypass.stderr, "forced lean_single bypass lacks E-SRB-000")

    advisory_manifest = write_policy(
        root,
        [policy_row(".", "scientific-package-candidate")],
        "advisory.tsv",
    )
    advisory_elf = root / "advisory-output"
    advisory_receipt = root / "advisory.json"
    run(
        [
            str(MADAROS),
            "--science-boundary",
            "advisory",
            "--science-manifest",
            str(advisory_manifest),
            "--emit-boundary-receipt",
            str(advisory_receipt),
            "build",
            str(hello),
            "-o",
            str(advisory_elf),
        ]
    )
    advisory_data = json.loads(advisory_receipt.read_text(encoding="ascii"))
    check(advisory_elf.is_file(), "advisory UNKNOWN changed successful compiler status")
    check(advisory_data["verdict"] == "UNKNOWN", "advisory candidate was not recorded as UNKNOWN")

    invalid = write(root / "invalid.sio", "fn main( -> i64 { 0 }\n")
    baseline_failure = run([str(MADAROS), "check", str(invalid)], expected={1, 101})
    advisory_failure = run(
        [
            str(MADAROS),
            "--science-boundary",
            "advisory",
            "--science-manifest",
            str(advisory_manifest),
            "check",
            str(invalid),
        ],
        expected={1, 101},
    )
    check(
        advisory_failure.returncode == baseline_failure.returncode,
        "advisory mode changed the compiler's semantic failure status",
    )

    off_receipt = root / "off.json"
    off_data, _ = evaluate(hello, manifest, off_receipt, mode="off")
    check(off_data["mode"] == "off" and off_data["verdict"] == "UNKNOWN", "off receipt became authoritative")


def assert_registry_quarantine() -> None:
    direct = registry_serve._search("pbpk")
    check(bool(direct), "read-only registry catalog search returned no PBPK fixture")
    legacy_keys = {
        "epistemic_score_pct",
        "gum_compliant",
        "regulatory_ready",
        "provenance_level",
        "tier",
    }
    check(not (legacy_keys & set(direct[0])), "registry catalog still emits scalar authority fields")

    server = HTTPServer(("127.0.0.1", 0), registry_serve.RegistryHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(
            f"{base}/api/v1/search?q=pbpk&min_score=100&regulatory=true",
            timeout=5,
        ) as response:
            payload = json.loads(response.read())
        check(payload["total"] == len(direct), "legacy filters still influence registry search")
        check(
            payload["limitations"] == ["legacy_score_and_regulatory_filters_are_ignored"],
            "registry search does not disclose ignored legacy filters",
        )

        request = urllib.request.Request(
            f"{base}/api/v1/packages",
            data=json.dumps({"name": "legacy", "epistemic_score_pct": 100}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=5)
            raise AssertionError("score-based registry publish was accepted")
        except urllib.error.HTTPError as error:
            check(error.code == 501, "disabled registry publish returned the wrong status")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def assert_static_contracts() -> None:
    for schema in (
        ROOT / "schemas/sounio.claim-contract.v1.schema.json",
        ROOT / "schemas/sounio.package-boundary-receipt.v1.schema.json",
    ):
        data = json.loads(schema.read_text(encoding="utf-8"))
        check(isinstance(data, dict) and "$schema" in data, f"invalid JSON schema: {schema.name}")

    for rel in (
        "packages/epistemic-core/sounio.toml",
        "packages/sounio-units/sounio.toml",
        "packages/sounio-formats/sounio.toml",
        "packages/sounio-io-primitives/sounio.toml",
    ):
        path = ROOT / rel
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        check("science" in data, f"active package lacks [science]: {rel}")
        check("epistemic" not in data, f"active package still emits [epistemic]: {rel}")

    help_text = run([str(SOUC), "--help"]).stdout
    for flag in (
        "--science-boundary",
        "--science-manifest",
        "--claim-contract",
        "--emit-boundary-receipt",
    ):
        check(flag in help_text, f"public help lacks {flag}")

    for rel in (
        "docs/ecosystem/REGISTRY_ARCHITECTURE.md",
        "docs/ecosystem/PKG_MANAGER_SOTA_POSITION.md",
        "docs/ecosystem/ECOSYSTEM_ROADMAP_2026.md",
        "docs/ecosystem/CURATED_PACKAGES.md",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for legacy in (
            "epistemic-score",
            "gum-compliant",
            "regulatory-ready",
            "confidence-threshold",
        ):
            check(legacy not in text, f"active ecosystem documentation still recommends {legacy}: {rel}")

    registry_client = (ROOT / "self-hosted/compiler/pkg/registry_client.sio").read_text(encoding="utf-8")
    check("below minimum" not in registry_client, "registry client still gates publication by score")
    check("&min_score=" not in registry_client, "registry client still filters resolution by score")
    scorer = (ROOT / "self-hosted/compiler/pkg/scorer.sio").read_text(encoding="utf-8")
    check("package will be marked" not in scorer, "legacy scorer still controls registry status")
    check("REGULATORY tier" not in scorer, "legacy scorer still claims regulatory authority")
    package_cli = (ROOT / "self-hosted/compiler/pkg/cli.sio").read_text(encoding="utf-8")
    check("pkg_score_from_manifest" not in package_cli, "normal package CLI still computes a scalar authority score")
    check("registry.sounio.org" not in package_cli, "demonstrative package CLI still presents a public registry")
    compiler_main = (ROOT / "self-hosted/compiler/main.sio").read_text(encoding="utf-8")
    for forbidden in (
        "pkg_score_from_manifest",
        "High-quality package",
        "Excellent epistemic quality",
        "registry.sounio.org",
    ):
        check(forbidden not in compiler_main, f"active package CLI still presents legacy authority: {forbidden}")
    preview_server = (ROOT / "scripts/dev/registry_serve.py").read_text(encoding="utf-8")
    check("registry+https://registry.sounio.org" not in preview_server, "preview catalog presents itself as a public registry")


def main() -> int:
    print("SOUNIO_SCIENCE_BOUNDARY_GATE_START")
    check(ATTESTOR.is_file() and os.access(ATTESTOR, os.X_OK), "attestor is not executable")
    check(MADAROS.is_file() and SOUC.is_file() and RAW_MADAROS.is_file(), "compiler surfaces are absent")
    assert_raw_ast_transitive_closure()
    with tempfile.TemporaryDirectory(prefix="sounio-science-boundary-") as temporary:
        work = Path(temporary)
        assert_matrix(work / "matrix")
        assert_unknowns(work / "unknown")
        assert_manifest_and_claims(work / "claims")
        assert_receipts_and_elf(work / "receipt")
    assert_registry_quarantine()
    assert_static_contracts()
    print(f"package-boundary-receipt tests={TESTS}")
    print("SOUNIO_SCIENCE_BOUNDARY_GATE_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        print(f"SOUNIO_SCIENCE_BOUNDARY_GATE_FAIL reason={error}", file=sys.stderr)
        raise SystemExit(1)

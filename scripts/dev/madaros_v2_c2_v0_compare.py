#!/usr/bin/env python3
"""Compare two pinned ENIR executions and emit a bounded C2ReceiptV0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROFILE_CONTRACTS = {
    "eisa_v1+dd64_expansion": {"profile": 1, "error_kind": 1, "limbs": 2},
    "eisa_v2+qd128_expansion": {"profile": 2, "error_kind": 2, "limbs": 4},
}
REALISED_FORMATS = {
    (1, 1): "dd64_expansion",
    (2, 2): "qd128_expansion",
}
FULL_FIELDS = [
    "value_bits",
    "correction0_bits",
    "correction1_bits",
    "correction2_bits",
    "correction3_bits",
    "uncertainty_bits",
    "status",
    "gate_class",
    "branch_poisoned",
    "frail_branches",
]
VALUE_FIELDS = ["value_bits"]
REQUIRED_EVENT_FIELDS = {
    "schema",
    "stage",
    "module",
    "module_hash",
    "ordinal",
    "site",
    "value_id",
    "value_bits",
    "error0_bits",
    "error1_bits",
    "uncertainty_bits",
    "status",
    "gate_class",
    "branch_poisoned",
    "frail_branches",
    "source_span",
}


class EvidenceError(Exception):
    pass


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def enir_hash_l64(data: bytes) -> int:
    value = 14695981
    for byte in data:
        value = (value * 257 + byte) % 1000000007
    return value


def parse_expectations(values: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise EvidenceError(f"invalid expectation: {item}")
        key, digest = item.split("=", 1)
        if not key or key in expected or len(digest) != 64:
            raise EvidenceError(f"invalid expectation: {item}")
        expected[key] = digest
    return expected


def parse_artifact(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    lines = data.decode("ascii").splitlines()
    if not lines:
        raise EvidenceError(f"empty ENIR artifact: {path}")
    header = lines[0].split("|")
    if len(header) != 5 or header[0] != "enir":
        raise EvidenceError(f"malformed ENIR header: {path}")
    type_rows = [line.split("|") for line in lines if line.startswith("type|0|")]
    if len(type_rows) != 1 or len(type_rows[0]) < 4:
        raise EvidenceError(f"missing canonical ENIR type row: {path}")
    values: dict[int, int] = {}
    provenance: dict[int, int] = {}
    gate_ops: list[tuple[int, int]] = []
    observations: list[tuple[int, str, int, int]] = []
    end_rows: list[list[str]] = []
    for line in lines[1:]:
        parts = line.split("|")
        if parts[0] == "value":
            if len(parts) != 14:
                raise EvidenceError(f"malformed canonical ENIR value row: {path}")
            value_id = int(parts[1])
            if value_id in values:
                raise EvidenceError(f"duplicate canonical ENIR value id: {path}")
            values[value_id] = int(parts[13])
        elif parts[0] == "prov":
            if len(parts) != 7:
                raise EvidenceError(f"malformed canonical ENIR provenance row: {path}")
            provenance_id = int(parts[1])
            if provenance_id in provenance:
                raise EvidenceError(f"duplicate canonical ENIR provenance id: {path}")
            provenance[provenance_id] = int(parts[2])
        elif parts[0] == "op":
            if len(parts) != 11:
                raise EvidenceError(f"malformed canonical ENIR op row: {path}")
            if int(parts[2]) == 7:
                gate_ops.append((int(parts[1]), int(parts[5])))
        elif parts[0] == "obs":
            if len(parts) != 5:
                raise EvidenceError(f"malformed canonical ENIR observation row: {path}")
            observations.append((int(parts[1]), parts[2], int(parts[3]), int(parts[4])))
        elif parts[0] == "end2":
            end_rows.append(parts)
    if len(end_rows) != 1 or len(end_rows[0]) != 11:
        raise EvidenceError(f"missing canonical ENIR end2 row: {path}")
    declared_observation_count = int(end_rows[0][9])
    if not gate_ops or len(gate_ops) != len(observations) or len(observations) != declared_observation_count:
        raise EvidenceError(f"gate, observation, and end2 counts disagree: {path}")
    declared_events: list[tuple[int, int, int, int]] = []
    for index, (observation_id, observation_module, ordinal, kind) in enumerate(observations):
        if observation_id != index or ordinal != index or observation_module != header[3] or kind != 0:
            raise EvidenceError(f"noncanonical ENIR gate observation manifest: {path}")
        site, value_id = gate_ops[index]
        provenance_id = values.get(value_id)
        if provenance_id is None or provenance_id not in provenance:
            raise EvidenceError(f"gate value lacks canonical provenance: {path}")
        declared_events.append((ordinal, site, provenance[provenance_id], value_id))
    return {
        "sha256": sha256_path(path),
        "module_hash_l64": enir_hash_l64(data),
        "schema": int(header[1]),
        "stage": int(header[2]),
        "module": header[3],
        "profile": int(header[4]),
        "error_kind": int(type_rows[0][3]),
        "declared_observation_count": declared_observation_count,
        "declared_events": declared_events,
    }


def parse_fields(line: str, path: Path) -> tuple[str, dict[str, str]]:
    parts = line.split("|")
    tag = parts[0]
    fields: dict[str, str] = {}
    for item in parts[1:]:
        if item.count("=") != 1:
            raise EvidenceError(f"malformed trace field in {path}: {item!r}")
        key, value = item.split("=", 1)
        if not key or not value or key in fields:
            raise EvidenceError(f"invalid trace field in {path}: {item!r}")
        fields[key] = value
    return tag, fields


def parse_trace(path: Path, limb_count: int) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    completion: dict[str, str] | None = None
    trace_module: str | None = None
    trace_module_hash: int | None = None
    for line in path.read_text(encoding="ascii").splitlines():
        tag, fields = parse_fields(line, path)
        if tag == "enir-exec":
            missing = sorted(REQUIRED_EVENT_FIELDS - fields.keys())
            if missing:
                raise EvidenceError(f"trace event omitted {','.join(missing)}: {path}")
            if fields["schema"] != "2" or fields["stage"] != "2":
                raise EvidenceError(f"trace event has unexpected schema or stage: {path}")
            event_module = fields["module"]
            event_module_hash = int(fields["module_hash"])
            if trace_module is None:
                trace_module = event_module
                trace_module_hash = event_module_hash
            elif event_module != trace_module or event_module_hash != trace_module_hash:
                raise EvidenceError(f"trace events do not share one module identity: {path}")
            corrections: list[str] = [fields["error0_bits"], fields["error1_bits"]]
            for index in (2, 3):
                key = f"error{index}_bits"
                if index < limb_count:
                    if key not in fields:
                        raise EvidenceError(f"trace event omitted {key}: {path}")
                    corrections.append(fields[key])
                else:
                    if key in fields:
                        raise EvidenceError(f"trace event unexpectedly contains {key}: {path}")
                    corrections.append("not_applicable")
            events.append(
                {
                    "ordinal": int(fields["ordinal"]),
                    "site": int(fields["site"]),
                    "source_span": int(fields["source_span"]),
                    "value_id": int(fields["value_id"]),
                    "value_bits": fields["value_bits"],
                    "correction_bits": corrections,
                    "uncertainty_bits": fields["uncertainty_bits"],
                    "status": fields["status"],
                    "gate_class": fields["gate_class"],
                    "branch_poisoned": fields["branch_poisoned"],
                    "frail_branches": fields["frail_branches"],
                }
            )
        elif tag == "enir-exec-ok":
            if completion is not None:
                raise EvidenceError(f"duplicate completion receipt: {path}")
            completion = fields
        else:
            raise EvidenceError(f"unexpected trace row {tag!r}: {path}")
    if completion is None:
        raise EvidenceError(f"trace lacks completion receipt: {path}")
    if completion.get("module") != trace_module or int(completion.get("module_hash", "-1")) != trace_module_hash:
        raise EvidenceError(f"trace completion does not bind its event module: {path}")
    if int(completion.get("observations", "-1")) != len(events):
        raise EvidenceError(f"trace observation count mismatch: {path}")
    ordinals = [event["ordinal"] for event in events]
    if ordinals != list(range(len(events))):
        raise EvidenceError(f"trace ordinals are not canonical: {path}")
    return {
        "sha256": sha256_path(path),
        "events": events,
        "completion": completion,
        "module": trace_module,
        "module_hash_l64": trace_module_hash,
    }


def identity_key(event: dict[str, Any]) -> tuple[int, int, int, int]:
    return event["ordinal"], event["site"], event["source_span"], event["value_id"]


def projected(event: dict[str, Any], projection: str) -> dict[str, str]:
    out = {"value_bits": event["value_bits"]}
    if projection == "value-bits-only":
        return out
    for index, value in enumerate(event["correction_bits"]):
        out[f"correction{index}_bits"] = value
    for key in ("uncertainty_bits", "status", "gate_class", "branch_poisoned", "frail_branches"):
        out[key] = event[key]
    return out


def field_diff(left: dict[str, str], right: dict[str, str], projection: str) -> str | None:
    fields = VALUE_FIELDS if projection == "value-bits-only" else FULL_FIELDS
    for field in fields:
        a = left[field]
        b = right[field]
        if a == b:
            continue
        # A profile without the extra correction limb has no value there. A
        # realised +0 limb on the other profile is not a numerical divergence.
        if field in ("correction2_bits", "correction3_bits") and {a, b} <= {"not_applicable", "0"}:
            continue
        return field
    return None


def base_receipt(args: argparse.Namespace, identities: dict[str, Any]) -> dict[str, Any]:
    observed = VALUE_FIELDS if args.projection == "value-bits-only" else FULL_FIELDS
    blind_spots = [
        "non-gate intermediate operation values",
        "optimizer and production codegen transformations",
        "native ISA and hardware execution",
        "physical, biological, psychiatric, and clinical interpretation",
    ]
    if args.projection == "value-bits-only":
        blind_spots[:0] = [
            "dd64 and qd128 expansion correction limbs",
            "uncertainty, status, gate class, and branch state",
        ]
    evidence_binding = {
        "inputs": identities,
        "compiler_revisions": [args.compiler_revision_a, args.compiler_revision_b],
        "environments": [args.environment_a, args.environment_b],
        "requested_semantics": [args.requested_a, args.requested_b],
        "run_statuses": [args.run_a_status, args.run_b_status],
        "observation_projection": args.projection,
    }
    receipt: dict[str, Any] = {
        "receipt_version": "C2ReceiptV0",
        "evidence_identity": canonical_hash(evidence_binding),
        "source_identity": {
            "run_a_sha256": identities["source_a"],
            "run_b_sha256": identities["source_b"],
        },
        "compiler_identity": {
            "run_a_revision": args.compiler_revision_a,
            "run_b_revision": args.compiler_revision_b,
            "run_a_sha256": identities["compiler_a"],
            "run_b_sha256": identities["compiler_b"],
        },
        "toolchain_identity": {
            "seed_sha256": identities["seed"],
            "build_lock_sha256": identities["build_lock"],
            "comparator_sha256": identities["comparator"],
        },
        "environment_identity": {"run_a": args.environment_a, "run_b": args.environment_b},
        "observation_projection": args.projection,
        "intervention_dimensions": ["numeric representation and precision"],
        "observed_fields": observed,
        "blind_spots": blind_spots,
        "alignment_status": "NOT_ATTEMPTED",
        "first_divergence": None,
        "comparison_status": "BLOCKED",
        "classification_basis": "classification not completed",
        "integrity_status": {"status": "UNCHECKED", "checks": []},
        "run_a": {
            "requested_semantics": args.requested_a,
            "realised_semantics": None,
            "transformation_path": "lower-v1 -> ENIR interpreter",
            "fallback_path": "unknown until integrity verification",
            "artifact_identity": identities["artifact_a"],
            "trace_identity": identities["trace_a"],
            "execution_status": "UNTRUSTED",
        },
        "run_b": {
            "requested_semantics": args.requested_b,
            "realised_semantics": None,
            "transformation_path": "lower-v2 -> ENIR interpreter",
            "fallback_path": "unknown until integrity verification",
            "artifact_identity": identities["artifact_b"],
            "trace_identity": identities["trace_b"],
            "execution_status": "UNTRUSTED",
        },
        "claim_scope": (
            "Pinned ENIR interpreter comparison only; dd64 and qd128 are expansion arithmetic. "
            "No IEEE f128/f256 support, native hardware result, or physical/clinical claim."
        ),
    }
    return receipt


def run_record(
    requested: str,
    artifact: dict[str, Any] | None,
    artifact_sha: str,
    trace_sha: str,
    status: int,
    command: str,
) -> dict[str, Any]:
    return {
        "requested_semantics": requested,
        "realised_semantics": (
            {
                "profile": artifact["profile"],
                "error_kind": artifact["error_kind"],
                "format": REALISED_FORMATS.get((artifact["profile"], artifact["error_kind"]), "unknown"),
            }
            if artifact is not None
            else None
        ),
        "transformation_path": command,
        "fallback_path": "none",
        "artifact_identity": artifact_sha,
        "trace_identity": trace_sha,
        "execution_status": "EXECUTED" if status == 0 else f"FAILED({status})",
    }


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="ascii")
    temporary.replace(path)


def compare(args: argparse.Namespace) -> int:
    paths = {
        "source_a": Path(args.source_a),
        "source_b": Path(args.source_b),
        "compiler_a": Path(args.compiler_a),
        "compiler_b": Path(args.compiler_b),
        "artifact_a": Path(args.artifact_a),
        "artifact_b": Path(args.artifact_b),
        "trace_a": Path(args.trace_a),
        "trace_b": Path(args.trace_b),
        "seed": Path(args.seed),
        "build_lock": Path(args.build_lock),
        "comparator": Path(args.comparator),
    }
    try:
        identities = {key: sha256_path(path) for key, path in paths.items()}
        expected = parse_expectations(args.expect)
    except (OSError, EvidenceError) as error:
        receipt = {
            "receipt_version": "C2ReceiptV0",
            "comparison_status": "BLOCKED",
            "alignment_status": "NOT_ATTEMPTED",
            "integrity_status": {"status": "FAILED", "checks": [], "reason": str(error)},
            "claim_scope": "No comparison claim: evidence inputs could not be bound.",
        }
        write_receipt(Path(args.receipt), receipt)
        return 2

    receipt = base_receipt(args, identities)
    required_expectations = set(identities)
    missing_expectations = sorted(required_expectations - set(expected))
    unexpected_expectations = sorted(set(expected) - required_expectations)
    comparator_mismatch = Path(args.comparator).resolve() != Path(__file__).resolve()
    if missing_expectations or unexpected_expectations or comparator_mismatch:
        receipt["integrity_status"] = {
            "status": "FAILED",
            "checks": [],
            "missing_expectations": missing_expectations,
            "unexpected_expectations": unexpected_expectations,
            "comparator_path_matches_execution": not comparator_mismatch,
        }
        receipt["comparison_status"] = "BLOCKED"
        receipt["classification_basis"] = "integrity expectations do not exactly cover every bound file identity"
        write_receipt(Path(args.receipt), receipt)
        return 2
    checks: list[dict[str, str]] = []
    failed_checks: list[str] = []
    for key, digest in expected.items():
        actual = identities.get(key)
        outcome = "pass" if actual == digest else "fail"
        checks.append({"input": key, "expected_sha256": digest, "actual_sha256": actual or "missing", "result": outcome})
        if outcome == "fail":
            failed_checks.append(key)
    if failed_checks:
        receipt["integrity_status"] = {"status": "FAILED", "checks": checks, "failed_inputs": failed_checks}
        receipt["comparison_status"] = "BLOCKED"
        receipt["classification_basis"] = "evidence identity mismatch"
        write_receipt(Path(args.receipt), receipt)
        return 2
    receipt["integrity_status"] = {"status": "VERIFIED", "checks": checks}

    artifact_a: dict[str, Any] | None = None
    artifact_b: dict[str, Any] | None = None
    try:
        artifact_a = parse_artifact(paths["artifact_a"])
        artifact_b = parse_artifact(paths["artifact_b"])
        contract_a = PROFILE_CONTRACTS[args.requested_a]
        contract_b = PROFILE_CONTRACTS[args.requested_b]
    except (OSError, ValueError, KeyError, EvidenceError) as error:
        receipt["comparison_status"] = "BLOCKED"
        receipt["classification_basis"] = f"artifact/profile evidence invalid: {error}"
        write_receipt(Path(args.receipt), receipt)
        return 0

    receipt["run_a"] = run_record(
        args.requested_a, artifact_a, identities["artifact_a"], identities["trace_a"],
        args.run_a_status, "lower-v1 -> ENIR interpreter",
    )
    receipt["run_b"] = run_record(
        args.requested_b, artifact_b, identities["artifact_b"], identities["trace_b"],
        args.run_b_status, "lower-v2 -> ENIR interpreter",
    )

    controlled = (
        identities["source_a"] == identities["source_b"]
        and identities["compiler_a"] == identities["compiler_b"]
        and args.compiler_revision_a == args.compiler_revision_b
        and args.environment_a == args.environment_b
    )
    if not controlled:
        receipt["comparison_status"] = "INCOMPARABLE"
        receipt["classification_basis"] = "an undeclared source, compiler, revision, or environment dimension differs"
        write_receipt(Path(args.receipt), receipt)
        return 0

    realised_a = (
        artifact_a["stage"] == 2
        and artifact_a["profile"] == contract_a["profile"]
        and artifact_a["error_kind"] == contract_a["error_kind"]
    )
    realised_b = (
        artifact_b["stage"] == 2
        and artifact_b["profile"] == contract_b["profile"]
        and artifact_b["error_kind"] == contract_b["error_kind"]
    )
    if args.run_a_status != 0 or args.run_b_status != 0 or not realised_a or not realised_b:
        receipt["comparison_status"] = "BLOCKED"
        receipt["classification_basis"] = "a requested run failed or did not realise its declared expansion-arithmetic profile"
        write_receipt(Path(args.receipt), receipt)
        return 0

    try:
        trace_a = parse_trace(paths["trace_a"], contract_a["limbs"])
        trace_b = parse_trace(paths["trace_b"], contract_b["limbs"])
    except (OSError, ValueError, EvidenceError) as error:
        receipt["comparison_status"] = "BLOCKED"
        receipt["classification_basis"] = f"trace evidence invalid: {error}"
        write_receipt(Path(args.receipt), receipt)
        return 0
    receipt["run_a"]["execution_summary"] = trace_a["completion"]
    receipt["run_b"]["execution_summary"] = trace_b["completion"]
    trace_artifact_bound = (
        trace_a["module"] == artifact_a["module"]
        and trace_a["module_hash_l64"] == artifact_a["module_hash_l64"]
        and trace_b["module"] == artifact_b["module"]
        and trace_b["module_hash_l64"] == artifact_b["module_hash_l64"]
    )
    if not trace_artifact_bound:
        receipt["comparison_status"] = "BLOCKED"
        receipt["classification_basis"] = "an execution trace is not bound to its canonical ENIR artifact"
        write_receipt(Path(args.receipt), receipt)
        return 0

    keys_a = [identity_key(event) for event in trace_a["events"]]
    keys_b = [identity_key(event) for event in trace_b["events"]]
    if keys_a != artifact_a["declared_events"] or keys_b != artifact_b["declared_events"]:
        receipt["alignment_status"] = "UNALIGNED"
        receipt["comparison_status"] = "UNALIGNED"
        receipt["classification_basis"] = "trace identities do not equal the artifact-declared gate/provenance manifest"
        write_receipt(Path(args.receipt), receipt)
        return 0

    if keys_a != keys_b:
        receipt["alignment_status"] = "UNALIGNED"
        receipt["comparison_status"] = "UNALIGNED"
        receipt["classification_basis"] = "gate events do not have a one-to-one (ordinal, site, source_span, value_id) identity"
        write_receipt(Path(args.receipt), receipt)
        return 0

    receipt["alignment_status"] = "ALIGNED"
    for event_a, event_b in zip(trace_a["events"], trace_b["events"]):
        projected_a = projected(event_a, args.projection)
        projected_b = projected(event_b, args.projection)
        different_field = field_diff(projected_a, projected_b, args.projection)
        if different_field is None:
            continue
        receipt["comparison_status"] = "DIVERGED"
        receipt["classification_basis"] = f"first aligned observed field difference: {different_field}"
        receipt["first_divergence"] = {
            "stage": "ENIR interpreter",
            "operation_identity": {
                "ordinal": event_a["ordinal"],
                "site": event_a["site"],
                "source_span": event_a["source_span"],
                "value_id": event_a["value_id"],
            },
            "differing_field": different_field,
            "run_a": projected_a,
            "run_b": projected_b,
        }
        write_receipt(Path(args.receipt), receipt)
        return 0

    receipt["comparison_status"] = "OBSERVED_EQUIVALENT"
    receipt["classification_basis"] = "no aligned event differs under the declared bounded projection"
    write_receipt(Path(args.receipt), receipt)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-a", "source-b", "compiler-a", "compiler-b",
        "artifact-a", "artifact-b", "trace-a", "trace-b",
        "seed", "build-lock", "comparator",
    ):
        result.add_argument(f"--{name}", required=True)
    result.add_argument("--requested-a", choices=sorted(PROFILE_CONTRACTS), required=True)
    result.add_argument("--requested-b", choices=sorted(PROFILE_CONTRACTS), required=True)
    result.add_argument("--compiler-revision-a", required=True)
    result.add_argument("--compiler-revision-b", required=True)
    result.add_argument("--environment-a", required=True)
    result.add_argument("--environment-b", required=True)
    result.add_argument("--run-a-status", type=int, default=0)
    result.add_argument("--run-b-status", type=int, default=0)
    result.add_argument(
        "--projection", choices=("full-epistemic", "value-bits-only"),
        default="full-epistemic",
    )
    result.add_argument("--expect", action="append", default=[], metavar="NAME=SHA256")
    result.add_argument("--receipt", required=True)
    return result


def main() -> int:
    try:
        return compare(parser().parse_args())
    except EvidenceError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())

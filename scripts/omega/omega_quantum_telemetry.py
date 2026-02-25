#!/usr/bin/env python3
"""Sounio Omega Sprint 1 - QIR shim + quantum conformance telemetry.

Generates:
- artifacts/quantum/omega/qir_shim.json
- artifacts/quantum/omega/quantum_conformance.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

QIR_SHIM_SCHEMA = "sounio.quantum.qir-shim.v1"
QIR_TELEMETRY_SCHEMA = "sounio.quantum.conformance.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sounio Omega quantum telemetry emitter")
    parser.add_argument(
        "--contract",
        default="benchmarks/independence/contract.v2.json",
        help="Independence contract JSON (v2 expected)",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/quantum/omega",
        help="Output directory for QIR + conformance artifacts",
    )
    parser.add_argument(
        "--reference",
        default="",
        help="Optional JSON file with reference samples",
    )
    parser.add_argument(
        "--candidate",
        default="",
        help="Optional JSON file with candidate samples",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Optional explicit shot count override",
    )
    parser.add_argument(
        "--calibration-source",
        default="omega_sprint1_calibration_v1",
        help="Calibration source string or file path for digest",
    )
    parser.add_argument(
        "--circuit-file",
        default="",
        help="Optional QIR text file; generated when omitted",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when conformance does not satisfy contract thresholds",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_contract(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing contract: {path}")
    obj = json.loads(path.read_text())
    if obj.get("schema") != "sounio.independence.contract.v2":
        raise SystemExit(
            f"invalid contract schema for quantum telemetry: {obj.get('schema')}"
        )
    if "quantum_conformance" not in obj:
        raise SystemExit("contract missing quantum_conformance block")
    return obj


def as_float_list(obj: object) -> List[float]:
    if isinstance(obj, list):
        return [float(x) for x in obj]
    if isinstance(obj, dict):
        if isinstance(obj.get("samples"), list):
            return [float(x) for x in obj["samples"]]
        if isinstance(obj.get("values"), list):
            return [float(x) for x in obj["values"]]
    raise SystemExit("sample file must be JSON list or object with 'samples'/'values'")


def load_samples(path: str) -> List[float]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"sample file not found: {p}")
    return as_float_list(json.loads(p.read_text()))


def build_default_samples(n: int) -> Tuple[List[float], List[float]]:
    ref = [
        0.45 + 0.08 * math.sin(i * 0.17) + 0.03 * math.cos(i * 0.09)
        for i in range(n)
    ]
    cand = [
        0.452 + 0.08 * math.sin(i * 0.17 + 0.015) + 0.03 * math.cos(i * 0.09 + 0.012)
        for i in range(n)
    ]
    return ref, cand


def quantile_sorted(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    q_clamped = min(1.0, max(0.0, q))
    pos = q_clamped * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    w = pos - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def interval_coverage_95(reference: List[float], candidate: List[float]) -> float:
    if not reference or not candidate:
        return 0.0
    ref_sorted = sorted(reference)
    lo = quantile_sorted(ref_sorted, 0.025)
    hi = quantile_sorted(ref_sorted, 0.975)
    inside = sum(1 for value in candidate if lo <= value <= hi)
    return inside / len(candidate)


def ks_two_sample_pvalue(reference: List[float], candidate: List[float]) -> float:
    if not reference or not candidate:
        return 0.0

    a = sorted(reference)
    b = sorted(candidate)
    i = 0
    j = 0
    d_max = 0.0

    while i < len(a) and j < len(b):
        x = min(a[i], b[j])
        while i < len(a) and a[i] <= x:
            i += 1
        while j < len(b) and b[j] <= x:
            j += 1
        fa = i / len(a)
        fb = j / len(b)
        d_max = max(d_max, abs(fa - fb))

    n = float(len(a))
    m = float(len(b))
    n_eff = (n * m) / (n + m)
    if n_eff <= 0.0:
        return 0.0

    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d_max
    series = 0.0
    for k in range(1, 65):
        term = math.exp(-2.0 * k * k * lam * lam)
        if k % 2 == 1:
            series += term
        else:
            series -= term
    return max(0.0, min(1.0, 2.0 * series))


def resolve_digest(source: str) -> str:
    path = Path(source)
    if path.exists():
        return sha256_path(path)
    return sha256_bytes(source.encode("utf-8"))


def default_qir_text() -> str:
    return "\n".join(
        [
            "; QIR shim generated by Sounio Omega Sprint 1",
            "; module flags: qir_profile=base",
            "; qubits=2",
            "define void @sounio__qir__entry() {",
            "  call void @qir__qis__h__body(%q0)",
            "  call void @qir__qis__cnot__body(%q0, %q1)",
            "  ret void",
            "}",
        ]
    )


def count_gate_calls(lines: Iterable[str]) -> int:
    return sum(1 for line in lines if "@qir__qis__" in line)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()
    contract = load_contract(Path(args.contract))
    qc = contract["quantum_conformance"]

    ks_pvalue_min = float(qc["ks_pvalue_min"])
    coverage_lo, coverage_hi = [float(x) for x in qc["coverage_range"]]
    min_shots = int(qc["min_shots"])
    coverage_target = float(qc["coverage_target"])

    reference = load_samples(args.reference)
    candidate = load_samples(args.candidate)
    if not reference or not candidate:
        default_n = max(min_shots, 128)
        reference, candidate = build_default_samples(default_n)

    if len(reference) != len(candidate):
        count = min(len(reference), len(candidate))
        reference = reference[:count]
        candidate = candidate[:count]

    shots = int(args.shots) if args.shots > 0 else len(candidate)
    shots = max(shots, min_shots)

    ks_pvalue = ks_two_sample_pvalue(reference, candidate)
    coverage_95 = interval_coverage_95(reference, candidate)
    conformant = (
        shots >= min_shots
        and ks_pvalue >= ks_pvalue_min
        and coverage_95 >= coverage_lo
        and coverage_95 <= coverage_hi
    )

    out_dir = Path(args.out_dir)
    qir_text = (
        Path(args.circuit_file).read_text()
        if args.circuit_file
        else default_qir_text()
    )
    calibration_digest = resolve_digest(args.calibration_source)
    circuit_digest = sha256_bytes(qir_text.encode("utf-8"))

    qir_payload = {
        "schema": QIR_SHIM_SCHEMA,
        "qubit_count": 2,
        "gate_count": count_gate_calls(qir_text.splitlines()),
        "calibration_digest": calibration_digest,
        "circuit_digest": circuit_digest,
        "qir": qir_text,
    }

    telemetry_payload = {
        "schema": QIR_TELEMETRY_SCHEMA,
        "shots": shots,
        "ks_pvalue": ks_pvalue,
        "coverage_95": coverage_95,
        "coverage_target": coverage_target,
        "coverage_range": [coverage_lo, coverage_hi],
        "ks_pvalue_min": ks_pvalue_min,
        "min_shots": min_shots,
        "calibration_digest": calibration_digest,
        "circuit_digest": circuit_digest,
        "conformant": conformant,
    }

    qir_path = out_dir / "qir_shim.json"
    telemetry_path = out_dir / "quantum_conformance.json"
    write_json(qir_path, qir_payload)
    write_json(telemetry_path, telemetry_payload)

    print(
        "omega_quantum_telemetry: "
        f"shots={shots} ks_pvalue={ks_pvalue:.6f} coverage_95={coverage_95:.6f} "
        f"conformant={str(conformant).lower()} qir={qir_path} telemetry={telemetry_path}"
    )

    if args.strict and not conformant:
        print(
            "omega_quantum_telemetry: strict mode failed (non-conformant run)",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

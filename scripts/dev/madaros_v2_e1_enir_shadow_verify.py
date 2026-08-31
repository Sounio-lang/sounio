#!/usr/bin/env python3
"""Independent E1 ENIR verifier and METRON corpus-manifest derivation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SECTION_ORDER = {"enir": 0, "type": 1, "value": 2, "prov": 3, "policy": 4, "op": 5, "obs": 6, "end": 7}
FIELD_COUNTS = {"enir": 5, "type": 8, "value": 14, "prov": 7, "policy": 5, "op": 11, "obs": 5, "end": 8}
NUMERIC_FIELDS = {
    "enir": (1, 2, 4),
    "type": tuple(range(1, 8)),
    "value": tuple(range(1, 14)),
    "prov": tuple(range(1, 7)),
    "policy": (1, 3, 4),
    "op": tuple(range(1, 11)),
    "obs": (1, 3, 4),
    "end": tuple(range(1, 8)),
}
FUEL_PROGRAMS = {"v1_fuel", "v1_fuel_high", "v2_fuel"}


class VerificationError(ValueError):
    pass


def fail(message: str) -> None:
    raise VerificationError(message)


def parse_i64(token: str) -> int:
    if not re.fullmatch(r"-?(0|[1-9][0-9]*)", token):
        fail(f"non-canonical integer: {token!r}")
    value = int(token)
    if not -(2**63) <= value < 2**63:
        fail(f"integer outside i64: {token}")
    return value


def parse_artifact(raw: bytes) -> dict[str, Any]:
    if not raw or not raw.endswith(b"\n") or b"\r" in raw or b"\x00" in raw:
        fail("artifact must be nonempty LF text with one final newline")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        fail(f"artifact is not ASCII: {exc}")
    rows: list[tuple[str, list[str], list[int | None]]] = []
    prior_section = -1
    for line_no, line in enumerate(text[:-1].split("\n"), 1):
        fields = line.split("|")
        tag = fields[0]
        if tag not in SECTION_ORDER:
            fail(f"line {line_no}: unknown tag {tag!r}")
        if len(fields) != FIELD_COUNTS[tag] or any(not field for field in fields):
            fail(f"line {line_no}: wrong field shape for {tag}")
        if any(any(ord(ch) < 33 or ord(ch) > 126 for ch in field) for field in fields):
            fail(f"line {line_no}: invalid field byte")
        section = SECTION_ORDER[tag]
        if section < prior_section or (tag in {"enir", "end"} and section == prior_section):
            fail(f"line {line_no}: non-canonical section order")
        prior_section = section
        ints: list[int | None] = [None] * len(fields)
        for index in NUMERIC_FIELDS[tag]:
            ints[index] = parse_i64(fields[index])
        rows.append((tag, fields, ints))
    if not rows or rows[0][0] != "enir" or rows[-1][0] != "end":
        fail("header/end missing or misplaced")
    return {"raw": raw, "text": text, "rows": rows}


def rows_for(parsed: dict[str, Any], tag: str) -> list[tuple[list[str], list[int | None]]]:
    return [(fields, ints) for row_tag, fields, ints in parsed["rows"] if row_tag == tag]


def iv(ints: list[int | None], index: int) -> int:
    value = ints[index]
    assert value is not None
    return value


def f64_class(bits: int) -> int:
    unsigned = bits & ((1 << 64) - 1)
    sign = unsigned >> 63
    exponent = (unsigned >> 52) & 0x7FF
    fraction = unsigned & ((1 << 52) - 1)
    if exponent == 0:
        return sign if fraction == 0 else 2
    if exponent == 0x7FF:
        if fraction == 0:
            return 4 + sign
        return 6 if fraction & (1 << 51) else 7
    return 3


def verify_semantics(parsed: dict[str, Any]) -> dict[str, Any]:
    header_fields, header = rows_for(parsed, "enir")[0]
    if (iv(header, 1), iv(header, 2), header_fields[3], iv(header, 4)) != (1, 1, "eisa_metron_shadow", 2):
        fail("unexpected E1 header")

    types = rows_for(parsed, "type")
    values = rows_for(parsed, "value")
    provenance = rows_for(parsed, "prov")
    policies = rows_for(parsed, "policy")
    ops = rows_for(parsed, "op")
    observations = rows_for(parsed, "obs")
    footer = rows_for(parsed, "end")[0][1]
    counts = (len(types), len(values), len(provenance), len(policies), len(ops), len(observations))
    if tuple(iv(footer, i) for i in range(1, 7)) != counts:
        fail("footer counts do not match parsed sections")
    if counts != (3, 10, 10, 1, 16, 39) or iv(footer, 7) != 30:
        fail(f"E1 exact counts violated: sections={counts} programs={iv(footer, 7)}")

    type_by_id: dict[int, tuple[int, int, int, int]] = {}
    for ordinal, (_, row) in enumerate(types):
        if iv(row, 1) != ordinal or iv(row, 2) != 4:
            fail("type IDs/value kind must be canonical f64")
        error, uncertainty, status, prov, profile = (iv(row, i) for i in range(3, 8))
        if error not in range(4) or uncertainty not in range(5) or status != 1 or prov != 1 or profile not in range(4):
            fail(f"invalid type semantics at type {ordinal}")
        type_by_id[ordinal] = (error, uncertainty, status, profile)
    if {(v[0], v[1], v[3]) for v in type_by_id.values()} != {(0, 0, 2), (1, 1, 1), (2, 1, 2)}:
        fail("exact/DD64/QD128 and exact/GUM1 profile axes are incomplete")

    policy_ids = set()
    for ordinal, (fields, row) in enumerate(policies):
        if (iv(row, 1), fields[2], iv(row, 3), iv(row, 4)) != (ordinal, "eisa_10_100_count_only", 0, 0):
            fail("invalid EISA policy")
        policy_ids.add(ordinal)

    provenance_ids = set()
    for ordinal, (_, row) in enumerate(provenance):
        if iv(row, 1) != ordinal or iv(row, 2) < 0 or iv(row, 4) < 0:
            fail(f"invalid provenance {ordinal}")
        if iv(row, 3) < -1 or iv(row, 5) < -1 or iv(row, 6) < -1:
            fail(f"invalid optional provenance field {ordinal}")
        if iv(row, 6) >= 0 and iv(row, 6) not in policy_ids:
            fail(f"unresolved provenance policy {ordinal}")
        provenance_ids.add(ordinal)

    value_by_id: dict[int, int] = {}
    known_by_id: dict[int, int] = {}
    for ordinal, (_, row) in enumerate(values):
        value_id, type_id, known = iv(row, 1), iv(row, 2), iv(row, 3)
        if value_id != ordinal or type_id not in type_by_id or iv(row, 13) not in provenance_ids or known not in {0, 1}:
            fail(f"invalid value identity/reference {ordinal}")
        bits, fp_class = iv(row, 4), iv(row, 5)
        errors = tuple(iv(row, i) for i in range(6, 10))
        uncertainty_bits, status, reason = iv(row, 10), iv(row, 11), iv(row, 12)
        if status not in range(4) or (status == 0) != (reason == -1):
            fail(f"invalid status/reason at value {ordinal}")
        if known and (fp_class not in range(8) or fp_class != f64_class(bits)):
            fail(f"known value has invalid or non-derived IEEE class at value {ordinal}")
        if not known and (bits != 0 or fp_class != -1 or any(errors) or uncertainty_bits != 0):
            fail(f"unknown value carries fabricated static payload at value {ordinal}")
        error_kind, uncertainty_kind, _, _ = type_by_id[type_id]
        if error_kind == 0 and any(errors):
            fail(f"exact value carries error payload at value {ordinal}")
        if error_kind == 1 and any(errors[2:]):
            fail(f"DD64 value carries QD128-only limbs at value {ordinal}")
        if uncertainty_kind == 0 and uncertainty_bits != 0:
            fail(f"exact value carries uncertainty at value {ordinal}")
        value_by_id[value_id] = type_id
        known_by_id[value_id] = known

    produced: dict[int, int] = {}
    opcode_set = set()
    for ordinal, (_, row) in enumerate(ops):
        op_id, opcode, result, type_id, a, b, target, policy, slot, fuel = (iv(row, i) for i in range(1, 11))
        if op_id != ordinal or opcode not in range(14) or fuel != 1:
            fail(f"invalid operation identity/opcode/fuel at op {ordinal}")
        opcode_set.add(opcode)
        produces = opcode in {0, 1, 2, 3, 4, 5, 6, 9}
        if produces:
            if result not in value_by_id or result in produced or type_id != value_by_id[result]:
                fail(f"invalid SSA result at op {ordinal}")
            produced[result] = type_id
        elif result != -1 or type_id != -1:
            fail(f"non-producing op has result/type at op {ordinal}")
        if opcode == 0 and (a != -1 or b != -1 or not known_by_id[result]):
            fail(f"invalid econst at op {ordinal}")
        if opcode == 1 and slot < 0:
            fail(f"invalid eload at op {ordinal}")
        if opcode in {2, 3, 4, 5} and (a not in produced or b not in produced):
            fail(f"binary op violates def-before-use at op {ordinal}")
        if opcode in {6, 7, 8, 9, 11, 12} and a not in produced:
            fail(f"unary/control op violates def-before-use at op {ordinal}")
        if opcode == 8 and slot < 0:
            fail(f"invalid estore at op {ordinal}")
        if opcode in {10, 11, 12} and target not in range(len(ops)):
            fail(f"invalid branch target at op {ordinal}")
        if opcode in {7, 11, 12} and policy not in policy_ids:
            fail(f"invalid policy reference at op {ordinal}")
        if opcode in {11, 12} and type_by_id[produced[a]][1] not in {0, 1}:
            fail(f"branch consumes non-branchable uncertainty at op {ordinal}")
    if opcode_set != set(range(14)) or set(produced) != set(value_by_id):
        fail("opcode coverage or declared/produced SSA set is incomplete")

    manifest: list[tuple[str, int, int]] = []
    seen_programs: set[str] = set()
    previous = None
    expected_ordinal = 0
    for ordinal, (fields, row) in enumerate(observations):
        obs_id, name, obs_ordinal, kind = iv(row, 1), fields[2], iv(row, 3), iv(row, 4)
        if obs_id != ordinal or kind not in {0, 1}:
            fail(f"invalid observation {ordinal}")
        if name != previous:
            if name in seen_programs:
                fail(f"non-contiguous observation group {name}")
            seen_programs.add(name)
            expected_ordinal = 0
        if obs_ordinal != expected_ordinal:
            fail(f"invalid observation ordinal for {name}")
        manifest.append((name, obs_ordinal, kind))
        previous, expected_ordinal = name, expected_ordinal + 1
    return {"counts": counts, "program_count": len(seen_programs), "manifest": manifest, "opcodes": sorted(opcode_set)}


def function_body(source: str, name: str) -> str:
    match = re.search(rf"\bfn\s+{re.escape(name)}\s*\([^)]*\)[^{{]*\{{", source)
    if not match:
        fail(f"corpus function not found: {name}")
    start = match.end()
    depth = 1
    index = start
    in_string = False
    escaped = False
    while index < len(source):
        ch = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start:index]
        index += 1
    fail(f"unterminated corpus function: {name}")
    raise AssertionError


def derive_corpus_manifest(corpus_path: Path) -> list[tuple[str, int, int]]:
    source = corpus_path.read_text(encoding="utf-8")
    main = function_body(source, "main")
    calls = re.findall(r"run_(one|img)\(\s*([A-Za-z0-9_]+)\(\)\s*\)", main)
    if len(calls) != 30:
        fail(f"METRON corpus must call exactly 30 programs, found {len(calls)}")
    manifest: list[tuple[str, int, int]] = []
    for runner, function in calls:
        body = function_body(source, function)
        if runner == "one":
            name = function.removesuffix("_src")
            count = len(re.findall(r'add_line\s*\([^\n]*"gate\s', body))
            kind = 0
        else:
            name = function.removesuffix("_img")
            kind = 1 if name in FUEL_PROGRAMS else 0
            if "rump_build(" in body:
                count = len(re.findall(r"\bop_egate\s*\(", function_body(source, "rump_build")))
            else:
                count = len(re.findall(r"\bop_egate\s*\(", body))
            if kind == 1 and count == 0:
                count = 1
        if count <= 0:
            fail(f"program {name} has no source-derived gate/fuel observation")
        manifest.extend((name, ordinal, kind) for ordinal in range(count))
    if len(manifest) != 39:
        fail(f"METRON corpus must derive exactly 39 observations, found {len(manifest)}")
    return manifest


def rolling_hash(raw: bytes) -> int:
    value = 14695981
    for byte in raw:
        value = (value * 257 + byte) % 1_000_000_007
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--native-receipt", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    raw = args.artifact.read_bytes()
    parsed = parse_artifact(raw)
    verified = verify_semantics(parsed)
    corpus_manifest = derive_corpus_manifest(args.corpus)
    if verified["manifest"] != corpus_manifest:
        expected = Counter(name for name, _, _ in corpus_manifest)
        actual = Counter(name for name, _, _ in verified["manifest"])
        fail(f"artifact/corpus manifest mismatch: expected={dict(expected)} actual={dict(actual)}")

    l64 = rolling_hash(raw)
    native_receipt = ""
    if args.native_receipt:
        native_receipt = args.native_receipt.read_text(encoding="ascii").strip()
        match = re.fullmatch(
            r"enir-shadow-ok\|hash=(\d+)\|types=3\|values=10\|provenance=10\|policies=1\|ops=16\|programs=30\|observations=39",
            native_receipt,
        )
        if not match or int(match.group(1)) != l64:
            fail("native receipt does not agree with independent hash/counts")

    program_counts = Counter(name for name, _, _ in corpus_manifest)
    receipt = {
        "schema": "madaros.v2.e1.enir-shadow-receipt/1",
        "shadow_only": True,
        "compiler_owned_native_model": True,
        "canonical_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_l64": l64,
        "canonical_bytes": len(raw),
        "canonical_lines": len(parsed["rows"]),
        "counts": {"types": 3, "values": 10, "provenance": 10, "policies": 1, "ops": 16, "programs": 30, "observations": 39},
        "opcodes": verified["opcodes"],
        "numeric_axes": ["exact", "dd64", "qd128", "gum1", "ieee754_class", "status", "provenance"],
        "corpus_source": str(args.corpus),
        "program_observation_counts": dict(program_counts),
        "native_receipt": native_receipt,
        "fallback_used": False,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.write_text(rendered, encoding="utf-8")
    print(f"E1_ENIR_SHADOW_VERIFY_PASS sha256={receipt['canonical_sha256']} l64={l64} programs=30 observations=39")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, VerificationError) as exc:
        print(f"E1_ENIR_SHADOW_VERIFY_FAIL: {exc}")
        raise SystemExit(1)

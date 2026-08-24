#!/usr/bin/env bash
# epistemic_shape_ratchet_gate.sh — freeze the live Epistemic / Knowledge
# value shapes. A new shape, or a vanished frozen shape, fails.
#
# Why this exists
# ---------------
# The same noun denotes structurally different objects. stdlib Epistemic is
# {val, variance, confidence}; four dissertation/example files re-declare
# Epistemic as {value, variance, label}. label:0 is "measured" (strongest
# provenance). confidence:0 is the weakest possible claim. No diagnostic
# separates them. Which form is canonical is a founder ruling. Until that
# ruling, a further form must not appear.
#
# What is frozen
# --------------
# The SET of field-signature shapes, not the declaration count. A new file
# that repeats a frozen shape is allowed. A new field list is not. Changing
# the only site of a frozen shape is a lost shape and also fails.
#
# What is matched
# ---------------
# Line-anchored `struct Epistemic`, `struct Knowledge`, `struct KnowledgeTypeInfo`
# (optional `pub`, optional `<...>`). Word boundary after the name, so
# EpistemicOrderedMap / EpistemicNeuralNetwork / KnowledgeARIMA /
# KnowledgeConstraint are NOT this noun. Those are other types.
#
# What is excluded, and why
# -------------------------
# archive/              — historical evolution, not live code
# bootstrap/            — C→Sounio bootstrap chain, frozen
# self-hosted/bootstrap/ — the frozen self-hosted seed (same class)
#
# Usage
#   bash scripts/ci/epistemic_shape_ratchet_gate.sh
#   bash scripts/ci/epistemic_shape_ratchet_gate.sh --self-test
#   bash scripts/ci/epistemic_shape_ratchet_gate.sh --write-reference
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REF="${EPISTEMIC_SHAPE_RATCHET_REF:-$ROOT/scripts/ci/epistemic_shape_ratchet.tsv}"
ARTIFACT="${EPISTEMIC_SHAPE_RATCHET_ARTIFACT:-/tmp/epistemic_shape_ratchet.json}"
MODE=check

while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) MODE=selftest; shift ;;
    --write-reference) MODE=write; shift ;;
    --ref) REF="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v python3 >/dev/null 2>&1 || {
  echo "EPISTEMIC_SHAPE_RATCHET_FAIL reason=python3_missing" >&2
  exit 1
}

export ROOT REF ARTIFACT MODE
python3 - <<'PY'
import json
import os
import pathlib
import re
import sys
import tempfile

root = pathlib.Path(os.environ["ROOT"])
ref_path = pathlib.Path(os.environ["REF"])
artifact = pathlib.Path(os.environ["ARTIFACT"])
mode = os.environ["MODE"]

# Exact names that denote the epistemic *value*. Prefix Knowledge* without
# this closed set would freeze KnowledgeARIMA / KnowledgeConstraint / …
NAME_RE = re.compile(
    r"^(?P<indent>\s*)(?P<vis>pub\s+)?struct\s+"
    r"(?P<name>Epistemic|Knowledge|KnowledgeTypeInfo)"
    r"(?P<gen><[^;{]*>)?\s*\{",
    re.M,
)
FIELD_RE = re.compile(
    r"(?:pub\s+)?(?P<fname>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<ftype>[^,\n]+)"
)
SKIP_PREFIXES = (
    "archive/",
    "bootstrap/",
    "self-hosted/bootstrap/",
)


def skip_path(rel: str) -> bool:
    return any(rel == p[:-1] or rel.startswith(p) for p in SKIP_PREFIXES)


def strip_line_comment(line: str) -> str:
    if "//" in line:
        return line[: line.index("//")]
    return line


def parse_fields(after_brace: str) -> list[str]:
    depth = 1
    i = 0
    while i < len(after_brace) and depth:
        ch = after_brace[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                after_brace = after_brace[:i]
                break
        i += 1
    cleaned = []
    for line in after_brace.splitlines():
        cleaned.append(strip_line_comment(line))
    text = "\n".join(cleaned)
    fields = []
    for m in FIELD_RE.finditer(text):
        fname = m.group("fname")
        if fname in {"pub", "struct", "fn", "let", "var"}:
            continue
        ftype = re.sub(r"\s+", "", m.group("ftype").strip().rstrip(","))
        if not ftype:
            continue
        fields.append(f"{fname}:{ftype}")
    return fields


def scan(base: pathlib.Path, extra_files=None):
    hits = []
    for p in sorted(base.rglob("*.sio")):
        rel = p.relative_to(base).as_posix()
        if skip_path(rel):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in NAME_RE.finditer(text):
            # Do not match inside a line comment.
            line_start = text.rfind("\n", 0, m.start()) + 1
            prefix = text[line_start : m.start()]
            if "//" in prefix:
                continue
            line = text.count("\n", 0, m.start()) + 1
            fields = parse_fields(text[m.end() :])
            shape = ",".join(fields)
            hits.append(
                {
                    "path": rel,
                    "line": line,
                    "name": m.group("name") + (m.group("gen") or "").replace(" ", ""),
                    "shape": shape,
                }
            )
    for p in extra_files or []:
        p = pathlib.Path(p)
        text = p.read_text(encoding="utf-8", errors="replace")
        rel = str(p)
        for m in NAME_RE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            fields = parse_fields(text[m.end() :])
            shape = ",".join(fields)
            hits.append(
                {
                    "path": rel,
                    "line": line,
                    "name": m.group("name") + (m.group("gen") or "").replace(" ", ""),
                    "shape": shape,
                }
            )
    return hits


def load_frozen(path: pathlib.Path) -> list[str]:
    shapes = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        shapes.append(line)
    return shapes


def write_frozen(path: pathlib.Path, shapes: set[str]) -> None:
    lines = [
        "# epistemic_shape_ratchet.tsv — frozen Epistemic/Knowledge/KnowledgeTypeInfo shapes",
        "# One field-signature per line: name:type,name:type,...",
        "# Types have whitespace collapsed. Do not add a shape without a founder ruling.",
        "# Generated by scripts/ci/epistemic_shape_ratchet_gate.sh --write-reference",
    ]
    for s in sorted(shapes):
        lines.append(s)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit(status, total, passed, failed, not_run, extra):
    metrics = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "not_run": not_run,
    }
    payload = {
        "schema": "sounio.epistemic-shape-ratchet-gate.v1",
        "status": status,
        "metrics": metrics,
        **extra,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"status={status}")
    print(
        "metrics "
        f"{{total={total}, passed={passed}, failed={failed}, not_run={not_run}}}"
    )
    print(f"artifact={artifact}")
    if status != "pass":
        raise SystemExit(1)


hits = scan(root)
live_shapes = {h["shape"] for h in hits}

if mode == "write":
    write_frozen(ref_path, live_shapes)
    print(f"wrote {ref_path} shapes={len(live_shapes)} declarations={len(hits)}")
    raise SystemExit(0)

if not ref_path.is_file():
    print(f"EPISTEMIC_SHAPE_RATCHET_FAIL reason=reference_missing path={ref_path}", file=sys.stderr)
    emit("fail", 1, 0, 1, 0, {"reason": "reference_missing"})

frozen = set(load_frozen(ref_path))
new_shapes = sorted(live_shapes - frozen)
lost_shapes = sorted(frozen - live_shapes)

print(f"EPISTEMIC_SHAPE_RATCHET_SCAN declarations={len(hits)} shapes={len(live_shapes)} frozen={len(frozen)}")
for h in hits:
    print(f"  site {h['path']}:{h['line']} {h['name']} shape={h['shape']}")

failed_msgs = []
if new_shapes:
    for s in new_shapes:
        sites = [h for h in hits if h["shape"] == s]
        where = ", ".join(f"{h['path']}:{h['line']}" for h in sites)
        failed_msgs.append(f"new_shape file={where} shape={s}")
        print(f"EPISTEMIC_SHAPE_RATCHET_FAIL new_shape file={where} shape={s}", file=sys.stderr)
if lost_shapes:
    for s in lost_shapes:
        failed_msgs.append(f"lost_shape shape={s}")
        print(f"EPISTEMIC_SHAPE_RATCHET_FAIL lost_shape shape={s}", file=sys.stderr)

# Declaration-level: each live site whose shape is frozen is a pass.
decl_pass = sum(1 for h in hits if h["shape"] in frozen)
decl_fail = sum(1 for h in hits if h["shape"] not in frozen)

if mode == "selftest":
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="epi-shape-ratchet-"))
    novel = tmp / "seventh_novel_shape.sio"
    frozen_copy = tmp / "repeat_frozen_shape.sio"
    novel.write_text(
        "struct Epistemic {\n    value: f64,\n    mystery: i64,\n}\n",
        encoding="utf-8",
    )
    frozen_copy.write_text(
        "struct Epistemic {\n    val: f64,\n    variance: f64,\n    confidence: i64,\n}\n",
        encoding="utf-8",
    )
    novel_hits = scan(root, extra_files=[novel])
    novel_shapes = {h["shape"] for h in novel_hits} - frozen
    novel_ok = any(
        h["path"] == str(novel) and h["shape"] == "value:f64,mystery:i64"
        for h in novel_hits
    ) and "value:f64,mystery:i64" in novel_shapes
    repeat_hits = scan(root, extra_files=[frozen_copy])
    repeat_new = {h["shape"] for h in repeat_hits} - frozen
    repeat_ok = (
        any(h["path"] == str(frozen_copy) and h["shape"] == "val:f64,variance:f64,confidence:i64" for h in repeat_hits)
        and not repeat_new
    )
    print(
        "EPISTEMIC_SHAPE_RATCHET_SELFTEST "
        f"positive_new_shape_refused={str(novel_ok).lower()} "
        f"positive_file={novel} positive_shape=value:f64,mystery:i64 "
        f"negative_frozen_shape_allowed={str(repeat_ok).lower()} "
        f"negative_file={frozen_copy}"
    )
    if not novel_ok:
        failed_msgs.append("selftest_positive_did_not_refuse")
        print("EPISTEMIC_SHAPE_RATCHET_FAIL selftest_positive_did_not_refuse", file=sys.stderr)
    if not repeat_ok:
        failed_msgs.append("selftest_negative_did_not_allow_frozen_shape")
        print("EPISTEMIC_SHAPE_RATCHET_FAIL selftest_negative_did_not_allow_frozen_shape", file=sys.stderr)
    # Two extra checks.
    extra_pass = int(novel_ok) + int(repeat_ok)
    extra_fail = 2 - extra_pass
    total = len(hits) + 2
    passed = decl_pass + extra_pass
    failed = decl_fail + extra_fail + len(lost_shapes)
    status = "pass" if not failed_msgs and failed == 0 else "fail"
    emit(
        status,
        total,
        passed,
        failed,
        0,
        {
            "declarations": hits,
            "frozen_shapes": sorted(frozen),
            "new_shapes": new_shapes,
            "lost_shapes": lost_shapes,
            "selftest": {
                "positive_new_shape_refused": novel_ok,
                "positive_file": str(novel),
                "positive_shape": "value:f64,mystery:i64",
                "negative_frozen_shape_allowed": repeat_ok,
                "negative_file": str(frozen_copy),
            },
            "failures": failed_msgs,
        },
    )
    raise SystemExit(0 if status == "pass" else 1)

total = len(hits)
passed = decl_pass
failed = decl_fail + len(lost_shapes)
status = "pass" if not failed_msgs and failed == 0 else "fail"
emit(
    status,
    total,
    passed,
    failed,
    0,
    {
        "declarations": hits,
        "frozen_shapes": sorted(frozen),
        "new_shapes": new_shapes,
        "lost_shapes": lost_shapes,
        "failures": failed_msgs,
    },
)
PY

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOADER="self-hosted/compiler/module_loader.sio"

fail() {
  printf '[soir-v4-thinlink-cache] FAIL: %s\n' "$*" >&2
  exit 1
}

[[ $# -eq 0 ]] || fail "unexpected argument: $1"
[[ -f "$LOADER" ]] || fail "missing module loader: $LOADER"
command -v python3 >/dev/null 2>&1 || fail python3_missing

python3 - "$LOADER" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")


def fail(reason: str) -> None:
    raise AssertionError(reason)


def function_body(name: str) -> str:
    match = re.search(r"(?m)^(?:pub\s+)?fn\s+" + re.escape(name) + r"\s*\(", source)
    if match is None:
        fail(f"missing_function_{name}")

    start = source.find("{", match.end())
    if start < 0:
        fail(f"missing_body_{name}")

    depth = 0
    in_string = False
    escaped = False
    line_comment = False
    block_comment = False
    pos = start
    while pos < len(source):
        char = source[pos]
        next_char = source[pos + 1] if pos + 1 < len(source) else ""

        if line_comment:
            if char == "\n":
                line_comment = False
            pos += 1
            continue
        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                pos += 2
            else:
                pos += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            pos += 1
            continue
        if char == "/" and next_char == "/":
            line_comment = True
            pos += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            pos += 2
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : pos + 1]
        pos += 1

    fail(f"unterminated_function_{name}")
    return ""


def require(condition: bool, reason: str) -> None:
    if not condition:
        fail(reason)


try:
    receipt = function_body("thin_emit_ir_cache_quarantine_receipt")
    require(
        "[thinlink-ir-cache] schema=1 format=SOIR-v4 operation=" in receipt,
        "quarantine_receipt_prefix_missing",
    )
    require("status=quarantined authority=0" in receipt, "quarantine_authority_not_zero")
    require("reason=semantic-roundtrip-unproven" in receipt, "quarantine_reason_missing")

    ir_write = function_body("thin_try_write_ir_cache")
    require(
        'thin_emit_ir_cache_quarantine_receipt("write", path)' in ir_write,
        "ir_write_receipt_missing",
    )
    require(re.search(r"\bfalse\s*\}$", ir_write) is not None, "ir_write_not_unconditionally_false")
    for forbidden in ("serialize_ir_module(", "write_file(", "io_write_"):
        require(forbidden not in ir_write, f"ir_write_performs_io_{forbidden}")

    for name in ("thin_try_read_ir_cache", "thin_try_read_ir_cache_no_stats"):
        ir_read = function_body(name)
        require(
            'thin_emit_ir_cache_quarantine_receipt("read", path)' in ir_read,
            f"{name}_receipt_missing",
        )
        require("(false, ir_empty_module())" in ir_read, f"{name}_miss_result_missing")
        require("(true," not in ir_read, f"{name}_can_return_hit")
        for forbidden in ("thin_read_small_file(", "deserialize_ir_module(", "read_file("):
            require(forbidden not in ir_read, f"{name}_performs_io_{forbidden}")

    require("deserialize_ir_module" not in source, "v4_deserializer_still_reachable")
    require(
        "use ir::serialize::{serialize_ir_module}" in source,
        "non_cache_serializer_import_changed",
    )

    read_mode = function_body("thin_cache_mode_can_read")
    write_mode = function_body("thin_cache_mode_can_write")
    require('str_eq(cache_mode, "read")' in read_mode, "read_mode_missing")
    require('str_eq(cache_mode, "rw")' in read_mode, "rw_read_mode_missing")
    require('str_eq(cache_mode, "write")' in write_mode, "write_mode_missing")
    require('str_eq(cache_mode, "rw")' in write_mode, "rw_write_mode_missing")

    summary = function_body("thin_load_or_build_module_summary")
    unit = function_body("thin_build_compiled_unit")
    ir_callers = summary + "\n" + unit
    require(ir_callers.count("thin_try_read_ir_cache(") == 3, "unexpected_ir_read_call_count")
    require(ir_callers.count("thin_try_write_ir_cache(") == 3, "unexpected_ir_write_call_count")

    read_guard = re.compile(
        r"if\s+str_len\(cache_dir\)\s*>\s*0\s*&&\s*"
        r"thin_cache_mode_can_read\(cache_mode\)\s*\{\s*"
        r"let\s+[A-Za-z0-9_]+\s*=\s*thin_try_read_ir_cache\(",
        re.S,
    )
    write_guard = re.compile(
        r"if\s+str_len\(cache_dir\)\s*>\s*0\s*&&\s*"
        r"thin_cache_mode_can_write\(cache_mode\)\s*\{\s*"
        r"let\s+_\s*=\s*thin_try_write_ir_cache\(",
        re.S,
    )
    require(len(read_guard.findall(ir_callers)) == 3, "ir_read_call_bypasses_cache_mode")
    require(len(write_guard.findall(ir_callers)) == 3, "ir_write_call_bypasses_cache_mode")

    soir_kinds = re.findall(
        r'thin_cache_path\(cache_dir,\s*"(summary|body|object)",\s*[^,\n]+,\s*"\.soir"\)',
        ir_callers,
    )
    require(sorted(soir_kinds) == ["body", "object", "summary"], "unexpected_persistent_ir_cache_paths")

    binary_write = function_body("thin_try_write_binary_cache")
    binary_read = function_body("thin_try_read_binary_cache")
    require("io_write_huge_binary(path, bytes, len)" in binary_write, "binary_write_disabled")
    require("thin_read_small_file(path)" in binary_read, "binary_read_disabled")
    require("(true, read_pair.1, read_pair.2)" in binary_read, "binary_hit_result_disabled")
    require("thin_emit_ir_cache_quarantine_receipt" not in binary_write, "binary_write_quarantined")
    require("thin_emit_ir_cache_quarantine_receipt" not in binary_read, "binary_read_quarantined")

    compiler = function_body("compile_multimodule_native_advanced")
    require(
        'thin_cache_path(cache_dir, "link", link_digest, ".elf")' in compiler,
        "binary_link_cache_path_missing",
    )
    require("thin_try_read_binary_cache(link_cache_path" in compiler, "binary_link_read_missing")
    require("thin_try_write_binary_cache(link_cache_path" in compiler, "binary_link_write_missing")
except AssertionError as exc:
    print(f"[soir-v4-thinlink-cache] FAIL: source_contract_{exc}", file=sys.stderr)
    raise SystemExit(1)
PY

printf '%s\n' \
  'SOIR_V4_THINLINK_CACHE_QUARANTINE_RECEIPT status=pass evidence=source-contract modes=read,write,rw ir_read=always-miss ir_write=disabled source_rebuild=required binary_cache=preserved authority=0'
printf '%s\n' \
  'SOIR_V4_THINLINK_CACHE_QUARANTINE_BOUNDARY semantic_roundtrip=unproven runtime=not-run v5_v6_compatibility=not-claimed'
printf '%s\n' 'SOIR_V4_THINLINK_CACHE_QUARANTINE_PASS'

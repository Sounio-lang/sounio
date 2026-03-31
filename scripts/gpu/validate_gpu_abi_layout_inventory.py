#!/usr/bin/env python3
import json
from pathlib import Path
import sys

ALLOWED_SUPPORT_CLASSES = {
    "gpu-surface-supported",
    "gpu-lowering-supported",
    "gpu-compile-proof",
    "gpu-sim-runtime-supported",
    "gpu-hardware-runtime-supported",
    "gpu-explicit-unsupported",
}

ALLOWED_STAGES = {
    "surface",
    "lowering",
    "descriptor-runtime",
    "sim-reference",
    "hardware-info",
}

REQUIRED_ENTRY_IDS = {
    "public-launch-surface",
    "public-intrinsics-explicit-unsupported",
    "kernel-param-lowering",
    "ptx-runtime-launch-1d",
    "sim-reference-launch-descriptor",
    "hardware-runtime-attestation",
}


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_string(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} must be a non-empty string")
    return value.strip()


def require_string_list(value, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        fail(f"{label} must be a non-empty list")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            fail(f"{label}[{index}] must be a non-empty string")
        result.append(item.strip())
    return result


def main() -> int:
    if len(sys.argv) != 3:
        fail("usage: validate_gpu_abi_layout_inventory.py <inventory.json> <repo_root>")

    inventory_path = Path(sys.argv[1])
    repo_root = Path(sys.argv[2]).resolve()
    if not inventory_path.is_file():
        fail(f"missing inventory: {inventory_path}")

    obj = json.loads(inventory_path.read_text(encoding="utf-8"))
    if obj.get("schema") != "sounio.gpu.abi_layout_inventory.v1":
        fail(f"unexpected schema in {inventory_path}")

    required_support_classes = obj.get("required_support_classes")
    if not isinstance(required_support_classes, list):
        fail("required_support_classes must be a list")
    if set(required_support_classes) != ALLOWED_SUPPORT_CLASSES:
        fail("required_support_classes must match the canonical GPU support classes exactly")

    entries = obj.get("entries")
    if not isinstance(entries, list) or not entries:
        fail("entries must be a non-empty list")

    seen_ids = set()
    seen_support_classes = set()

    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            fail(f"entries[{entry_index}] must be an object")

        entry_id = require_string(entry.get("entry_id"), f"entries[{entry_index}].entry_id")
        if entry_id in seen_ids:
            fail(f"duplicate entry_id {entry_id}")
        seen_ids.add(entry_id)

        stage = require_string(entry.get("stage"), f"entries[{entry_index}].stage")
        if stage not in ALLOWED_STAGES:
            fail(f"entries[{entry_index}].stage has unknown value {stage}")

        support_class = require_string(
            entry.get("support_class"), f"entries[{entry_index}].support_class"
        )
        if support_class not in ALLOWED_SUPPORT_CLASSES:
            fail(f"entries[{entry_index}].support_class has unknown value {support_class}")
        seen_support_classes.add(support_class)

        require_string(entry.get("summary"), f"entries[{entry_index}].summary")
        require_string_list(
            entry.get("descriptor_fields"), f"entries[{entry_index}].descriptor_fields"
        )
        require_string_list(entry.get("evidence_paths"), f"entries[{entry_index}].evidence_paths")
        require_string_list(entry.get("notes"), f"entries[{entry_index}].notes")

        symbol_refs = entry.get("symbol_refs")
        if not isinstance(symbol_refs, list) or not symbol_refs:
            fail(f"entries[{entry_index}].symbol_refs must be a non-empty list")

        for ref_index, ref in enumerate(symbol_refs):
            if not isinstance(ref, dict):
                fail(f"entries[{entry_index}].symbol_refs[{ref_index}] must be an object")
            rel_path = require_string(
                ref.get("path"),
                f"entries[{entry_index}].symbol_refs[{ref_index}].path",
            )
            symbols = require_string_list(
                ref.get("symbols"),
                f"entries[{entry_index}].symbol_refs[{ref_index}].symbols",
            )
            target = repo_root / rel_path
            if not target.is_file():
                fail(f"entries[{entry_index}].symbol_refs[{ref_index}] missing file {rel_path}")
            content = target.read_text(encoding="utf-8", errors="replace")
            for symbol in symbols:
                if symbol not in content:
                    fail(
                        f"entries[{entry_index}].symbol_refs[{ref_index}] missing symbol "
                        f"{symbol!r} in {rel_path}"
                    )

        for evidence_index, evidence_path in enumerate(entry["evidence_paths"]):
            target = repo_root / evidence_path
            if not target.is_file():
                fail(
                    f"entries[{entry_index}].evidence_paths[{evidence_index}] missing file "
                    f"{evidence_path}"
                )

    if seen_ids != REQUIRED_ENTRY_IDS:
        missing = sorted(REQUIRED_ENTRY_IDS - seen_ids)
        extra = sorted(seen_ids - REQUIRED_ENTRY_IDS)
        fail(f"inventory entry set mismatch missing={missing} extra={extra}")

    if seen_support_classes != ALLOWED_SUPPORT_CLASSES:
        missing = sorted(ALLOWED_SUPPORT_CLASSES - seen_support_classes)
        fail(f"inventory must cover every support class; missing={missing}")

    print(
        "gpu abi inventory ok: "
        f"path={inventory_path} entries={len(entries)} support_classes={len(seen_support_classes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

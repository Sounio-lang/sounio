#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ALL_SUPPORT_CLASSES = {
    "gpu-surface-supported",
    "gpu-lowering-supported",
    "gpu-compile-proof",
    "gpu-sim-runtime-supported",
    "gpu-hardware-runtime-supported",
    "gpu-explicit-unsupported",
}

KIND_SPECS = {
    "surface-lowering": {
        "header": [
            "case_id",
            "support_class",
            "mode",
            "program",
            "expected_marker",
        ],
        "support_classes": {
            "gpu-surface-supported",
            "gpu-lowering-supported",
            "gpu-explicit-unsupported",
        },
        "modes": {"check-pass", "build-pass", "check-fail"},
        "path_field": "program",
    },
    "compile-proof": {
        "header": [
            "case_id",
            "support_class",
            "mode",
            "program",
            "expected_marker",
        ],
        "support_classes": {
            "gpu-surface-supported",
            "gpu-compile-proof",
        },
        "modes": {"check-pass", "check-fail"},
        "path_field": "program",
    },
    "sim-runtime": {
        "header": [
            "case_id",
            "support_class",
            "program",
            "expected_exit",
            "expected_stdout",
        ],
        "support_classes": {"gpu-sim-runtime-supported"},
        "path_field": "program",
    },
}


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_rows(manifest_path: Path):
    rows = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if not rows:
                rows.append(("__header__", line[1:].strip().split("\t")))
            continue
        rows.append(("row", raw_line.split("\t")))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=sorted(KIND_SPECS.keys()))
    parser.add_argument("manifest")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root = Path(args.root).resolve()
    if not manifest_path.is_file():
        fail(f"missing manifest: {manifest_path}")

    spec = KIND_SPECS[args.kind]
    rows = parse_rows(manifest_path)
    if not rows:
        fail(f"manifest has no content: {manifest_path}")
    if rows[0][0] != "__header__":
        fail(f"manifest {manifest_path} must begin with a tab-separated header comment")

    header = rows[0][1]
    if header != spec["header"]:
        fail(
            f"manifest {manifest_path} header mismatch: expected "
            + "\t".join(spec["header"])
        )

    seen_case_ids = set()
    for index, row in enumerate(rows[1:], start=2):
        if row[0] != "row":
            continue
        values = row[1]
        if len(values) != len(header):
            fail(f"{manifest_path}:{index}: expected {len(header)} columns, got {len(values)}")
        item = dict(zip(header, values))

        case_id = item["case_id"].strip()
        if not case_id:
            fail(f"{manifest_path}:{index}: case_id must not be empty")
        if case_id in seen_case_ids:
            fail(f"{manifest_path}:{index}: duplicate case_id {case_id}")
        seen_case_ids.add(case_id)

        support_class = item["support_class"].strip()
        if support_class not in ALL_SUPPORT_CLASSES:
            fail(f"{manifest_path}:{index}: unknown support_class {support_class}")
        if support_class not in spec["support_classes"]:
            fail(
                f"{manifest_path}:{index}: support_class {support_class} is not allowed "
                f"for kind {args.kind}"
            )

        if "modes" in spec:
            mode = item["mode"].strip()
            if mode not in spec["modes"]:
                fail(f"{manifest_path}:{index}: invalid mode {mode}")

        path_field = spec["path_field"]
        path_value = item[path_field].strip()
        if not path_value:
            fail(f"{manifest_path}:{index}: {path_field} must not be empty")
        if not (root / path_value).is_file():
            fail(f"{manifest_path}:{index}: missing file {path_value}")

        if args.kind == "sim-runtime":
            try:
                int(item["expected_exit"])
            except ValueError:
                fail(f"{manifest_path}:{index}: expected_exit must be an integer")

    print(f"manifest ok: kind={args.kind} path={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

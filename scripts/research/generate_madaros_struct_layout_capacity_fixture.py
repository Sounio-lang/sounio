#!/usr/bin/env python3
"""Generate an imported field-access fixture at a precise layout count."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom-layouts", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = args.custom_layouts
    if count < 1 or count > 256:
        raise SystemExit("--custom-layouts must be between 1 and 256")

    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    library_lines = ["//@ run-pass", ""]
    for index in range(count):
        library_lines.extend(
            [
                f"pub struct Layout{index} {{",
                f"    field{index}: i64,",
                "}",
                "",
            ]
        )

    final_index = count - 1
    witness_value = 900000 + count
    total_layouts = count + 1  # lowerer_new_with_epistemic pre-registers Knowledge.
    marker = (
        "PASS struct_layout_capacity "
        f"custom_layouts={count} catalog_layouts={total_layouts}"
    )
    main_lines = [
        "//@ run-pass",
        "",
        f"use layout_capacity_lib::{{Layout{final_index}}}",
        "",
        "fn main() -> i64 with IO {",
        f"    let witness = Layout{final_index} {{ field{final_index}: {witness_value} }}",
        f"    if witness.field{final_index} != {witness_value} {{ return 7 }}",
        f'    println("{marker}")',
        "    0",
        "}",
        "",
    ]

    (output_dir / "layout_capacity_lib.sio").write_text(
        "\n".join(library_lines), encoding="ascii"
    )
    (output_dir / "layout_capacity_main.sio").write_text(
        "\n".join(main_lines), encoding="ascii"
    )
    (output_dir / "expected_marker.txt").write_text(
        marker + "\n", encoding="ascii"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

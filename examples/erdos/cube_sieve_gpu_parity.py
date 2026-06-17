#!/usr/bin/env python3
"""Check cube-propagation parity between the CPU reference and a GPU-lane producer.

The GPU lane is search plumbing only.  This checker deliberately compares the
untrusted backend against the deterministic CPU producer and preserves the
non-promotable manifest boundary.  A real GPU wrapper can be plugged in as any
executable that accepts:

    <edge-file> <k> <cube-file>

and writes a `cube_sieve_propagation_manifest v1` manifest to stdout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from validate_cube_sieve_manifest import ManifestError, validate


DEFAULT_CPU_PRODUCER = Path(__file__).with_name("cube_sieve_propagation_manifest.py")


def canonical_manifest_lines(text: str) -> list[str]:
    validate(text)
    return [line.rstrip() for line in text.splitlines()]


def run_producer(
    command: list[str], edge_file: Path, k: int, cube_file: Path, label: str
) -> tuple[str, list[str]]:
    proc = subprocess.run(
        command + [str(edge_file), str(k), str(cube_file)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no output"
        raise RuntimeError(f"{label} producer failed with exit {proc.returncode}: {detail}")
    try:
        canonical = canonical_manifest_lines(proc.stdout)
    except ManifestError as exc:
        raise RuntimeError(f"{label} producer emitted invalid manifest: {exc}") from exc
    return proc.stdout, canonical


def write_if_requested(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("edge_file", type=Path)
    parser.add_argument("k", type=int)
    parser.add_argument("cube_file", type=Path)
    parser.add_argument(
        "--cpu-producer",
        type=Path,
        default=DEFAULT_CPU_PRODUCER,
        help="reference producer with CLI: edge_file k cube_file",
    )
    parser.add_argument(
        "--backend-producer",
        type=Path,
        default=DEFAULT_CPU_PRODUCER,
        help="GPU-lane producer with the same CLI; defaults to CPU reference for local smoke",
    )
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    if args.k <= 0 or args.k >= 62:
        raise SystemExit("error: k must satisfy 0 < k < 62")
    if not args.edge_file.is_file():
        raise SystemExit(f"error: missing edge file: {args.edge_file}")
    if not args.cube_file.is_file():
        raise SystemExit(f"error: missing cube file: {args.cube_file}")
    if not args.cpu_producer.is_file():
        raise SystemExit(f"error: missing CPU producer: {args.cpu_producer}")
    if not args.backend_producer.is_file():
        raise SystemExit(f"error: missing backend producer: {args.backend_producer}")

    try:
        cpu_text, cpu_lines = run_producer(
            [sys.executable, str(args.cpu_producer)], args.edge_file, args.k, args.cube_file, "cpu"
        )
        backend_text, backend_lines = run_producer(
            [sys.executable, str(args.backend_producer)],
            args.edge_file,
            args.k,
            args.cube_file,
            "backend",
        )
    except RuntimeError as exc:
        print(f"cube_sieve_gpu_parity: FAIL: {exc}", file=sys.stderr)
        return 1

    if args.out_dir is not None:
        write_if_requested(args.out_dir / "cpu.manifest", cpu_text)
        write_if_requested(args.out_dir / "backend.manifest", backend_text)

    if cpu_lines != backend_lines:
        print("cube_sieve_gpu_parity: FAIL: canonical manifest mismatch", file=sys.stderr)
        for i, (a, b) in enumerate(zip(cpu_lines, backend_lines), 1):
            if a != b:
                print(f"first_mismatch_line={i}", file=sys.stderr)
                print(f"cpu={a}", file=sys.stderr)
                print(f"backend={b}", file=sys.stderr)
                break
        if len(cpu_lines) != len(backend_lines):
            print(
                f"cpu_line_count={len(cpu_lines)} backend_line_count={len(backend_lines)}",
                file=sys.stderr,
            )
        return 1

    print("cube_sieve_gpu_parity v1")
    print("trust_boundary=backend_untrusted__cpu_parity_only__drat_lrat_lean_verified_required")
    print(f"edge_file={args.edge_file}")
    print(f"k={args.k}")
    print(f"cube_file={args.cube_file}")
    print(f"cpu_producer={args.cpu_producer}")
    print(f"backend_producer={args.backend_producer}")
    print(f"canonical_line_count={len(cpu_lines)}")
    print("verified_claim=none")
    print("geometry_claim=none")
    print("proof_artifact_sha256=NONE")
    print("promotion_gate=REJECT_NONE_PROOF_ARTIFACT")
    print("promotable=0")
    print("status=GPU_PARITY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

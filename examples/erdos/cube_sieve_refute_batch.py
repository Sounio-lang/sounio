#!/usr/bin/env python3
"""Run souc_sat cube-unit refutations over a batch of cubes.

This is proof-artifact plumbing for graph-colouring search, not a Euclidean
chromatic-number witness. Each cube is turned into original CNF unit clauses by
souc_sat, then the emitted deletion-free DRAT is converted to the repo-local LRAT
format. Real promotion still requires Lean-checked SAT/LRAT plus exact Euclidean
geometry.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from cube_sieve_batch_manifest import parse_batch, sha256_file, write_cube_file
from cube_sieve_propagation_manifest import parse_edge_file


ROOT = Path(__file__).resolve().parents[2]
SOUC_SAT_SRC = Path(__file__).with_name("souc_sat.sio")
DEFAULT_SOUC = ROOT / "artifacts" / "self-hosted" / "souc-self-hosted-x86_64"
DEFAULT_CONVERTER = Path(__file__).with_name("drup_to_lrat_rup.py")


def parse_cnf_header(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="ascii") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if len(parts) == 4 and parts[:2] == ["p", "cnf"]:
                return int(parts[2]), int(parts[3])
            raise RuntimeError(f"{path}: expected p cnf header, got {line!r}")
    raise RuntimeError(f"{path}: missing p cnf header")


def parse_dimacs_units(path: Path) -> set[int]:
    units: set[int] = set()
    pending: list[int] = []
    saw_header = False
    with path.open("r", encoding="ascii") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if parts[0] == "p":
                saw_header = True
                continue
            if not saw_header:
                raise RuntimeError(f"{path}:{lineno}: DIMACS clause before header")
            for tok in parts:
                lit = int(tok)
                if lit == 0:
                    if len(pending) == 1:
                        units.add(pending[0])
                    pending = []
                else:
                    pending.append(lit)
    if pending:
        raise RuntimeError(f"{path}: DIMACS clause missing terminating 0")
    return units


def has_deletions(path: Path) -> bool:
    with path.open("r", encoding="ascii") as f:
        return any(re.match(r"^\s*d\s", line) for line in f)


def lrat_has_empty_clause(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="ascii") as f:
        return any(re.match(r"^\d+\s+0(\s+|$)", line.strip()) for line in f)


def converter_reported_empty(stderr: str) -> bool:
    return re.search(
        r"^wrote .*\(vars=\d+, original=\d+, additions=\d+, deletions=unsupported, empty=1\)$",
        stderr,
        re.MULTILINE,
    ) is not None


def dimacs_lit(v: int, c: int, k: int) -> int:
    return v * k + c + 1


def run_checked(cmd: list[str], cwd: Path, label: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"{label} failed with exit {proc.returncode}: {detail}")
    return proc


def compile_solver(souc: Path, out_dir: Path) -> Path:
    elf = out_dir / "souc_sat.elf"
    run_checked([str(souc), str(SOUC_SAT_SRC), str(elf)], out_dir, "souc_sat compile")
    elf.chmod(0o755)
    return elf


def refute_cube(
    *,
    elf: Path,
    converter: Path,
    edge_file: Path,
    k: int,
    seed: int,
    use_lrb: int,
    sb_mode: int,
    cube_id: str,
    cube: list[tuple[int, int]],
    base_clause_count: int,
    out_dir: Path,
) -> dict[str, str]:
    cube_dir = out_dir / cube_id
    cube_dir.mkdir(parents=True, exist_ok=True)
    cube_path = cube_dir / f"{cube_id}.cube"
    write_cube_file(cube_path, cube)

    proc = run_checked(
        [
            str(elf),
            str(seed),
            str(k),
            str(use_lrb),
            str(sb_mode),
            str(edge_file),
            str(cube_path),
        ],
        cube_dir,
        f"souc_sat cube {cube_id}",
    )
    stdout_path = cube_dir / f"{cube_id}.souc_sat.stdout"
    stderr_path = cube_dir / f"{cube_id}.souc_sat.stderr"
    stdout_path.write_text(proc.stdout, encoding="ascii")
    stderr_path.write_text(proc.stderr, encoding="ascii")
    if "UNSAT" not in proc.stdout:
        raise RuntimeError(f"souc_sat cube {cube_id} exited 0 without UNSAT marker")

    raw_cnf = cube_dir / "souc_sat_worker.cnf"
    raw_drat = cube_dir / "souc_sat_worker.drat"
    if not raw_cnf.is_file() or not raw_drat.is_file():
        raise RuntimeError(f"souc_sat cube {cube_id} did not emit CNF/DRAT artifacts")
    if has_deletions(raw_drat):
        raise RuntimeError(f"souc_sat cube {cube_id} emitted deletion records")

    cnf_path = cube_dir / f"{cube_id}.cnf"
    drat_path = cube_dir / f"{cube_id}.drat"
    lrat_path = cube_dir / f"{cube_id}.lrat"
    shutil.move(str(raw_cnf), cnf_path)
    shutil.move(str(raw_drat), drat_path)
    converter_proc = run_checked(
        [sys.executable, str(converter), str(cnf_path), str(drat_path), str(lrat_path)],
        cube_dir,
        f"RUP-to-LRAT converter cube {cube_id}",
    )
    converter_stdout = cube_dir / f"{cube_id}.converter.stdout"
    converter_stderr = cube_dir / f"{cube_id}.converter.stderr"
    converter_stdout.write_text(converter_proc.stdout, encoding="ascii")
    converter_stderr.write_text(converter_proc.stderr, encoding="ascii")
    if not converter_reported_empty(converter_proc.stderr):
        raise RuntimeError(
            f"converter cube {cube_id} did not report final empty proof: "
            f"{converter_proc.stderr.strip()}"
        )
    if not lrat_has_empty_clause(lrat_path):
        raise RuntimeError(f"converter cube {cube_id} emitted no LRAT empty-clause row")

    cnf_vars, cnf_clauses = parse_cnf_header(cnf_path)
    assignments = ",".join(f"{v}:{c}" for v, c in cube)
    unit_lits = ",".join(str(dimacs_lit(v, c, k)) for v, c in cube)
    cnf_units = parse_dimacs_units(cnf_path)
    missing_units = [str(dimacs_lit(v, c, k)) for v, c in cube if dimacs_lit(v, c, k) not in cnf_units]
    if missing_units:
        raise RuntimeError(f"cube {cube_id} CNF missing unit clauses: {','.join(missing_units)}")
    if len(cnf_units) != len(cube):
        raise RuntimeError(
            f"cube {cube_id} CNF has unexpected extra unit clauses: "
            f"expected {len(cube)}, got {len(cnf_units)}"
        )
    return {
        "assignments": assignments,
        "unit_lits": unit_lits,
        "cube_assignment_count": str(len(cube)),
        "cube": str(cube_path.relative_to(out_dir)),
        "cube_sha256": sha256_file(cube_path),
        "cnf": str(cnf_path.relative_to(out_dir)),
        "cnf_sha256": sha256_file(cnf_path),
        "drat": str(drat_path.relative_to(out_dir)),
        "drat_sha256": sha256_file(drat_path),
        "lrat": str(lrat_path.relative_to(out_dir)),
        "lrat_sha256": sha256_file(lrat_path),
        "stdout_sha256": sha256_file(stdout_path),
        "converter_stderr_sha256": sha256_file(converter_stderr),
        "cnf_vars": str(cnf_vars),
        "cnf_clauses": str(cnf_clauses),
        "expected_cnf_clauses": str(base_clause_count + len(cube)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("edge_file", type=Path)
    parser.add_argument("k", type=int)
    parser.add_argument("cube_batch", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--souc", type=Path, default=DEFAULT_SOUC)
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-lrb", type=int, default=1)
    parser.add_argument(
        "--sb-mode",
        type=int,
        default=0,
        choices=[0],
        help="only plain colourCNF mode is composable in this batch-cover rung",
    )
    args = parser.parse_args()

    if args.k <= 0:
        raise SystemExit("error: k must be positive")
    if args.use_lrb not in (0, 1):
        raise SystemExit("error: --use-lrb must be 0 or 1")
    if not args.edge_file.is_file():
        raise SystemExit(f"error: missing edge file: {args.edge_file}")
    if not args.souc.is_file():
        raise SystemExit(f"error: missing souc compiler: {args.souc}")
    if not args.converter.is_file():
        raise SystemExit(f"error: missing converter: {args.converter}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cubes = parse_batch(args.cube_batch)
        n, m, _edges = parse_edge_file(args.edge_file)
        for cube_id, cube in cubes:
            for v, c in cube:
                if not (0 <= v < n):
                    raise RuntimeError(f"cube {cube_id} vertex out of range: {v}")
                if not (0 <= c < args.k):
                    raise RuntimeError(f"cube {cube_id} colour out of range: {c}")
        expected_vars = n * args.k
        if args.sb_mode != 0:
            raise RuntimeError("cube-cover composition currently supports only --sb-mode=0")
        base_clause_count = n + m * args.k
        elf = compile_solver(args.souc, args.out_dir)
        rows: list[tuple[str, dict[str, str]]] = []
        for cube_id, cube in cubes:
            row = refute_cube(
                elf=elf,
                converter=args.converter,
                edge_file=args.edge_file.resolve(),
                k=args.k,
                seed=args.seed,
                use_lrb=args.use_lrb,
                sb_mode=args.sb_mode,
                cube_id=cube_id,
                cube=cube,
                base_clause_count=base_clause_count,
                out_dir=args.out_dir,
            )
            if row["cnf_vars"] != str(expected_vars):
                raise RuntimeError(
                    f"cube {cube_id} CNF var count {row['cnf_vars']} != expected {expected_vars}"
                )
            if row["cnf_clauses"] != row["expected_cnf_clauses"]:
                raise RuntimeError(
                    f"cube {cube_id} CNF clause count {row['cnf_clauses']} "
                    f"!= expected {row['expected_cnf_clauses']}"
                )
            rows.append((cube_id, row))
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("cube_sieve_refute_batch v1")
    print("trust_boundary=producer_untrusted__lean_lrat_and_cover_required_for_promotion")
    print("output=dimacs_cube_refutation_batch_summary")
    print("formula_kind=colourCNF")
    print(f"edge_path={args.edge_file}")
    print(f"edge_sha256={sha256_file(args.edge_file)}")
    print(f"n={n}")
    print(f"m={m}")
    print(f"k={args.k}")
    print(f"expected_vars={expected_vars}")
    print(f"base_clause_count={base_clause_count}")
    print(f"cube_batch_path={args.cube_batch}")
    print(f"cube_batch_sha256={sha256_file(args.cube_batch)}")
    print(f"out_dir={args.out_dir}")
    print(f"souc_path={args.souc}")
    print(f"souc_sha256={sha256_file(args.souc)}")
    print(f"souc_sat_source={SOUC_SAT_SRC}")
    print(f"souc_sat_source_sha256={sha256_file(SOUC_SAT_SRC)}")
    print(f"souc_sat_elf=souc_sat.elf")
    print(f"souc_sat_elf_sha256={sha256_file(args.out_dir / 'souc_sat.elf')}")
    print(f"converter_path={args.converter}")
    print(f"converter_sha256={sha256_file(args.converter)}")
    print(f"seed={args.seed}")
    print(f"use_lrb={args.use_lrb}")
    print(f"sb_mode={args.sb_mode}")
    print(f"cube_count={len(rows)}")
    print(f"solver_unsat_count={len(rows)}")
    print(f"lrat_artifact_count={len(rows)}")
    print("failed_count=0")
    print("subproblem_artifact=cnf_plus_cube_units_with_deletion_free_drat_and_lrat")
    print("proof_checker=repo_local_rup_to_lrat_converter_only")
    print("formal_proof_checker=none")
    print("verified_claim=none")
    print("global_unsat_claim=none")
    print("geometry_claim=none")
    print("cover_certificate_sha256=NONE")
    print("promotion_gate=REJECT_NONE_CUBE_COVER_CERTIFICATE")
    print("promotable=0")
    for index, (cube_id, fields) in enumerate(rows):
        print(
            "cube "
            f"index={index} id={cube_id} assignments={fields['assignments']} "
            f"unit_lits={fields['unit_lits']} "
            f"cube_assignment_count={fields['cube_assignment_count']} cube={fields['cube']} "
            f"cube_sha256={fields['cube_sha256']} cnf={fields['cnf']} "
            f"cnf_sha256={fields['cnf_sha256']} drat={fields['drat']} "
            f"drat_sha256={fields['drat_sha256']} lrat={fields['lrat']} "
            f"lrat_sha256={fields['lrat_sha256']} cnf_vars={fields['cnf_vars']} "
            f"cnf_clauses={fields['cnf_clauses']} "
            f"expected_cnf_clauses={fields['expected_cnf_clauses']} drat_deletions=0 "
            f"stdout_sha256={fields['stdout_sha256']} "
            f"converter_stderr_sha256={fields['converter_stderr_sha256']}"
        )
    print("status=subproblem_lrat_artifacts_emitted_unpromotable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

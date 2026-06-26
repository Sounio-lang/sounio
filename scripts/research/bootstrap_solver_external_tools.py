#!/usr/bin/env python3
"""Record and optionally prepare external solver/tool availability.

The script is intentionally user-space only. It never invokes apt, sudo, or
system package managers. With --prepare-missing it builds small solver tools
under SOUNIO_SOLVER_TOOL_ROOT, then records the exact executable and source
commit used by the external-corpus pilot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SAT_SOLVERS = ("z3", "cadical", "kissat", "minisat", "glucose", "cryptominisat5")
SMT_SOLVERS = ("z3", "cvc5", "yices-smt2")
OPB_SOLVERS = ("roundingsat", "open-wbo", "open-wbo_static", "scip", "sat4j-pb", "minisat+", "minisatp", "pbsolver")


@dataclass(frozen=True)
class BuildSpec:
    solver: str
    repo_url: str
    commit: str
    exe_rel: tuple[str, ...]
    build: Callable[[Path], None]


@dataclass(frozen=True)
class Tool:
    domain: str
    solver: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], cwd: Path, timeout: int = 900) -> None:
    completed = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed in {cwd}\n{completed.stdout}")


def build_autoconf_make(src: Path) -> None:
    run_cmd(["./configure"], src)
    run_cmd(["make", f"-j{os.cpu_count() or 2}"], src)


def build_roundingsat(src: Path) -> None:
    if (src / "CMakeLists.txt").exists() and shutil.which("cmake"):
        build_dir = src / "build"
        build_dir.mkdir(exist_ok=True)
        try:
            run_cmd(["cmake", "-DCMAKE_BUILD_TYPE=Release", "-Dsoplex=OFF", ".."], build_dir, timeout=900)
            run_cmd(["make", f"-j{os.cpu_count() or 2}"], build_dir, timeout=1800)
            return
        except RuntimeError:
            artifact = src / "build" / "roundingsat"
            url = "https://gitlab.com/MIAOresearch/software/roundingsat/-/jobs/artifacts/master/raw/build/roundingsat?job=build-linux"
            urllib.request.urlretrieve(url, artifact)
            artifact.chmod(0o755)
            (src / "build" / "roundingsat.artifact_source").write_text(url + "\n", encoding="utf-8")
            return
    if (src / "Cargo.toml").exists() and shutil.which("cargo"):
        run_cmd(["cargo", "build", "--release"], src, timeout=1800)
        return
    if (src / "build.sh").exists():
        run_cmd(["bash", "build.sh"], src, timeout=1800)
        return
    if (src / "Makefile").exists():
        run_cmd(["make", f"-j{os.cpu_count() or 2}"], src, timeout=1800)
        return
    raise RuntimeError("roundingsat checkout has no supported build entrypoint")


BUILD_SPECS = {
    "cadical": BuildSpec(
        solver="cadical",
        repo_url="https://github.com/arminbiere/cadical.git",
        commit="7b99c07f0bcab5824a5a3ce62c7066554017f641",
        exe_rel=("build/cadical", "cadical"),
        build=build_autoconf_make,
    ),
    "kissat": BuildSpec(
        solver="kissat",
        repo_url="https://github.com/arminbiere/kissat.git",
        commit="8af8e56f174b778aef3aa45af9f739b2a5f492c2",
        exe_rel=("build/kissat", "kissat"),
        build=build_autoconf_make,
    ),
    "roundingsat": BuildSpec(
        solver="roundingsat",
        repo_url="https://gitlab.com/MIAOresearch/software/roundingsat.git",
        commit="d4edbf7908a9bb951fd181940919e0f3ac7ab1ee",
        exe_rel=("build/roundingsat", "target/release/roundingsat", "roundingsat"),
        build=build_roundingsat,
    ),
}


def git_commit(src: Path) -> str:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def prepare_tool(spec: BuildSpec, tool_root: Path, log_dir: Path) -> None:
    src = tool_root / "src" / spec.solver
    bin_dir = tool_root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"prepare_{spec.solver}.log"
    try:
        if not src.exists():
            run_cmd(["git", "clone", spec.repo_url, str(src)], tool_root, timeout=900)
        current = git_commit(src)
        if current != spec.commit:
            run_cmd(["git", "fetch", "--all", "--tags"], src, timeout=900)
            run_cmd(["git", "checkout", spec.commit], src, timeout=300)
        exe_candidates = [src / rel for rel in spec.exe_rel]
        if not any(path.exists() and os.access(path, os.X_OK) for path in exe_candidates):
            spec.build(src)
        exe = next((path for path in exe_candidates if path.exists() and os.access(path, os.X_OK)), None)
        if exe is None:
            raise RuntimeError(f"{spec.solver}: build completed but executable not found in {spec.exe_rel}")
        link = bin_dir / spec.solver
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(exe, link)
        log_path.write_text(f"status=OK\nrepo={spec.repo_url}\ncommit={git_commit(src)}\nexecutable={exe}\nlink={link}\n", encoding="utf-8")
    except Exception as exc:
        log_path.write_text(f"status=ERROR\nrepo={spec.repo_url}\ncommit={spec.commit}\nerror={exc}\n", encoding="utf-8")


def prepare_python_zstandard(tool_root: Path, log_dir: Path) -> None:
    py_dir = tool_root / "python"
    log_path = log_dir / "prepare_python_zstandard.log"
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import zstandard; print(zstandard.__version__)"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            env={**os.environ, "PYTHONPATH": str(py_dir)},
        )
        if probe.returncode != 0:
            run_cmd([sys.executable, "-m", "pip", "install", "--quiet", "--target", str(py_dir), "zstandard"], tool_root, timeout=900)
            probe = subprocess.run(
                [sys.executable, "-c", "import zstandard; print(zstandard.__version__)"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
                env={**os.environ, "PYTHONPATH": str(py_dir)},
            )
        if probe.returncode != 0:
            raise RuntimeError(probe.stdout)
        log_path.write_text(f"status=OK\npython_path={py_dir}\nzstandard_version={probe.stdout.strip()}\n", encoding="utf-8")
    except Exception as exc:
        log_path.write_text(f"status=ERROR\npython_path={py_dir}\nerror={exc}\n", encoding="utf-8")


def solver_version(path: str | None) -> str:
    if not path:
        return ""
    probes = ([path, "--version"], [path, "-version"], [path, "-h"], [path, "--help"])
    for probe in probes:
        try:
            completed = subprocess.run(probe, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = (completed.stdout + completed.stderr).strip().splitlines()
        if lines:
            return lines[0][:200]
    return "version_unavailable"


def find_tool(name: str, tool_root: Path) -> tuple[str, str]:
    for candidate in (tool_root / "bin" / name, tool_root / name):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate), "user-space-cache"
    path = shutil.which(name)
    if path:
        return path, "path"
    return "", "missing"


def rows(tool_root: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for domain, names in (("sat", SAT_SOLVERS), ("smt", SMT_SOLVERS), ("opb", OPB_SOLVERS)):
        for solver in names:
            key = (domain, solver)
            if key in seen:
                continue
            seen.add(key)
            path, source = find_tool(solver, tool_root)
            digest = ""
            if path:
                try:
                    digest = sha256_file(Path(path))
                except OSError:
                    digest = "sha256_unavailable"
            marker = tool_root / "src" / solver / "build" / f"{solver}.artifact_source"
            if source == "user-space-cache" and marker.exists():
                source = "user-space-cache-prebuilt-artifact"
            elif source == "user-space-cache" and solver in BUILD_SPECS:
                commit = git_commit(tool_root / "src" / solver)
                if commit:
                    digest = commit
            out.append(
                {
                    "domain": domain,
                    "solver": solver,
                    "path": path,
                    "available": "1" if path else "0",
                    "version": solver_version(path),
                    "source": source,
                    "sha256_or_commit": digest,
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for solver_availability.csv")
    parser.add_argument("--tool-root", type=Path, default=None, help="User-space cache root")
    parser.add_argument("--prepare-missing", action="store_true", help="Build cadical, kissat, and roundingsat in the user-space cache when missing")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tool_root = args.tool_root or Path(os.environ.get("SOUNIO_SOLVER_TOOL_ROOT", "~/.cache/sounio/solver-tools")).expanduser()
    out_dir = args.out_dir or Path(f"/tmp/sounio-solver-tools-{stamp}")
    tool_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare_missing:
        prep_rows = rows(tool_root)
        available = {row["solver"] for row in prep_rows if row["available"] == "1"}
        for solver in ("cadical", "kissat", "roundingsat"):
            if solver not in available:
                prepare_tool(BUILD_SPECS[solver], tool_root, out_dir)
        prepare_python_zstandard(tool_root, out_dir)

    csv_path = out_dir / "solver_availability.csv"
    fieldnames = ["domain", "solver", "path", "available", "version", "source", "sha256_or_commit"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows(tool_root))

    manifest = out_dir / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "schema=sounio.solver.external_tools.bootstrap.v1",
                f"timestamp_utc={stamp}",
                f"tool_root={tool_root}",
                f"solver_availability={csv_path}",
                f"prepare_missing={1 if args.prepare_missing else 0}",
                "note=user-space discovery/build only; no apt/root/system install attempted.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(manifest.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

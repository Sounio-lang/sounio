#!/usr/bin/env python3
"""Mechanical vacuous-fixture census for scripts/ci/*_gate.sh.

Positive control: scripts/ci/fixtures/vacuity_positive_control/vacuous_seed_gate.sh
must be classified VACUOUS or this instrument exits 2.
"""
from __future__ import annotations

import os
import re
import sys
from glob import iglob
from pathlib import Path

root = Path(os.environ.get("ROOT_DIR", Path(__file__).resolve().parents[2])).resolve()
out = Path(os.environ.get("OUT_DIR", "/tmp/sounio-vacuous-fixture-sweep"))
out.mkdir(parents=True, exist_ok=True)
seed_rel = "scripts/ci/fixtures/vacuity_positive_control/vacuous_seed_gate.sh"

# Quoted repo-relative paths with known extensions (reject junk like "sort)").
LIT = re.compile(
    r"""['"]((?:tests|scripts|stdlib|examples|benchmarks|formal)/[A-Za-z0-9_./-]+\.(?:sio|sh|tsv|csv|json|md|lean|ptx|txt))['"]"""
)
# "$FIXTURES/foo.sio" or "$ROOT_DIR/tests/..." concatenations.
VAR_PATH = re.compile(
    r"""["']?\$(?:\{)?(?:ROOT_DIR|ROOT|REPO_ROOT|FIXTURES|FIXTURE_DIR|GOLDEN_DIR|TEST_DIR)(?:\})?/([A-Za-z0-9_./-]+\.(?:sio|sh|tsv|csv|json|md|lean|ptx|txt))["']?"""
)
# FIXTURES="$ROOT_DIR/tests/..."
DIR_ASSIGN = re.compile(
    r"""(?:FIXTURES|FIXTURE_DIR|GOLDEN_DIR|TEST_DIR)=["']?(?:\$\{?ROOT_DIR\}?/)?([A-Za-z0-9_./-]+)["']?"""
)
# for name in <iterable> ; do  — last token may be a glob.
FOR = re.compile(
    r"""for\s+\w+\s+in\s+([^\n;]+?)\s*;\s*do"""
)


def exists_file(rel: str) -> bool:
    if not rel or ".." in rel or rel.startswith("/"):
        return False
    return (root / rel).is_file()


def expand_glob(pattern: str) -> list[str]:
    pattern = pattern.strip().strip("'\"")
    if not pattern or ".." in pattern or pattern.startswith("/") or "**" in pattern:
        return []
    if "$" in pattern:
        return []
    hits: list[str] = []
    for match in iglob(str(root / pattern)):
        p = Path(match)
        if p.is_file():
            hits.append(str(p.relative_to(root)))
            if len(hits) > 500:
                break
    return hits


def analyze(gp: Path) -> tuple:
    rel = str(gp.relative_to(root))
    text = gp.read_text(encoding="utf-8", errors="replace")
    claim = "(no header claim)"
    for line in text.splitlines()[:40]:
        if line.startswith("#") and not line.startswith("#!"):
            body = line.lstrip("# ").strip()
            if len(body) > 24 and not body.startswith("GATE_"):
                claim = body[:140]
                break

    files: set[str] = set()
    missing: set[str] = set()
    empty: set[str] = set()
    n_patterns = 0

    # Resolve FIXTURES= dirs for later $FIXTURES/file joins
    fixture_dirs: list[str] = []
    for m in DIR_ASSIGN.finditer(text):
        d = m.group(1).rstrip("/")
        if d and ".." not in d:
            fixture_dirs.append(d)

    def consider(path: str) -> None:
        nonlocal n_patterns
        if not path or ")" in path or "=" in path or " " in path:
            return
        if not path.endswith(
            (".sio", ".sh", ".tsv", ".csv", ".json", ".md", ".lean", ".ptx", ".txt")
        ) and "*" not in path:
            return
        n_patterns += 1
        if any(ch in path for ch in "*?["):
            hits = expand_glob(path)
            if hits:
                files.update(hits)
            else:
                empty.add(path)
            return
        if exists_file(path):
            files.add(path)
        else:
            missing.add(path)

    for m in LIT.finditer(text):
        consider(m.group(1))

    for m in VAR_PATH.finditer(text):
        tail = m.group(1)
        # Prefer joining with assigned FIXTURES dirs when present
        if fixture_dirs and not tail.startswith(("tests/", "scripts/", "stdlib/")):
            for d in fixture_dirs:
                consider(f"{d}/{tail}")
        else:
            # $ROOT_DIR/tests/... already captured as group full path-ish
            if tail.startswith(("tests/", "scripts/", "stdlib/", "examples/", "benchmarks/", "formal/")):
                consider(tail)
            elif fixture_dirs:
                for d in fixture_dirs:
                    consider(f"{d}/{tail}")

    # Bare "$FIXTURES/foo.sio" where FIXTURES is tests/compiler/...
    for m in re.finditer(
        r"""\$FIXTURES/([A-Za-z0-9_./-]+\.(?:sio|sh|tsv|csv|json))""", text
    ):
        tail = m.group(1)
        if fixture_dirs:
            for d in fixture_dirs:
                consider(f"{d}/{tail}")
        else:
            consider(tail)

    for m in FOR.finditer(text):
        body = m.group(1).strip()
        if "/" not in body and "*" not in body and "?" not in body:
            continue
        toks = re.findall(r'"[^"]+"|\'[^\']+\'|[^\s]+', body)
        if not toks:
            continue
        target = toks[-1].strip("'\"")
        if target.startswith("$") and "FIXTURES" in target:
            # $FIXTURES/*.sio
            tail = target.split("FIXTURES/", 1)[-1] if "FIXTURES/" in target else target
            if fixture_dirs:
                for d in fixture_dirs:
                    consider(f"{d}/{tail}" if not tail.startswith(d) else tail)
            continue
        if target.startswith("$"):
            continue
        consider(target)

    n_files, n_miss, n_empty = len(files), len(missing), len(empty)
    if n_patterns == 0:
        vacuous, reason = False, "no_fixture_patterns"
    elif n_files > 0:
        vacuous, reason = False, "non_vacuous"
    elif n_empty > 0 or n_miss > 0:
        vacuous, reason = True, "zero_files_empty_or_missing"
    else:
        vacuous, reason = False, "unresolved"

    return (
        rel,
        vacuous,
        reason,
        n_patterns,
        n_files,
        n_miss,
        n_empty,
        ",".join(sorted(files)[:5]),
        ",".join(sorted(missing)[:5]),
        ",".join(sorted(empty)[:5]),
        claim,
    )


def write_tsv(path: Path, rows: list[tuple]) -> None:
    hdr = (
        "gate\tvacuous\treason\tn_patterns\tn_files\tn_missing\tn_empty_globs\t"
        "sample_files\tsample_missing\tsample_empty_globs\tclaim\n"
    )
    with path.open("w", encoding="utf-8") as fh:
        fh.write(hdr)
        for r in rows:
            fh.write(
                "\t".join(
                    [
                        r[0],
                        "yes" if r[1] else "no",
                        r[2],
                        str(r[3]),
                        str(r[4]),
                        str(r[5]),
                        str(r[6]),
                        r[7],
                        r[8],
                        r[9],
                        r[10].replace("\t", " "),
                    ]
                )
                + "\n"
            )


def main() -> int:
    gates = sorted(root.glob("scripts/ci/*_gate.sh"))
    if (root / seed_rel).is_file():
        gates.append(root / seed_rel)

    rows = [analyze(p) for p in gates]
    vac_rows = [r for r in rows if r[1]]
    seed_hit = any(r[0].endswith("vacuous_seed_gate.sh") and r[1] for r in vac_rows)

    write_tsv(out / "all_gates_fixture_status.tsv", rows)
    write_tsv(out / "vacuous_gates.tsv", vac_rows)

    print("=== gate_vacuous_fixture_sweep ===")
    print(f"scanned_gates={len(rows)}")
    print(f"vacuous_gates={len(vac_rows)}")
    print(f"positive_control_seed={seed_rel}")
    print(f"positive_control_fired={'yes' if seed_hit else 'NO'}")
    print(f"all_tsv={out / 'all_gates_fixture_status.tsv'}")
    print(f"vacuous_tsv={out / 'vacuous_gates.tsv'}")
    for r in rows:
        if "madaros_visibility_context_gate" in r[0]:
            print(
                f"seed_case_visibility_gate={r[0]} vacuous={r[1]} files={r[4]} "
                f"missing={r[5]} empty={r[6]} reason={r[2]} "
                f"files_sample={r[7]} missing_sample={r[8]}"
            )
    print("--- VACUOUS ---")
    for r in vac_rows:
        print(
            f"VACUOUS\t{r[0]}\tfiles={r[4]}\tmissing={r[5]}\tempty_globs={r[6]}\t"
            f"{r[2]}\tmiss={r[8][:140]}\tempty={r[9][:140]}"
        )

    (out / "status.env").write_text(
        f"SCANNED={len(rows)}\nVACUOUS={len(vac_rows)}\nSEED_FIRED={'1' if seed_hit else '0'}\n",
        encoding="utf-8",
    )
    return 0 if seed_hit else 2


if __name__ == "__main__":
    raise SystemExit(main())

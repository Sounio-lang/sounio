#!/usr/bin/env python3
"""Deterministic gate for Rust-term leakage in user-facing text artifacts.

This script is CI-friendly:
- deterministic target expansion and output ordering
- explicit non-zero exit when leakage is detected
- allowlist support for internal/dev-only contexts
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_TARGET_DIRS = (
    "crates/souc/tests/golden/help",
    "crates/souc/tests/golden/errors",
    "crates/souc/tests/golden/suggestions",
    "crates/souc/tests/golden/warnings",
    "crates/souc/tests/golden/scenarios",
    "crates/souc/tests/golden/json",
)

DEFAULT_TEXT_EXTENSIONS = {".txt", ".json"}
LOG_EXTENSIONS = {".log", ".stderr", ".stdout", ".txt", ".err", ".out"}
DEFAULT_MAX_FILE_BYTES = 2_000_000

# Order matters: keep specific patterns before broader ones to avoid duplicates.
BLOCKED_TERM_PATTERNS = (
    re.compile(r"\bcrates\.io\b", re.IGNORECASE),
    re.compile(r"\bCargo\.(?:toml|lock)\b", re.IGNORECASE),
    re.compile(r"\brustc\b", re.IGNORECASE),
    re.compile(r"\brustup\b", re.IGNORECASE),
    re.compile(r"\brustfmt\b", re.IGNORECASE),
    re.compile(r"\bclippy\b", re.IGNORECASE),
    re.compile(r"\bcargo\b", re.IGNORECASE),
    re.compile(r"\bcrates?\b", re.IGNORECASE),
    re.compile(r"\brust\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class AllowRule:
    path_pattern: re.Pattern[str]
    term_pattern: re.Pattern[str]
    source_file: Path
    line_number: int


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    term: str
    excerpt: str


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Gate user-facing help/error/output text for Rust-term leakage "
            "(cargo/crate/rustc/etc.)."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Repository root used for default targets and path normalization.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="PATH",
        help="File or directory to scan (repeatable).",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        metavar="LOG_FILE",
        help="Log file to scan (repeatable).",
    )
    parser.add_argument(
        "--logs-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory containing generated logs (*.log/*.stderr/*.stdout/*.txt).",
    )
    parser.add_argument(
        "--no-default-targets",
        action="store_true",
        help="Skip built-in golden fixture targets.",
    )
    parser.add_argument(
        "--allowlist",
        action="append",
        default=[],
        metavar="FILE",
        help="Allowlist TSV file (path_regex<TAB>term_regex<TAB>reason). Repeatable.",
    )
    parser.add_argument(
        "--no-default-allowlist",
        action="store_true",
        help="Ignore scripts/baselines/cultural_fidelity_allowlist.tsv.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help="Skip files larger than this byte threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic fixture self-test and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print scan target and skip details.",
    )
    return parser.parse_args(argv)


def normalize_display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def collect_files_from_dir(
    directory: Path,
    *,
    extensions: set[str] | None,
) -> list[Path]:
    files: list[Path] = []
    if not directory.exists() or not directory.is_dir():
        return files
    for candidate in sorted(directory.rglob("*")):
        if not candidate.is_file():
            continue
        if extensions is not None and candidate.suffix.lower() not in extensions:
            continue
        files.append(candidate)
    return files


def expand_targets(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for target in paths:
        if target.is_dir():
            expanded.extend(collect_files_from_dir(target, extensions=None))
        elif target.is_file():
            expanded.append(target)
    return expanded


def load_allowlist_rules(files: Sequence[Path]) -> list[AllowRule]:
    rules: list[AllowRule] = []
    for allowlist_file in files:
        if not allowlist_file.exists():
            raise ValueError(f"allowlist file not found: {allowlist_file}")
        if not allowlist_file.is_file():
            raise ValueError(f"allowlist path is not a file: {allowlist_file}")
        for line_number, raw_line in enumerate(
            allowlist_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = raw_line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"{allowlist_file}:{line_number}: expected at least 2 tab-separated fields"
                )
            path_regex = parts[0].strip()
            term_regex = parts[1].strip()
            if not path_regex or not term_regex:
                raise ValueError(
                    f"{allowlist_file}:{line_number}: path and term regex must be non-empty"
                )
            try:
                path_pattern = re.compile(path_regex)
                term_pattern = re.compile(term_regex, re.IGNORECASE)
            except re.error as error:
                raise ValueError(
                    f"{allowlist_file}:{line_number}: invalid regex: {error}"
                ) from error
            rules.append(
                AllowRule(
                    path_pattern=path_pattern,
                    term_pattern=term_pattern,
                    source_file=allowlist_file,
                    line_number=line_number,
                )
            )
    return rules


def is_binary_content(data: bytes) -> bool:
    return b"\x00" in data[:4096]


def spans_overlap(start: int, end: int, existing: list[tuple[int, int]]) -> bool:
    for other_start, other_end in existing:
        if start < other_end and end > other_start:
            return True
    return False


def is_allowlisted(path_text: str, term_text: str, rules: Sequence[AllowRule]) -> bool:
    for rule in rules:
        if rule.path_pattern.search(path_text) and rule.term_pattern.search(term_text):
            return True
    return False


def scan_file(path: Path, path_text: str, allow_rules: Sequence[AllowRule]) -> list[Finding]:
    data = path.read_bytes()
    if is_binary_content(data):
        return []
    text = data.decode("utf-8", errors="replace")
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        taken_spans: list[tuple[int, int]] = []
        for pattern in BLOCKED_TERM_PATTERNS:
            for match in pattern.finditer(line):
                start = match.start()
                end = match.end()
                if spans_overlap(start, end, taken_spans):
                    continue
                matched_text = match.group(0)
                if is_allowlisted(path_text, matched_text, allow_rules):
                    taken_spans.append((start, end))
                    continue
                excerpt = line.strip()
                if len(excerpt) > 180:
                    excerpt = f"{excerpt[:177]}..."
                findings.append(
                    Finding(
                        path=path_text,
                        line=line_number,
                        column=start + 1,
                        term=matched_text,
                        excerpt=excerpt,
                    )
                )
                taken_spans.append((start, end))
    return findings


def scan_paths(
    files: Sequence[Path],
    *,
    root: Path,
    allow_rules: Sequence[AllowRule],
    max_file_bytes: int,
    verbose: bool,
) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    skipped_count = 0
    for file_path in files:
        if not file_path.exists() or not file_path.is_file():
            skipped_count += 1
            if verbose:
                print(f"[skip] missing or non-file: {file_path}")
            continue
        if file_path.stat().st_size > max_file_bytes:
            skipped_count += 1
            if verbose:
                print(
                    f"[skip] too large ({file_path.stat().st_size} bytes > {max_file_bytes}): {file_path}"
                )
            continue
        path_text = normalize_display_path(file_path, root)
        file_findings = scan_file(file_path, path_text, allow_rules)
        findings.extend(file_findings)
        if verbose:
            print(f"[scan] {path_text}")
    findings.sort(key=lambda f: (f.path, f.line, f.column, f.term.lower()))
    return findings, skipped_count


def build_scan_target_list(args: argparse.Namespace) -> list[Path]:
    root = args.root.resolve()
    targets: list[Path] = []

    if not args.no_default_targets:
        for rel_dir in DEFAULT_TARGET_DIRS:
            targets.extend(collect_files_from_dir(root / rel_dir, extensions=DEFAULT_TEXT_EXTENSIONS))

    if args.path:
        user_paths = [Path(p).resolve() for p in args.path]
        targets.extend(expand_targets(user_paths))

    if args.log:
        user_logs = [Path(p).resolve() for p in args.log]
        targets.extend(expand_targets(user_logs))

    if args.logs_dir:
        for directory in args.logs_dir:
            targets.extend(
                collect_files_from_dir(Path(directory).resolve(), extensions=LOG_EXTENSIONS)
            )

    # Deterministic ordering and de-duplication.
    unique = sorted({path.resolve() for path in targets}, key=lambda p: p.as_posix())
    return unique


def run_self_test(root: Path, max_file_bytes: int) -> int:
    fixture_dir = root / "scripts" / "fixtures" / "cultural_fidelity"
    pass_file = fixture_dir / "pass_user_output.txt"
    fail_file = fixture_dir / "fail_user_output.txt"
    dev_file = fixture_dir / "dev_build.txt"
    fixture_allowlist = fixture_dir / "allowlist.tsv"

    if not pass_file.exists() or not fail_file.exists() or not dev_file.exists():
        print("SELF_TEST FAIL: fixture files are missing", file=sys.stderr)
        return 1

    no_rules: list[AllowRule] = []
    allow_rules = load_allowlist_rules([fixture_allowlist])

    pass_findings, _ = scan_paths(
        [pass_file],
        root=root,
        allow_rules=no_rules,
        max_file_bytes=max_file_bytes,
        verbose=False,
    )
    fail_findings, _ = scan_paths(
        [fail_file],
        root=root,
        allow_rules=no_rules,
        max_file_bytes=max_file_bytes,
        verbose=False,
    )
    dev_findings, _ = scan_paths(
        [dev_file],
        root=root,
        allow_rules=allow_rules,
        max_file_bytes=max_file_bytes,
        verbose=False,
    )

    ok = True
    if pass_findings:
        ok = False
        print("SELF_TEST FAIL: pass fixture unexpectedly leaked terms", file=sys.stderr)
    if not fail_findings:
        ok = False
        print("SELF_TEST FAIL: fail fixture did not trigger leakage detection", file=sys.stderr)
    if dev_findings:
        ok = False
        print("SELF_TEST FAIL: allowlisted dev fixture still reported leakage", file=sys.stderr)

    if not ok:
        return 1

    print("SELF_TEST PASS cultural_fidelity_gate")
    return 0


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    root = args.root.resolve()

    if args.self_test:
        return run_self_test(root, args.max_file_bytes)

    allowlist_files: list[Path] = []
    if not args.no_default_allowlist:
        allowlist_files.append(root / "scripts" / "baselines" / "cultural_fidelity_allowlist.tsv")
    allowlist_files.extend(Path(p).resolve() for p in args.allowlist)

    try:
        allow_rules = load_allowlist_rules(allowlist_files)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    targets = build_scan_target_list(args)
    if not targets:
        print(
            "error: no files to scan. Provide --path/--log/--logs-dir or omit --no-default-targets.",
            file=sys.stderr,
        )
        return 2

    findings, skipped_count = scan_paths(
        targets,
        root=root,
        allow_rules=allow_rules,
        max_file_bytes=args.max_file_bytes,
        verbose=args.verbose,
    )

    print(
        f"CULTURAL_FIDELITY scanned_files={len(targets)} skipped_files={skipped_count} findings={len(findings)}"
    )
    if not findings:
        print("CULTURAL_FIDELITY_RESULT PASS")
        return 0

    print("CULTURAL_FIDELITY_RESULT FAIL")
    for finding in findings:
        print(f"{finding.path}:{finding.line}:{finding.column}: leaked_term={finding.term!r} line={finding.excerpt}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

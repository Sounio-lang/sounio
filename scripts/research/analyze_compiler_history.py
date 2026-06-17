#!/usr/bin/env python3
"""Extract training pairs from self-hosted/ git history for Pilar V (Self-Improving Compiler Loop).

Produces datasets/sounio-ai-compiler-patches/compiler_patches.jsonl.
Usage: python3 scripts/research/analyze_compiler_history.py [--repo /workspace/sounio]
"""
import argparse, hashlib, json, subprocess, sys
from collections import defaultdict
from pathlib import Path

OUT_RELPATH = "datasets/sounio-ai-compiler-patches/compiler_patches.jsonl"

SUBSYSTEMS = [
    ("lexer",     ("lexer",)),
    ("parser",    ("parser",)),
    ("check",     ("check", "types", "effects", "linear")),
    ("ir",        ("ir", "hlir", "sir", "opt", "egraph", "rewrite")),
    ("native",    ("native", "codegen", "x86", "elf", "emit")),
    ("gpu",       ("gpu", "ptx", "wgsl", "vulkan")),
    ("wasm",      ("wasm",)),
    ("bootstrap", ("bootstrap", "boot")),
    ("vm",        ("vm", "interp")),
    ("lsp",       ("lsp",)),
    ("effects",   ("effect",)),
]

PATCH_INSTRUCTION = (
    "You are a Sounio compiler engineer. Given a bug report and the old compiler code, "
    "produce the corrected replacement code."
)
UNDERSTAND_INSTRUCTION = (
    "You are a Sounio compiler engineer. Given a compiler diff (old → new), "
    "explain what the change does and why."
)


def chash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def classify(paths: list[str]) -> str:
    joined = " ".join(paths).lower()
    for name, keys in SUBSYSTEMS:
        if any(k in joined for k in keys):
            return name
    return "check"


def git_log(repo: Path) -> str:
    cmd = [
        "git", "log",
        "--diff-filter=M", "--follow", "-p", "--no-merges",
        '--format=COMMIT_SEP%H%n%s%n%b',
        "--", "self-hosted/",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
    return r.stdout


def parse_commits(raw: str) -> list[dict]:
    """Split on COMMIT_SEP and parse each block into a structured dict."""
    blocks = raw.split("COMMIT_SEP")
    commits = []
    for block in blocks:
        if not block.strip():
            continue
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        commit_hash = lines[0].strip()
        subject = lines[1].strip() if len(lines) > 1 else ""
        # body = lines between subject and first diff line
        body_lines, diff_lines = [], []
        in_diff = False
        for line in lines[2:]:
            if line.startswith("diff --git"):
                in_diff = True
            if in_diff:
                diff_lines.append(line)
            else:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        hunks = parse_hunks(diff_lines)
        commits.append({"hash": commit_hash, "subject": subject, "body": body, "hunks": hunks})
    return commits


def parse_hunks(diff_lines: list[str]) -> list[dict]:
    """Parse unified diff lines into per-hunk dicts."""
    hunks, cur_file, cur_hunk = [], None, None
    for line in diff_lines:
        if line.startswith("+++ b/"):
            cur_file = line[6:].strip()
            cur_hunk = None
        elif line.startswith("@@ "):
            if cur_hunk is not None and cur_file:
                hunks.append(cur_hunk)
            cur_hunk = {"file": cur_file, "header": line, "removed": [], "added": []}
        elif cur_hunk is not None:
            if line.startswith("-") and not line.startswith("---"):
                cur_hunk["removed"].append(line[1:])
            elif line.startswith("+") and not line.startswith("+++"):
                cur_hunk["added"].append(line[1:])
    if cur_hunk is not None and cur_hunk.get("file"):
        hunks.append(cur_hunk)
    return hunks


def is_meaningful(hunks: list[dict]) -> tuple[bool, list[str], int]:
    """Return (ok, file_list, lines_changed). Filters non-.sio and size bounds."""
    if not hunks:
        return False, [], 0
    files = list({h["file"] for h in hunks})
    if any(not f.endswith(".sio") for f in files):
        return False, files, 0
    changed = sum(len(h["removed"]) + len(h["added"]) for h in hunks)
    return 3 <= changed <= 200, files, changed


def build_records(commits: list[dict]) -> list[dict]:
    seen: set[str] = set()
    records = []
    for c in commits:
        ok, files, changed = is_meaningful(c["hunks"])
        if not ok:
            continue
        subsys = classify(files)
        old_code = "\n".join(
            "\n".join(h["removed"]) for h in c["hunks"] if h["removed"]
        ).strip()
        new_code = "\n".join(
            "\n".join(h["added"]) for h in c["hunks"] if h["added"]
        ).strip()
        if not old_code or not new_code:
            continue
        key = chash(old_code + new_code)
        if key in seen:
            continue
        seen.add(key)

        description = c["subject"]
        if c["body"]:
            description += "\n\n" + c["body"]

        base = {
            "commit": c["hash"][:7],
            "subsystem": subsys,
            "lines_changed": changed,
            "files": files,
        }
        records.append({**base,
            "task": "compiler_patch",
            "instruction": PATCH_INSTRUCTION,
            "input": f"Bug report:\n{description}\n\nOld code:\n{old_code}",
            "output": new_code,
        })
        records.append({**base,
            "task": "compiler_understand",
            "instruction": UNDERSTAND_INSTRUCTION,
            "input": f"Old code:\n{old_code}\n\nNew code:\n{new_code}",
            "output": description,
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="/workspace/sounio")
    args = ap.parse_args()
    repo = Path(args.repo)
    out = repo / OUT_RELPATH
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Running git log on self-hosted/…", file=sys.stderr)
    raw = git_log(repo)
    commits = parse_commits(raw)
    print(f"Total commits parsed: {len(commits)}", file=sys.stderr)

    records = build_records(commits)
    used = len({r["commit"] for r in records})

    by_subsystem: dict[str, int] = defaultdict(int)
    by_task: dict[str, int] = defaultdict(int)
    with out.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
            by_subsystem[rec["subsystem"]] += 1
            by_task[rec["task"]] += 1

    print(f"\nWrote {len(records)} records → {out}")
    print(f"  commits used: {used} / {len(commits)}")
    print("  by subsystem:")
    for k, v in sorted(by_subsystem.items()):
        print(f"    {k}: {v}")
    print("  by task:")
    for k, v in sorted(by_task.items()):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()

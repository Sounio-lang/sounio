"""Words added to IDENTIFIER_OK since origin/main, by parsed set difference.

A line-based diff is wrong for this list: it is wrapped, so adding one word
rewrites the whole line and every word on it reads as added. The first version
of this check marked `match`, `let` and `loop` as newly declared IDENTIFIER_OK
when they are RESERVED, which would have excused a real failure on any of them.
"""
import re
import subprocess

GATE = "scripts/ci/parser_keyword_classification_gate.sh"


def identifier_ok(text: str) -> set:
    body = re.search(r"IDENTIFIER_OK = \{(.*?)\n\}", text, re.S)
    return set(re.findall(r'"([a-z_]+)"', body.group(1))) if body else set()


def main() -> int:
    here = identifier_ok(open(GATE).read())
    try:
        base = identifier_ok(subprocess.run(
            ["git", "show", f"origin/main:{GATE}"],
            capture_output=True, text=True, timeout=30, check=False).stdout)
    except Exception:
        base = set()
    if not here:
        # The pattern stopped matching. Declare nothing new rather than declare
        # everything new: a leniency list that swallowed the whole set would
        # turn every real failure into PENDING-REBUILD.
        return 0
    print(" ".join(sorted(here - base)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

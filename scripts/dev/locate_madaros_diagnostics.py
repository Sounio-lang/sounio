#!/usr/bin/env python3
"""Attribute Madaros diagnostics to a file and line.

Madaros prints `error[EXXX] at <start>..<end>` and NEVER prints span.file_id -- it is hardcoded
to 0 at 31 of the 37 sites that construct a Span, and the tree has no file-id counter and no
id-to-path table. Offsets are correct per file (verified with a two-module witness carrying one
undeclared variable each: 958 in the main, 263 in the helper), so across a 105-module closure an
offset alone cannot say which file it belongs to.

This recovers the attribution offline: walk the `use` closure of an entry file, then for each
diagnostic span look for closure files long enough to contain it whose bytes at [start:end] look
like source text. A span that fits exactly one file is attributed; one that fits several is
reported as ambiguous rather than guessed.

    python3 scripts/dev/locate_madaros_diagnostics.py <build.log> [entry.sio] [--code E137]

Limits worth knowing before trusting the output. A degenerate `start == 0` is not one code's
quirk -- on a self-compile of compiler/main.sio these carry it in 100% of occurrences:

    E175 (before the pub campaign: 6181)   E012  71   E009  9
    E004    8    E038  4    E011  2    E010  1

so ~95 diagnostics plus every E175 cannot be attributed by span at all, and the script reports
them as "start==0 (unlocatable)" instead of guessing. For E175, use the function name that
checker_report_private_fn_inplace prints. E002 spans cover expressions rather than identifiers,
so most land in "unmatched" -- also reported, never silently dropped.
"""
import collections
import os
import re
import sys

IDENT = re.compile(rb"^[A-Za-z_][A-Za-z0-9_]*$")
USE = re.compile(r"^\s*use\s+([A-Za-z_][A-Za-z0-9_:]*)")
# The printer splits the code and the span across lines: "error[E137\n] at 12\n..34\n: msg".
EVENT = re.compile(r"error\[E(\d+)\n\] at (-?\d+)\n\.\.(-?\d+)\n: ([^\n]*)")


def module_to_file(mod):
    for root in ("self-hosted", "stdlib"):
        cand = os.path.join(root, mod.replace("::", "/") + ".sio")
        if os.path.exists(cand):
            return cand
    return None


def use_closure(entry):
    seen, queue = set(), [entry]
    while queue:
        path = queue.pop()
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        with open(path, errors="replace") as handle:
            for line in handle:
                match = USE.match(line)
                if not match:
                    continue
                segments = match.group(1).split("::")
                # `use a::b::{x, y}` names module a::b, but `use a::b::c::*` may name a::b::c --
                # try the longest prefix that is a file.
                for k in range(len(segments), 0, -1):
                    found = module_to_file("::".join(segments[:k]))
                    if found:
                        queue.append(found)
                        break
    return sorted(seen)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--code"):
            only = a.split("=", 1)[1].lstrip("E") if "=" in a else None
    if not args:
        print(__doc__)
        return 2
    log = args[0]
    entry = args[1] if len(args) > 1 else "self-hosted/compiler/main.sio"

    text = open(log, errors="replace").read()
    events = [
        (m.group(1), int(m.group(2)), int(m.group(3)), m.group(4))
        for m in EVENT.finditer(text)
    ]
    if not events:
        print(f"no spanned diagnostics found in {log}")
        return 1

    files = use_closure(entry)
    raw = {}
    for path in files:
        try:
            raw[path] = open(path, "rb").read()
        except OSError:
            pass

    by_code = collections.Counter(code for code, _, _, _ in events)
    print(f"log={log}  entry={entry}  closure={len(raw)} files  spanned diagnostics={len(events)}")
    print("  by code: " + "  ".join(f"E{c}:{n}" for c, n in sorted(by_code.items())))

    located = collections.defaultdict(list)
    stats = collections.Counter()
    for code, start, end, msg in events:
        if only and code != only:
            continue
        if start == 0:
            stats[f"E{code} start==0 (unlocatable)"] += 1
            continue
        hits = []
        for path, data in raw.items():
            if end <= len(data) and IDENT.match(data[start:end]):
                hits.append((path, data[start:end].decode(), data.count(b"\n", 0, start) + 1))
        if len(hits) == 1:
            stats[f"E{code} located"] += 1
            located[code].append(hits[0])
        elif hits:
            stats[f"E{code} ambiguous({len(hits)})"] += 1
            located[code].append(hits[0] + (f"ambiguous:{len(hits)}",))
        else:
            stats[f"E{code} unmatched"] += 1

    print("\nattribution:")
    for key, n in sorted(stats.items()):
        print(f"  {n:5d}  {key}")

    for code, hits in sorted(located.items()):
        print(f"\n=== E{code}: {len(hits)} attributed ===")
        for path, n in collections.Counter(h[0] for h in hits).most_common():
            print(f"  {n:5d}  {path}")
        names = collections.Counter(h[1] for h in hits)
        print("  symbols: " + " ".join(f"{name}({n})" for name, n in names.most_common(12)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

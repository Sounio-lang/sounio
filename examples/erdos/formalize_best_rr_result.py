#!/usr/bin/env python3
"""
examples/erdos/formalize_best_rr_result.py

Read a compact-disk sweep log from stdlib/research/erdos90_optimize.sio,
pick the best (n, count, rr, nsq) record, and emit a Lean theorem in
formal/lean4/SounioErdos90PlanarLowerBound.lean.

Usage:
    python3 examples/erdos/formalize_best_rr_result.py \
        /path/to/optimize_rr10000.log \
        >> formal/lean4/SounioErdos90PlanarLowerBound.lean

Then add the theorem to the lakefile if it is a new module, or edit the
existing `erdos90_compact_disk_u*` theorem.
"""
import re
import sys
from pathlib import Path


def parse_log(path: str):
    records = []
    cur = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'n=(\d+)', line)
            if m:
                if cur and 'count' in cur:
                    records.append((cur['n'], cur.get('harb', 0), cur['count'], cur['N'], cur['x100']))
                cur = {'n': int(m.group(1))}
            elif re.search(r'\b(harb|count|N|x100)=', line):
                for k, v in re.findall(r'(\w+)=(\d+)', line):
                    cur[k] = int(v)
        if cur and 'count' in cur:
            records.append((cur['n'], cur.get('harb', 0), cur['count'], cur['N'], cur['x100']))
    return records


def best_by_count(records):
    return max(records, key=lambda r: r[2])


def best_by_ratio(records):
    return max(records, key=lambda r: r[4])


def theorem_name(n: int) -> str:
    return f"erdos90_compact_disk_u{n}"


def emit_theorem(n: int, count: int, nsq: int, rr: int | None = None) -> str:
    rr_hint = f" (disk x²+y² ≤ {rr})" if rr is not None else ""
    return f"""
/- **New explicit lower bound:** among the {n} integer points with x² + y² ≤ {rr or "?"}{rr_hint}
    there are at least {count} pairs at squared distance {nsq}.
    Certified by `native_decide` in `countUnitSq (compactDiskZ2 {rr or "?"}) {nsq}`. -/
theorem {theorem_name(n)} :
    countUnitSq (compactDiskZ2 {rr or "?"}) {nsq} ≥ {count} := by
  native_decide
"""


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <sweep.log> [--count|--ratio] [--rr RR]", file=sys.stderr)
        sys.exit(1)
    log_path = sys.argv[1]
    mode = '--count'
    rr = None
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ('--count', '--ratio'):
            mode = arg
        elif arg == '--rr':
            i += 1
            rr = int(sys.argv[i])
        i += 1
    records = parse_log(log_path)
    if not records:
        print(f"error: no records in {log_path}", file=sys.stderr)
        sys.exit(1)
    if mode == '--ratio':
        rec = best_by_ratio(records)
    else:
        rec = best_by_count(records)
    n, _, count, nsq, x100 = rec
    print(emit_theorem(n, count, nsq, rr))
    print(f"-- selected record: n={n} count={count} N={nsq} x100={x100}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Balance the Sounio test suite across N GitHub Actions matrix shards by
historical per-test wall-clock time, and keep that history current.

Bootstrap: with no corpus (or an incomplete one), every test not yet measured
gets the corpus's own median weight (1.0 if the corpus is empty) -- shards are
already a correct PARTITION on day one (every test assigned to exactly one
shard), just not time-balanced until scripts/ci/plan_test_shards.py refresh
has run at least once. See .github/workflows/shard-timing-refresh.yml.

Subcommands:
    plan --corpus FILE --all-tests FILE --shards N --index I --out FILE
        Partition the paths in --all-tests (one relative path per line, the
        exact format scripts/dev/run_sio_test_suite_v2.sh --list-tests emits)
        into N shards by greedy LPT (longest processing time first) bin
        packing, and write shard I's paths to --out -- itself the exact
        format --test-list consumes. Deterministic: identical inputs always
        produce an identical partition (ties broken by path).

    refresh --corpus FILE --junit FILE
        Ingest a JUnit XML report (the harness's own --format junit output)
        and rewrite --corpus with each <testcase name="X" time="T"> as
        {"X.sio": T} (JUnit strips the .sio suffix the harness's own basename
        does not carry). Existing corpus entries not present in this JUnit run
        are kept unchanged, so a run that only covers a subset still refines
        the corpus rather than discarding history for the rest.

The corpus is keyed by basename (with .sio), not full relative path: the
partition itself always operates on full relative paths (never conflated), the
basename key is only used to look up a *weight* for one, and a same-named
test in two different directories sharing one weight is an acceptable
imprecision for a load-balancing heuristic, never a correctness issue (no test
can be dropped, duplicated, or misidentified by this ambiguity).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import xml.etree.ElementTree as ET


def load_corpus(path: str) -> dict[str, float]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}


def cmd_plan(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    default_weight = statistics.median(corpus.values()) if corpus else 1.0

    with open(args.all_tests, encoding="utf-8") as f:
        all_tests = [line.strip() for line in f if line.strip()]

    if not all_tests:
        print("plan_test_shards: --all-tests file has no entries -- refusing to plan an empty partition", file=sys.stderr)
        return 1
    if args.shards < 1:
        print(f"plan_test_shards: --shards must be >= 1, got {args.shards}", file=sys.stderr)
        return 2
    if not (0 <= args.index < args.shards):
        print(f"plan_test_shards: --index {args.index} out of range for --shards {args.shards}", file=sys.stderr)
        return 2

    def weight(path: str) -> float:
        return corpus.get(os.path.basename(path), default_weight)

    # LPT: heaviest first, ties broken by path for determinism. Greedily
    # assign each to whichever shard currently carries the least load -- the
    # standard, simple approximation for balanced multiway partitioning; exact
    # optimal packing is not needed for a load-balancing heuristic.
    ordered = sorted(all_tests, key=lambda p: (-weight(p), p))
    loads = [0.0] * args.shards
    shards: list[list[str]] = [[] for _ in range(args.shards)]
    for path in ordered:
        i = min(range(args.shards), key=lambda i: loads[i])
        loads[i] += weight(path)
        shards[i].append(path)

    # Deterministic within-shard order too, independent of assignment order.
    result = sorted(shards[args.index])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for path in result:
            f.write(path + "\n")

    print(
        f"plan_test_shards: shard {args.index}/{args.shards}: {len(result)} tests, "
        f"estimated {loads[args.index]:.1f}s (default_weight={default_weight:.3f}, "
        f"corpus covers {sum(1 for p in all_tests if os.path.basename(p) in corpus)}/{len(all_tests)})",
        file=sys.stderr,
    )
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)

    tree = ET.parse(args.junit)
    n_seen = 0
    for testcase in tree.getroot().iter("testcase"):
        name = testcase.get("name")
        time_s = testcase.get("time")
        if not name or time_s is None:
            continue
        try:
            corpus[f"{name}.sio"] = float(time_s)
            n_seen += 1
        except ValueError:
            continue

    if n_seen == 0:
        print(f"plan_test_shards: refresh: {args.junit} has no usable <testcase> entries -- refusing to write an unchanged/empty corpus", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.corpus)) or ".", exist_ok=True)
    with open(args.corpus, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(corpus.items())), f, indent=2)
        f.write("\n")

    print(f"plan_test_shards: refresh: updated {n_seen} entries from {args.junit}; corpus now has {len(corpus)} entries", file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--corpus", required=True)
    p_plan.add_argument("--all-tests", required=True)
    p_plan.add_argument("--shards", type=int, required=True)
    p_plan.add_argument("--index", type=int, required=True)
    p_plan.add_argument("--out", required=True)
    p_plan.set_defaults(func=cmd_plan)

    p_refresh = sub.add_parser("refresh")
    p_refresh.add_argument("--corpus", required=True)
    p_refresh.add_argument("--junit", required=True)
    p_refresh.set_defaults(func=cmd_refresh)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

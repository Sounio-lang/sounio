#!/usr/bin/env python3
"""Validate deterministic cube-propagation trail manifests.

This is a producer-side format/consistency checker, not a SAT or geometry proof.
It replays domain-propagation trail metadata emitted by the fixed Sounio
skeleton or the DIMACS cube producer and rejects malformed DIMACS literals,
binary edge reasons, domain transitions, conflict summaries, or accidental
promotable markers.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


CUBE_ASSIGN_RE = re.compile(
    r"^\s+cube_assignment index=(?P<idx>\d+) vertex=(?P<v>\d+) colour=(?P<c>\d+)$"
)
EDGE_RE = re.compile(r"^\s+edge (?P<u>\d+) (?P<v>\d+)$")
PRECOLOUR_RE = re.compile(
    r"^\s+precolour vertex=(?P<v>\d+) colour=(?P<c>\d+) "
    r"colour_valid=(?P<valid>\d+) bounded_encoding_supported=(?P<bounded>\d+) "
    r"lean_var=(?P<var>\d+) dimacs_lit=(?P<lit>\d+)$"
)
FACT_RE = re.compile(r"^\s+rup_fact_clause=(?P<lit>\d+) 0$")
TRAIL_RE = re.compile(
    r"^\s+trail_step=(?P<step>\d+) op=remove reason=edge\((?P<src>\d+),(?P<dst>\d+)\) "
    r"source_singleton_colour=(?P<colour>\d+) target_vertex=(?P<target>\d+) "
    r"before_domain=(?P<before>\d+) after_domain=(?P<after>\d+) "
    r"removed_dimacs_lit=(?P<removed>\d+)$"
)
REASON_RE = re.compile(r"^\s+rup_reason_clause=-(?P<src>\d+) -(?P<dst>\d+) 0$")


class ManifestError(RuntimeError):
    pass


def dimacs_lit(v: int, c: int, k: int) -> int:
    return v * k + c + 1


def singleton_colour(mask: int) -> int:
    if mask == 0 or mask & (mask - 1):
        raise ManifestError(f"domain is not singleton: {mask}")
    c = 0
    while mask & 1 == 0:
        mask >>= 1
        c += 1
    return c


def parse_key(lines: list[str], key: str) -> str:
    prefix = f"  {key}="
    matches = [line[len(prefix) :] for line in lines if line.startswith(prefix)]
    if len(matches) != 1:
        raise ManifestError(f"expected exactly one {key}= line, got {len(matches)}")
    return matches[0]


def parse_nat_key(lines: list[str], key: str) -> int:
    raw = parse_key(lines, key)
    if not re.fullmatch(r"\d+", raw):
        raise ManifestError(f"{key} must be a nonnegative integer, got {raw!r}")
    return int(raw)


def parse_hex_sha256(lines: list[str], key: str) -> str:
    raw = parse_key(lines, key)
    if not re.fullmatch(r"[0-9a-f]{64}", raw):
        raise ManifestError(f"{key} must be a lowercase SHA256 hex string, got {raw!r}")
    return raw


def complete_graph_edges(n: int) -> set[tuple[int, int]]:
    return {(u, v) for u in range(n) for v in range(n) if u != v}


def section(lines: list[str], name: str) -> list[str]:
    marker = f"section={name}"
    try:
        start = lines.index(marker)
    except ValueError as exc:
        raise ManifestError(f"missing {marker}") from exc
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("section=") or lines[i].startswith("promotion_gate="):
            end = i
            break
    return lines[start:end]


def validate_global_markers(lines: list[str], header: str, output: str) -> None:
    required = {
        header,
        "trust_boundary=search_untrusted__drat_lrat_lean_verified_required",
        f"output={output}",
        "promotion_gate=REJECT_NONE_PROOF_ARTIFACT",
        "promotable=0",
        "status=manifest_emitted_unpromotable",
    }
    missing = sorted(required.difference(lines))
    if missing:
        raise ManifestError(f"missing global markers: {missing}")
    forbidden = {"promotion_gate=READY", "promotable=1"}
    present = sorted(forbidden.intersection(lines))
    if present:
        raise ManifestError(f"forbidden promotable markers present: {present}")


def validate_section_params(sec: list[str]) -> tuple[int, int, int, set[tuple[int, int]]]:
    graph_family = parse_key(sec, "graph_family")
    n = parse_nat_key(sec, "n")
    k = parse_nat_key(sec, "k")
    cube_count = parse_nat_key(sec, "cube_assignment_count")
    if n <= 0:
        raise ManifestError("n must be positive")
    if k <= 0 or k >= 62:
        raise ManifestError(f"k must satisfy 0 < k < 62, got {k}")
    if cube_count <= 0 or cube_count > n:
        raise ManifestError(f"cube_assignment_count must be in 1..n, got {cube_count}")
    if graph_family == "complete_graph":
        graph_id = parse_nat_key(sec, "graph_id")
        if graph_id != 0:
            raise ManifestError(f"unsupported graph_id for complete_graph smoke: {graph_id}")
        return n, k, cube_count, complete_graph_edges(n)
    if graph_family == "dimacs_edge":
        parse_hex_sha256(sec, "edge_sha256")
        m = parse_nat_key(sec, "m")
        edge_count = parse_nat_key(sec, "edge_count")
        if edge_count != m:
            raise ManifestError(f"edge_count={edge_count} does not match m={m}")
        undirected: set[tuple[int, int]] = set()
        for line in sec:
            if edge := EDGE_RE.match(line):
                u = int(edge["u"])
                v = int(edge["v"])
                if not (0 <= u < n) or not (0 <= v < n) or u == v:
                    raise ManifestError(f"bad edge row: {line}")
                key = (u, v) if u < v else (v, u)
                if key in undirected:
                    raise ManifestError(f"duplicate edge row: {key}")
                undirected.add(key)
        if len(undirected) != edge_count:
            raise ManifestError(f"edge_count={edge_count}, found {len(undirected)} edge rows")
        return n, k, cube_count, undirected | {(v, u) for u, v in undirected}
    raise ManifestError(f"unsupported graph_family: {graph_family}")


def validate_precolours(
    sec: list[str], n: int, k: int, cube_count: int, require_distinct_colours: bool
) -> list[tuple[int, int]]:
    cube_rows: list[tuple[int, int, int]] = []
    assignments: list[tuple[int, int]] = []
    fact_literals: list[int] = []
    for line in sec:
        if m := CUBE_ASSIGN_RE.match(line):
            idx = int(m["idx"])
            v = int(m["v"])
            c = int(m["c"])
            if not (0 <= v < n) or not (0 <= c < k):
                raise ManifestError(f"cube assignment out of range: vertex={v} colour={c}")
            cube_rows.append((idx, v, c))
        elif m := PRECOLOUR_RE.match(line):
            v = int(m["v"])
            c = int(m["c"])
            if not (0 <= v < n) or not (0 <= c < k):
                raise ManifestError(f"precolour out of range: vertex={v} colour={c}")
            lit = dimacs_lit(v, c, k)
            if int(m["valid"]) != 1 or int(m["bounded"]) != 1:
                raise ManifestError(f"invalid precolour guard: {line}")
            if int(m["var"]) != lit - 1 or int(m["lit"]) != lit:
                raise ManifestError(f"bad precolour variable encoding: {line}")
            assignments.append((v, c))
        elif m := FACT_RE.match(line):
            fact_literals.append(int(m["lit"]))

    if len(cube_rows) != cube_count:
        raise ManifestError(f"cube_assignment_count={cube_count}, found {len(cube_rows)} rows")
    expected_indices = list(range(cube_count))
    got_indices = [idx for idx, _v, _c in cube_rows]
    if got_indices != expected_indices:
        raise ManifestError(f"cube assignment indices are not contiguous: {got_indices}")
    cube_assignments = [(v, c) for _idx, v, c in cube_rows]
    seen_vertices: set[int] = set()
    seen_colours: set[int] = set()
    for v, c in cube_assignments:
        if v in seen_vertices:
            raise ManifestError(f"duplicate cube assignment for vertex {v}")
        seen_vertices.add(v)
        if require_distinct_colours and c in seen_colours:
            raise ManifestError(f"duplicate cube assignment colour {c}")
        seen_colours.add(c)
    if assignments != cube_assignments:
        raise ManifestError(
            f"precolour rows do not match cube assignments: {assignments} vs {cube_assignments}"
        )
    expected_facts = [dimacs_lit(v, c, k) for v, c in cube_assignments]
    if fact_literals != expected_facts:
        raise ManifestError(f"unexpected fact clauses: {fact_literals}")
    return cube_assignments


def validate_negated_cube(sec: list[str], assignments: list[tuple[int, int]], k: int) -> None:
    clause = parse_key(sec, "rup_clause_negated_cube")
    expected = " ".join(f"-{dimacs_lit(v, c, k)}" for v, c in assignments) + " 0"
    if clause != expected:
        raise ManifestError(f"bad negated cube clause: got {clause!r}, expected {expected!r}")


def validate_trail(
    sec: list[str],
    assignments: list[tuple[int, int]],
    n: int,
    k: int,
    graph_edges: set[tuple[int, int]],
    require_trail: bool,
) -> tuple[int, list[int], int]:
    all_colours = (1 << k) - 1
    domains = [all_colours for _ in range(n)]
    for v, c in assignments:
        domains[v] = 1 << c

    trail_len = 0
    conflict = 0
    conflict_vertex = -1
    i = 0
    while i < len(sec):
        line = sec[i]
        m = TRAIL_RE.match(line)
        if not m:
            i += 1
            continue
        if i + 1 >= len(sec):
            raise ManifestError(f"trail step missing reason clause: {line}")
        reason = REASON_RE.match(sec[i + 1])
        if not reason:
            raise ManifestError(f"trail step has malformed reason clause: {sec[i + 1]}")

        step = int(m["step"])
        src = int(m["src"])
        dst = int(m["dst"])
        colour = int(m["colour"])
        target = int(m["target"])
        before = int(m["before"])
        after = int(m["after"])
        removed = int(m["removed"])
        if step != trail_len + 1:
            raise ManifestError(f"non-contiguous trail step: got {step}, expected {trail_len + 1}")
        if not (0 <= src < n) or not (0 <= dst < n) or not (0 <= colour < k):
            raise ManifestError(f"trail value out of range at step {step}")
        if target != dst:
            raise ManifestError(f"target vertex disagrees with edge destination: {line}")
        if (src, dst) not in graph_edges:
            raise ManifestError(f"non-graph edge reason: {(src, dst)}")
        if before != domains[dst]:
            raise ManifestError(f"bad before-domain at step {step}: got {before}, expected {domains[dst]}")
        src_domain = domains[src]
        if src_domain == 0 or src_domain & (src_domain - 1):
            raise ManifestError(f"source vertex {src} domain is not singleton at step {step}")
        if singleton_colour(src_domain) != colour:
            raise ManifestError(f"bad source singleton colour at step {step}")
        if before & (1 << colour) == 0:
            raise ManifestError(f"removed colour {colour} not present before step {step}")
        expected_after = before & (all_colours ^ (1 << colour))
        if after != expected_after:
            raise ManifestError(f"bad after-domain at step {step}: got {after}, expected {expected_after}")
        if removed != dimacs_lit(dst, colour, k):
            raise ManifestError(f"bad removed literal at step {step}: got {removed}")
        if int(reason["src"]) != dimacs_lit(src, colour, k):
            raise ManifestError(f"bad source literal in reason at step {step}")
        if int(reason["dst"]) != dimacs_lit(dst, colour, k):
            raise ManifestError(f"bad target literal in reason at step {step}")

        domains[dst] = after
        trail_len = step
        if after == 0:
            conflict = 1
            conflict_vertex = dst
        i += 2

    if require_trail and trail_len == 0:
        raise ManifestError("no trail steps found")
    return trail_len, domains, conflict_vertex if conflict else -1


def validate_summary(
    sec: list[str],
    trail_len: int,
    domains: list[int],
    conflict_vertex: int,
    require_conflict: bool,
    expected_passes: str | None = None,
    expected_guard: str | None = None,
) -> None:
    expected_pairs = {
        "trail_len": str(trail_len),
        "conflict": "1" if conflict_vertex >= 0 else "0",
        "conflict_vertex": str(conflict_vertex),
        "final_domains": ",".join(str(d) for d in domains),
    }
    if expected_passes is not None:
        expected_pairs["propagation_passes"] = expected_passes
    else:
        parse_nat_key(sec, "propagation_passes")
    if expected_guard is not None:
        expected_pairs["termination_guard_tripped"] = expected_guard
    else:
        guard = parse_key(sec, "termination_guard_tripped")
        if guard not in {"0", "1"}:
            raise ManifestError(f"termination_guard_tripped must be 0 or 1, got {guard!r}")
    for key, expected in expected_pairs.items():
        got = parse_key(sec, key)
        if got != expected:
            raise ManifestError(f"bad {key}: got {got!r}, expected {expected!r}")
    if require_conflict and conflict_vertex < 0:
        raise ManifestError("expected a propagation conflict but replay found none")
    if not require_conflict:
        hard_cube = parse_key(sec, "hard_cube")
        expected_hard_cube = "0" if conflict_vertex >= 0 else "1"
        if hard_cube != expected_hard_cube:
            raise ManifestError(f"bad hard_cube: got {hard_cube!r}, expected {expected_hard_cube!r}")


def validate(text: str) -> None:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    if "cube_sieve_skeleton v0" in lines:
        validate_global_markers(lines, "cube_sieve_skeleton v0", "deterministic_cube_manifest")
        sec = section(lines, "complete_graph_cube_propagation_smoke")
        require_distinct_colours = True
        require_trail = True
        require_conflict = True
        expected_passes = "1"
        expected_guard = "0"
    elif "cube_sieve_propagation_manifest v1" in lines:
        validate_global_markers(
            lines, "cube_sieve_propagation_manifest v1", "dimacs_cube_propagation_manifest"
        )
        sec = section(lines, "dimacs_cube_propagation")
        require_distinct_colours = False
        require_trail = False
        require_conflict = False
        expected_passes = None
        expected_guard = None
    else:
        raise ManifestError("unknown cube propagation manifest header")
    n, k, cube_count, graph_edges = validate_section_params(sec)
    if parse_key(sec, "verified_claim") != "none":
        raise ManifestError("cube propagation manifest must not claim a verified theorem")
    if parse_key(sec, "geometry_claim") != "none":
        raise ManifestError("cube propagation manifest must not claim geometry")
    if parse_key(sec, "proof_artifact_sha256") != "NONE":
        raise ManifestError("cube propagation manifest must keep proof_artifact_sha256=NONE")
    assignments = validate_precolours(sec, n, k, cube_count, require_distinct_colours)
    validate_negated_cube(sec, assignments, k)
    trail_len, domains, conflict_vertex = validate_trail(
        sec, assignments, n, k, graph_edges, require_trail
    )
    validate_summary(
        sec,
        trail_len,
        domains,
        conflict_vertex,
        require_conflict,
        expected_passes,
        expected_guard,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", help="manifest output file; stdin if omitted")
    args = parser.parse_args()
    text = Path(args.manifest).read_text() if args.manifest else sys.stdin.read()
    try:
        validate(text)
    except ManifestError as exc:
        print(f"cube_sieve_manifest_validator: FAIL: {exc}", file=sys.stderr)
        return 1
    print("cube_sieve_manifest_validator: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

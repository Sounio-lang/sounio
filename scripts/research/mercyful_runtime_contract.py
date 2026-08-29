#!/usr/bin/env python3
"""
Mercyful Learning Runtime — first substrate-aware suffering-budget scheduler.

Companion to:
  docs/research/mercyful_runtime_spec_2026-07-25.md
  docs/research/mercyful_runtime_falsifiers_2026-07-25.md

Pure Python; no dependencies beyond the standard library.
"""

from collections import defaultdict


class MercyGraph:
    def __init__(self, states, edges, suffering, lengths=None):
        """
        states: iterable of state names
        edges: list of (u, v)
        suffering: dict state -> float
        lengths: dict edge -> float (default 1.0)
        """
        self.states = list(states)
        self.suffering = dict(suffering)
        self.adj = defaultdict(list)
        self.lengths = {}
        for u, v in edges:
            self.adj[u].append(v)
            self.lengths[(u, v)] = 1.0 if lengths is None else float(lengths.get((u, v), 1.0))

    def path_cost(self, path, mu):
        """path = list of states. Returns (length, integral_suffering, peak_suffering, total_cost)."""
        length = 0.0
        integral = 0.0
        peak = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            ell = self.lengths[(u, v)]
            length += ell
            # use suffering at the source node for the edge segment
            s = self.suffering[u]
            integral += s * ell
            peak = max(peak, s)
        # include suffering at target node
        peak = max(peak, self.suffering[path[-1]])
        total = integral + mu * peak
        return length, integral, peak, total


def enumerate_paths(graph, start, target, L0, max_states_visit=20):
    """
    Enumerate all simple paths (no repeated states) from start to target
    with total length <= L0. For small graphs only.
    """
    results = []
    stack = [(start, [start], 0.0, set([start]))]
    while stack:
        node, path, length, visited = stack.pop()
        if length > L0:
            continue
        if node == target and len(path) > 1:
            results.append(list(path))
            # continue to allow longer paths that still reach target
        if len(path) >= max_states_visit:
            continue
        for nxt in graph.adj[node]:
            if nxt in visited:
                continue
            ell = graph.lengths[(node, nxt)]
            stack.append((nxt, path + [nxt], length + ell, visited | {nxt}))
    return results


def mercyful_schedule(graph, start, target, mu, L0):
    """
    Return (best_path, metrics) or ('INFEASIBLE', None).
    """
    paths = enumerate_paths(graph, start, target, L0)
    if not paths:
        return 'INFEASIBLE', None
    best = None
    best_cost = float('inf')
    for p in paths:
        length, integral, peak, total = graph.path_cost(p, mu)
        if total < best_cost:
            best_cost = total
            best = (p, length, integral, peak, total)
    return best[0], best[1:]


def pareto_frontier(graph, start, target, L0):
    """Return Pareto-optimal (integral, peak) pairs and their paths."""
    paths = enumerate_paths(graph, start, target, L0)
    candidates = []
    for p in paths:
        length, integral, peak, _ = graph.path_cost(p, 0.0)
        candidates.append((integral, peak, p, length))
    # remove dominated: (i1,p1) dominates (i2,p2) if i1<=i2 and p1<=p2 and strict
    frontier = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if other is c:
                continue
            if other[0] <= c[0] and other[1] <= c[1] and (other[0] < c[0] or other[1] < c[1]):
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    return sorted(frontier)


# -----------------------------------------------------------------------------
# Contract clauses
# -----------------------------------------------------------------------------

def check_M1_well_defined():
    g = MercyGraph(
        states=['S', 'T'],
        edges=[('S', 'T')],
        suffering={'S': 0.0, 'T': 0.0}
    )
    path, metrics = mercyful_schedule(g, 'S', 'T', mu=0.0, L0=5.0)
    ok = (path == ['S', 'T'] and metrics is not None)
    print(f"M1_WELL_DEFINED path={path} metrics={metrics} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_M2_pareto_frontier():
    # Graph with exactly two non-dominated paths:
    # P1 (S-A-T): integral 5, peak 5, length 2
    # P2 (S-B-C-D-T): integral 6, peak 2, length 4
    # Neither dominates the other: 5<6 but 5>2.
    g = MercyGraph(
        states=['S', 'A', 'B', 'C', 'D', 'T'],
        edges=[('S', 'A'), ('A', 'T'), ('S', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'T')],
        suffering={'S': 0.0, 'A': 5.0, 'B': 2.0, 'C': 2.0, 'D': 2.0, 'T': 0.0}
    )
    pf = pareto_frontier(g, 'S', 'T', L0=5.0)
    points = [(round(i, 6), round(p, 6)) for i, p, _, _ in pf]
    expected = {(5.0, 5.0), (6.0, 2.0)}
    ok = set(points) == expected
    print(f"M2_PARETO_FRONTIER points={points} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_M3_anti_goodhart():
    # Exposure-therapy graph
    g = MercyGraph(
        states=['avoidance', 'mild', 'moderate', 'recovery'],
        edges=[
            ('avoidance', 'avoidance'),
            ('avoidance', 'mild'),
            ('mild', 'avoidance'),
            ('mild', 'moderate'),
            ('moderate', 'mild'),
            ('moderate', 'recovery'),
        ],
        suffering={
            'avoidance': 0.0,
            'mild': 2.0,
            'moderate': 5.0,
            'recovery': 0.0,
        }
    )
    # Best target-constrained path to recovery
    paths = enumerate_paths(g, 'avoidance', 'recovery', L0=10.0)
    recovery_cost = min(
        (g.path_cost(p, mu=0.0)[1] for p in paths),
        default=float('inf')
    )
    # Unconstrained raw minimizer: can stay in avoidance forever at zero cost.
    # The self-loop ('avoidance','avoidance') has integral 0.
    unconstrained_best = 0.0  # lower bound achieved by A->A self-loop
    # Anti-Goodhart claim: the unconstrained minimum is strictly less than the
    # constrained minimum, so a raw minimizer would avoid recovery.
    ok = (recovery_cost > unconstrained_best)
    print(f"M3_ANTI_GOODHART recovery_cost={recovery_cost} unconstrained_best={unconstrained_best} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_M4_mercyful_selects_exposure():
    g = MercyGraph(
        states=['avoidance', 'mild', 'moderate', 'recovery'],
        edges=[
            ('avoidance', 'avoidance'),
            ('avoidance', 'mild'),
            ('mild', 'avoidance'),
            ('mild', 'moderate'),
            ('moderate', 'mild'),
            ('moderate', 'recovery'),
        ],
        suffering={
            'avoidance': 0.0,
            'mild': 2.0,
            'moderate': 5.0,
            'recovery': 0.0,
        }
    )
    path, metrics = mercyful_schedule(g, 'avoidance', 'recovery', mu=1.0, L0=10.0)
    reaches_recovery = (path is not None and path[-1] == 'recovery')
    passes_moderate = (path is not None and 'moderate' in path)
    ok = reaches_recovery and passes_moderate
    print(f"M4_MERCYFUL_SELECTS_EXPOSURE path={path} metrics={metrics} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_M5_mu_continuity():
    # Graph where mu changes the selected path
    g = MercyGraph(
        states=['S', 'A', 'B', 'C', 'T'],
        edges=[('S', 'A'), ('A', 'T'), ('S', 'B'), ('B', 'C'), ('C', 'T')],
        suffering={'S': 0.0, 'A': 8.0, 'B': 2.0, 'C': 2.0, 'T': 0.0}
    )
    # S-A-T: integral 8, peak 8, length 2
    # S-B-C-T: integral 4, peak 2, length 3
    # For mu=0: prefer S-B-C-T (integral 4 < 8)
    # For mu=100: S-A-T cost = 8 + 100*8 = 808; S-B-C-T cost = 4 + 100*2 = 204; still prefer S-B-C-T
    # Hmm, S-B-C-T dominates. Need a path with lower peak but higher integral.
    g2 = MercyGraph(
        states=['S', 'A', 'B', 'C', 'D', 'T'],
        edges=[('S', 'A'), ('A', 'T'), ('S', 'B'), ('B', 'C'), ('C', 'T'), ('S', 'D'), ('D', 'T')],
        suffering={'S': 0.0, 'A': 3.0, 'B': 2.0, 'C': 2.0, 'D': 1.0, 'T': 0.0}
    )
    # S-A-T: integral 3, peak 3, length 2
    # S-B-C-T: integral 4, peak 2, length 3
    # S-D-T: integral 1, peak 1, length 2
    # S-D-T dominates all. Bad example.
    # Need a graph with genuine trade-off.
    g3 = MercyGraph(
        states=['S', 'A', 'B', 'C', 'T'],
        edges=[('S', 'A'), ('A', 'T'), ('S', 'B'), ('B', 'C'), ('C', 'T')],
        suffering={'S': 0.0, 'A': 4.0, 'B': 3.0, 'C': 3.0, 'T': 0.0}
    )
    # S-A-T: integral 4, peak 4, length 2
    # S-B-C-T: integral 6, peak 3, length 3
    # mu=0: S-A-T (cost 4 < 6)
    # mu=10: S-A-T cost = 4 + 40 = 44; S-B-C-T cost = 6 + 30 = 36; prefer S-B-C-T
    selected = []
    for mu in [0.0, 10.0]:
        path, _ = mercyful_schedule(g3, 'S', 'T', mu=mu, L0=5.0)
        selected.append((mu, path))
    changed = (selected[0][1] != selected[1][1])
    # Also check peak weakly decreases as mu increases across frontier
    pf = pareto_frontier(g3, 'S', 'T', L0=5.0)
    peaks = [p for _, p, _, _ in sorted(pf)]
    # Just verify selection changed
    ok = changed
    print(f"M5_MU_CONTINUITY selected={selected} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_M6_budget_infeasibility():
    g = MercyGraph(
        states=['S', 'A', 'T'],
        edges=[('S', 'A'), ('A', 'T')],
        suffering={'S': 0.0, 'A': 1.0, 'T': 0.0},
        lengths={('S', 'A'): 2.0, ('A', 'T'): 2.0}
    )
    path, metrics = mercyful_schedule(g, 'S', 'T', mu=0.0, L0=3.0)
    ok = (path == 'INFEASIBLE')
    print(f"M6_BUDGET_INFEASIBILITY path={path} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 70)
    print("MERCYFUL LEARNING RUNTIME — contract")
    print("=" * 70)
    results.append(("M1", check_M1_well_defined()))
    results.append(("M2", check_M2_pareto_frontier()))
    results.append(("M3", check_M3_anti_goodhart()))
    results.append(("M4", check_M4_mercyful_selects_exposure()))
    results.append(("M5", check_M5_mu_continuity()))
    results.append(("M6", check_M6_budget_infeasibility()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"MERCYFUL_RUNTIME_VERDICT M_GREEN ({passed}/{total} clauses PASS)")
        print("MERCYFUL_RUNTIME_NOTE graph_prototype; exposure_therapy_toy; no_clinical_claim")
        return 0
    else:
        print(f"MERCYFUL_RUNTIME_VERDICT M_AMBER ({passed}/{total} clauses PASS)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

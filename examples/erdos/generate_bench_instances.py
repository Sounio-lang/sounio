#!/usr/bin/env python3
"""
Generate standard graph-colouring benchmark instances as DIMACS edge files.

Generates:
  - Mycielski graphs M_n (triangle-free, chi=n) — classic hard colouring instances
  - Queen graphs Q_n (queen placement on n×n board)
  - Complete graphs K_n (pigeonhole)

All output as DIMACS 'p edge N M' format, 1-indexed vertices.
"""
import sys, os

outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/colour_bench"
os.makedirs(outdir, exist_ok=True)

def write_graph(name, adj):
    """adj: dict vertex -> set of neighbors. Writes DIMACS edge file."""
    verts = sorted(adj.keys())
    edges = set()
    for u in verts:
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
            elif v < u:
                edges.add((v, u))
    n = len(verts)
    # remap to 1-indexed
    with open(os.path.join(outdir, f"{name}.edge"), "w") as f:
        f.write(f"p edge {n} {len(edges)}\n")
        for u, v in sorted(edges):
            f.write(f"e {u+1} {v+1}\n")
    print(f"  {name}: {n} vertices, {len(edges)} edges")

def mycielski(level):
    """Generate Mycielski graph M_k. M_2 = K_2 (edge), M_3 = C_5 (chi=3), etc."""
    if level < 2:
        return {}
    # M_2 = single edge
    adj = {0: {1}, 1: {0}}
    for _ in range(level - 2):
        verts = sorted(adj.keys())
        n = len(verts)
        # new vertices: n copies u_i, plus one apex w
        # u_i = n + i, w = 2*n
        w = 2 * n
        for i in range(n):
            u = n + i
            adj[u] = set()
        adj[w] = set()
        for i in range(n):
            u = n + i
            old_i = verts[i]
            for nb in adj[old_i]:
                # connect u_i to the copy of each neighbor of v_i
                nb_idx = verts.index(nb)
                u_nb = n + nb_idx
                adj[u].add(u_nb)
                adj[u_nb].add(u)
            # connect u_i to apex
            adj[u].add(w)
            adj[w].add(u)
            # remove old edges involving v_i (they stay, but u_i connects to copies of neighbors)
        # Keep old vertices and their edges too
    return adj

def queen_graph(n):
    """Queen graph Q_n: vertices are squares of n×n board, edges if queens attack."""
    verts = {}
    for r in range(n):
        for c in range(n):
            verts[r * n + c] = set()
    for r1 in range(n):
        for c1 in range(n):
            for r2 in range(n):
                for c2 in range(n):
                    if r1 == r2 and c1 == c2:
                        continue
                    v1 = r1 * n + c1
                    v2 = r2 * n + c2
                    if r1 == r2 or c1 == c2 or abs(r1-r2) == abs(c1-c2):
                        verts[v1].add(v2)
    return verts

def complete_graph(n):
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            adj[i].add(j)
            adj[j].add(i)
    return adj

print("Generating benchmark instances:")
print("  Mycielski graphs (triangle-free, chi = level):")
for level in range(3, 8):
    g = mycielski(level)
    write_graph(f"mycielski_{level}", g)

print("  Queen graphs:")
for n in [5, 6, 7, 8]:
    g = queen_graph(n)
    write_graph(f"queen_{n}", g)

print("  Complete graphs:")
for n in [5, 6, 7]:
    g = complete_graph(n)
    write_graph(f"complete_{n}", g)

print("Done.")

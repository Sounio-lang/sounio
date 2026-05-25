#!/usr/bin/env python3
"""Cube-and-conquer for k-colourability of the campaign graphs (Slurm-parallel).

Cubes = the proper k-colourings of a chosen CLIQUE in the graph. Every proper
k-colouring of G restricts to a proper k-colouring of the clique, so the cubes
COVER all colourings:  G is k-colourable  <=>  some cube extends  <=>  some
(CNF + cube) is SAT.  Hence: ALL cubes UNSAT  <=>  G is NOT k-colourable.

Modes:
  split  --edges E.json --colors k --clique-size m --out cubes.jsonl
  solve  --edges E.json --colors k --cube '<json list of [vertex,colour]>' [--timeout T]
Self-test: `python3 cnp_cube.py selftest` (K6 must be all-UNSAT for k=5; a triangle
graph K3 must be SAT for k=5).
"""
import sys, json, itertools, argparse

def clique_find(n, adj, m):
    # greedy: grow a clique from the highest-degree vertex
    deg=sorted(range(n), key=lambda v:-len(adj[v]))
    for start in deg:
        cl=[start]
        for v in deg:
            if v in cl: continue
            if all(v in adj[u] for u in cl): cl.append(v)
            if len(cl)==m: return cl
    return None  # no clique of size m

def proper_colorings_of(clique, adj, k):
    # all assignments colour: clique->{0..k-1} proper on induced edges
    cubes=[]
    for assign in itertools.product(range(k), repeat=len(clique)):
        ok=True
        for i in range(len(clique)):
            for j in range(i+1,len(clique)):
                if clique[j] in adj[clique[i]] and assign[i]==assign[j]: ok=False; break
            if not ok: break
        if ok: cubes.append(list(zip(clique,assign)))
    return cubes

def build_cnf(n, edges, k, cube=None):
    cl=[]
    for v in range(n):
        cl.append([v*k+c+1 for c in range(k)])
        for c in range(k):
            for d in range(c+1,k): cl.append([-(v*k+c+1),-(v*k+d+1)])
    for a,b in edges:
        for c in range(k): cl.append([-(a*k+c+1),-(b*k+c+1)])
    if cube:
        for (v,c) in cube: cl.append([v*k+c+1])   # unit clause: vertex v gets colour c
    return n*k, cl

def solve(nv, cl, timeout=None):
    from pysat.formula import CNF; from pysat.solvers import Solver
    f=CNF(); f.extend(cl); s=Solver(name='cadical153', bootstrap_with=f)
    r=s.solve(); s.delete(); return r

def adj_of(n, edges):
    adj=[set() for _ in range(n)]
    for a,b in edges: adj[a].add(b); adj[b].add(a)
    return adj

def main():
    if len(sys.argv)>1 and sys.argv[1]=="selftest":
        # K6: 6-clique -> chi=6 -> NOT 5-colourable -> all cubes UNSAT
        n=6; E=[(i,j) for i in range(6) for j in range(i+1,6)]; adj=adj_of(n,E)
        cq=clique_find(n,adj,3); cubes=proper_colorings_of(cq,adj,5)
        res=[solve(*build_cnf(n,E,5,cube)) for cube in cubes]
        print(f"K6 k=5: clique={cq} cubes={len(cubes)} any_SAT={any(res)} -> "
              f"{'5-colourable' if any(res) else 'NOT 5-colourable (chi>=6) [CORRECT]'}")
        # K3 (triangle): chi=3 -> 5-colourable -> some cube SAT
        n=3; E=[(0,1),(0,2),(1,2)]; adj=adj_of(n,E)
        cq=clique_find(n,adj,3); cubes=proper_colorings_of(cq,adj,5)
        res=[solve(*build_cnf(n,E,5,cube)) for cube in cubes]
        print(f"K3 k=5: cubes={len(cubes)} any_SAT={any(res)} -> "
              f"{'5-colourable [CORRECT]' if any(res) else 'NOT (wrong!)'}")
        return
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='mode',required=True)
    sp=sub.add_parser('split'); sp.add_argument('--edges',required=True); sp.add_argument('--n',type=int,required=True)
    sp.add_argument('--colors',type=int,default=5); sp.add_argument('--clique-size',type=int,default=3); sp.add_argument('--out',required=True)
    so=sub.add_parser('solve'); so.add_argument('--edges',required=True); so.add_argument('--n',type=int,required=True)
    so.add_argument('--colors',type=int,default=5); so.add_argument('--cube',required=True); so.add_argument('--timeout',type=int,default=0)
    a=ap.parse_args()
    E=[tuple(e) for e in json.load(open(a.edges))]; adj=adj_of(a.n,E)
    if a.mode=='split':
        cq=clique_find(a.n,adj,a.clique_size)
        if cq is None: print(json.dumps({"error":f"no {a.clique_size}-clique"})); sys.exit(1)
        cubes=proper_colorings_of(cq,adj,a.colors)
        with open(a.out,'w') as f:
            for cube in cubes: f.write(json.dumps([[int(v),int(c)] for v,c in cube])+"\n")
        print(json.dumps({"clique":cq,"ncubes":len(cubes)}))
    else:
        cube=json.loads(a.cube)
        nv,cl=build_cnf(a.n,E,a.colors,cube)
        r=solve(nv,cl,a.timeout if a.timeout>0 else None)
        print(json.dumps({"cube":cube,"sat":r}))

if __name__=="__main__": main()

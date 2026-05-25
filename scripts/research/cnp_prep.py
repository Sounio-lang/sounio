#!/usr/bin/env python3
# Build the closure graph for (size,rcap), dump edges.json + n, then split into
# cube-and-conquer cubes (proper k-colourings of a clique). Reuses the worker + cnp_cube.
import sys, json, argparse, importlib.util
def load_mod(name, path):
    spec=importlib.util.spec_from_file_location(name, path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
ap=argparse.ArgumentParser()
ap.add_argument('--dir', required=True)         # staged dir with worker.py, cnp_cube.py, cnp510.*
ap.add_argument('--size', type=int, required=True); ap.add_argument('--rcap', type=float, required=True)
ap.add_argument('--colors', type=int, default=5); ap.add_argument('--clique-size', type=int, default=3)
a=ap.parse_args()
w=load_mod('w', a.dir+'/worker.py'); cube=load_mod('cube', a.dir+'/cnp_cube.py')
V0,MOVES=w.load(a.dir)
VV=w.closure(V0,MOVES,a.size,a.rcap); n=len(VV)
E=w.complete_edges(VV)
ej=f"{a.dir}/edges_{a.size}.json"; json.dump([[a,b] for a,b in E], open(ej,'w'))
adj=cube.adj_of(n,E); cq=cube.clique_find(n,adj,a.clique_size)
cubes=cube.proper_colorings_of(cq,adj,a.colors)
cj=f"{a.dir}/cubes_{a.size}.jsonl"
with open(cj,'w') as f:
    for cb in cubes: f.write(json.dumps([[int(v),int(c)] for v,c in cb])+"\n")
print(json.dumps({"size":a.size,"n":n,"edges":len(E),"clique":cq,"ncubes":len(cubes),
                  "edges_json":ej,"cubes_jsonl":cj}))

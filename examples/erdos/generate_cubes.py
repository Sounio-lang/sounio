#!/usr/bin/env python3
"""
Generate cube files for cube-and-conquer on G529 4-colouring.

Cubes fix the colours of selected high-degree vertices.
4^N cubes for N cubed vertices, 4 colours each.

Output: cube_XXXX.cube files in WORKDIR, plus manifest.txt.
"""
import sys, os, itertools

edge_file = sys.argv[1] if len(sys.argv) > 1 else "degrey_529.edge"
workdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/cubes_g529"

# 0-indexed vertices to cube on (high-degree, not precoloured)
cube_vertices = [188, 189, 221]  # 1-indexed: 189, 190, 222

os.makedirs(workdir, exist_ok=True)

k = 4
manifest = open(os.path.join(workdir, "manifest.txt"), "w")
idx = 0
for colours in itertools.product(range(k), repeat=len(cube_vertices)):
    cubefile = os.path.join(workdir, f"cube_{idx:04d}.cube")
    with open(cubefile, "w") as f:
        for v, c in zip(cube_vertices, colours):
            f.write(f"{v} {c}\n")
    manifest.write(f"cube_{idx:04d}.cube\n")
    idx += 1
manifest.close()
print(f"Generated {idx} cubes on vertices {cube_vertices} in {workdir}")

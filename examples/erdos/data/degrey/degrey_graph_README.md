# de Grey graph dataset

This archive contains a reconstructed dataset for the 1581-vertex unit-distance de Grey graph from Aubrey D. N. J. de Grey's 2018 construction for the Hadwiger–Nelson problem.

## Files

- `degrey_graph_vertices.csv` — vertex table with numeric coordinates (`id,x,y`).
- `degrey_graph_edges.csv` — undirected edge list (`source,target`), 1-indexed.
- `degrey_graph_edges.dimacs` — DIMACS edge format.
- `degrey_graph_adjacency.txt` — adjacency list, 1-indexed.
- `degrey_graph_graph.json` — JSON graph object containing vertices and edges.
- `gen_degrey.py` — Python script used to reconstruct the dataset from the construction in de Grey 2018.

## Integrity checks

- Vertices: 1581
- Edges: 7877
- Construction: unit-distance graph in the Euclidean plane; edges are all vertex pairs at squared Euclidean distance 1 within numerical tolerance.

## Degree distribution

- degree 4: 24 vertices
- degree 6: 212 vertices
- degree 7: 202 vertices
- degree 8: 236 vertices
- degree 9: 88 vertices
- degree 10: 224 vertices
- degree 11: 192 vertices
- degree 12: 137 vertices
- degree 13: 48 vertices
- degree 14: 72 vertices
- degree 16: 24 vertices
- degree 17: 48 vertices
- degree 18: 48 vertices
- degree 20: 24 vertices
- degree 60: 2 vertices

## Notes

The vertex ordering is deterministic but not necessarily identical to Wolfram GraphData's internal ordering. Coordinates are stored as decimal approximations. The construction follows the explicit seven-step generation in de Grey's paper and was verified against the published vertex/edge counts.

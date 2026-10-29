# Independent frontier observations

These fixtures preserve a specific old accepted compiler/current SDK diagnostic, plus a separate graph observability probe. They do not change the original stdlib tests or qualify the failed root05 compiler.

- `graph_metrics.sio` constructs the same four-edge graph as `test_graph_types_e2e.sio`, and prints actual node/edge counts, degrees, all adjacency pairs and the neighbor of node3.
- `graph_reference.cpp` independently constructs the undirected adjacency sets in C++23 and requires exactly the expected25 output lines. This companion adds observations; it does not replace the original test.
- `mutate.cpp` produces wrong-value, dropped-row and extra-row validator controls for both Graph and DP54.
- The observability recipe extracts the actual `invalid_control` function from the current SDK's `tests/reference/dp54_dense_events_reference.cpp`, hashes its literal body and compiles it with a strict final-marker wrapper. It does not introduce a second DP54 algorithm.
- `recipes/` are the exact executed one-shot Slurm recipes, with absolute custody paths and private toolchain references. The original tests remain in the sealed current SDK. Do not redirect these recipes to a different compiler silently.

See `docs/evidence/stdlib-wave4-independent-frontiers/README.md` for the source/compiler distinction, outcomes, original identities and durable custody.

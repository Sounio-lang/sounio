<!-- docs:meta
topic_id: repo.examples.cognitive-ossm.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.examples.cognitive-ossm.readme
-->

# Cognitive O-SSM (Canonical Sounio Lane)

This directory is the canonical `sounio-lang/sounio` side of the SWOW-EN O-SSM integration.

What is here:

- `cognitive_ossm.sio`
  - runnable architecture smoke for the octonionic update rule itself
- `run_regimes.sio`
  - bounded parity runner that reads the compact SWOW bundle exported from `hyperbolic-semantic-networks/data/cpc2026/sounio_input/`
- `export_results.sio`
  - emits a small manifest for the bounded parity outputs

Important scope boundary:

- The **full paper-scale O-SSM artifacts** live in the Python mirror:
  - `hyperbolic-semantic-networks/code/cpc2026/ossm_reference_simulator.py`
- The Sounio lane is currently an **executable parity / smoke path**, not the source of the 10,000 x 500 full-run CSVs.
- The parity runner intentionally defaults to the `*_nodes_parity.csv` bundle, which is a bounded 64 x 64 subset exported by:
  - `hyperbolic-semantic-networks/code/cpc2026/ossm_bridge/export_to_sounio.py`

Recommended commands from the `sounio-lang/sounio` root:

```bash
./artifacts/omega/souc-bin/souc-linux-x86_64-gpu run examples/cognitive_ossm/cognitive_ossm.sio
./artifacts/omega/souc-bin/souc-linux-x86_64-gpu run examples/cognitive_ossm/run_regimes.sio -- --max-trajectories 8 --max-steps 64
./artifacts/omega/souc-bin/souc-linux-x86_64-gpu run examples/cognitive_ossm/export_results.sio
```

Expected output directory:

- `examples/cognitive_ossm/results/`

Expected files:

- `ossm_parity_{regime}.csv`
- `ossm_parity_summary_{regime}.csv`
- `ossm_parity_manifest.csv`

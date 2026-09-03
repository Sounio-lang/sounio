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
  - legacy bounded smoke runner; its `process_regime` path does not execute the full reference recurrence and must not be used as parity evidence
- `run_ossm_native_reference.sio`
  - self-contained byte-level implementation of the reference recurrence
  - compiles and runs through a rebuilt Madaros native-v2 compiler
  - the evidence gate exercises a bounded `2 trajectories x 8 steps` runtime only
- `export_results.sio`
  - emits a small manifest for the bounded parity outputs

Important scope boundary:

- The **full paper-scale O-SSM artifacts** live in the Python mirror:
  - `hyperbolic-semantic-networks/code/cpc2026/ossm_reference_simulator.py`
- The Sounio lane currently provides architecture smokes, checkable reference source, and epistemic receipts. It is **not** the source of the 10,000 x 500 full-run CSVs.
- The legacy runner defaults to the `*_nodes_parity.csv` bundle, a bounded 64 x 64 subset exported by:
  - `hyperbolic-semantic-networks/code/cpc2026/ossm_bridge/export_to_sounio.py`
- The committed n=100/n=1000 native JSON files are historical pre-parser-fix reruns and are explicitly excluded from parity claims.
- A prior corrected-parser omega 1.0.0-beta.4 receipt reported an absolute delta of `2.03e-10`; omega was not available for re-verification on 2026-07-11.

Recommended commands from the `sounio-lang/sounio` root:

```bash
./bin/souc check examples/cognitive_ossm/run_ossm_native_reference.sio
CPC2026_SCIENTIFIC_REPO=/workspace/hyperbolic-semantic-networks \
CPC2026_MADAROS_RAW_BIN=/tmp/rebuilt-madaros \
  bash scripts/ci/cpc2026_yale_evidence_gate.sh
uv run --with numpy python scripts/research/cpc2026_ossm_subset_audit.py
```

The bounded native run proves compiler/runtime execution and JSON production. It
does not replace or independently replicate the frozen Python `n=10,000` result.

Historical bounded output directory:

- `examples/cognitive_ossm/results/`

Expected files:

- `ossm_parity_{regime}.csv`
- `ossm_parity_summary_{regime}.csv`
- `ossm_parity_manifest.csv`

These small legacy CSVs are smoke artifacts, not the paper-scale statistics.
See `docs/research/cpc2026_yale_evidence_dossier_2026-07-11.md` for the complete claim ledger.

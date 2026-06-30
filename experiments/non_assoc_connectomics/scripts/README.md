# Reference scripts for `octonion_arc_findings.md`

NumPy/PyTorch reference implementations backing the quantitative claims in
`../octonion_arc_findings.md`. These are research/methods references (not a packaged
pipeline): some carry absolute cache paths from the session and expect the ABIDE
CC200 `.1D` files under `artifacts/research/abide/`; adjust `CACHE`/`ABIDE` constants
before re-running. `fc_baseline.py` builds the FC cache the others reuse.

The canonical, self-contained, runnable artifact is the Sounio program
`examples/physics/octonion_mass_delta.sio` (mass δ-consistency + cross-sector held-out
+ GUM uncertainty + octonion-associator check), which reproduces the physics results
bit-identically and ties to `formal/OctonionAssociator.lean`.

| script | produces (section in findings doc) |
|---|---|
| `fc_baseline.py` | full-FC linear LOSO baseline 65.9% (§2.2) |
| `fc_vs_octonion.py` | matched-input linear vs O-SSM/H-SSM (§2.3) |
| `octonion_positive_control.py` | double dissociation 99.5 vs 55.7 (§3.1) |
| `degree_matched_control.py`, `mlp_baseline.py` | degree/capacity controls (§3.2) |
| `assoc_ablation.py` | trained ablation, weak readout +4.1 (§3.3) |
| `assoc_ablation_fair.py` | fair ablation +35.4, pre-committed rule (§3.3) |
| `assoc_ablation_fc.py`, `learned_embedding_fc.py` | brain re-tests −1.78 / +0.28 (§2.4) |
| `koide_jordan_test.py` | Koide diamond + δ²=3/8 bridge (§4.2) |
| `singh_tableI_test.py` | Singh Table I vs PDG (§4.3) |
| `delta_consistency.py` | δ-cluster + null p<1e-5 (§4.4) |
| `heldout_crosssector.py` | quark→lepton held-out prediction (§4.5) |
| `assignment_enumeration.py` | bounds the assignment freedom: only ~2/81 cross-sector-consistent, both at √(3/8) (§4.7) |
| `centers_from_charges.py` | sector centers = electric charges, recovered mass-blind; compound-ratio forward checks (§4.8) |

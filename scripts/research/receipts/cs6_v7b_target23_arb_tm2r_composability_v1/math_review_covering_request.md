# Math review request: full-support covering B -> C

Review the proposed rigorous analyzer, verifier, gate, and contract:

- `scripts/research/cs6_v7b_target23_arb_tm2r_composability_analyze.py`
- `scripts/research/cs6_v7b_target23_arb_tm2r_composability_verify.py`
- `scripts/research/cs6_v7b_target23_arb_tm2r_composability_gate.sh`
- `scripts/research/cs6_v7b_target23_arb_tm2r_composability_covering_contract_v1.txt`

The four complete carrier receipts are still running on Slurm. Review the
mathematical method and fail-closed implementation, not any positive numerical
claim. In particular:

1. Is restricting each terminal Taylor model to local `xi=-1` or `xi=+1`
   mathematically valid for the global source faces after exact affine binary
   reconditioning, when every terminal lineage forms a complete cover?
2. Do strict opposite unstable face signs plus the entire stable image inside
   the target entry interval imply degree `+1` in unstable dimension one
   without a monotonicity hypothesis?
3. Is the source chart determinant for h-set B
   `U_RADIUS*S_RADIUS/(ROW_U_X^2+ROW_U_Y^2)` and the analogous target formula
   correct for the row-coordinate convention used by `target_hset()`?
4. Is the Poincare determinant enclosure
   `exp(ell) * initial_normal / final_normal * source_chart_determinant`, then
   divided by the target chart determinant, correct under the existing worker
   convention?
5. Identify any unsound trust in serialized flags, incomplete face-cover
   validation, interval-correlation issue, sign error, or verifier gap.

Return concrete findings ordered by severity. Do not promote recurrence,
chaos, novelty, priority, or an open-problem solution.

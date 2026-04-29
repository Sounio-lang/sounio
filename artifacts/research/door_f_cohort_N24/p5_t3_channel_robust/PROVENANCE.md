# P5.5 / PROTOCOL §2 T3 — Channel-subset robustness

Seed: 20260421 (PROTOCOL-frozen in plan_channel_draws.py)

Plan (plan_channel_draws.py): 24 patients × 100 random 16-channel subsets
drawn independently per patient via numpy.random.SeedSequence(20260421)
with patient-level spawn keys.

Execution: all 2400 (patient, draw) .sio files compiled and run locally
against bin/souc 2026-04-21 build; no Slurm dependency after OrangeFS
PVFS2 concurrent-write corruption was detected and bypassed. Total
wall-clock 120 s at 6-way parallelism.

Decision rule (PROTOCOL §2 T3): robust iff ≥ 95 % of the 100 draws yield
a cohort-median with the canonical positive sign, separately for dip and
for spike. Both must pass.

Result: dip_sign_preserved = 1.000, spike_sign_preserved = 1.000. PASS.

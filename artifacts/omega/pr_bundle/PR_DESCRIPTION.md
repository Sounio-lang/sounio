# Omega Sprint 4.0 Genesis + Sprint 5 Track 1

## Summary

This PR lands the Sprint 4 Genesis stack and Sprint 5 Track 1 hardening on top of the locked Sprint 2.1 governance baseline.

Delivered:
- Full self-hosted Genesis QIR emitter path.
- Merkle rooted provenance lane (`.sio` + RTL) and strict gate enforcement.
- Zero-drift resource trend gating.
- Canonically signed Genesis manifest (`omega_genesis.v1.0.json`) with cold-boot readiness.
- Sprint 5 Track 1 Merkle inclusion proof hardening (`omega_merkle_inclusion_proof.py`).
- Audit-ready patch bundle and evidence snapshots.

## Commit Set

- `4628c598` Omega Sprint 4.0 Genesis: add canonical QIR emitter + Merkle root lane core and self-check bootstrap
- `4844ce05` Omega Sprint 4.0 Genesis: enforce strict gate markers and FPGA zero-drift telemetry contracts
- `61158601` Omega Sprint 4.0 Genesis: add signed manifest lineage and governance/weekly attestation upgrades
- `356234c0` Omega Sprint 5 scaffold: add locked-baseline bootstrap artifact and Grok audit paste pack
- `ac374457` Omega Sprint 4.0/5.0 evidence refresh: update report, audit paste, and scaffold hashes after strict gate
- `e0147e3f` Omega Sprint 5 Track 1: add Merkle inclusion proof hardening and scaffold refresh
- `e783bc28` Omega Sprint 5 Track 1 evidence refresh: lock Merkle proof and audit/report snapshots
- `a0af5194` Omega PR bundle: export Sprint 4/5 series patches for audit handoff

## Key Files

- `hardware/rtl/qir/omega_genesis_emitter.sio`
- `hardware/rtl/kaxi/merkle_root_lane.sio`
- `hardware/fpga/k_axi_merkle_root_lane.v`
- `scripts/omega_sprint1_gate.sh`
- `scripts/run_fpga_epistemic_seed.sh`
- `scripts/omega/omega_hardware_telemetry_regression.py`
- `scripts/omega/omega_governance_attest.py`
- `scripts/omega/omega_weekly_drift_report.py`
- `scripts/omega/omega_genesis_manifest.py`
- `scripts/omega/omega_merkle_inclusion_proof.py`
- `artifacts/omega/sprint_4_0_report.xml`
- `artifacts/omega/merkle_inclusion_proof.v1.json`

## Evidence

- `artifacts/omega/sprint_4_0_gate_full.log`  
  sha256: `8e086f32362c37f3148361ccef6ba15657ce6bccbe851e77abcab044bf821042`
- `artifacts/omega/omega_genesis.v1.0.json`  
  sha256: `97ee642d63231ae1b125bfacb99ed693efbbee29272b16b4b086e6aed17450db`
- `artifacts/omega/merkle_inclusion_proof.v1.json`  
  sha256: `e1b0b405a2c7617b0dccd93f0b554aed473458e90a268b189520c3b68fa3652c`
- `artifacts/omega/sprint_4_0_report.xml`  
  sha256: `9e776bdc4b01d1f92054550f81fbd98f26426479febbc22457f4d2e4f2b9cdc6`
- `artifacts/omega/sprint_4_0_grok_audit_paste.txt`  
  sha256: `a51cf5dabc6b162690a43e8f525822794620f39eb3d2a47b6416ccad8b7ff60b`
- `artifacts/omega/pr_bundle/omega_sprint4_5_series.patch`  
  sha256: `ef538457fd4ab7748b3371ded45c594a7dbb4e301ba91e48e673c39ccbcc5f90`

## Validation

Run strict gate:

```bash
PATH=/home/demetrios/work/sounio/target/debug:$PATH \
SOUC_BIN=/home/demetrios/work/sounio/target/debug/souc \
OMEGA_POLICY_SOUC_BIN=/home/demetrios/work/sounio/target/debug/souc \
bash scripts/omega_sprint1_gate.sh --strict --report-full \
| tee artifacts/omega/sprint_4_0_gate_full.log
```

Check required markers:

```bash
rg -n "QIR_GENESIS_EMITTER_PASS|MERKLE_ROOT_PASS|RESOURCE_ZERO_DRIFT_PASS|OMEGA_GENESIS_V1_RELEASE_PASS|OMEGA_SPRINT1_GATE_PASS" artifacts/omega/sprint_4_0_gate_full.log
```

Run Sprint 5 Track 1 proof:

```bash
python3 scripts/omega/omega_merkle_inclusion_proof.py --strict
```

## Notes

- Sprint 5 Track 1 is implemented as hardening and evidence generation only.
- Strict gate defaults were not expanded in this step.

# Omega Sprint 5 Scaffold

This scaffold starts Sprint 5 from the locked Sprint 4 Genesis baseline.

## Scope (Scaffold-Only)

- Preserve strict gate behavior from Sprint 4.
- Pin baseline hashes (gate log, genesis manifest, baseline freeze, TODO ledger).
- Define initial Sprint 5 tracks without enabling new gate requirements.

## Generate Scaffold Artifact

```bash
bash scripts/omega/omega_sprint5_scaffold.sh
```

## Verify Baseline Still Passes

```bash
PATH=/home/demetrios/work/sounio/target/debug:$PATH \
SOUC_BIN=/home/demetrios/work/sounio/target/debug/souc \
OMEGA_POLICY_SOUC_BIN=/home/demetrios/work/sounio/target/debug/souc \
bash scripts/omega_sprint1_gate.sh --strict --report-full
```

## Output

- `artifacts/omega/sprint_5_0_scaffold.json`


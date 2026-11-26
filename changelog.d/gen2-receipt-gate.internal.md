CI Decision now requires a gen2 fixed-point receipt for the compiler state at the head. A state is identified by its gen2 inputs key: sha256 over the git object ids of every input listed in `scripts/ci/gen2_receipt_inputs.txt`, plus that file's parameters. The list is fail-closed: compiler, stdlib, bootstrap ELFs, the fixed-point gate, the build script and their helpers. The receipt is `docs/evidence/gen2-fixpoint/<key>.receipt` and records:
- the full commit, which the gate fetches if absent and refuses if it cannot;
- the Slurm job and node;
- `gate_rc=0` and `ir_parity=pass`;
- full sha256 of gen1, gen2 and gen3, with gen2 required to equal gen3;
- MaxRSS and wall time.

Checked by `scripts/ci/gen2_receipt_gate.sh` in Fast Checks, with sixteen controls (`--selftest`). The receipt is a signed-off attestation checked for consistency, not a re-execution.

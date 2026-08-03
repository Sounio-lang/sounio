1. [BLOCKER] The verifier's `recompute_determinant` uses `Decimal(fields[f"{prefix}_Q0_AREA"])` but the worker only emits `Q0_AREA` as a single field, and the result column `Q0_AREA` is populated from `fields.get("FINE_Q0_AREA")`, so the coarse recomputation will fail with a KeyError when `COARSE_Q0_AREA` is missing.
   - scripts/research/cs6_v7b_target23_decimal_center_replay_verify.py: recompute_determinant()
   - The verifier will crash on every leaf, making the entire verification pipeline non-functional.
   - Fix: Use a single `Q0_AREA` field for both resolutions, or emit `COARSE_Q0_AREA` and `FINE_Q0_AREA` from the worker.

2. [BLOCKER] The worker's `final_normal` computation uses `second[0] * second[1] - second[2] - Decimal(ZS)`, but the contract's `NORMAL_VELOCITY_ON_SECTION=x*y-zs` does not include the `-w` term, so the final normal is computed with an extra `-w` term that is not part of the section definition.
   - scripts/research/cs6_v7b_target23_decimal_center_replay_worker.py: integrate()
   - This produces a systematically different determinant than the contract specifies, invalidating the pointwise falsification against the CAPD enclosures.
   - Fix: Remove `- second[2]` from the final_normal computation to match `x*y - zs`.

3. [BLOCKER] The `build_plan` function in the runner and verifier reads `leaf.input_sha256` from the manifest, but the `load_leaves` function from `cs6_v7b_target23_prospective_epistemic_replay_run` is not shown, and there is no verification that the manifest's `INPUT_SHA256` field actually binds to the leaf coordinates used in the worker command.
   - scripts/research/cs6_v7b_target23_decimal_center_replay_run.py: build_plan()
   - A malicious or corrupted manifest could provide coordinates different from those hashed into `INPUT_SHA256`, allowing the worker to run on different orbits than those frozen in the contract.
   - Fix: Recompute the leaf input hash from the actual coordinates and verify it matches `leaf.input_sha256` before building the plan.

4. [MAJOR] The `verify` function checks `stderr` is empty but does not verify that the worker's `stdout` contains exactly the expected fields; `parse_fields` accepts any key-value lines, so a worker that prints extra fields (e.g., `MUTATED=true`) would still pass if the required fields are present.
   - scripts/research/cs6_v7b_target23_decimal_center_replay_verify.py: verify()
   - The mutation test M09 (append_first_stdout) would not be detected because the verifier only checks for the presence of required keys, not the absence of unexpected ones.
   - Fix: In `parse_fields`, reject any key not in a predefined allowlist of worker output fields.

5. [MAJOR] The `verify` function does not verify that the `command.txt` file's first element (the Python executable) matches the actual interpreter used; it only checks `command[1] == "-B"` and `Path(command[2]).name == WORKER_REL.name`, so a different Python binary could be substituted.
   - scripts/research/cs6_v7b_target23_decimal_center_replay_verify.py: verify()
   - This weakens the binding of the worker source to the execution environment, as a different interpreter with modified Decimal semantics could be used.
   - Fix: Verify the absolute path of the executable matches the expected system Python, or hash the interpreter binary.

6. [MAJOR] The `verify` function does not check that the `provenance/python-runtime.txt` file's `PYTHON_VERSION` matches the actual Python version used for the run; it only checks `PYTHON_DECIMAL_IMPLEMENTATION` and `CAPD_IMPORTED`.
   - scripts/research/cs6_v7b_target23_decimal_center_replay_verify.py: verify()
   - A different Python version could have different Decimal rounding behavior, invalidating the precision claims.
   - Fix: Compare the recorded `PYTHON_VERSION` against the verifier's own Python version.

7. [MAJOR] The `write_summary` function in the runner hardcodes `INDEPENDENT_POINTWISE_SCOUT_COMPLETED=false`, but the `verify` function expects `INDEPENDENT_POINTWISE_SCOUT_COMPLETED=true` in the summary after successful verification; this is never updated in the receipt.
   - scripts/research/cs6_v7b_target23_decimal_center_replay_run.py: write_summary()
   - The summary file will always say `false`, causing the gate script's `grep -qx 'INDEPENDENT_POINTWISE_SCOUT_COMPLETED=true'` to fail.
   - Fix: Update the summary file after verification, or remove the `false` value and let the verifier set it.

8. [MAJOR] The Slurm script's return mechanism uses a raw TCP socket with no authentication or integrity check beyond the SHA256 in the header; a man-in-the-middle could substitute a fake result archive with a matching SHA256 if the archive is small enough to brute-force or if the SHA256 is leaked.
   - scripts/research/cs6_v7b_target23_decimal_center_replay_slurm_job.sh: result return
   - The result integrity is not cryptographically bound to the Slurm job or the worker source.
   - Fix: Sign the result archive with a private key whose public key is pinned in the contract.

9. [MAJOR] The `verify` function's `minimum_margin` calculation uses `min(fine - lower, upper - fine)` but does not verify that `fine` is strictly inside the interval; if `fine` equals an endpoint, the margin is zero, which could indicate a boundary case that the CAPD certificate does not support.
   - scripts/research/cs6_v7b_target23_decimal_center_replay_verify.py: verify()
   - A zero margin means the pointwise determinant is exactly on the enclosure boundary, which is not a strict containment and could be numerically unstable.
   - Fix: Require `margin > 0` and fail if any leaf has zero margin.

10. [MINOR] The worker's `center` function computes the center of the leaf as `-r + (index + 0.5) * (2r) / 2^depth`, but the contract specifies `FROZEN_LEAF_MANIFEST` with specific leaf coordinates; there is no cross-check between the manifest's leaf coordinates and the `center` function's computed values.
    - scripts/research/cs6_v7b_target23_decimal_center_replay_worker.py: center()
    - If the manifest's leaf definitions differ from this formula (e.g., different origin or basis vectors), the worker would integrate the wrong orbit.
    - Fix: In `build_plan`, recompute the center from the manifest's depth/index and verify it matches the manifest's coordinates.

11. [MINOR] The `verify` function checks `SLURM_CPUS_PER_TASK == "32"` but does not verify that the actual number of cores used by the runner matches; the runner's `--jobs` argument is not checked against the Slurm allocation.
    - scripts/research/cs6_v7b_target23_decimal_center_replay_verify.py: verify()
    - A CPU count mismatch could affect timing but not correctness; still, it weakens the reproducibility claim.
    - Fix: Verify the runner's `--jobs` argument in the command line matches the Slurm allocation.

12. [MINOR] The `mutations.py` script's M05 mutation changes `FINE_DETERMINANT` to `-1E-20`, but the verifier's `recompute_determinant` would still compute a different value, so the mutation is detected; however, M06 changes `ABSOLUTE_DETERMINANT_DELTA` to `0`, which would pass if the recomputed delta is also zero (which is unlikely), but the mutation test does not verify the failure signature is meaningful.
    - scripts/research/cs6_v7b_target23_decimal_center_replay_mutations.py: mutations()
    - The mutation tests only check that verification fails, not that it fails for the right reason, so a mutation that causes a different but still failing path would pass the gate.
    - Fix: For each mutation, verify the failure message matches an expected pattern.

13. [MINOR] The Slurm script's `tar -cf` command does not use `--sort=name` or `--mtime`, so the archive's byte order and timestamps are non-deterministic, meaning the `RESULT_SHA256` cannot be reproduced by a third party.
    - scripts/research/cs6_v7b_target23_decimal_center_replay_slurm_job.sh: archive creation
    - The SHA256 of the result archive is not reproducible, weakening the provenance claim.
    - Fix: Use `tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner` for deterministic archives.

14. [MINOR] The `verify` function does not check that the `provenance/slurm-control-plane.txt` file is non-empty or contains the expected `scontrol` output format; it only checks the `slurm-context.txt` file.
    - scripts/research/cs6_v7b_target23_decimal_center_replay_verify.py: verify()
    - The control-plane output is not validated, so a forged or empty file would pass.
    - Fix: Parse `slurm-control-plane.txt` and verify it contains the expected job ID and node list.

15. [NIT] The contract specifies `COARSE_DECIMAL_PRECISION=50` and `FINE_DECIMAL_PRECISION=80`, but the worker's `integrate` function uses `localcontext` with these precisions only for the integration; the `exp()` call in the determinant computation uses the same context, which is correct, but the `combine` function uses `sum(..., Decimal(0))` which may lose precision in the sum ordering.
    - scripts/research/cs6_v7b_target23_decimal_center_replay_worker.py: combine()
    - The sum order is deterministic but not guaranteed to be optimal for precision; this is a minor numerical concern.
    - Fix: Document the sum ordering or use a more precise accumulation method (e.g., `math.fsum` for Decimal).

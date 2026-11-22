<!-- docs:meta
topic_id: repo.docs.audit.native-compile-gates-scale-contract-20261010
authority: repo_only
audience: users
last_validated: 2026-10-10
validated_by: Codex metadata review; audit provenance 15e8184ef3 (2026-10-10)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.native-compile-gates-scale-contract-20261010
-->

# Runtime threshold contract: verified result and remaining backend failure

The candidate is isolated in `/workspace/.tmp/compiler-gates-20261010`, based on published main `45f53364c6f66a4fdcfd1788e2abbd93b70a4106`. No push or merge occurred, and no lower/IR/serialize file was edited.

## Verified

* Job13357: Div-only source-built Madaros checked the driver normally with a64MiB stack, zeroE035 and the six existingE012. The first check under8MiB crashed; its absence-of-E035 label was withdrawn.
* Job13366(v3): Madaros rebuilt from exact source; all50 driver modules checked successfully, eliminating bothE012 andE035. The driver compiled into anELF, but runtime crashed before emitting guards for all18cases. This is a failed runtime result, preserved separately from source checking.
* Job13387(v4): same target source compiled through Madaros still crashed at the sameRIP; the source-built seed route executed the first mg/g witness. The complete seed matrix found13passes and5failures. The scanner failures were diagnosed and fixed, rather than counted as successes.
* Job13388(v5): driver built from exact v5 source using the source-tracking seed. All18cases passed:17runtime cases had a positive emitted-guard counter and expected exit; dimension mismatch refused compilation with diagnostic and noELF. Slurm completed with exit0 in27seconds. This is seed-backend runtime evidence, not a claim that the Madaros-generated driver executes.

## Change

The existing f64 AST value now reaches a native f64-bit load without inventing decimal scale fields. Units convert threshold from its declared unit into the field's registered unit using normalized positive rational factors, then compare directly, preserving zero and negative bounds. Purei64 fields retain integer comparison, including9007199254740993. Unsupported conversion, missing/incompatible units and nonfinite thresholds refuse compilation. Registered unit annotations are f64 in this frontend; no integer unit representation was invented.

Signed literals are parsed as Knowledge thresholds. Both native token scanners now recognize signed numeric thresholds. Signature/body scans track constraint braces so a comparison `>` cannot consume the outer generic bracket and hide a factory function. The existing precompute wrapper performs refusal checks without parsing the source twice.

## Provenance

Artifacts:`/orangefs/training/sounio-foundry/runs/threshold-contract-v5-20261010/results`.

* Base source archiveSHA256:`2e70a9e3765fe313ca3dc735106f091979fa2537dfda0618cb91d134f4d6d9a4`.
* V5 source/fixture overlaySHA256:`f894629717f686d02cfa6f61a30421e280bebbf9613edba40198d2864d351f90`.
* Source-tracking seedSHA256:`09a2fe17be9f1819be1fed1cc8df49bddbf0b278d74bdd579e919ca83605b739`.
* V5 driver compiled by seedSHA256:`72ea02d3829d8ea77a3b9fc5527578be835c1bf7203456334cf56fa3d299135c`.
* MatrixSHA256:`3b0ddadf53d118952740c8244119236d4f412e53161a5e86cfd7f3e7c06f6bdb`.

Exact source hashes, matching worktree readback:

```
707ea5681d51906584cf971b4deb0205e8a5fc7ed6fc1790e40dae4c0653d0cf  self-hosted/check/knowledge_context.sio
c0fb69482eeb57cc1e02d3fddc9f1eff1634cc022c533c847c185bb9c0fc1e18  self-hosted/compiler/knowledge_runtime_guard_lowering_plan.sio
82db31bc2979dd5ad3f7230b3dca75b682182466f0ae7e0ae8d1b9d92115900c  self-hosted/compiler/native_compile_driver.sio
0cab0082c2dc048f48282b6913ce9db5daea973d209f91ab308989f3a8b74536  self-hosted/parser/types.sio
```

## Backend failure requiring integration owner

Madaros-generated native driver dereferences a null indexed base atRIP`0x68cd03`: `mov(%rax,%rbx,8),%rax`, withrax0/rbx6. The same crash remains in v4 after simplifying preflight; the identical source works via the seed route. Debugging evidence is in v3 `results/driver-crash-gdb.log` and v4 `results/madaros-gdb.log`. No claim of preexistence is made without a working exact baseline.

## External review

DevPass Qwen delivered a real review and found two generated identifier mistakes; both were source-confirmed and fixed before the accepted v5 run. Other speculative suggestions were rejected against code. GLM exhausted its reasoning/output budget without a visible review; that call is not counted as a completed review. Details:`Threshold-Candidate-Model-Review-Checked-20261010.md`.

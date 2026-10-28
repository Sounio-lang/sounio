# Conditional ownership for logical RHS

Production delta:45 lines in self-hosted/check/check.sio (39 added,6 removed), based on root05359 plus the exact constant-scope predecessor8df245b3d4, adopted locally as31bd57c592. SourceProfile Worker hooks and the constant-scope changes are preserved.

Both active BinaryExpr spines and the unused legacy fields helper route &&/|| RHS through one shared helper. LHS remains unconditional. The helper enters the same BorrowEnv and namespace branch contexts used by if, checks RHS once, rewinds to the skip path, and uses the existing linear/affine/namespace merges. Conditional linear consumption emits E040; affine consumption on either path remains consumed for later-use checking. Branch depth also preserves the existing conservative loan-liveness rules.

The helper has one exit. It restores the caller's immutable BorrowEnv frame links using an explicit header constructor and restores namespace depth after merging. E397 remains a storage error and stays sticky. Effects remain conservative and literals receive the existing if semantics; no Lean certificate or runtime-lowering change is claimed.

## Qualification

-19294 builds baseline and candidate checker components from the same source context, both including the30-line constant fix. All51 existing cases preserve verdicts and diagnostics; all7 wrong-output controls pass. All10 constant-scope cases remain unchanged.
-19325 passes19 focused cases with a C++23 verifier and7 adversarial output controls. RHS-only linear consumption changes baseline0 to E040 for &&, ||, conservative literal cases and12/24 nested RHS. Definite LHS consumption and pure/nested boolean/counter programs remain valid. Affine reuse and conditional-loan violations preserve E039/E038.
-The namespace fixture's before trace retains its original known set; after tracing shows env_join=-1 and the final union=-1. This verifies the surviving binding as well as executing the join.
-19326 appends a test-only bridge to an isolated copy of check.sio; the production prefix is hashed separately. Both inplace and byvalue paths pass normal RHS, RHS type error and a deliberately injected byte-accounting overflow after the branch allocation. This is resource-budget injection, not an allocator OOM claim.
-The cleanup mutant removes only publication of the restored BorrowEnv header. Both fault cases then return91 for leaked branch depth, while normal/type-error cases still pass. Real fault cases retain E397 with the exact overflow reason and balanced frames.
-19335 final validation COMPLETED0:19 cases/7 output controls,6 real cleanup probes,4 nonfault mutant controls and2 omitted-cleanup refusals. Actual namespace sabotage makes the verifier refuse the missing join. Source/input guards pass.

The first diagnostic namespace fixture used epsilon annotations rejected by the current checker with E001/E002. Those logs remain in19294. The new focused fixture infers only x/y from the same values, keeping expected0 and the required namespace-top checks. No existing witness was edited. The initial diagnostic manifest also contained an empty producer; it was not accepted as a gate. The final harness rejects blank names, missing files, duplicates, extra/missing cases and nonsemantic statuses.

## Existing-checker use

Inside Slurm:

    bash tests/fixtures/wave4_logical_ownership/gate_existing.sh CHECKER BASELINE_RESULTS FRESH_RESULTS

This compiles only the C++23 verifier and uses an already-built checker for all19 cases. Baseline results are retained at /scratch/sounio-stdlib-wave3-20261027/services/logical-ownership03/focused/before. The standalone verifier CLI is reference --verify BEFORE_RESULTS AFTER_RESULTS, with --self-test for the seven invalid-output controls. FOCUSED.tsv has no header and contains name<TAB>exit; logs are name.stdout with SOUNIO_NS_TRACE=1.

Compiler bootstrap was not duplicated. Runtime short-circuit lowering and the single integrated raw build belong to root's ADR014 integration. The counter fixture's runtime assertions remain for that integration; this receipt qualifies checking, ownership and cleanup only.

All sources, ELF files, logs and bridges remain under /scratch/sounio-stdlib-wave3-20261027/services/logical-ownership02 and logical-ownership03. Durable custody was completed by Slurm19346; see custody-19346.md for archive contents and SHA256.

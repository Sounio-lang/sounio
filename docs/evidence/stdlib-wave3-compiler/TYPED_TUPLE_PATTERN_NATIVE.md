# Typed tuple patterns: native qualification

Tuple match arms previously had no PatTuple branch in lower_match_arm_ref. The lowerer accepted the program and skipped those arm bodies. The original returned_loan_pattern_forward program compiled but failed natively with status139; an older scope-only witness could pass without executing its tuple body.

The checker now publishes the concrete tuple scrutinee type keyed by module owner, canonical function identity and source span. Lowering consumes that ephemeral metadata to project each element and bind scalar/reference pointees. Missing, inconsistent or unsupported metadata refuses output. The cache is reset by the owning compilation entrypoint; the auxiliary ontology pass preserves it.

Supported patterns include nested tuples, scalar and supported reference bindings, wildcards, integer/bool literals, unit-enum tags and guards. Whole aggregate value bindings and unproved reference representations remain refused. This does not open generic borrowed storage or HeapVec.

## Real raw compiler replay

Slurm **18900 COMPLETED0**, elapsed21m47, builds the canonical compiler entrypoint and runs the expanded strict gate. Durable directory:

`/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/typed-tuple-raw-28aea`

- Baseline: git archive28aea8743e7d9f4b458d40be700b15d30bd09695; archive SHA256 e9c6b767c5ab7ccf567325e685503194a25c0b0740e083c3a004b9ab19fabb96.
- Overlay: ONLY checked tuple cache/lowering, the missing EnumVariantEntry payload_fields initializer, fixtures and51-case gate; SHA2569f4a5d71a0a7663f367b027fc8db3f39a98356ab7d1d5f5804bc2a2da339c868.
- Raw ELF SHA256 **c02a0d212761d1954959f53832689d44887da96a048749dbff4a3976c268c3dd**.
- Parent cast/exit fix and semantic read_f64 fix are included through the baseline. Capacity WIP, WorkerFn WIP, root preseed fix, storage backend and trusted origins are excluded.
- Strict check matrix:51 cases,21 positive and30 negative, PASS.
- Strict compile matrix: the same51cases and diagnostics, PASS. All negative cases produce no ELF.
- Every one of the21positive programs executes with status0 and the expected marker.
- Seven wrong-output controls pass for each raw-check and compile verifier.
- Three original plain-string/enum-string/combined-string controls compile and execute with status0.
- Final feature, source, seed, compiler and archive guards all pass. Source custody also rechecks the earlier failed component snapshot.

The original forwarding witness now executes and prints RETURNED_LOAN_PATTERN_FORWARD_OK. Five new witnesses require observable tuple-arm execution: literal selection and guard fallthrough, mixed f64-reference/scalar projection, nested tuples, a struct-reference projection with another layout declared first, and a unit-enum tag inside a tuple. Each new witness compiles with the prior raw0b6 compiler but exits1, because its required arm effect is absent; all five execute correctly with rawc02a.

## Preserved component boundaries

Slurm18892 stops at two draft `>>=None` lexical errors and the existing missing enum initializer; no runtime claim follows. Slurm18893, after those corrections, compiles the actual checker and passes all46loan verdicts and seven wrong-output controls. Its tuple component refuses because an auxiliary ontology pass reset the checked-type cache.

Slurm18896 preserves the auxiliary-pass correction. Actual checker/lowerer/native component execution succeeds for the original forwarding witness, literal/guard witness and enum-tag witness. The richer three witnesses refuse unresolved-local/pointee metadata in that reduced driver. Their sources are unchanged in canonical raw18900, which compiles and executes all five correctly. The reduced-driver refusals remain archived and are not relabeled as passes.

The isolated EnumVariantEntry initializer is necessary because the struct already contains payload_fields. No capacity constants or allocation strategies are included.

## Remaining gates

Rawc02a is a qualified compiler for these51cases. It is not an integrated root build containing the later dead-external-preseed fix, storage hooks or trusted origins. Owner-view import/body-lowering, generic storage, HeapVec, WorkerFn and whole-stdlib acceptance retain their separate gates.

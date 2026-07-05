<!-- docs:meta
topic_id: repo.docs.research.madaros-v2-s3-hlir-serialization-2026-07-04
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.madaros-v2-s3-hlir-serialization-2026-07-04
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Madaros v2 S3 HLIR serialization

Status: S3 is complete for the compiler-native HLIR JSON/hash/roundtrip
boundary. S4 is ready to consume the HLIR artifact, but S4 optimization is not
implemented here.

## Implemented surface

- `bin/madaros --emit-hlir <source>` emits clean
  `madaros.hlir.module/0.2` JSON to stdout.
- The emitter uses the real compiler path: parse source, run the module
  front-end check, lower with `hlir_lower_module`, then serialize the
  `HlirModule`.
- `bin/madaros s3-receipt <source> [--out-dir OUT]` emits:
  - `<case>.s3.hlir.json`
  - `<case>.s3.receipt.json`
- `scripts/dev/madaros_v2_s3_gate.sh` validates representative S3 witnesses.
- `scripts/dev/madaros_v2_s4_preflight_gate.sh` consumes the S3 gate artifacts
  and proves they are usable as the input contract for the next optimizer lane.

## Gate contract

The S3 gate rejects:

- stdout banners or diagnostics before the JSON object
- invalid JSON
- non-deterministic re-emission
- mismatched module/function/block/instruction counts
- empty instruction bodies
- missing required ops, terminators, calls, or constant kinds for the manifest
  cases

The manifest lives at `tests/madaros/v2_s3/manifest.tsv`.

Current cases:

- `hello`: string literal, direct call, return
- `recursion_fact`: binary ops, direct recursive call, branch, conditional
  branch, returns
- `gpu_ptx_combo`: minimal GPU/PTX import-combination witness source carried
  forward from the S1/S2 gates

## Validation command

```bash
env MADAROS_RAW_BIN=$PWD/artifacts/self-hosted/madaros \
  SOUNIO_STDLIB_PATH=$PWD/stdlib \
  bash scripts/dev/madaros_v2_s3_gate.sh
```

Observed local result on 2026-07-04:

```text
[madaros-v2-s3] ok case=hello fns=1 instrs=3 hlir_sha=8b0f6e4f43e2
[madaros-v2-s3] ok case=recursion_fact fns=2 instrs=11 hlir_sha=b29607f429c0
[madaros-v2-s3] ok case=gpu_ptx_combo fns=1 instrs=2 hlir_sha=1f7594d8dfb4
[madaros-v2-s3] PASS: native HLIR JSON deterministic, parseable, roundtrippable, S4-ready
```

## S4 boundary

S3 now provides the artifact S4 needs: a stable HLIR JSON byte hash and a
canonical JSON roundtrip hash. S4 still needs a separate implementation and
gate for e-graph/E-KAN rewrite receipts, proof obligations, exact fallback
semantics, and domain-bounded approximation validation.

The S4-ready claim is executable through:

```bash
env MADAROS_RAW_BIN=$PWD/artifacts/self-hosted/madaros \
  SOUNIO_STDLIB_PATH=$PWD/stdlib \
  bash scripts/dev/madaros_v2_s4_preflight_gate.sh
```

Observed local result on 2026-07-04:

```text
[madaros-v2-s4-preflight] ok cases=3 ops=binary,call_direct,const sha=7bf28f714080
[madaros-v2-s4-preflight] PASS: S3 HLIR receipts are consumable by S4; S4 optimizer remains future work
```

The preflight receipt uses schema `madaros.v2.s4.preflight/0.1` and records
`s4_ready = true` with `s4_implemented = false`.

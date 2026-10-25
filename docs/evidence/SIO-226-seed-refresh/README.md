<!-- docs:meta
topic_id: repo.docs.evidence.sio-226-seed-refresh.readme
authority: repo_only
audience: users
last_validated: 2026-10-25
validated_by: Slurm17539
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.sio-226-seed-refresh.readme
-->

# Private-field compiler seed refresh

The private-field source fix in PR3119 changed the compiler defined by
`self-hosted/compiler/lean_single.sio`, while the committed bootstrap seed and
its old receipt still described the previous fixed point. ContractsCore job
121075845369, run40162035330 failed the canonical comparison: committed MD5
`bb26fcd2aebf6e0170b6d249db3cb149`, self-compile MD5
`358cfd5e688c73145bd5967cf48d9335`. The expected hash and comparison gate were
not changed to accept this drift.

## Actual source and generation identities

The compiler source SHA-256 is
`702b4bf38036b11d61929a6135af5432de75859ff656361aa95c0dd2bd80b9a5`.
It is byte-identical to the earlier source-built private compiler qualification
17374 and unchanged from PR3119 before this repair. The metadata-only port is
commit `ebb906a9ce522fe7505168772717619b3cf7ebf9`; it contains no compiler-source
or workflow edit.

Fresh Slurm job **17539** ran on `cpu-ops`, UID1000, one CPU and4GiB, and
completed **0:0 in 00:01:32**. Its preserved root is
`/scratch/sounio-stdlib-inference-scale-20261024/SIO226-seed01`.

| Generation | MD5 | SHA-256 |
| --- | --- | --- |
| g0, previous committed seed | bb26fcd2aebf6e0170b6d249db3cb149 | 8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc |
| g1, g0 compiling the patched source | 358cfd5e688c73145bd5967cf48d9335 | 13003a8b9b1bf4bfc64ad8d13eb0795d722888fabef5bbb2913b4f7b83bb454b |
| g2, g1 compiling the same source | 358cfd5e688c73145bd5967cf48d9335 | 13003a8b9b1bf4bfc64ad8d13eb0795d722888fabef5bbb2913b4f7b83bb454b |
| g3, g2 compiling the same source | 358cfd5e688c73145bd5967cf48d9335 | 13003a8b9b1bf4bfc64ad8d13eb0795d722888fabef5bbb2913b4f7b83bb454b |

All generation files remain on the worker. The receipt declares the g2/g3
settle pair and records the complete g0–g3 chain. The freshly derived g2 is also
byte-identical to the retained17374 Stage2 artifact. This is a new derive run,
not a copied receipt or a substituted generation. Installed seed size is
12170605 bytes. The old895 seed, receipt and all historical runs are preserved.

## Discriminating validation

The job first reproduces the original failing canonical comparison with the old
committed seed on this exact patched source. It then derives all three new
generations, requires g2==g3 and the retained13003 comparison, installs g2 only
inside the frozen job tree, and emits a source-bound receipt.

The unchanged canonical comparison shell and unchanged `verify_lean_seed.sh`
then pass, including actual native self-reproduction and deterministic repeated
compilation. The canonical gate invokes the full provenance leg and its real
wrong-source control, which must refuse. After the final receipt records the
observed pass/pass verdicts, both gates run again and pass. The exact final
staged source catalog also matches the refreshed workspace files after copying.

The full private-field gate passes44 E259 matrix cases,22 on each of x86_64 and
AArch64, plus the byte-preserved original Phonon refusal. X86 public controls
execute natively. AArch64 positives are cross-compiled ELF checks, not native
execution. The existing AArch64 public-global E001 limitation remains explicit.
Full-corpus, the composed scientific library and diverse double-compilation are
separate qualifications and are not inferred from this focused result.

## Metadata runtime and preserved policy

Receipt emission/checking and the wrong-source fixture now use Node.js for JSON,
file hashing and identity bookkeeping. The active refresh recipe is routed to
`scripts/dev/write_seed_receipt.mjs`. The old Python source remains unmodified
as historical data; no Python was executed for this repair.

The fixture suite checks six accepted legacy cases,14 specifically rejected
provenance cases, an exact ASCII JSON roundtrip of the previously committed
receipt, emitter schema/field order, SHA-first/MD5-fallback policy and refusal to
emit an unsettled receipt. The numerical compiler path and canonical equality
criterion are unchanged. This preserves the prior checker policy; it does not
claim to strengthen all existing schema checks.

An isolated existing Linux Node22.22.2 ELF was staged without global installation,
SHA-256 `81925c0995b5c1427b5d538e6a90ca2fdc4daffb786b09af749beaf7369d4e90`.
Actual worker dynamic libraries and the Node ELF are hashed before/after; their
ldd output and hashes are in the evidence. All immutable staged compiler source,
helpers, tests and stdlib inputs also verify before/after. Only the intended
job-local installed seed and receipt change during the derivation workflow.

## Compact custody

`seed-refresh-evidence.tar.gz` is455216 bytes, SHA-256
`4d43f65f9cbcd8f1534d8c29af3e8484ef67ecc2c63748f8b976c878f535a00d`.
It contains raw gate logs, metadata controls, source/compiler/runtime identities,
complete generation hashes, the old and final receipts, the private-field
matrix and submission/terminal records. It was independently extracted and its
internal manifest verified after copying to the workspace. The archive contains
no ELF or toolchain payload; full generation artifacts remain on the worker.

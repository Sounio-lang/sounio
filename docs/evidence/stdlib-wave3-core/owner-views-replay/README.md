# Qualified raw compiler and original owner-view replay

The isolated raw compiler built by Slurm 18931 accepts and executes both
original consumers that previously failed on dead imported function slots.
No witness, parameter, assertion or expected lifetime diagnostic was changed.

Raw compiler:

`/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/core-preseed-raw-18900/results/madaros`

SHA256:
`f3ceba1af545a006ab61c66dafde4bbc5eb63c0b1463f63d6ed2f3dd853cb332`.

## Source and build boundary

The starting snapshot is the immutable, qualified Slurm 18900 source: head
28aea8743e7d9f4b458d40be700b15d30bd09695 plus the isolated tuple overlay later
committed as 26f0791562d0eccd2dc779c2e6ef52379e393576.
The source owner authorized this adoption. All original source, feature,
archive, compiler and seed manifests passed before copying.

Only the lower.sio patch from root commit
a7eb6b2ae798073590a4c29cb529bb9d4eacf23e was then applied. It adds the two merged
external-signature guards and the Alloc annotation. The enum initializer
correction was already present, verified by the exact ir.sio digest
`fae9922021030bb0af0bf7b435bd02190e97cf0b36b5fe6aa4f589dde343c11b`.
Reverse-patch checks prove that checker67d and cast/exit489 were already
included; those files were not overwritten. The a7 patch applied cleanly and
its complete postimage passed a reverse-patch check.

Root storage and origin WIP is excluded. This artifact does not qualify those
features, WorkerFn certification, or all standard-library modules.

The build used the same legacy positional seed and modular build invocation as
18900. [raw18931/PROVENANCE.txt](raw18931/PROVENANCE.txt),
[SEED.sha256](raw18931/SEED.sha256), and
[PATCHES.sha256](raw18931/PATCHES.sha256) preserve its identities.
The final source-manifest SHA256 is
`c48f850f05ce80e2ce51264c8dd309cb55244fc3b7d61d17542c6ef7f443bd32`.
Compiler, source, seed, patch and original-snapshot guards passed after the
build and tests.

## Executed results

| Job | State | Result |
| --- | --- | --- |
| 18931 | COMPLETED, exit 0, 22m27 | Built raw f3ce and passed the unchanged returned-loan matrix. |
| 18944 | COMPLETED, exit 0, 9s | Required baseline failures with raw c02: owner 41 stubs, facade 31 stubs; lifetime negatives retained. |
| 18945 | COMPLETED, exit 0, 10s | Same original consumers check, compile and execute successfully with f3ce; lifetime negatives retained. |
| 18953 | COMPLETED, exit 0, 5s | Additional subview, block-escape and Utf8Text-close negatives rejected by the intended diagnostics on c02. |
| 18972 | COMPLETED, exit 0, 5s | Forged private ByteView/Utf8View fields rejected by E259 on c02. |

Job 18931 repeats all 51 original matrix checks and compilation decisions:
21 positive native executions and 30 negative cases without an ELF.
Both independent C++23 matrix verifiers report success; their wrong-output
controls also pass. See [raw check receipt](raw18931/reference.stdout) and
[compile receipt](raw18931/compile-reference.stdout).

The before/after consumer comparisons are:

| Original consumer | c02 before a7 | f3ce after a7 |
| --- | --- | --- |
| owner_views | Check 0; compile 1, exactly 41 missing-body stubs | Check 0; compile 0; native 0, OWNER_VIEWS_OK |
| core_generic_lib_facade | Check 0; compile 1, exactly 31 missing-body stubs | Check 0; compile 0; native 0, CORE_GENERIC_LIB_FACADE_OK |
| owner_views_mutation | Check/compile 1, E038, no ELF | Check/compile 1, E038, no ELF |
| owner_views_close | Check/compile 1, E038, no ELF | Check/compile 1, E038, no ELF |

The f4 refusal guard was retained. The missing-body registrations were removed
at their source. Production type metadata and live imported function bodies
remain exercised by the successful consumers.

The owner witness retains its eight original bytes, including embedded NUL,
two-byte UTF-8 and a four-byte emoji. It checks byte/subview boundaries and
relative UTF-8 positions. Original witness identities are pinned in
[ORIGINAL_OWNER_WITNESSES.sha256](ORIGINAL_OWNER_WITNESSES.sha256).
The fully qualified facade witness is pinned in
[QUALIFIED_FACADE.sha256](QUALIFIED_FACADE.sha256).

The two original consumer source closures were copied independently from their
preserved failed snapshots. Only the wrapper's derived compiler verification
stamp was primed before each source identity capture. Source, raw compiler,
platform ABI112 library and executable hashes passed their final guards.
Executable hashes are retained in [owner receipt](after18945/owner-ELF.sha256)
and [facade receipt](after18945/facade-ELF.sha256).

The additional negative probes cover provenance forwarding through subviews,
Result escaping a local owner block, closing a Utf8Text with a live view, and
private-field constructor access. These are frontend/compile refusal evidence
on the qualified c02 compiler. Their extension to the new compiler and public
facade integration remain separate API promotion work; the original consumer
replay alone does not establish every lifetime or visibility property.

## Durable custody

All reconstruction inputs, compiler ELF, original failures and guards are
retained at:

`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/owner-views-replay01`

| Archive | SHA256 |
| --- | --- |
| base-28aea.tar.gz | e9c6b767c5ab7ccf567325e685503194a25c0b0740e083c3a004b9ab19fabb96 |
| tuple-26f-overlay.tar.gz | 9f4a5d71a0a7663f367b027fc8db3f39a98356ab7d1d5f5804bc2a2da339c868 |
| compiler-18931.tar.gz | 69cffa75d38c73fa768ca2024a697da85c36987984626bff0b40875287d52c12 |
| consumer-source-closures.tar.gz | 6ce79b475ae0013e532aff49a664e0e0dfee7b982bb55f4f120e6cff52db839a |
| replay-and-negative-receipts.tar.gz | 317d79c7c8d750bae5fa108564e59075e64eb41ef7cf65c187dc4624845a0ae3 |

Archive digests were verified after writing to durable storage. Reconstruct
compiler source by extracting the base and tuple overlay, then applying the
recorded a7 lower patch. The None initializer, checker67d and cast489 are already
present. The compiler archive includes the exact worker and complete results,
including compiler identity and build logs. The consumer archive contains both
original library/wrapper/witness closures; the receipt archive includes their
separate before/after gates and the additional probe inputs. No full production
lower.sio or ir.sio is copied into this Git commit.


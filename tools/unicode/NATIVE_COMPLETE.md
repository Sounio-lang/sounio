# Complete native core corpus receipts

Every row below names the actual executed ELF. Focused smoke, C++ oracle,
a complete native manifest and a future integrated root/compiler acceptance
are distinct claims. Original fixtures and expected rows were retained.

| Corpus | Array / complete collector | Files | Rows | Native ELF SHA256 |
| --- | --- | ---: | ---: | --- |
| Base Unicode after the codepoint-normalizer refactor |18641 /18642 |21971 |11248160 |70fd483dc981fc0b7443114c6b56cd325689bc012dcce3f468496f521e638fe2 |
| UCA complete extended domain |18643 /18644 |2307 |1180750 |70fd483dc981fc0b7443114c6b56cd325689bc012dcce3f468496f521e638fe2 |
| CLDR49 public default corpus, format2 |18612 /18613 |763 |390251 |9b26bf5f358cdfa8ee3e6fdc4b2e4d5fb4f29f33cf37836b9d9f3a12659c1b8e |
| CLDR49 public480-setting branch corpus, format2 |18614 /18615 |4154 |2126400 |9b26bf5f358cdfa8ee3e6fdc4b2e4d5fb4f29f33cf37836b9d9f3a12659c1b8e |

All arrays and complete collectors above completed with exit0. Each uses all8
declared partitions; duplicate/missing row/part or wrong ELF identity is an error.
The final base/UCA collectors additionally require successful outer gate STATUS
and pin those receipt bytes, exact source, manifest, every fixture byte, compiler
and platform before/after. They use the new gate's default gate receipt format.
The custom CLDR worker receipts use their explicit legacy-worker metadata format.

Base Unicode covers every20171 official normalization row with its five inputs
and four forms, all3309 official default extended-grapheme/word/sentence rows,
and11224680 exhaustive case/fold rows across every valid scalar and approved
locale/operation plus contexts. This rerun replaces the missing complete-base
coverage after the shared codepoint normalizer refactor; the earlier3ed ELF
receipt is preserved but is not retagged as the changed implementation.

The UCA fixture prefix retains every original1176850 source row byte-for-byte,
plus explicit strength5/FFFE controls. Its low-level codepoint entry handles
0..10FFFF including surrogate implicit elements without replacement. Every
valid-scalar operand is also checked against the strict Utf8Text public entry
under the same options. The original chunk413 row39 witness is retained.

CLDR native defaults cover all154 bundled public locale/type entries and the
complete declared source-derived default corpus. The native setting corpus
covers all154 entries and480 configurations on the declared branch probes.
It is not the complete source-derived corpus at every option: that separate
C++23/ICU79.1 gate18540/18550 covers169 source entries and219386880 comparisons.
No C++ or native subset is relabeled as the other.

## Immutable identities

The base/UCA source manifest is
f7cfe0a90fe988c7781c6d6925b31d1cc6a9b1b4a0341ddb5f726e2923d8d97b,
captured by the final facade-inclusive smoke18636. The frozen source includes
the separately qualified byte_buffer_budget and BIDI c9fcb10 dependencies.
The fresh conformance ELF is byte-identical to the prior complete UCA ELF,
and was nevertheless rerun on both complete manifests.

Compiler:
a2c91344ce5ada4d68e9db4521342248b0dc8511e442c021d0f7140d35344915.
Platform ABI112:
de01597868c09725c2c26651d23203043c65fdc04a3e5e96f45dfe4a51d494c4.

Base manifest:
a872fecac523bbee18932592fdcb7fce9e932d79e8609508c07a97a447df46fd.
Base complete hash list:
86bf093265d707b93bda4e655b5ca94cdd130be7dc0608c281078f394a9df1ac.
UCA manifest:
0686152cf10c85f00576fa32fd87f93c8f6090af80923e75e9096e41c7a367bd.
UCA complete hash list:
100bb1d193628303bedab967769902e70d4e5ad7f643433f268c2a9cd5ae3e70.

CLDR format2 immutable asset:
9ceb2499c85f9704e4b27cfd01fc51dde2078ad7b1e721063a1fe69d92fb32ee.
Default manifest:
6f5e6003a0813992ff0fb970212206c5fa7665541e5100905e02352a6950b6ae.
Default complete hash list:
3a9987638ea0bd81c3a486037477cd5e3a8e899fc7f34fd6a5a4ecf502bafe78.
Setting manifest:
eb1ebece158c358b29b32f7379151551b861593404bad55aad24ee75b79b382d.
Setting complete hash list:
2c4e2f520f9e8b44504a5eec6b2c0f281823fd99cbef4028c1ab4043358d089d.

## Subsequent source and boundaries

Locale ELF9b26 predates the subsequent private-use alias-target parser fix.
That change preserves the original four-case positive witness:
native1 before18671, native0 after18674. It does not change the immutable
asset or CE/key emission. Its complete metadata concordance has a separate
receipt; the constructor change is not silently covered by the older ELF.

This is component acceptance on the pinned a2c backend and named source
snapshots. The final generic compiler, canonical imported Option/Result,
owner-bound returned borrow carriers, generic HeapVec storage/movement, BIDI's
independent oracle discrepancy, and integrated whole-stdlib zero-failure result
remain independent gates. Legacy primitive C-string/Unicode-char capability
witnesses retain their separate unresolved status.


## Complete constructor metadata concordance

Slurm18677 generated72 fixtures/9190 rows twice with byte-identical output:
all502 language,1 script,640 territory and2 variant aliases, every7868
likely-subtag row,23 type aliases and154 bundled public locale/type entries.
The independent C++23 registry supplies the expected observable resolved
locale/type. This is metadata concordance, separate from collation-key
comparisons and from an ICU factory's different locale fallback policy.

Slurm18688 completed0 after checking, compiling and executing the public
Sounio constructor on the complete manifest. All72 fixtures ran once, all9190
rows matched, the complete collector passed, and original source, fixture,
tool, compiler, runtime and executed-ELF hashes passed before/after.
Executed ELF:
9133c81286febe1641e6629434aa73e996d9de6240d828c8cad79d029bc7845c.
Exact locale source:
fb1c8fb614038dd597837ed2e16bc4b338692229527da86693832c89233cd33e.
Manifest:
852cf7a9ea503f88c63648fff37ae158f15299abe087e1295ece0166685e3b3f.
Fixture hash list:
95031c9a90bfaa8742b3761c79f2ceb2da95e4f2452843421a080a1ac3fe2085.
Compiled-source manifest:
a19851a22924b846456fce8782932200d2c37d04e916099846ff2d94d5690d38.

The constructor now binary-searches the pinned ordered alias tables and
locale language prefixes; Slurm18687 independently checked all ordering
preconditions (1145 aliases,292 locale metadata rows,7868 likely rows).
It also proved all7 earlier-qualified CE sections byte-identical:
16236888 bytes and154 locale identity references. No CE algorithm, weight,
fixture, expected identity or resource deadline was weakened.
Earlier metadata attempts18678/18680/18685 and their diagnostic crash/timeout
receipts remain preserved; their results are not relabeled as passes.

This closes the pinned component metadata gate. A single final integrated
compiler/source execution of all Unicode/CLDR/BIDI and the full stdlib remains
a separate acceptance requirement.

## Public guarded metadata replay

Slurm18714 completed0:0 in3m30s using the tracked
`tests/stdlib/wave3-core/unicode_locale_metadata.sio` harness. Check, compile,
first-chunk native execution, all72 fixtures/9190 rows through the public
`locale-metadata` gate, and complete collection all passed.
Executed ELF:
97e111445bc2409501a6be89feb468aa0c176e3c88d0c8b87e23820a8147221d.
Compiled-source manifest:
f76f007592f3873fb3864e54a217cd8aedbee3b2c03ce5e267483e53741297a7.

The unchanged earlier gate rejected the new metadata family with status64;
the new adapter completed the same declared metadata protocol.
Duplicate part, wrong family, wrong compiled-source identity and a private
part copy whose outer STATUS alone was changed to1 were each rejected.
All original source, compiler, runtime, executable and archive guards passed.
The isolated source snapshot includes the already-qualified root
byte_buffer_budget dependency without editing the services worktree.
This is full declared constructor-metadata acceptance, not a new full
collation-key or BIDI result.

# CLDR49 locale collation

The separate `text::locale_collation` API bundles CLDR49 tailoring built from
official XML rules and FractionalUCA18. Production code is Sounio. The C++23
builder parses and applies rules, computes canonical decomposition from UCD18,
and emits immutable data; it neither links ICU nor exports ICU collation
elements. ICU4C79.1 is an independent test oracle only.

The existing `text::collation` DUCET root key API and bytes are unchanged.
Locale keys use a separate linear owner, `LocaleCollationKey`; comparing keys
with different resolved locale/type/settings returns status7. Encoding1 uses
opaque i64 cells, including group ranks and full primary weights. Persisted
keys must retain their UCA18.0.0, CLDR49, encoding1 and resolved collator identity.

## Public contract

`unicode_locale_collator(locale,type,options,max_work_bytes)` copies request
bytes temporarily and returns a non-owning immutable configuration. Inputs are
strict Utf8Text, locale identifiers are ASCII and at most255 bytes, and the
explicit collation type is at most64 bytes. Names accept CLDR identifiers or
hyphenated tags, case-insensitive aliases, Unicode co selectors, Unicode
va-posix, and syntactically valid private-use suffixes. Other extensions and
keywords fail explicitly. Conflicting explicit/inline co selectors, duplicate
variants/keywords, malformed identifiers and private collations fail.

Canonicalization uses all502 language,1 script,640 territory and2 variant
aliases, plus7868 likely-subtag rows from the pinned CLDR data. Language
inference for explicit und is excluded. Main inheritance parents apply except
the main nonlikelyScript rule; explicit collation parents take precedence.
Bundle matching removes redundant scripts, considers matching variant subsets
(longest then lexical), and follows the versioned parent chain. Type fallback
is requested type, search for unavailable search variants, default type, then
standard. The resulting locale/type are always observable.

Options use -1 to inherit. Supported overrides are strength1..5 (primary through
identical), alternate0/1 (non-ignorable/shifted), backwards_secondary0/1,
case_level0/1, case_first0/1/2 (off/lower/upper), max_variable0..3
(space/punct/symbol/currency). `unicode_locale_resolved_options` returns the
effective settings; its all-inherit value on an invalid collator accompanies the
collator's nonzero status. Version getters identify the pinned data and key
format. `unicode_locale_bundled_count/bundled_collator` enumerate all154 bundled
public locale/type configurations. Enumeration indices are not persistent IDs.

`unicode_locale_collation_key(c,input,max_key_bytes,max_work_bytes)` always
normalizes to NFD, preserving canonical equivalence and accepting non-FCD
scalar sequences. It handles root/local contractions, expansions, prefixes,
discontiguous matching, implicit elements, script reordering, variable weights,
backwards secondary, case options and the identical level. Strict UTF8 inputs
retain embedded U+0000. Each key owns its buffer and requires
`unicode_locale_key_close`; source borrows do not escape.

Key and work budgets separately charge requested payload capacities. Work is
quartered between normalized output, normalization transient storage, scalar
storage and CE storage. Input and returned key have separate budgets. A failed
key has zero length, preserves input bytes, and must still be closed. Status0
means success,1 invalid identifier/settings/budget argument,2 exhausted budget,
3 allocation failure,4 bounds,5 UTF8,6 invalid data/version,7 key identity
mismatch. A zero work budget can process empty input when the key budget permits
its level separators.

## Asset and source custody

Format2 asset `stdlib/text/data/cldr49-locales.bin` SHA256
`9ceb2499c85f9704e4b27cfd01fc51dde2078ad7b1e721063a1fe69d92fb32ee`
is16507440 bytes. Its12 sections contain strings, scalar sequences, CEs, mappings,
154 locale configurations, settings, reorder boundaries, Han implicit ranks,
default/parent metadata, type aliases, locale aliases and likely subtags.
The explicit generated-data limit is24MiB. The compiler embeds it as immutable
read-only bytes with extent and SHA custody.

Generation Slurm18605 completed0, repeat outputs byte-identical, official
UCD/CLDR source and oracle-library guards intact. C++23 preservation gate18608
compared all7 qualified semantic sections:16236888 bytes and all154 locale
identity references unchanged relative to format1. Alias/likely metadata is
new; this does not claim that an older native ELF contains the new constructor.

## Qualification boundaries

C++ independent CE oracle18532 plus complete setting array18540/collector18550
passed169 source entries,219386880 comparisons across480 configurations.
The deliberately wrong primary-level implementation was rejected.

Native ELF09001b2a554cf2f625393a26013c02097118cc3e7a2ea6b8245f71480366e013
passed all154 public default corpora:763 chunks/390251 rows,18579/18595. Its
format1 constructor predates the full locale metadata. The full native setting
branch corpus (4154 chunks/2126400 rows) and final-format native rerun have
separate receipts; their completion is not implied by this default result.

Native18606 passed the format2 Sounio canonicalization and adversarial
contracts, all154 bundled configurations, key identity/budgets/bounds,
embedded NUL, strict surrogate rejection and input preservation. Reuse after
key close was rejected with E039. A private source mutant omitting primary
weights compiled and failed the unchanged native assertion (status1).
ELFs: canonical7e152770ba0ff4e49fb8dbc3b9caa0ff10bfd25365e2f26d864c1413e5692662;
adversarialdd1a5d141c730dd8addbb3454c9870e597f21d0396864c80b28990b513f6525e.
Raw compilera2c91344ce5ada4d68e9db4521342248b0dc8511e442c021d0f7140d35344915;
runtime ABI112de01597868c09725c2c26651d23203043c65fdc04a3e5e96f45dfe4a51d494c4.

Canonical C++ oracle18598 passed15 request controls and public source
reachability. Raw ICU factory behavior for en-US-POSIX-fonipa failed18594:
it does not choose the specified longest matching variant subset. That original
witness remains unchanged. The explicit standard-bundle oracle separately uses
ICU's available-locale inventory and likely-subtag queries to model the specified
variant subset chain; it does not rewrite expectations to the production result.
The en-u-va-posix missing-region adapter failure18597 is also preserved.

Non-FCD witness18562 (0418 0345 0306 versus048B) compared unconditional NFD with
an ICU normalization-OFF oracle. Adjudication18564 retained every codepoint and
proved original/NFD agreement with ICU normalizationON, matching the public
contract. Subsequent corpus generation18565 explicitly enables normalizationON.

This tranche does not claim final coherent whole-Unicode or whole-stdlib
acceptance. BIDI, generic ownership/compiler prerequisites, final corpus source
identity, official CI authority and unresolved legacy primitive-string/char
capabilities retain independent gates.

Final-format default native corpus18612/18613 completed0:763 files/390251 rows,
all8 partitions, ELF9b26bf5f358cdfa8ee3e6fdc4b2e4d5fb4f29f33cf37836b9d9f3a12659c1b8e,
asset9ceb2499c85f9704e4b27cfd01fc51dde2078ad7b1e721063a1fe69d92fb32ee.
The complete collector also passed through the public core gate18633 with
wrong-source/fixture/family/missing-part negatives.

Baseline-format1 complete native setting branch corpus18584/18596 completed0:
4154 files/2126400 rows, all8 partitions, ELF09001b2a554cf2f625393a26013c02097118cc3e7a2ea6b8245f71480366e013.
It covers every154 public entry and480 settings on the declared branch corpus;
the complete source-derived corpus at all480 settings is independently qualified
in C++23. Final-format2 full setting execution18614/18615 remains separate until
its collector closes.


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

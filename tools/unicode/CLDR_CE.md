# CLDR49 own CE compiler and oracle receipts

The production compiler is cldr_build.cpp plus cldr_data_generate.cpp.
It consumes the pinned official FractionalUCA.txt, CLDR collation XML/imports,
and UCD18 canonical decomposition data. It does not link ICU or extract
production CEs from ICU. cldr_graph_oracle.cpp is a separate ICU79.1-only
comparison oracle.

The CE graph retains exact fractional root weights and explicit case values.
Relations insert ordered nodes between root anchors, including before-strength
resets, expansions, contextual prefixes, canonical decomposition, discontiguous
contractions, starred ranges, imports and suppressContractions. Secondary and
tertiary ignorable domains use the official well-formedness boundaries. The
root Han domain follows all 101,997 entries of the official radical order.

Root single-character canonical decompositions flow through the active
tailoring rather than synthetic root contractions. Explicit root contractions
and canonical closure of script-boundary mappings remain available. Fractional
FDD0 bookkeeping rows are metadata, not regular input-text mappings; actual
tailoring index mappings beginning with FDD0 are retained.

Normal output is bounded to 24 MiB and is deterministic byte-for-byte. The
current candidate has 16,241,368 bytes, 58,774 shared root entries, 296,619 total
mapping rows, 193,016 interned CEs and 154 public locale/type entries. Private
incomplete collations and alt variants participate in import/oracle validation
but are not public constructor choices.

The internal binary format1 stores strings, scalar sequences, 32-byte CEs
(primary/secondary/tertiary weights, quaternary and case), sorted mapping
rows, public locale records, settings, resolved reorder boundaries, radical
ordinals, explicit locale defaults/parents and collation-type aliases.
It is a generated Unicode table format, not an interchange serialization API.

CLDR49 source archive SHA256:
1104a80925a97d6c1cb7a480ac5fa15650edf67dc7eff63c86a3cc065e801778
UCD18 source archive SHA256:
726d30c1398ef4dfaa1217faf0760521c052ed1cdf1c87908947e2d8e82ed362
Candidate binary SHA256:
9829ed45c4ecfec8f880894147a1f3bc506f14f12911c8f8cd978ce9125e1271

Reproducible allocated-job entry point:

    ICU_PREFIX=/isolated/icu79.1/install \
      bash tools/unicode/check_cldr_ce.sh /new/absolute/output

Slurm 18532 passed the strict source/input/ICU library guards, five malformed
fractional-source controls, deterministic generation, all 169 source entries
with their imported texts, complete root mappings and radical domain, and an
independent wrong-implementation control which clears primary weights.
The default bundled-setting oracle checks 457,056 adjacent comparisons.

The exact sealed oracle from that gate (SHA256
6e56260be6a7f40a10a1d6a150e814fbd65967da0867933e31959d8814ce5797)
then passed Slurm array18540, all eight partitions. Complete collector18550
passed all 169 entries, 219,386,880 comparisons and 480 configurations:

- Strength primary through identical.
- Alternate non-ignorable and shifted.
- Backwards secondary off and on.
- Case level off and on.
- Case first off, lower and upper.
- maxVariable space, punctuation, symbol and currency.

The complete collector rejects missing or duplicate partitions, wrong oracle
identity, changed sample domains/counts and empty baselines. Slurm 18550 also
runs missing/duplicate/wrong-identity/empty negative controls. The first two
collector attempts had C++ warning-as-error failures before collection; their
original sources and logs remain in cldr-options-collect01/02.

The full option source-domain checks include all 169 public/private/alt rule
entries. These are CE-construction and C++ key-encoding receipts. They are not
native Sounio locale-key conformance, complete locale canonicalization,
whole-Unicode acceptance or whole-stdlib acceptance. The generated binary
remains a candidate until the Sounio consumer passes its independent corpus.

Closed receipt directories on DL380:

- /scratch/sounio-stdlib-wave3-20261027/core/cldr-ce-gate01
- /scratch/sounio-stdlib-wave3-20261027/core/cldr-options02
- /scratch/sounio-stdlib-wave3-20261027/core/cldr-options-collect04

Earlier failing aa/standard, cu/standard, root/Hangul-boundary and root/FDD0
witnesses remain intact in cldr-build07/08/09/10 and subsequent closed attempts.
No original code point, rule, sample row or oracle expectation was replaced.

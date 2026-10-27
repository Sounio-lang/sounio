The committed unicode18.bin contains immutable Unicode 18.0.0 core UCD data. Production algorithms read embedded data in Sounio; ICU is confined to the independent C++23 test oracle. Binary size is 1,289,504 bytes, SHA256 78f8090d9d5897f4662973af25825d3d87a540dfc4d5fda95aa67f6ab8bca62e.

ucd18-source.tar.gz contains unmodified official UCD files and Unicode License V3, with fixed tar metadata and gzip header. Source archive SHA256: 726d30c1398ef4dfaa1217faf0760521c052ed1cdf1c87908947e2d8e82ed362. Each extracted file is checked against UCD18-SOURCES.sha256. The official source root is [Unicode 18 UCD](https://www.unicode.org/Public/18.0.0/ucd/); ReadMe explicitly identifies final 18.0.0 data. Original copyrights and comments remain. The archive also preserves official normalization, grapheme, word, sentence, and BIDI conformance inputs.

Compile tools/unicode/unicode_generate.cpp with C++23 and invoke it with UCD_DIRECTORY OUTPUT.bin REPORT.tsv. It rejects a ReadMe that does not identify final Unicode 18.0.0. Run tools/unicode/check_tables.sh NEW_OUTPUT_DIRECTORY with ICU_PREFIX pointing at an isolated ICU4C 79.1 build inside a Slurm allocation. That gate verifies every source file, compares the committed binary, tests repeat determinism, and runs the independent oracle. ICU headers and libraries must come from the same build; the oracle rejects mismatched ICU, Unicode, or CLDR versions.

The first eight bytes are ASCII SOUNUCD1. Fields at byte offsets 8, 12, 16 and 20 are unsigned little endian 32 bit values: format version 1, Unicode version 0x00120000, section count 8, and reserved zero. Eight directory records follow at byte offset 24. Each is four unsigned little endian 32 bit values: section ID, record width in bytes, record count, and absolute byte offset. Fields are independent of C++ object layout and host endianness.

| ID | Record bytes | Meaning |
| --- | ---: | --- |
| 1 | 1 | ASCII string pool with NUL terminated entries; other string fields are offsets in this pool |
| 2 | 16 | Property canonical name, first range index, range count, default value string |
| 3 | 12 | Inclusive code point start, inclusive end, value string |
| 4 | 28 | Mapping kind, code point, compatibility flag, first scalar index, scalar count, condition string, reserved zero |
| 5 | 4 | Mapping scalar pool |
| 6 | 12 | Canonical composition starter, second scalar, composite; sorted by first two fields |
| 7 | 8 | Normalized property alias, canonical property name |
| 8 | 12 | Canonical property name, normalized value alias, canonical value |

Property and value aliases follow UAX44 loose matching. String IDs are local to the pinned version and are not stable application identifiers. Missing ranges apply before explicit ranges; adjacent equal values are merged. Code point properties include surrogate categories; text decoders/transforms accept scalar values only. Numeric values retain exact UCD rational strings rather than the approximate decimal column.

Mapping kinds: 1 decomposition (flag 0 canonical, 1 compatibility); 3 simple uppercase; 4 simple lowercase; 5 simple titlecase; 6 default full case fold (C/F); 7 Turkic T override; 8 full lowercase; 9 full titlecase; 10 full uppercase. Full mappings preserve SpecialCasing conditions and empty mappings. Absent mappings mean identity. Hangul decomposition/composition is algorithmic. Composition excludes every Full_Composition_Exclusion code point.

The table contains 105 core UCD properties, 64,335 merged ranges, 12,512 mappings and 961 composition pairs. It includes segmentation, emoji, Indic conjunct, normalization, script, joining, BIDI, bracket, mirroring, numeric, age, width and line break property data. UAX14 line breaking is outside the accepted algorithm scope. Character names, algorithmic names, named sequences and the separate Unihan lexical database are not exposed by this format; source files in the archive are not themselves API completion claims.

The oracle checks all 1,114,112 code points for 82 binary/integer properties exposed by ICU (91,357,184 comparisons), every valid scalar under default and Turkic full folding (2,224,128 cases), 4,562 simple mappings, and all 20,171 official normalization rows in four forms. It compares table-derived normalization and ICU independently to official expectations (806,840 checks). It rejects an intentionally corrupted ASCII uppercase category and an unpinned UCD version. The other 23 properties lie outside ICU's binary/integer property API; their presence is not a claim of independent oracle coverage.

This qualifies generated tables and the C++23 oracle only. Compiler embedding, narrow pointers, Sounio APIs/normalization, default UAX29 segmentation, UCA/CLDR collation and UAX9 BIDI require separate executable gates.

The extended oracle also checks 8,979,744 upper/lower case transformations across default, Turkish, Azeri and Lithuanian rules, including contextual sigma, soft dotted contexts and combining marks. The default UAX29 table-derived algorithms and ICU independently pass every official Unicode18 row: 853 extended grapheme, 1,944 word and 512 sentence cases. Unicode18 GB9c is the Linker Extend* / Consonant rule; no preceding consonant is required. Unknown SpecialCasing conditions are a generator error, with an explicit rejection control in the gate. These are C++23 table/oracle checks; native Sounio execution remains a separate acceptance gate.


## UCA18 / CLDR49 root collation data

`uca18-cldr49.bin` is generated only from the pinned official DUCET18 allkeys and CLDR49 root allkeys sources. Production code never calls ICU. ICU79.1 is an independent test oracle.

The source archive retains official files and licenses, plus the complete official UCA18 conformance text files. Official inputs are https://www.unicode.org/Public/18.0.0/uca/allkeys.txt, https://www.unicode.org/Public/18.0.0/uca/CollationTest.zip, and CLDR49 core.zip. The latter archive digest is78403d8d6fb1fed6e208bf9001bf5be00bb5abfd9f8110dae062bf49309d6a89. All extracted source bytes have per-file hashes inside the bundled archive.

Format:8-byte SOUNUCA1 magic, u32 format1, UCA0x00120000, section_count4, CLDR49, then4 directory records(id,width,count,offset). All integers are little-endian. Section1 entries(width16) stores(key_first,key_count,CE_first,CE_count); section2(width4) is key scalar pool; section3(width16) is CE(primary,secondary,tertiary,flags); section4(width16) is implicit-range(start,end,lead,shared_lead_minimum_offset). Flags bit0 is DUCET variable, bit1 is CLDR variable. An explicit entry with zero CE count is an ignorable mapping, not a missing mapping.

Completely zero p/s/t CEs are removed as a semantic normalization. The original official CE streams remain in the source archive. This is not byte-lossless reconstruction. Slurm18321 verified equivalent keys before/after compaction for every source key at strengths1..5, shifted off/on and backwards secondary off/on, in both root views (1,618,680 comparisons). It also passed both complete official sorted corpora (212,066 rows each),752,720 focused ICU comparisons and wrong-weight/shifted/version rejection controls.

Run `ICU_PREFIX=... tools/unicode/check_collation_data.sh /absolute/new/output` under Slurm to regenerate twice, compare the committed asset, verify source hashes, run conformance and reject bad implementations. This data gate does not qualify Sounio native collation or bundled locale tailorings.


## CLDR49 rule parser preparation

`cldr49-source.tar.gz` contains the pinned selected official CLDR49 sources and license, extracted from the official core.zip identified above. It includes all135 collation XML files;169 entries include private, proposed and short variants. The AST preserves variant and default-type identity.

`cldr_generate.cpp` is currently a strict intermediate rule parser and inventory tool. It does not yet produce production tailoring weights. Unsupported grammar, unknown settings and invalid supported-setting values fail explicitly. `cldr_oracle.cpp` is test-only ICU79.1 validation: it reconstructs rules from the AST and checks exact sort-key equality against the original source rules on318,628 probes across all169 entries. It exports no ICU-derived production CE data.

Slurm18359 passed the complete standalone parser/oracle/source-hash gate. Run `ICU_PREFIX=... tools/unicode/check_cldr_ast.sh /absolute/new/output` under Slurm to reproduce it. Own CE/rank construction, locale/import resolution, reorder/case/radical metadata and Sounio native locale conformance remain explicit pending work.


CLDR registry preparation now preserves locale-level defaults even for files containing no collation entries, such as zh_Hant. An empty explicit-default field means inheritance. Collation-component parents are processed independently of main-component nonlikelyScript parents, as specified in [UTS35 Core49](https://www.unicode.org/reports/tr35/tr35-79/tr35.html). Type fallback and private-import visibility follow [UTS35 Collation49](https://www.unicode.org/reports/tr35/tr35-79/tr35-collation.html).

Slurm18375 revalidated all318,628 AST sort-key probes and a deterministic registry/import graph over135 locales and169 entries. Co aliases come from the pinned BCP47 collation file added to the source package. Missing imports, cycles, private public requests and instruction-budget overflow fail. Eleven explicit locale/type fallback controls include zh_Hant and yue. This is still generator preparation: no production tailoring CE tables or runtime locale API are claimed.


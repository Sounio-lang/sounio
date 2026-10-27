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


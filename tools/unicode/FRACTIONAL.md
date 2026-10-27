# CLDR49 fractional root input

cldr_fractional.cpp reads the pinned official common/uca/FractionalUCA.txt
from cldr49-source.tar.gz without ICU. It preserves each weight byte and
its width, including the two case bits, instead of translating the root
into the older 16-bit allkeys representation.

The reader retains normalized source structures for mappings, explicit
prefixes, character references and reference override fields, all logical
reset anchors (including wildcard metadata), fixed constants, script
reorder metadata, and general-category metadata. The original source
archive remains authoritative for comments and original text.

The radical list must enumerate exactly the Unified_Ideograph domain,
without duplicates, in all 214 consecutive radical records. Reference
targets must be present or be implicitly mapped Unified_Ideograph
characters. A later CE builder must resolve references and reject cycles;
the input reader alone does not claim to construct locale weights.

Build and run inside an allocated Slurm job:

    g++ -std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation \
      tools/unicode/cldr_fractional.cpp -o fractional
    ./fractional SELECTED/common/uca/FractionalUCA.txt report.tsv

The executable also creates five intentionally invalid inputs under
fractional-negatives beside the report. All must be rejected: wrong UCA
version, duplicate radical ideograph, oversized weight, out-of-range
reference, and unknown directive.

Qualified reader source SHA256:
ee4986d5e6b317b2fbe31c26ebcdb0646625a51f70191754254f9ef67ae12c5e

Slurm 18478, cldr-build04, completed the reader and its repeat comparison
and all reader input/source hash guards successfully:
60,895 mappings, 65,166 CEs, 101,997 Unified_Ideograph characters and the
same exact radical domain, 4 contextual prefixes, 1,587 references,
18 logical anchors, 186 reorder tokens, 11 fixed constants.

The larger CE graph experiment in the same job failed its separate ICU
semantic oracle. This reader receipt is not a locale collation verdict.
No production locale table or Sounio locale API is delivered by this file.

# SIO-175 native prepared-contour evidence

This package accompanies `devpass/SIO-175-prepared-contours`, based on remote main
`c21b0e139830ca4949f80cba62887a8922a972d7`. It is a reviewable implementation and
focused native acceptance record; it does not imply merge, full-repository CI,
scientific-model calibration, or a universal contour performance claim.

## Final acceptance

Slurm **17201**, `SIO175-full05`, completed `0:0` with gate `STATUS=0`.
The explicit raw compiler SHA is
`a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3`,
provided from current-main CI run 40144792765, artifact 12580970584,
Witness job 121024346990. The portable gate records the actual raw compiler and
canonical launcher hashes separately. Earlier retained-d74 evidence remains in
scratch and is not substituted for this current-main compiler result.

- Four production libraries checked successfully.
- Three fixtures checked, compiled, and ran successfully: legacy contour,
  legacy exact predicate, and the new prepared contour.
- Independent C++23 concordance passed for 96 `cpp_int` exact dyadic predicates,
  legacy contour topology/pixels, and the new 15 segments / 732 nonzero native
  Canvas pixels. Quarter-grid and flat-edge examples have independent explicit
  expected geometry; the oracle does not call Sounio geometry code.
- The new fixture covers ambiguous positive/negative/zero saddles, T junction,
  three-zero L, flat/masked/shared-edge ownership, degenerate endpoint counting,
  strided input, cropped endpoints and fully outside provenance, repeated prepare,
  exact capacity refusal, late invalid data preserving the capture, closed input
  owners and mutated source spec/levels before export, and an invalid SVG ID
  emitting zero bytes.
- Coordinate, captured-level, and missing-completion mutant copies were each
  rejected with exit 1. They are explicitly non-native controls.
- Strict seed import-closure compilation passed. Seed SHA:
  `53b64a871490a6daf64fb2e56d62482c590b00d9f0ec94f0d621cefc2b341fd2`.
- The full functional input hash check passed unchanged.

The fixture's complete native nonzero pixel stream is decoded to a PPM after
concordance. `preview/actual-canvas.png` is a viewing-only `sips` conversion of
those Canvas pixels. It was viewed at its complete 920×200 bounds: cross,
separated branches, T/L shapes, shared-edge verticals, and clipped diagonal are
visible. This is a renderer regression diagram, not a publication-layout example.
`preview/actual-native.svg` is the actual emitted SVG section of the fixture;
its diagnostic text nodes are retained. No SVG redraw generated the Canvas image.

## Before/after observations

Slurm **17196**, `SIO175-bench03`, completed `0:0` with benchmark `STATUS=0`.
For each grid the actual order was **AB / BA / AB**, where A is original c21
source and B is the prepared implementation. Both used the same compiler above,
the same sampled linear field and quarter-grid isovalue, the same 257² Canvas,
and one Canvas plus one SVG export. Process measurements include setup and both
exports, not an isolated kernel. No statistics were calculated on a reduced field.

| Cells per side | Original elapsed seconds, pairs 1/2/3 | Prepared elapsed seconds, pairs 1/2/3 | Original peak KiB, pairs 1/2/3 | Prepared peak KiB, pairs 1/2/3 |
| --- | --- | --- | --- | --- |
| 256 | 0.36 / 0.32 / 0.33 | 0.11 / 0.11 / 0.13 | 328192 / 328192 / 328192 | 14344 / 13832 / 14344 |
| 1024 | unavailable: exit 182 in all three | 1.73 / 1.71 / 1.72 | 744448 / 744448 / 744960 at failure | 214024 / 214024 / 213504 |

Each original 1024² process reports `madaros: handles full (allocation of 64
bytes)` at the existing 4,194,304 managed-object limit. Failed process times
(0.87 seconds each) are preserved as failure receipts, **not speed comparisons**.
No arena limit was raised. The original 256² runs completed and their actual SVG
geometry passed the same independent checker as the new path.

The prepared runs retain 511 segments for 256² and 2047 for 1024²; their owner
capacities retain 65544 and 262152 payload bytes respectively. Preparation reports
131072 and 2097152 actual cell evaluations (count pass plus commit pass), with
zero saddle-predicate evaluations for this linear field. Both exports report zero
cell and predicate evaluations. The 96-case separate predicate gate supplies exact
saddle coverage; this benchmark does not measure a saddle-dense grid. Independent
C++ exact rational quarter-grid projection/provenance matched every successful
benchmark SVG. Large-grid Canvas checksums are preserved as observations; the
independent full Canvas pixel oracle is the smaller functional fixture above.

## Replay and custody

Portable focused gate, executed on Slurm uid 1000 with a 524288-KiB stack:

```sh
MADAROS_RAW_BIN=/path/to/pinned/madaros.elf \
SOUNIO_SEED_COMPILER=/path/to/seed \
bash scripts/stdlib/prepared_contour_gate.sh /fresh/output/path
```

`contour_bench.sio` and `contour_bench_legacy.sio` hold the same raw field setup.
Set only the literal side to 256 or 1024 as in the archived `replay/*.sio` files.
The legacy source imports only the original APIs and is compiled against base c21.
`prepared_contour_bench_reference.cpp SIDE NATIVE_STDOUT` validates the actual
benchmark geometry independently. The archived worker prepare/dispatch scripts
record the observed source and compiler locations; replay must select available
worker paths, not assume those retained paths exist elsewhere.

`receipts.tar.gz` is **395030 bytes**, SHA-256
`941269dd81b6413b56fa02ad0e3b0693dc2a9e6cf31695cf37b9ffc024dd79c0`.
It contains compact functional/benchmark results, compiler/input hashes,
resources, native stdout, C++ outcomes, controls, source hashes, previews and
scheduler receipts. No ELF, cache or full source archive is included. Its internal
`MANIFEST.sha256` was verified before packing and after fresh extraction:

```sh
mkdir /fresh/receipt-directory
tar -xzf receipts.tar.gz -C /fresh/receipt-directory
(cd /fresh/receipt-directory && sha256sum -c MANIFEST.sha256)
```

Original worker paths remain immutable:
`/scratch/sounio-stdlib-inference-scale-20261024/SIO175-full05`,
`SIO175-bench03`, and `SIO175-custody02`. Earlier attempts are retained separately;
check01/check02 integration argument/linear-close diagnostics are included in the
archive history. An initial custody attempt used the host's disabled accounting
client; custody02 captures actual worker accounting instead and preserves that
operational error. No numerical acceptance criterion was relaxed.

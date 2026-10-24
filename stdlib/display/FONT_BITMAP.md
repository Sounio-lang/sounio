# Canonical native bitmap font

`display::font` renders the public-domain `font8x8_basic` table. The source is
Daniel Hepper's font8x8 collection, derived from Marcel Sondaar's IBM public-domain
VGA font data, pinned to commit
`8e279d2d864e79128e96188a6b9526cfa3fbfef9`.

Primary files, preserved with their original attribution:

- [Encoding and license](https://github.com/dhepper/font8x8/blob/8e279d2d864e79128e96188a6b9526cfa3fbfef9/README):
  vendored as `tests/reference/data/font8x8_README.txt`, SHA256
  `885cc022fa9c844695b364e77e6a18ab2be74678a181827f1ce893071205cb65`.
- [Primary bytes](https://github.com/dhepper/font8x8/blob/8e279d2d864e79128e96188a6b9526cfa3fbfef9/font8x8_basic.h):
  vendored unmodified as `tests/reference/data/font8x8_basic.h`, SHA256
  `49d8df366296b203ca3211bc0672cf2a762135bf12710735b6292756b19dffd5`.

The reference header is test/generation input, not a runtime dependency.
C++23 packs each eight-byte glyph into the production constants. The initial
generated function was calculated under Slurm16413 and preserved with SHA256
`0eb373ad247f9f2d0b91ac6c658d244c05acac5a015629ef58e2e2ef0915dc3b`.

## Public packed-mask and drawing contract

`font_glyph(ch:i64)->i64` returns one eight-by-eight bitmap for ASCII32..127;
unsupported values return the blank mask. The low byte is the TOP row; bit0
in each row is its LEFTMOST pixel. Byte r occupies bits8*r through8*r+7.
This is a bit-mask contract: a mask with its highest bit set can have a negative
i64 representation. It is not a numeric intensity or a pointer.

`font_draw_char` and `display::canvas::canvas_draw_char` retain their signatures and
top-left origin. Set bits write opaque packed0x00RRGGBB pixels, clipped to the
raster. Unset bits leave the existing pixel unchanged; a glyph does not clear or
fill its entire cell rectangle. There is no alpha blend or hidden background.
`font_draw_string` / `canvas_draw_text` retain eight-pixel horizontal advance,
nine-pixel newline advance, existing wrap behavior and clipping. No Unicode
shaping, proportional font metrics or equality with an external SVG font's
individual pixels is implied.

The raw glyph integers intentionally change from the former mixed table to
the pinned canonical data. Clients that persist or compare raw masks must use
the new table; function signatures and cell/advance geometry do not change.
This is a source-font correction, not an image flip or an update of screenshot
goldens.

## Discriminating evidence and scope

Before the correction, Slurm16413 compared the actual native glyph integers and
all pixels of the96-character grid, clipped glyphs and a multiline string
against the old SOURCE's row/column convention. Source/backend parity passed
for96 glyphs and6832 pixels. Thus the compiler and export did reproduce the
source, while the primary canonical comparison failed (first differing packed
glyph33). The native primary anchor and canonical C++ comparison were expected
failures; old source, executable, inputs and outputs remain preserved.

The new source normalizes the entire table and uses the primary LSB-left
sampler. Replacing only the column shift would have broken some old entries
that used the opposite convention. Reversing image rows would have hidden the
defect and changed scientific geometry; no images were edited.

`tests/run-pass/display_font_orientation.sio` dumps all96 actual masks and
complete grid/clipping/string rasters before its canonical anchor assertion.
It deliberately remains eligible for both Stage2 and Madaros, as the public
legacy drawing API is supported by both. `display_font_orientation_reference.cpp`
has three distinct modes:

- `--source-parity OLD_FONT_SOURCE ACTUAL_STDOUT`: reconstruct the old source's
  MSB-left convention and verify native masks AND every raster pixel.
- `--canonical PRIMARY_HEADER ACTUAL_STDOUT`: reconstruct the pinned byte table
  independently in C++23 and check all masks/pixels with LSB-left orientation.
- `--emit-font PRIMARY_HEADER`: emit the numerically packed production function.

After-source Slurm16417 passed the canonical C++ comparison for all96 glyphs
and6832 pixels, including clipping and multiline/string advance. Slurm16422
re-exported both actual scientific Canvas images; their glyph orientation and
readability were inspected independently. The [focused evidence
bundle](../../docs/evidence/stdlib-inference-document-2026-10-23/README.md)
preserves before/after receipts, corrected actual PNGs, SVGs and source/compiler
identities. These focused results are distinct from a full legacy display/replay
suite, a whole-repository gate or human acceptance.

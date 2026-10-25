# Canonical RGB8 palettes and checked scientific color scales

`viz::colormap` now supplies the canonical 256-entry viridis, plasma, inferno and
magma data through the existing channel and packed-RGB function names. The former
eight-anchor and degree-six approximations remain available under explicitly
named `*_approx_r`, `*_approx_g`, `*_approx_b`, and `*_approx_u32` functions. This is
an intentional numerical correction: a canonical name and its heatmap metadata
now refer to the published listed palette rather than a sparse fit.

## Primary data, license, and byte policy

The immutable input is [BIDS/colormap at
bc549477db0c12b54a5928087552ad2cf274980f](https://github.com/BIDS/colormap/blob/bc549477db0c12b54a5928087552ad2cf274980f/colormaps.py),
by Nathaniel J. Smith, Stéfan van der Walt, and, for viridis, Eric Firing. The
source explicitly releases its code and palettes under CC0. Its exact source
bytes and license are vendored in `tests/reference/data/canonical_palettes`.
The data is also checked against [Matplotlib source at
39b27ddaafe636f06776a2044b5df05d07648dc4](https://github.com/matplotlib/matplotlib/blob/39b27ddaafe636f06776a2044b5df05d07648dc4/lib/matplotlib/_cm_listed.py),
with its license retained. The latter source is stored compressed to avoid an
unnecessary uncompressed duplicate. These `.py.txt` assets are **data inputs**;
no Python interpreter or Python numerical implementation is used.

The input files retain their published floating channel values. Runtime colors
are quantized separately as `floor(255 * channel)`, producing opaque RGB8. The
C++23 converter checks exact decimal-rational quantization against binary64
multiplication and verifies all 3,072 floating channels against the Matplotlib
source before accepting all 1,024 RGB triplets. For example, the high viridis blue
channel is 0.143936: its RGB8 value is 36 under this rule, rather than the old
rounded anchor value 37. Floating canonical data, quantized RGB bytes, and
browser/font rasterization are distinct representations.

`colormap_data.sio` contains a generated balanced immutable lookup, packing two
RGB values in each positive 48-bit integer. There are no runtime LUT arrays,
array constructors, file reads, per-cell records, interpolation fits, or hidden
payload allocations on the canonical lookup path. The C++23 gate regenerates
this source and compares it byte-for-byte. `palette_lut_rgb8(palette, index)`
returns packed `0xRRGGBB` for IDs 1=viridis, 2=plasma, 3=inferno, 4=magma and
indices 0..255, or -1 for an invalid ID/index.

Normalized convenience functions use the declared listed policy `floor(256*t)`
for `0 <= t < 1`, with `t == 1` selecting bin 255. Their existing public
signatures/effects are retained. Values below/above [0,1], including infinities,
clamp to endpoints. NaN uses the documented opaque gray `0x808080`. This legacy
convenience produces a color, not a data-availability status; scientific input
validation should use the checked scale API. Explicit approximation functions
preserve the old algorithms and outputs for compatibility, including their
allocation characteristics; they are not the canonical large-field path.

## Checked scale and availability contract

`viz::palette::palette_scale(id, minimum, maximum, under_rgb, over_rgb,
missing_rgb, constant_index)` constructs an immutable scale with private fields.
`palette_scale_defaults` uses endpoint under/over colors, gray missing color,
and constant bin 128. A valid scale has a recognized ID, finite ordered endpoints,
24-bit colors, and a constant index in 0..255. Inspect `palette_scale_status`:
0 means valid, 1 means invalid configuration. It is not a scientific certificate.

Construct `palette_sample_slot()` once, then call
`palette_map_into(&scale, value, available, &!sample)`. Every call clears its
result before evaluation. Sample statuses are:

| Status | Meaning |
| --- | --- |
| 0 | Color available under the declared policy |
| 1 | Invalid scale |
| 2 | Numeric normalization unavailable |
| 3 | Active datum is NaN or infinite |

Available colors carry an independent classification: 0 listed bin, 1 finite
under-range, 2 finite over-range, 3 masked/missing, 4 the declared value of a
constant domain. Special configured colors use index -1; unavailable results
have RGB placeholder 0, index -1 and classification -1 and must not be painted.
A configured RGB value 0 remains a legitimate black color when status/class say
so; it is not a replacement numeric observation.

A false availability flag selects the caller's missing color without inspecting
or interpreting the ignored value argument. Active nonfinite values are refused,
not silently promoted to missing, an endpoint, or a valid zero. Finite values
outside the range select the configured under/over colors. A constant range
selects its declared constant bin only for that represented value; finite values
below/above it still select under/over colors. Signed zero is the same constant
domain value. In-range normalization is affine; a half-scaled equivalent is used
when the represented full span overflows. No raw data, statistics, uncertainty
intervals, or biological quality are inferred from colors.

## Heatmaps and shared metadata

`dynamic_heatmap_palette_validate`, `dynamic_heatmap_palette_canvas_into`, and
`dynamic_heatmap_palette_svg_into` in `viz::dynamic_fields` borrow the checked
matrix, values, 0/1 cell mask and the same `PaletteScale`. The
`DynamicHeatmapSpec.color_min/color_max` must equal the scale's endpoints; a
mismatched legend/data scale is rejected. Construct `palette_field_work()` once
and reuse it. It contains one color sample and one integer rectangle.

The complete active prefix, mask, matrix fit, dimensions and normalization are
validated before any Canvas paint or SVG bytes. Masked-only NaNs are allowed;
active nonfinite data is refused. Matrix corners/cells use checked scalar
accessors after `numeric_matrix_fits`, and integer cell edges partition the
existing closed pixel-center viewport. No array or result object is created per
cell. More grid cells than pixels per axis return status 6; there is no implicit
density aggregation. Other statuses are 1 invalid request/scale/domain pairing,
2 unavailable storage, 3 invalid active data/mask, and 5 numeric color failure.

`paint_missing=false` leaves holes. With `true`, missing cells receive their
configured color and explicit class 3/index -1 metadata. `work.masked` counts all
masked cells; `missing_painted` is the subset actually painted, while
`active_painted` excludes them. `outside` counts attempted rectangles outside the
Canvas raster. Under/over/constant counters describe active color classifications.
These categories are not independent sample-quality metrics.

Existing `dynamic_heatmap_canvas/svg` retain their signatures, nonconstant
finite-span domain refusal, and masked-skip behavior. They now use canonical
viridis. SVG retains its palette/row-order labels and adds shared source,
quantization, bin, raw-domain and special-color attributes. New checked SVG cells
also carry source row/column/index, classification and palette index. The compact
numeric attributes use 15 significant digits; they are human-facing provenance,
not a binary64 round-trip serialization format.

## Horizontal colorbar and label layout

`viz::colorbar` provides `ColorBarSpec`, `colorbar_work`, `colorbar_validate`,
`colorbar_canvas_into` and `colorbar_svg_into`. The same immutable scale defines
both heatmap colors and colorbar metadata. The horizontal strip increases left
to right and emits all 256 listed bins; a nonconstant strip narrower than 256
pixel cells is explicitly refused with status 6. A constant domain renders one
uniform rectangle in the declared bin and one true-value label, not a fictitious
gradient or range.

Labels use the existing six-significant-digit axis formatter and canonical
8-pixel text-cell contract. Names are optional; endpoint labels are checked for
separation and complete document bounds before drawing. Indistinguishable
nonconstant labels return status 4; insufficient document footprint returns 3;
invalid configuration/raster returns 1. Canvas dimensions must match the declared
document. Invalid requests paint nothing and emit no SVG bytes. A compositor can
call the public validators for all layers before opening a document.

SVG uses the same integer cell boundaries and forces the corresponding text
widths. Canvas uses the existing bitmap font; browser font shapes/antialiasing
are not claimed to be pixel-identical. Palette names and numeric labels contain
no user-provided markup. Palette and scale metadata are emitted by the same
helpers for the heatmap and colorbar. Units and scientific captions remain the
caller's responsibility; neither palette nor colorbar creates confidence,
coverage, model assessment, or inference eligibility.

See `tests/stdlib/canonical_palettes/EVIDENCE.md` and
`scripts/stdlib/canonical_palette_gate.sh` for actual native/C++23 acceptance,
primary source hashes, replay and the before/after witness.

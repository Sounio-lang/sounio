# Compact SVG for clipped scientific lines

The existing clipped-line exporter now emits at most one SVG path per chart.
Each accepted segment is an independent `M x0 y0 L x1 y1` subpath. Endpoints,
ordering, color, one-pixel width, round caps and the closed-pixel-cell clipPath
are retained. There is no curve fitting, simplification, resampling or decimation.
NaN/infinity gaps and outside segments cannot acquire connecting edges because
every retained segment starts with M.

Zero-length segments remain M/L pairs with coincident endpoints. Round line caps
preserve their point geometry under SVG2. No Z or joined-polyline command is used.
Empty, one-sample, all-outside and all-invalid charts emit no path; their audit
counts and clipping metadata remain present.

Primary semantics:
- https://www.w3.org/TR/SVG2/paths.html
- https://www.w3.org/TR/SVG2/painting.html#LineCaps

Serialization is one streaming pass over the existing fixed-capacity series.
It allocates no coordinate list or path string in Sounio and no additional scene
storage. At most 1023 segments are emitted for a 1024-sample series.
Style attributes are emitted once rather than once per segment.
This applies to the opt-in clipped one-pixel line path. Other marks and legacy
unclipped polyline export keep their existing representation.

## Compatibility and verification boundary

SVG element type changes from line to path inside clipped plot groups.
Consumers inspecting XML should parse path subpaths instead of selecting line
elements there. Axis lines and other SVG markup are unchanged.

The C++23 comparator in tests/reference/svg_path_reference.cpp reads native logs,
extracts complete SVG documents and canonicalizes only clipped line/path shapes
into ordered segment records. It requires exact integer endpoints, stroke style,
one independent M/L pair per segment, audit counts and unchanged other SVG
content (including masks, axes, labels and legend). Compact groups must have at
most one path. It also measures actual document bytes and reduction percentages.

This comparator is intentionally restricted to this exporter's markup, not a
general XML or arbitrary SVG parser. It rejects unsupported commands and shapes.
Existing independent clipping/AA reference checks remain in the native suite;
the comparator checks serialization equivalence, not a new clipping algorithm.

The geometric stroke set is retained. Replacing several painted elements with a
compound path can change subpixel antialias/compositing at overlapping strokes.
Bitwise browser-raster parity is not claimed. Canvas code is unchanged and its
existing pixel checks still execute. Future alpha, wide-stroke, marker, CSS or
per-segment interaction semantics require a separate contract.

The degenerate-point SVG assertion formerly tied to line attributes in
viz_scientific_clip.sio moved to the new representation-specific viz_svg_paths.sio;
the earlier Canvas degenerate-point assertion remains.
See SCIENTIFIC_SVG_PATHS_EXECUTION_2026_10_17.md for accepted jobs, independent
comparison, raw/gzip sizes and limits. Artifact-size reductions do not establish
RSS, execution time, compiler-memory or binary-size improvements.

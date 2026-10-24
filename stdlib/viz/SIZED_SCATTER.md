# Explicit scatter marker radius

`dynamic_scatter_validate_sized`, `dynamic_scatter_canvas_sized` and
`dynamic_scatter_svg_sized` take the existing scatter arguments followed by
`radius: i64`. The accepted range is 1 through 4. A mark is a filled,
axis-aligned square of side `2 * radius + 1` integer pixels, centered at the
existing projected integer coordinate. This parameter is a visual size; it
represents no statistical uncertainty or density estimate.

`scientific_overlay_validate_sized`, `scientific_overlay_canvas_sized` and
`scientific_overlay_svg_sized` take the existing overlay arguments followed by
`primary_radius` and `secondary_radius`. The same series radius and color apply
in both panels and their legend. A line layer retains its existing stroke and
line legend. Both radius arguments are nevertheless validated.

Invalid radius returns status 1 from the scatter APIs and status 33 from the
overlay APIs. Global validation precedes SVG output and Canvas painting: a bad
second-layer radius or a late active nonfinite coordinate cannot leave a
partially rendered document. The borrowed payload, masks and domains retain
their existing contracts. No IR field or buffer ownership rule changes.

The original APIs delegate with radius 1. Their interior rendering and document
serialization are retained. One boundary defect is intentionally corrected for
all radii, including that default: an outside continuous center may truncate to
an integer center whose square still touches the plot. The inexpensive early
rejection therefore uses a conservative radius-plus-one margin; the final
integer rectangle intersection decides whether a footprint is painted. Centers
are not clamped onto the boundary. Nonfinite normalized coordinates produced by
extreme finite arithmetic are rejected as outside before integer conversion;
active nonfinite input coordinates remain a validation error.

The plot is a closed rectangle of integer pixel centers. Canvas intersects the
mark with both plot and raster. SVG emits the identical integer footprint as
pixel cells with half-integer origin and integer width/height. A radius-4 mark
has at most 81 candidate pixels before clipping. This bounded footprint does
not imply that the complete renderer is allocation-free or that arbitrarily
large payloads fit an execution budget.

Native evidence for the additive API is Slurm 16905, campaign_report_sized03:
48 independent radius/boundary cases, 36,864 actual raster pixels, 49 SVG roots,
full overlay geometry and legends, and three invalid-input zero-output probes.
The legacy-only four-edge witness compiles on the previous implementation and
exits 1 because all four visible footprints are missing; the corrected source
passes. Independent C++ verifies the projected centers and clipped rectangles.
An unchanged radius-1 campaign caller produces byte-identical stdout to its
previous accepted native execution. Existing scientific-document,
count-diagnostic and dynamic-field native/C++ checks also pass in this packet.
These are focused checks, not a full corpus or CI claim.

The real campaign report opts into radius 3. Its observations, masks, domains,
statistics and interpretation are unchanged; only the mark and legend squares
are larger. See `examples/viz_count_profile_campaign.txt` and the compact
campaign-report evidence bundle for the frozen-data provenance and limitations.

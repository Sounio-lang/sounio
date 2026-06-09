# Sounio Viz

`stdlib/viz` is the first native visual frontend layer for Sounio.

The v1 architecture is:

- `viz::ir`: fixed-capacity Visual IR owned by Sounio.
- `viz::viz_canvas`: immediate native renderer to `display::Canvas`.
- `viz::viz_html`: static HTML/SVG export renderer over the same IR.
- `viz::physchem`: Sounio data structures for molecules, bonds, scalar/vector fields, trajectories, and uncertainty overlays.
- `viz::{coord,chart,epiviz,sci}`: direct drawing helpers that the IR renderer can lower into.

Scientific meaning stays in Sounio. Units, epistemic variance, molecules, lattices, fields, simulation clocks, scene nodes, and interaction state are represented as Sounio data. Browser or terminal export surfaces display pixels or markup; they do not own the scientific model.

## V1 Constraints

- Fixed node/data arrays; no heap-heavy scene graph.
- Fixed rectangles and simple row/column layout metadata; no flexbox.
- CPU Canvas is the primary renderer.
- HTML/SVG is static export only.
- GPU rendering is deferred until general compiler GPU backend wiring is ready.

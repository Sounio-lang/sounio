# stdlib/graphics

Native pixel-based drawing and scientific raster plots (no external deps).

## Modules

| Path | Role |
|------|------|
| `graphics::drawing` | `Color`, `Point2D`, blending helpers |
| `graphics::surface` | 256×256 RGBA surface (by-value mutation) |
| `graphics::plot` / `scatter` / `heatmap` | Raster chart primitives |
| `graphics::quality` | Publication-oriented plots (markers, ECDF, error bars, …) |
| `graphics::export` | PPM/PNG export |
| `graphics::tile` | Tiled surfaces beyond 256×256 |

## Tests

- `tests/stdlib/graphics/test_graphics_core.sio` — drawing + surface (check-only)
- `tests/run-pass/graphics_*` — smoke tests for quality/gallery/tile paths
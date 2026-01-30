# Visual Examples Showcase

This directory contains the interactive HTML showcase for Sounio's visual terminal examples.

## Contents

- `index.html` - Interactive showcase with all visual examples, color swatches, and run commands

## Source

Generated from `examples/visual/SHOWCASE.html` in the main repository.

## Updating

To update the showcase after adding new visual examples:

```bash
# From repository root
cp examples/visual/SHOWCASE.html hugo/static/showcase/visual/index.html
```

## Viewing

After building the Hugo site:
- **Web**: https://souniolang.org/showcase/visual/
- **Local**: Open `hugo/public/showcase/visual/index.html` in a browser

## Related Pages

- Showcase markdown page: `hugo/content/showcases/visual.md`
- Visual examples source: `examples/visual/`
- SVG export utilities: `examples/visual/export_svg.py`, `examples/visual/ansi_to_svg.sh`

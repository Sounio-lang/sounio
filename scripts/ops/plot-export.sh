#!/usr/bin/env bash
# plot-export.sh — Convert SVG plots to PNG or PDF
#
# Usage:
#   bash scripts/ops/plot-export.sh <input.svg> <format>
#
# Arguments:
#   input.svg   Path to the SVG file
#   format      Output format: png or pdf
#
# Output:
#   Creates <input_basename>.<format> in the same directory as the input file.
#
# Tool preference (first available wins):
#   1. rsvg-convert  (librsvg — fast, accurate, lightweight)
#   2. inkscape       (full SVG spec support, slower)
#   3. convert        (ImageMagick — widely available, may rasterize poorly)
#
# Examples:
#   bash scripts/ops/plot-export.sh output/decay_curve.svg png
#   bash scripts/ops/plot-export.sh results/pk_profile.svg pdf

set -euo pipefail

# ============================================================================
# Argument validation
# ============================================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.svg> <format>"
    echo "  format: png | pdf"
    exit 1
fi

INPUT="$1"
FORMAT="$2"

if [ ! -f "$INPUT" ]; then
    echo "Error: input file not found: $INPUT"
    exit 1
fi

# Validate format
case "$FORMAT" in
    png|pdf) ;;
    *)
        echo "Error: unsupported format '$FORMAT' (use png or pdf)"
        exit 1
        ;;
esac

# Derive output path: same directory, same basename, new extension
DIR="$(dirname "$INPUT")"
BASE="$(basename "$INPUT" .svg)"
OUTPUT="${DIR}/${BASE}.${FORMAT}"

# Optional DPI for PNG (default 300 for publication quality)
DPI="${PLOT_EXPORT_DPI:-300}"

# ============================================================================
# Tool detection and conversion
# ============================================================================

convert_with_rsvg() {
    local args=()
    if [ "$FORMAT" = "png" ]; then
        args+=(-d "$DPI" -p "$DPI" -f png)
    else
        args+=(-f pdf)
    fi
    rsvg-convert "${args[@]}" -o "$OUTPUT" "$INPUT"
}

convert_with_inkscape() {
    local args=()
    if [ "$FORMAT" = "png" ]; then
        args+=(--export-type=png --export-dpi="$DPI")
    else
        args+=(--export-type=pdf)
    fi
    # Inkscape 1.x CLI
    inkscape "$INPUT" "${args[@]}" --export-filename="$OUTPUT" 2>/dev/null
}

convert_with_imagemagick() {
    local args=()
    if [ "$FORMAT" = "png" ]; then
        args+=(-density "$DPI" -background white -flatten)
    fi
    convert "${args[@]}" "$INPUT" "$OUTPUT"
}

# Try tools in preference order
if command -v rsvg-convert &>/dev/null; then
    echo "[plot-export] Using rsvg-convert"
    convert_with_rsvg
elif command -v inkscape &>/dev/null; then
    echo "[plot-export] Using Inkscape"
    convert_with_inkscape
elif command -v convert &>/dev/null; then
    echo "[plot-export] Using ImageMagick convert"
    convert_with_imagemagick
else
    echo "Error: no SVG converter found."
    echo "Install one of: librsvg (rsvg-convert), inkscape, imagemagick"
    exit 1
fi

# Verify output
if [ -f "$OUTPUT" ]; then
    SIZE=$(stat --printf="%s" "$OUTPUT" 2>/dev/null || stat -f "%z" "$OUTPUT" 2>/dev/null || echo "?")
    echo "[plot-export] $OUTPUT ($SIZE bytes)"
else
    echo "Error: conversion failed — output file not created"
    exit 1
fi

#!/bin/bash
# ANSI to SVG Converter for Sounio Visual Examples
#
# Converts ANSI-colored terminal output to publication-quality SVG
#
# Usage: ./ansi_to_svg.sh <example.sio> <output.svg>
# Example: ./ansi_to_svg.sh 06_octonion_color_table.sio octonion_table.svg

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <example.sio> <output.svg>"
    echo "Example: $0 06_octonion_color_table.sio octonion_table.svg"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Run Sounio example and capture output
echo "Running Sounio example: $INPUT_FILE"
ANSI_OUTPUT=$(../../target/debug/souc run "$INPUT_FILE" 2>&1)

# Convert ANSI to SVG using aha (ANSI HTML Adapter)
# If aha is not installed: sudo apt-get install aha (Ubuntu/Debian)
# Or: brew install aha (macOS)

if ! command -v aha &> /dev/null; then
    echo "Error: 'aha' not found. Installing..."
    echo "Ubuntu/Debian: sudo apt-get install aha"
    echo "macOS: brew install aha"
    echo ""
    echo "Alternative: Using ansi2html (Python)"
    if command -v ansi2html &> /dev/null; then
        echo "$ANSI_OUTPUT" | ansi2html --inline > "${OUTPUT_FILE%.svg}.html"
        echo "Created HTML: ${OUTPUT_FILE%.svg}.html"
        echo "You can convert HTML to SVG using a browser or Inkscape"
        exit 0
    fi
    exit 1
fi

# Generate HTML first
HTML_FILE="${OUTPUT_FILE%.svg}.html"
echo "$ANSI_OUTPUT" | aha --black --title "Sounio: $INPUT_FILE" > "$HTML_FILE"

echo "Created HTML: $HTML_FILE"

# Convert HTML to SVG using wkhtmltoimage or similar
# Note: This requires additional tools like wkhtmltopdf, Inkscape, or Chrome headless

if command -v wkhtmltoimage &> /dev/null; then
    wkhtmltoimage --format svg "$HTML_FILE" "$OUTPUT_FILE"
    echo "Created SVG: $OUTPUT_FILE"
elif command -v inkscape &> /dev/null; then
    inkscape "$HTML_FILE" --export-plain-svg="$OUTPUT_FILE"
    echo "Created SVG: $OUTPUT_FILE"
elif command -v google-chrome &> /dev/null; then
    google-chrome --headless --disable-gpu --screenshot="$OUTPUT_FILE" "$HTML_FILE"
    echo "Created PNG (SVG not supported by Chrome headless): $OUTPUT_FILE"
else
    echo "HTML created: $HTML_FILE"
    echo "To convert to SVG, install one of:"
    echo "  - wkhtmltoimage: https://wkhtmltopdf.org/"
    echo "  - Inkscape: https://inkscape.org/"
    echo "  - Or manually convert in a browser (File → Save As → SVG)"
fi

# Cleanup intermediate HTML if SVG was created
if [ -f "$OUTPUT_FILE" ] && [ "${OUTPUT_FILE##*.}" = "svg" ]; then
    # Keep HTML for reference
    echo "Kept intermediate HTML: $HTML_FILE"
fi

echo ""
echo "Done! You can now use $OUTPUT_FILE in papers, presentations, or documentation."

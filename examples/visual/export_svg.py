#!/usr/bin/env python3
"""
ANSI to SVG Converter for Sounio Visual Examples

Converts ANSI-colored terminal output to publication-quality SVG.
Pure Python implementation - no external dependencies except subprocess.

Usage:
    python3 export_svg.py 06_octonion_color_table.sio octonion_table.svg
    python3 export_svg.py --all  # Export all visual examples
"""

import subprocess
import sys
import re
from pathlib import Path

# ANSI color code mappings to RGB
ANSI_COLORS = {
    # Foreground colors
    '30': '#000000',  # Black
    '31': '#e74c3c',  # Red
    '32': '#2ecc71',  # Green
    '33': '#f1c40f',  # Yellow
    '34': '#3498db',  # Blue
    '35': '#9b59b6',  # Magenta
    '36': '#1abc9c',  # Cyan
    '37': '#ecf0f1',  # White
    '90': '#7f8c8d',  # Bright Black (Gray)
    '91': '#e74c3c',  # Bright Red
    '92': '#2ecc71',  # Bright Green
    '93': '#f39c12',  # Bright Yellow
    '94': '#3498db',  # Bright Blue
    '95': '#9b59b6',  # Bright Magenta
    '96': '#1abc9c',  # Bright Cyan
    '97': '#ecf0f1',  # Bright White

    # Background colors
    '40': '#000000',  # Black
    '41': '#e74c3c',  # Red
    '42': '#2ecc71',  # Green
    '43': '#f1c40f',  # Yellow
    '44': '#3498db',  # Blue
    '45': '#9b59b6',  # Magenta
    '46': '#1abc9c',  # Cyan
    '47': '#ecf0f1',  # White
    '100': '#7f8c8d', # Bright Black
    '101': '#e74c3c', # Bright Red
    '102': '#2ecc71', # Bright Green
    '103': '#f39c12', # Bright Yellow
    '104': '#3498db', # Bright Blue
    '105': '#9b59b6', # Bright Magenta
    '106': '#1abc9c', # Bright Cyan
    '107': '#ecf0f1', # Bright White
}

def run_sounio_example(sio_file):
    """Run a Sounio example and capture ANSI output"""
    souc_path = Path(__file__).parent.parent.parent / "target" / "debug" / "souc"

    if not souc_path.exists():
        print(f"Error: souc binary not found at {souc_path}")
        print("Build Sounio first: cargo build")
        sys.exit(1)

    result = subprocess.run(
        [str(souc_path), "run", str(sio_file)],
        capture_output=True,
        text=True
    )

    return result.stdout + result.stderr

def parse_ansi_codes(text):
    """Parse ANSI escape sequences and return styled segments"""
    segments = []
    current_fg = '#e0e0e0'  # Default foreground
    current_bg = None
    current_bold = False
    current_dim = False

    # Split by ANSI escape sequences
    parts = re.split(r'(\x1b\[[0-9;]*m)', text)

    for part in parts:
        if part.startswith('\x1b['):
            # Parse ANSI code
            codes = part[2:-1].split(';')
            for code in codes:
                if code == '0':  # Reset
                    current_fg = '#e0e0e0'
                    current_bg = None
                    current_bold = False
                    current_dim = False
                elif code == '1':  # Bold
                    current_bold = True
                elif code == '2':  # Dim
                    current_dim = True
                elif code in ['30', '31', '32', '33', '34', '35', '36', '37',
                             '90', '91', '92', '93', '94', '95', '96', '97']:
                    current_fg = ANSI_COLORS[code]
                elif code in ['40', '41', '42', '43', '44', '45', '46', '47',
                             '100', '101', '102', '103', '104', '105', '106', '107']:
                    current_bg = ANSI_COLORS[code]
        elif part:
            # Regular text
            segments.append({
                'text': part,
                'fg': current_fg,
                'bg': current_bg,
                'bold': current_bold,
                'dim': current_dim
            })

    return segments

def escape_xml(text):
    """Escape XML special characters"""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))

def ansi_to_svg(ansi_text, output_file):
    """Convert ANSI text to SVG"""
    lines = ansi_text.split('\n')

    # Calculate dimensions
    char_width = 10
    char_height = 18
    line_spacing = 20
    padding = 20

    max_line_length = max(len(line) for line in lines) if lines else 80
    width = max_line_length * char_width + 2 * padding
    height = len(lines) * line_spacing + 2 * padding

    # Start SVG
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    <!-- Sounio Visual Example - Generated from ANSI terminal output -->
    <style>
        .monospace {{ font-family: 'Monaco', 'Menlo', 'Consolas', monospace; font-size: 14px; }}
        .bold {{ font-weight: bold; }}
        .dim {{ opacity: 0.6; }}
    </style>

    <!-- Background -->
    <rect width="{width}" height="{height}" fill="#1a1a2e"/>

    <!-- Terminal output -->
    <g class="monospace">
'''

    y = padding + char_height
    for line in lines:
        segments = parse_ansi_codes(line)
        x = padding

        for seg in segments:
            text = seg['text']

            # Background rect if needed
            if seg['bg']:
                text_width = len(text) * char_width
                svg += f'        <rect x="{x}" y="{y - char_height + 3}" width="{text_width}" height="{char_height}" fill="{seg["bg"]}"/>\n'

            # Text element
            classes = "monospace"
            if seg['bold']:
                classes += " bold"
            if seg['dim']:
                classes += " dim"

            if text.strip():  # Only render non-whitespace
                svg += f'        <text x="{x}" y="{y}" fill="{seg["fg"]}" class="{classes}">{escape_xml(text)}</text>\n'

            x += len(text) * char_width

        y += line_spacing

    svg += '''    </g>
</svg>
'''

    with open(output_file, 'w') as f:
        f.write(svg)

    print(f"Created SVG: {output_file}")

def export_all_examples():
    """Export all visual examples to SVG"""
    visual_dir = Path(__file__).parent
    examples = sorted(visual_dir.glob("*.sio"))

    for example in examples:
        if example.name.startswith('12_'):  # Skip animation example
            continue

        output = example.with_suffix('.svg')
        print(f"\nExporting {example.name}...")

        try:
            ansi_output = run_sounio_example(example)
            ansi_to_svg(ansi_output, output)
        except Exception as e:
            print(f"Error exporting {example.name}: {e}")

def main():
    if len(sys.argv) == 2 and sys.argv[1] == '--all':
        export_all_examples()
    elif len(sys.argv) == 3:
        input_file = Path(sys.argv[1])
        output_file = Path(sys.argv[2])

        if not input_file.exists():
            print(f"Error: Input file '{input_file}' not found")
            sys.exit(1)

        print(f"Running Sounio example: {input_file}")
        ansi_output = run_sounio_example(input_file)

        print(f"Converting to SVG: {output_file}")
        ansi_to_svg(ansi_output, output_file)

        print(f"\nDone! Publication-quality SVG created at {output_file}")
    else:
        print(__doc__)
        sys.exit(1)

if __name__ == '__main__':
    main()

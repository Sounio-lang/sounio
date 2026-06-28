#!/usr/bin/env python3
"""Verify that a browser screenshot is nonblank and plausibly rendered."""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path


PNG_SIG = b"\x89PNG\r\n\x1a\n"


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def parse_png(path: Path) -> tuple[int, int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(PNG_SIG):
        raise ValueError("not a PNG file")
    pos = len(PNG_SIG)
    width = height = color_type = bit_depth = None
    compressed = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
        elif kind == b"IDAT":
            compressed.extend(chunk)
        elif kind == b"IEND":
            break
    if width is None or height is None or color_type is None or bit_depth != 8:
        raise ValueError("unsupported or missing PNG header")
    return width, height, color_type, zlib.decompress(bytes(compressed))


def unfilter(width: int, height: int, color_type: int, raw: bytes) -> bytes:
    channels_by_type = {2: 3, 6: 4}
    channels = channels_by_type[color_type]
    stride = width * channels
    out = bytearray(height * stride)
    src = 0
    for y in range(height):
        filter_type = raw[src]
        src += 1
        row = bytearray(raw[src : src + stride])
        src += stride
        prev_start = (y - 1) * stride
        cur_start = y * stride
        for x in range(stride):
            left = row[x - channels] if x >= channels else 0
            up = out[prev_start + x] if y > 0 else 0
            up_left = out[prev_start + x - channels] if y > 0 and x >= channels else 0
            if filter_type == 1:
                row[x] = (row[x] + left) & 0xFF
            elif filter_type == 2:
                row[x] = (row[x] + up) & 0xFF
            elif filter_type == 3:
                row[x] = (row[x] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                row[x] = (row[x] + paeth(left, up, up_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"unsupported filter {filter_type}")
        out[cur_start : cur_start + stride] = row
    return bytes(out)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_screenshot_pixels.py <screenshot.png>", file=sys.stderr)
        return 2
    width, height, color_type, raw = parse_png(Path(sys.argv[1]))
    pixels = unfilter(width, height, color_type, raw)
    channels = 4 if color_type == 6 else 3
    unique = set()
    bright = warm = cyan = 0
    samples = 0
    for i in range(0, len(pixels), channels * 137):
        r, g, b = pixels[i : i + 3]
        unique.add((r, g, b))
        samples += 1
        if max(r, g, b) > 96:
            bright += 1
        if r > 130 and g > 85 and b < 140:
            warm += 1
        if g > 120 and b > 120 and r < 140:
            cyan += 1
    bright_floor = max(10, samples // 25)
    if width < 900 or height < 600 or len(unique) < 128 or bright < bright_floor or warm < 5 or cyan < 5:
        print("SCREENSHOT_PIXEL_FAIL likely blank or under-rendered")
        print(f"dimensions={width}x{height} unique_samples={len(unique)} bright_samples={bright}/{samples} warm={warm} cyan={cyan} bright_floor={bright_floor}")
        return 1
    print("SCREENSHOT_PIXEL_PASS")
    print(f"dimensions={width}x{height} unique_samples={len(unique)} bright_samples={bright}/{samples} warm={warm} cyan={cyan} bright_floor={bright_floor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

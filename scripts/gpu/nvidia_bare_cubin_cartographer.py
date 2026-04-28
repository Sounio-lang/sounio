#!/usr/bin/env python3
"""Read-only cartography for existing NVIDIA CUBIN/SASS specimens.

The output is a specimen map for deriving and checking structural tables. It is
not a compiler intermediate and does not invoke vendor assemblers or compilers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path
from typing import Any


SHT_CUDA_INFO = 0x70000000
EM_CUDA = 190
SCHEMA = "sounio.gpu.nvidia-bare.cubin-cartography.v1"


def _read_c_string(blob: bytes, off: int) -> str:
    if off < 0 or off >= len(blob):
        return ""
    end = blob.find(b"\0", off)
    if end < 0:
        end = len(blob)
    return blob[off:end].decode("utf-8", errors="replace")


def _section_headers(blob: bytes) -> list[dict[str, Any]]:
    if len(blob) < 64 or blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        return []
    shoff = struct.unpack_from("<Q", blob, 40)[0]
    shentsize = struct.unpack_from("<H", blob, 58)[0]
    shnum = struct.unpack_from("<H", blob, 60)[0]
    shstrndx = struct.unpack_from("<H", blob, 62)[0]
    raw = []
    for idx in range(shnum):
        off = shoff + idx * shentsize
        if off + 64 > len(blob):
            break
        fields = struct.unpack_from("<IIQQQQIIQQ", blob, off)
        raw.append(
            {
                "index": idx,
                "name_offset": fields[0],
                "type": fields[1],
                "flags": fields[2],
                "addr": fields[3],
                "offset": fields[4],
                "size": fields[5],
                "link": fields[6],
                "info": fields[7],
                "align": fields[8],
                "entsize": fields[9],
            }
        )
    shstr = b""
    if 0 <= shstrndx < len(raw):
        sec = raw[shstrndx]
        shstr = blob[sec["offset"] : sec["offset"] + sec["size"]]
    for sec in raw:
        sec["name"] = _read_c_string(shstr, sec["name_offset"])
    return raw


def _nv_info_tags(blob: bytes, sec: dict[str, Any]) -> list[dict[str, Any]]:
    payload = blob[sec["offset"] : sec["offset"] + sec["size"]]
    tags: list[dict[str, Any]] = []
    i = 0
    while i + 1 < len(payload):
        fmt = payload[i]
        attr = payload[i + 1]
        if fmt == 0 and attr == 0:
            i += 1
            continue
        entry: dict[str, Any] = {
            "offset": i,
            "format": fmt,
            "attribute": attr,
            "attribute_hex": f"0x{attr:02x}",
        }
        if i + 4 <= len(payload):
            size = struct.unpack_from("<H", payload, i + 2)[0]
            entry["declared_size"] = size
            i += max(4, 4 + size)
        else:
            i += 2
        tags.append(entry)
    return tags


def _metadata_summary(blob: bytes, sections: list[dict[str, Any]]) -> dict[str, Any]:
    nv_info = [s for s in sections if s.get("name", "").startswith(".nv.info")]
    constant0 = [s for s in sections if s.get("name", "").startswith(".nv.constant0.")]
    text = [s for s in sections if s.get("name", "").startswith(".text.")]
    per_kernel = []
    for sec in nv_info:
        name = sec.get("name", "")
        if name == ".nv.info":
            continue
        kernel = name.removeprefix(".nv.info.")
        payload = blob[sec["offset"] : sec["offset"] + sec["size"]]
        tags = _nv_info_tags(blob, sec)
        per_kernel.append(
            {
                "kernel": kernel,
                "nv_info_section": name,
                "text_section": f".text.{kernel}" if any(s.get("name") == f".text.{kernel}" for s in text) else "",
                "constant0_section": (
                    f".nv.constant0.{kernel}"
                    if any(s.get("name") == f".nv.constant0.{kernel}" for s in constant0)
                    else ""
                ),
                "has_param_cbank": b"\x04\x0a" in payload,
                "has_cbank_param_size": b"\x03\x19" in payload,
                "has_kparam_info": b"\x04\x17" in payload,
                "has_exit_instr_offsets": b"\x04\x1c" in payload,
                "nv_info_tags": tags,
            }
        )
    return {
        "nv_info_section_count": len(nv_info),
        "constant0_section_count": len(constant0),
        "text_section_count": len(text),
        "kernels": per_kernel,
    }


def _inspect_cubin(path: Path, root: Path) -> dict[str, Any]:
    blob = path.read_bytes()
    header: dict[str, Any] = {"valid_elf64_le": False}
    if len(blob) >= 64 and blob[:4] == b"\x7fELF" and blob[4] == 2 and blob[5] == 1:
        header = {
            "valid_elf64_le": True,
            "osabi": blob[7],
            "abi_version": blob[8],
            "type": struct.unpack_from("<H", blob, 16)[0],
            "machine": struct.unpack_from("<H", blob, 18)[0],
            "machine_name": "NVIDIA CUDA architecture" if struct.unpack_from("<H", blob, 18)[0] == EM_CUDA else "unknown",
            "version": struct.unpack_from("<I", blob, 20)[0],
            "flags": struct.unpack_from("<I", blob, 48)[0],
            "flags_hex": f"0x{struct.unpack_from('<I', blob, 48)[0]:x}",
            "sm80_flag_present": (struct.unpack_from("<I", blob, 48)[0] & 0x550) == 0x550,
            "section_header_offset": struct.unpack_from("<Q", blob, 40)[0],
            "section_count": struct.unpack_from("<H", blob, 60)[0],
            "section_string_index": struct.unpack_from("<H", blob, 62)[0],
        }
    sections = _section_headers(blob)
    text_sections = [s for s in sections if s.get("name", "").startswith(".text.")]
    info_sections = [s for s in sections if s.get("name", "").startswith(".nv.info")]
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
        "elf_header": header,
        "metadata_summary": _metadata_summary(blob, sections),
        "text_sections": text_sections,
        "nv_info_sections": info_sections,
        "sections": sections,
    }


def _readelf_sections(path: Path) -> str:
    try:
        return subprocess.run(
            ["readelf", "-S", str(path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout
    except FileNotFoundError:
        return ""


def _inspect_sass(path: Path, root: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pairs: list[dict[str, str]] = []
    pending: dict[str, str] | None = None
    for line in text.splitlines():
        m = re.search(r"/\*([0-9a-fA-F]{4})\*/.*?/\*\s*(0x[0-9a-fA-F]{16})\s*\*/", line)
        if m:
            pending = {"offset": m.group(1), "lo": m.group(2)}
            continue
        m = re.search(r"/\*\s*(0x[0-9a-fA-F]{16})\s*\*/", line)
        if m and pending is not None:
            pending["hi"] = m.group(1)
            pairs.append(pending)
            pending = None
    sections = sorted(set(re.findall(r"\.section\s+([.\w]+)", text)))
    nvinfo_attrs = sorted(set(re.findall(r"EIATTR_([A-Z0-9_]+)", text)))
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
        "size_bytes": len(text),
        "headerflags": re.findall(r"\.headerflags\s+@\"([^\"]+)\"", text),
        "sections": sections,
        "nvinfo_attributes": nvinfo_attrs,
        "constant0_sections": [s for s in sections if s.startswith(".nv.constant0.")],
        "text_sections": [s for s in sections if s.startswith(".text.")],
        "sass_instruction_word_pairs": pairs[:64],
        "sass_instruction_word_pair_count": len(pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Map existing Sounio CUBIN/SASS specimens")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--out", default="artifacts/omega/nvidia_bare_cubin_cartography.v1.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cubins = sorted((root / "artifacts/sass/cubin").glob("*.cubin"))
    cubins += sorted((root / "artifacts/sass/omega").glob("*.cubin"))
    sass = sorted((root / "artifacts/sass/omega").glob("*.sass"))

    specimen = {
        "schema": SCHEMA,
        "purpose": "read_only_specimen_map_not_compiler_intermediate",
        "cubin_specimens": [_inspect_cubin(p, root) for p in cubins],
        "sass_specimens": [_inspect_sass(p, root) for p in sass],
        "tooling": {
            "readelf_available": bool(_readelf_sections(cubins[0])) if cubins else False,
            "inspection_only": ["readelf", "file", "cuobjdump", "nvdisasm"],
            "forbidden_for_proof": ["nvcc", "ptxas", "clang", "llc", "PTX compiler APIs"],
        },
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(specimen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        display_path = out.relative_to(root)
    except ValueError:
        display_path = out
    print(f"nvidia_bare_cubin_cartographer: wrote {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

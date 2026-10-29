#!/usr/bin/env python3
"""Generate a tiny Mixtral-shaped GGUF for llama_load_gguf smoke tests.

Tensor names follow llama.cpp MoE layout (blk.N.ffn_*_exps.weight Q4_K packs).
KV metadata uses llama.* keys read by llama_load_gguf in stdlib/llm/llama.sio.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

GGUF_VAL_UINT8 = 0
GGUF_VAL_INT8 = 1
GGUF_VAL_UINT16 = 2
GGUF_VAL_INT16 = 3
GGUF_VAL_UINT32 = 4
GGUF_VAL_INT32 = 5
GGUF_VAL_FLOAT32 = 6
GGUF_VAL_BOOL = 7
GGUF_VAL_STRING = 8
GGUF_VAL_ARRAY = 9
GGUF_VAL_UINT64 = 10
GGUF_VAL_INT64 = 11
GGUF_VAL_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_Q4_K = 12

GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q5_K = 13

QUANT_SPECS = {
    "q4k": (GGML_TYPE_Q4_K, "pack_q4k_bytes"),
    "q5k": (GGML_TYPE_Q5_K, "pack_q5k_bytes"),
    "q8_0": (GGML_TYPE_Q8_0, "pack_q8_0_bytes"),
}


def write_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_u32(key: str, val: int) -> bytes:
    return write_string(key) + struct.pack("<i", GGUF_VAL_UINT32) + struct.pack("<I", val)


def kv_f32(key: str, val: float) -> bytes:
    return write_string(key) + struct.pack("<i", GGUF_VAL_FLOAT32) + struct.pack("<f", val)


def kv_str(key: str, val: str) -> bytes:
    return write_string(key) + struct.pack("<i", GGUF_VAL_STRING) + write_string(val)


def f32_bytes(vals: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(vals), *vals)


def f32_to_f16_bits(v: float) -> int:
    if v <= 0.0:
        return 0
    exp = 15
    mant = v
    while mant >= 2.0 and exp < 30:
        mant /= 2.0
        exp += 1
    while mant < 1.0 and exp > 1:
        mant *= 2.0
        exp -= 1
    frac = int((mant - 1.0) * 1024.0)
    return (exp << 10) | (frac & 1023)


def pack_q4k_bytes(experts: int, m: int, k: int, fill: float) -> bytes:
    assert k % 256 == 0
    nqb = k // 256
    row_stride = nqb * 144
    expert_stride = m * row_stride
    buf = bytearray(experts * expert_stride)
    dbits = f32_to_f16_bits(fill / 15.0)
    for e in range(experts):
        for r in range(m):
            for ib in range(nqb):
                base = e * expert_stride + r * row_stride + ib * 144
                buf[base + 0] = dbits & 255
                buf[base + 1] = (dbits >> 8) & 255
                buf[base + 2] = 0
                buf[base + 3] = 0
                for s in range(4):
                    buf[base + 4 + s] = 1
                    buf[base + 8 + s] = 0
                    buf[base + 12 + s] = 1
                for q in range(128):
                    buf[base + 16 + q] = (8 << 4) | 8
    return bytes(buf)


def pack_q5k_bytes(experts: int, m: int, k: int, fill: float) -> bytes:
    assert k % 256 == 0
    nqb = k // 256
    row_stride = nqb * 176
    expert_stride = m * row_stride
    buf = bytearray(experts * expert_stride)
    dbits = f32_to_f16_bits(fill / 31.0)
    mbits = f32_to_f16_bits(0.0)
    for e in range(experts):
        for r in range(m):
            for ib in range(nqb):
                base = e * expert_stride + r * row_stride + ib * 176
                buf[base + 0] = dbits & 255
                buf[base + 1] = (dbits >> 8) & 255
                buf[base + 2] = mbits & 255
                buf[base + 3] = (mbits >> 8) & 255
                for s in range(12):
                    buf[base + 4 + s] = 1
                for q in range(128):
                    buf[base + 16 + q] = (8 << 4) | 8
    return bytes(buf)


def pack_q8_0_bytes(experts: int, m: int, k: int, fill: float) -> bytes:
    assert k % 32 == 0
    nqb = k // 32
    row_stride = nqb * 34
    expert_stride = m * row_stride
    buf = bytearray(experts * expert_stride)
    dbits = f32_to_f16_bits(fill)
    for e in range(experts):
        for r in range(m):
            for ib in range(nqb):
                base = e * expert_stride + r * row_stride + ib * 34
                buf[base + 0] = dbits & 255
                buf[base + 1] = (dbits >> 8) & 255
                for q in range(32):
                    buf[base + 2 + q] = 1
    return bytes(buf)


def pattern(n: int, scale: float) -> list[float]:
    return [scale * (((i % 7) + 1)) for i in range(n)]


def filled(n: int, v: float) -> list[float]:
    return [v] * n


def ggml_f32_matrix(rows: int, cols: int, data: list[float]) -> tuple[list[int], bytes]:
    # ggml stores [cols, rows] for rank-2 F32 (inner dim first).
    return [cols, rows], f32_bytes(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quant",
        choices=sorted(QUANT_SPECS.keys()),
        default="q4k",
        help="MoE expert pack dtype (gate/up/down)",
    )
    ap.add_argument(
        "--out",
        default="/workspace/llm-models/tiny-mixtral-moe.gguf",
        help="output GGUF file path",
    )
    args = ap.parse_args()
    quant = args.quant
    ggml_moe_type, pack_name = QUANT_SPECS[quant]
    pack_fn = globals()[pack_name]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    hidden = 256
    inter = 512
    moe_inter = 512
    layers = 2
    heads = 4
    kv_heads = 1
    head_dim = hidden // heads
    vocab = 64
    experts = 8
    slots = 2
    kvdim = kv_heads * head_dim

    kvs: list[bytes] = [
        kv_str("general.architecture", "llama"),
        kv_str("general.name", "tiny-mixtral-moe-fixture"),
        kv_u32("general.alignment", 32),
        kv_u32("llama.block_count", layers),
        kv_u32("llama.context_length", 512),
        kv_u32("llama.embedding_length", hidden),
        kv_u32("llama.feed_forward_length", moe_inter),
        kv_u32("llama.attention.head_count", heads),
        kv_u32("llama.attention.head_count_kv", kv_heads),
        kv_u32("llama.vocab_size", vocab),
        kv_u32("llama.expert_count", experts),
        kv_u32("llama.expert_used_count", slots),
        kv_f32("llama.rope.freq_base", 10000.0),
        kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
    ]
    kv_blob = b"".join(kvs)

    tensors: list[tuple[str, int, list[int], bytes]] = []

    def add_f32(name: str, shape: list[int], data: list[float]) -> None:
        tensors.append((name, GGML_TYPE_F32, shape, f32_bytes(data)))

    def add_moe(name: str, shape: list[int], raw: bytes) -> None:
        tensors.append((name, ggml_moe_type, shape, raw))

    emb = pattern(vocab * hidden, 0.001)
    add_f32("token_embd.weight", [hidden, vocab], emb)
    add_f32("output_norm.weight", [hidden], filled(hidden, 1.0))
    add_f32("output.weight", [hidden, vocab], pattern(vocab * hidden, 0.0001))

    for i in range(layers):
        bp = f"blk.{i}"
        add_f32(f"{bp}.attn_norm.weight", [hidden], filled(hidden, 1.0))
        add_f32(f"{bp}.ffn_norm.weight", [hidden], filled(hidden, 1.0))
        add_f32(f"{bp}.attn_q.weight", [hidden, hidden], pattern(hidden * hidden, 1e-6))
        add_f32(f"{bp}.attn_k.weight", [hidden, kvdim], pattern(kvdim * hidden, 1e-6))
        add_f32(f"{bp}.attn_v.weight", [hidden, kvdim], pattern(kvdim * hidden, 1e-6))
        add_f32(f"{bp}.attn_output.weight", [hidden, hidden], pattern(hidden * hidden, 1e-6))
        router = [0.00001 * (e + 1) for e in range(experts) for _ in range(hidden)]
        add_f32(f"{bp}.ffn_gate_inp.weight", [hidden, experts], router)
        fill = 0.02 * (i + 1)
        gate_raw = pack_fn(experts, moe_inter, hidden, fill)
        up_raw = pack_fn(experts, moe_inter, hidden, fill)
        down_raw = pack_fn(experts, hidden, moe_inter, fill)
        # Flattened expert rows: [k, experts * m] with k divisible by 256.
        add_moe(f"{bp}.ffn_gate_exps.weight", [hidden, experts * moe_inter], gate_raw)
        add_moe(f"{bp}.ffn_up_exps.weight", [hidden, experts * moe_inter], up_raw)
        add_moe(f"{bp}.ffn_down_exps.weight", [moe_inter, experts * hidden], down_raw)

    align = 32
    tensor_infos: list[bytes] = []
    data_parts: list[bytes] = []
    offset = 0
    for name, dtype, shape, raw in tensors:
        info = write_string(name)
        info += struct.pack("<I", len(shape))
        for d in shape:
            info += struct.pack("<q", d)
        info += struct.pack("<i", dtype)
        info += struct.pack("<Q", offset)
        tensor_infos.append(info)
        pad = (align - (len(raw) % align)) % align
        data_parts.append(raw + (b"\x00" * pad))
        offset += len(raw) + pad

    header = GGUF_MAGIC
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(kvs))
    header += kv_blob
    header += b"".join(tensor_infos)
    pad = (align - (len(header) % align)) % align
    header += b"\x00" * pad
    blob = header + b"".join(data_parts)
    out.write_bytes(blob)
    print(f"wrote {out} ({len(blob)} bytes)")
    print(
        f"  layers={layers} hidden={hidden} moe_inter={moe_inter} "
        f"experts={experts} slots={slots} vocab={vocab} quant={quant}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a tiny Mixtral-shaped fixture for llama_load_moe smoke tests.

Layout:
  <out>/config.json
  <out>/model.safetensors   (attn + norms + embed + MoE routers; f32)
  <out>/moe_q4k/layer{i}_{gate,up,down}.bin  (synthetic Q4_K packs)

Shapes stay Q4_K-aligned (hidden and moe_inter divisible by 256).
No HuggingFace / safetensors pip deps: hand-rolled .safetensors writer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path


def f32_bytes(vals: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(vals), *vals)


def write_safetensors(path: Path, tensors: dict[str, tuple[list[int], list[float]]]) -> None:
    # tensors: name -> (shape, flat f32 values)
    header: dict = {}
    blobs: list[bytes] = []
    offset = 0
    for name, (shape, data) in tensors.items():
        raw = f32_bytes(data)
        begin, end = offset, offset + len(raw)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [begin, end]}
        blobs.append(raw)
        offset = end
    meta = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # pad header to 8-byte alignment after the 8-byte length prefix
    pad = (8 - (len(meta) % 8)) % 8
    meta = meta + (b" " * pad)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(meta)))
        f.write(meta)
        for b in blobs:
            f.write(b)


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


def pack_q4k(path: Path, experts: int, m: int, k: int, fill: float) -> None:
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
    path.write_bytes(buf)


def pattern(n: int, scale: float) -> list[float]:
    return [scale * (((i % 7) + 1)) for i in range(n)]


def filled(n: int, v: float) -> list[float]:
    return [v] * n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="/workspace/llm-models/tiny-mixtral-moe",
        help="output model directory",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "moe_q4k").mkdir(exist_ok=True)

    hidden = 256
    inter = 512  # dense unused
    moe_inter = 512
    layers = 2
    heads = 4
    kv_heads = 1
    head_dim = hidden // heads
    vocab = 64
    experts = 8
    slots = 2
    kvdim = kv_heads * head_dim

    cfg = {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "hidden_size": hidden,
        "intermediate_size": inter,
        "moe_intermediate_size": moe_inter,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "vocab_size": vocab,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "num_local_experts": experts,
        "num_experts_per_tok": slots,
        "torch_dtype": "float32",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    tensors: dict[str, tuple[list[int], list[float]]] = {}
    tensors["model.embed_tokens.weight"] = ([vocab, hidden], pattern(vocab * hidden, 0.001))
    tensors["model.norm.weight"] = ([hidden], filled(hidden, 1.0))
    tensors["lm_head.weight"] = ([vocab, hidden], pattern(vocab * hidden, 0.0001))

    for i in range(layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = ([hidden], filled(hidden, 1.0))
        tensors[f"{p}.post_attention_layernorm.weight"] = ([hidden], filled(hidden, 1.0))
        # HF Linear weight [out, in]; ll_load_t will transpose
        tensors[f"{p}.self_attn.q_proj.weight"] = ([hidden, hidden], pattern(hidden * hidden, 1e-6))
        tensors[f"{p}.self_attn.k_proj.weight"] = ([kvdim, hidden], pattern(kvdim * hidden, 1e-6))
        tensors[f"{p}.self_attn.v_proj.weight"] = ([kvdim, hidden], pattern(kvdim * hidden, 1e-6))
        tensors[f"{p}.self_attn.o_proj.weight"] = ([hidden, hidden], pattern(hidden * hidden, 1e-6))
        tensors[f"{p}.block_sparse_moe.gate.weight"] = (
            [experts, hidden],
            [0.00001 * (e + 1) for e in range(experts) for _h in range(hidden)],
        )
        # Dense MLP fallback tensors (unused when MoE packs + GPU ready).
        tensors[f"{p}.mlp.gate_proj.weight"] = ([inter, hidden], filled(inter * hidden, 1e-6))
        tensors[f"{p}.mlp.up_proj.weight"] = ([inter, hidden], filled(inter * hidden, 1e-6))
        tensors[f"{p}.mlp.down_proj.weight"] = ([hidden, inter], filled(hidden * inter, 1e-6))

    write_safetensors(out / "model.safetensors", tensors)

    for i in range(layers):
        fill = 0.02 * (i + 1)
        pack_q4k(out / "moe_q4k" / f"layer{i}_gate.bin", experts, moe_inter, hidden, fill)
        pack_q4k(out / "moe_q4k" / f"layer{i}_up.bin", experts, moe_inter, hidden, fill)
        pack_q4k(out / "moe_q4k" / f"layer{i}_down.bin", experts, hidden, moe_inter, fill)

    st_size = (out / "model.safetensors").stat().st_size
    pack_size = sum(p.stat().st_size for p in (out / "moe_q4k").glob("*.bin"))
    print(f"wrote {out}")
    print(f"  config layers={layers} hidden={hidden} moe_inter={moe_inter} experts={experts}")
    print(f"  safetensors={st_size} bytes packs={pack_size} bytes")


if __name__ == "__main__":
    main()

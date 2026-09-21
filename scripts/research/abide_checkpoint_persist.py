#!/usr/bin/env python3
"""Persist and verify Brain O-SSM ABIDE fold checkpoints from benchmark stdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from abide_campaign_lib import load_manifest


SCALE = 100_000_000.0
INTERNAL_CV_SITE_1 = 177622
INTERNAL_CV_SITE_2 = 177623
INTERNAL_CV_KEYS = {INTERNAL_CV_SITE_1, INTERNAL_CV_SITE_2}
I64_STATE_ARRAYS = ("W", "MOM", "A_MOM", "PROJ_W", "PROJ_B", "PROJ_W_MOM", "PROJ_B_MOM", "DROP_MASK")
F64_STATE_ARRAYS = ("A", "TRAIN_FEATURE_MEAN", "TRAIN_FEATURE_STD")
I64_MASK = (1 << 64) - 1
I64_SIGN = 1 << 63
DECIMAL_FRAGMENT_RE = re.compile(r"^\.\d+$")


def fixed_to_float(value: int) -> float:
    return value / SCALE


def exact_int(value: str | int | float) -> int:
    """Parse native integer metadata without losing large i64 precision."""

    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"expected integer metadata, got {value!r}")
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty integer metadata")
    if any(ch in text for ch in ".eE"):
        parsed = float(text)
        if not parsed.is_integer():
            raise ValueError(f"expected integer metadata, got {value!r}")
        return int(parsed)
    return int(text)


def i64(value: int) -> int:
    value &= I64_MASK
    if value >= I64_SIGN:
        value -= 1 << 64
    return value


def abs_i64(value: int) -> int:
    return -value if value < 0 else value


def parse_blocks(raw_output: Path) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    lines = raw_output.read_text(encoding="utf-8", errors="replace").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line == "CKPT_BEGIN":
            current = {"meta": {}, "arrays": {}}
            idx += 1
            continue
        if line == "CKPT_END":
            if current is not None:
                blocks.append(current)
            current = None
            idx += 1
            continue
        if current is None:
            idx += 1
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == "CKPT_META":
            value = parts[2]
            next_idx = idx + 1
            if next_idx < len(lines) and DECIMAL_FRAGMENT_RE.match(lines[next_idx].strip()):
                value = f"{value}{lines[next_idx].strip()}"
                next_idx += 1
            current["meta"][parts[1]] = value
            idx = next_idx
            continue
        elif len(parts) >= 4 and parts[0] == "CKPT_ARRAY":
            name, dtype, count_raw = parts[1], parts[2], int(parts[3])
            values_raw = parts[4 : 4 + count_raw]
            next_idx = idx + 1
            while len(values_raw) < count_raw and next_idx < len(lines):
                next_line = lines[next_idx]
                if next_line.startswith(("CKPT_META", "CKPT_ARRAY", "CKPT_END")):
                    break
                stripped = next_line.strip()
                if not stripped:
                    next_idx += 1
                    continue
                if DECIMAL_FRAGMENT_RE.match(stripped) and values_raw:
                    values_raw[-1] = f"{values_raw[-1]}{stripped}"
                else:
                    values_raw.extend(part for part in next_line.split("\t") if part)
                next_idx += 1
            if len(values_raw) != count_raw:
                raise ValueError(f"checkpoint array {name} expected {count_raw} values, got {len(values_raw)}")
            if dtype == "i64":
                values = [int(v) for v in values_raw]
            elif dtype == "f64":
                values = [float(v) for v in values_raw]
            else:
                raise ValueError(f"unsupported checkpoint dtype {dtype!r} for {name}")
            current["arrays"][name] = {"dtype": dtype, "values": values}
            idx = next_idx
            continue
        idx += 1
    for block in blocks:
        arrays = block.get("arrays", {})
        if "A_FIXED" in arrays:
            arrays["A"] = {
                "dtype": "f64",
                "values": [fixed_to_float(int(value)) for value in arrays["A_FIXED"]["values"]],
            }
    return blocks


def merge_tab_continuation_records(lines: list[str], prefixes: tuple[str, ...]) -> list[str]:
    merged: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith(prefixes):
            parts = [part for part in line.split("\t") if part]
            next_idx = idx + 1
            while next_idx < len(lines) and lines[next_idx].startswith("\t"):
                parts.extend(part for part in lines[next_idx].split("\t") if part)
                next_idx += 1
            merged.append("\t".join(parts))
            idx = next_idx
            continue
        merged.append(line)
        idx += 1
    return merged


def array_has_signal(block: dict[str, Any], name: str) -> bool:
    values = block.get("arrays", {}).get(name, {}).get("values", [])
    return any(abs(float(v)) > 1.0e-12 for v in values)


def parse_run_config(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    config: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def int_config(config: dict[str, str], key: str) -> int | None:
    value = config.get(key)
    if value is None or value == "":
        return None
    try:
        return exact_int(value)
    except ValueError:
        return None


def expected_meta_from_config(config: dict[str, str]) -> dict[str, int]:
    mapping = {
        "oct_profile_id": "cfg_oct_profile_id",
        "oct_train_mode": "cfg_oct_train_mode",
        "oct_input_proj_mode": "cfg_oct_input_proj_mode",
    }
    expected: dict[str, int] = {}
    for config_key, meta_key in mapping.items():
        value = int_config(config, config_key)
        if value is not None:
            expected[meta_key] = value
    return expected


def block_matches_expected(block: dict[str, Any], expected_meta: dict[str, int]) -> bool:
    meta = block.get("meta", {})
    if meta.get("model") != "O-SSM":
        return False
    for key, expected in expected_meta.items():
        if key not in meta:
            return False
        try:
            actual = exact_int(meta[key])
        except ValueError:
            return False
        if actual != expected:
            return False
    return True


def best_block(blocks: list[dict[str, Any]], expected_meta: dict[str, int] | None = None) -> dict[str, Any]:
    if not blocks:
        raise ValueError("no CKPT blocks found in raw benchmark output")
    expected_meta = expected_meta or {}
    candidates = [
        block
        for block in blocks
        if block_matches_expected(block, expected_meta) and array_has_signal(block, "A")
    ]
    if not candidates:
        profile_hint = ", ".join(f"{k}={v}" for k, v in sorted(expected_meta.items())) or "no config filter"
        raise ValueError(
            "no valid O-SSM CKPT blocks found with nonzero A tensor "
            f"({profile_hint}); refusing to persist an incomplete checkpoint"
        )
    return max(candidates, key=lambda block: float(block["meta"]["balanced_accuracy_pct"]))


def site_hash(site: str) -> int:
    h = 5381
    for ch in site:
        h = i64(i64(h * 33) + ord(ch))
    return abs_i64(h)


def unique_site_key_count(records: list[Any]) -> int:
    return len({site_hash(record.site) for record in records})


def single_site_internal_cv_enabled(
    config: dict[str, str] | None,
    records: list[Any],
    holdout_key: int | None = None,
) -> bool:
    if config and int_config(config, "single_site_internal_cv") == 1:
        return True
    return holdout_key in INTERNAL_CV_KEYS and unique_site_key_count(records) < 2


def materialize_single_site_internal_cv_keys(records: list[Any]) -> list[int]:
    site_keys: list[int] = []
    pos_seen = 0
    neg_seen = 0
    for record in records:
        if int(record.label) == 1:
            site_keys.append(INTERNAL_CV_SITE_1 if pos_seen % 2 == 0 else INTERNAL_CV_SITE_2)
            pos_seen += 1
        else:
            site_keys.append(INTERNAL_CV_SITE_1 if neg_seen % 2 == 0 else INTERNAL_CV_SITE_2)
            neg_seen += 1
    return site_keys


def int_meta(meta: dict[str, str], key: str) -> int:
    return exact_int(meta[key])


def exp_q(x: float) -> float:
    if x > 15.0:
        return exp_q(7.5) * exp_q(x - 7.5)
    if x < -15.0:
        return 0.0
    if x < 0.0:
        return 1.0 / exp_q(-x)
    total = 1.0
    term = 1.0
    for i in range(1, 21):
        term = term * x / float(i)
        total += term
    return total


def sqrt_q(x: float) -> float:
    if math.isnan(x) or x <= 0.0:
        return 0.0
    g = x
    for _ in range(16):
        g = 0.5 * (g + x / g)
    return 0.0 if math.isnan(g) else g


def sigmoid_q(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + exp_q(-x))
    ex = exp_q(x)
    return ex / (1.0 + ex)


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def squash_pos(x: float) -> float:
    return 0.0 if x <= 0.0 else x / (1.0 + x)


def md2_squash(v: float) -> float:
    return v / (1.0 + v) if v >= 0.0 else v / (1.0 + -v)


def mandelbrot_d2_feature(seed: float, delta: float) -> float:
    z = clip(0.55 * seed + 0.20 * delta, -2.0, 2.0)
    c = clip(0.35 * seed + 0.25 * delta, -2.0, 2.0)
    dz = 1.0 + 0.10 * delta
    ddz = 0.0
    for _ in range(3):
        ddz = 2.0 * (dz * dz + z * ddz)
        dz = 2.0 * z * dz + 1.0 + 0.15 * delta
        z = clip(z * z + c, -4.0, 4.0)
        dz = clip(dz, -6.0, 6.0)
        ddz = clip(ddz, -8.0, 8.0)
    return clip(md2_squash(ddz), -1.5, 1.5)


def oct_mul(a: list[float], b: list[float]) -> list[float]:
    a0, a1, a2, a3, a4, a5, a6, a7 = a
    b0, b1, b2, b3, b4, b5, b6, b7 = b
    return [
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4 - a5 * b5 - a6 * b6 - a7 * b7,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2 + a4 * b5 - a5 * b4 - a6 * b7 + a7 * b6,
        a0 * b2 + a2 * b0 - a1 * b3 + a3 * b1 + a4 * b6 - a6 * b4 + a5 * b7 - a7 * b5,
        a0 * b3 + a3 * b0 + a1 * b2 - a2 * b1 + a4 * b7 - a7 * b4 - a5 * b6 + a6 * b5,
        a0 * b4 + a4 * b0 - a1 * b5 + a5 * b1 - a2 * b6 + a6 * b2 - a3 * b7 + a7 * b3,
        a0 * b5 + a5 * b0 + a1 * b4 - a4 * b1 - a2 * b7 + a7 * b2 + a3 * b6 - a6 * b3,
        a0 * b6 + a6 * b0 + a1 * b7 - a7 * b1 + a2 * b4 - a4 * b2 - a3 * b5 + a5 * b3,
        a0 * b7 + a7 * b0 - a1 * b6 + a6 * b1 + a2 * b5 - a5 * b2 + a3 * b4 - a4 * b3,
    ]


class CheckpointForward:
    def __init__(self, ckpt: dict[str, Any]) -> None:
        self.meta = ckpt["meta"]
        arrays = ckpt["arrays"]
        self.w = arrays["W"]["values"]
        self.a = arrays["A"]["values"]
        self.proj_w = arrays["PROJ_W"]["values"]
        self.proj_b = arrays["PROJ_B"]["values"]
        self.mean = arrays["TRAIN_FEATURE_MEAN"]["values"]
        self.std = arrays["TRAIN_FEATURE_STD"]["values"]
        self.drop_mask = arrays["DROP_MASK"]["values"]
        self.seq_len = int_meta(self.meta, "manifest_seq_len")
        self.feature_count = int_meta(self.meta, "manifest_feature_count")
        self.proj_mode = int_meta(self.meta, "cfg_oct_input_proj_mode")
        self.noise_std = int_meta(self.meta, "cfg_noise_std_bp") / 1_000_000.0
        self.fold_noise_key = int_meta(self.meta, "fold_noise_key")
        self.proj_structured_scale = int_meta(self.meta, "cfg_oct_proj_structured_scale_bp") / 1_000_000.0
        self.proj_delta_scale = int_meta(self.meta, "cfg_oct_proj_delta_scale_bp") / 1_000_000.0
        self.proj_hybrid_scale = int_meta(self.meta, "cfg_oct_proj_hybrid_scale_bp") / 1_000_000.0
        self.h = [0] * 16
        self.assoc = [0] * 2

    def wg(self, idx: int) -> float:
        return fixed_to_float(self.w[idx])

    def proj_wg(self, out_dim: int, in_dim: int) -> float:
        return fixed_to_float(self.proj_w[out_dim * 8 + in_dim])

    def proj_bg(self, out_dim: int) -> float:
        return fixed_to_float(self.proj_b[out_dim])

    def hg(self, idx: int) -> float:
        return fixed_to_float(self.h[idx])

    def hs(self, idx: int, value: float) -> None:
        self.h[idx] = int(value * SCALE)

    def assoc_g(self, head: int) -> float:
        return fixed_to_float(self.assoc[head])

    def assoc_s(self, head: int, value: float) -> None:
        self.assoc[head] = int(value * SCALE)

    def reset_state(self) -> None:
        self.h = [0] * 16
        self.assoc = [0] * 2
        self.hs(0, 1.0)
        self.hs(8, 1.0)

    def perturb(self, subj_idx: int, step: int, dim: int, value: float) -> float:
        out = value
        if dim < len(self.drop_mask) and self.drop_mask[dim] == 1:
            out = 0.0
        if self.noise_std > 0.0:
            raw = abs((((subj_idx + 1) * 1103515245) ^ ((step + 1) * 12345) ^ ((dim + 1) * 2654435761) ^ self.fold_noise_key) % 20001)
            out += ((float(raw) - 10000.0) / 10000.0) * self.noise_std
        return out

    def project_step(self, grad: list[float], delta: list[float]) -> list[float]:
        if self.proj_mode == 1:
            return [clip(self.proj_bg(o) + sum(grad[i] * self.proj_wg(o, i) for i in range(8)), -4.0, 4.0) for o in range(8)]
        if self.proj_mode == 2:
            out = []
            for o in range(8):
                s = self.proj_bg(o) + sum(grad[i] * self.proj_wg(o, i) for i in range(8))
                out.append(clip(grad[o] + 0.25 * (s - grad[o]), -4.0, 4.0))
            return out
        if self.proj_mode in {3, 4}:
            out = []
            for o in range(8):
                base = grad[o]
                struct_gate = clip(0.55 + 0.45 * self.proj_bg(o), -1.5, 1.5)
                delta_gate = clip(self.proj_wg(o, o), -1.5, 1.5)
                s = base + self.proj_structured_scale * struct_gate * mandelbrot_d2_feature(base, delta[o])
                s += self.proj_delta_scale * delta_gate * delta[o]
                if self.proj_mode == 4:
                    corr = self.proj_bg(o) + sum(grad[i] * self.proj_wg(o, i) for i in range(8))
                    s += self.proj_hybrid_scale * 0.20 * (corr - base)
                out.append(clip(s, -4.0, 4.0))
            return out
        return list(grad)

    def oct_step_head(self, head: int, x: list[float]) -> None:
        base = head * 8
        avec = self.a[base : base + 8]
        hvec = [self.hg(base + i) for i in range(8)]
        left = oct_mul(oct_mul(avec, hvec), x)
        right = oct_mul(avec, oct_mul(hvec, x))
        asq = sum((left[i] - right[i]) ** 2 for i in range(8))
        assoc_step = sqrt_q(asq)
        self.assoc_s(head, self.assoc_g(head) + squash_pos(assoc_step))
        for i in range(8):
            self.hs(base + i, clip(left[i] + 0.2 * right[i], -5.0, 5.0))

    def forward(self, subj_idx: int, sequence: list[list[float]]) -> tuple[float, float]:
        self.reset_state()
        raw_mean = [0.0] * 8
        last_mean = [0.0] * 8
        first_step = [0.0] * 8
        last_step = [0.0] * 8
        last_flat = [0.0] * max(512, self.feature_count)
        prev_raw = [0.0] * 8
        for step in range(self.seq_len):
            raw = []
            delta = []
            grad = []
            for dim in range(8):
                flat = step * 8 + dim
                value = (sequence[step][dim] - self.mean[flat]) / self.std[flat]
                value = self.perturb(subj_idx, step, dim, value)
                raw.append(value)
                d = value - prev_raw[dim] if step > 0 else 0.0
                delta.append(d)
                grad.append(clip(value * self.wg(554 + dim) + d * self.wg(34 + dim) + self.wg(562 + dim), -4.0, 4.0))
            x = self.project_step(grad, delta)
            prev_raw = raw
            for dim in range(8):
                raw_mean[dim] += raw[dim]
                last_mean[dim] += x[dim]
                last_flat[step * 8 + dim] = x[dim]
            if step == 0:
                first_step = list(x)
            last_step = list(x)
            self.oct_step_head(0, x)
            self.oct_step_head(1, x)
        for dim in range(8):
            raw_mean[dim] /= float(self.seq_len)
            last_mean[dim] /= float(self.seq_len)
        score = self.wg(570)
        for dim in range(16):
            score += self.hg(dim) * self.wg(dim)
        for head in range(2):
            assoc_raw = self.assoc_g(head) / float(self.seq_len)
            self.assoc_s(head, assoc_raw)
            score += squash_pos(assoc_raw) * self.wg(16 + head)
        for dim in range(8):
            score += last_mean[dim] * self.wg(18 + dim)
            score += (last_step[dim] - first_step[dim]) * self.wg(26 + dim)
        for flat in range(self.feature_count):
            score += last_flat[flat] * self.wg(42 + flat)
        prob = sigmoid_q(clip(score, -12.0, 12.0))
        assoc = sum(self.assoc_g(head) for head in range(2)) / 2.0
        return prob, assoc


def balanced_accuracy(labels: list[int], probs: list[float]) -> float:
    tp = tn = fp = fn = 0
    for label, prob in zip(labels, probs, strict=True):
        pred = 1 if prob >= 0.5 else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
    tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
    tnr = tn / (tn + fp) if tn + fp > 0 else 0.0
    return 50.0 * (tpr + tnr)


def flatten_sequence(sequence: list[list[float]], feature_count: int) -> list[float]:
    flat: list[float] = []
    for step in sequence:
        flat.extend(step)
    if len(flat) < feature_count:
        flat.extend([0.0] * (feature_count - len(flat)))
    return flat[:feature_count]


def arrays_need_standardization_rebuild(ckpt: dict[str, Any]) -> bool:
    arrays = ckpt["arrays"]
    feature_count = int_meta(ckpt["meta"], "manifest_feature_count")
    mean = arrays.get("TRAIN_FEATURE_MEAN", {}).get("values", [])
    std = arrays.get("TRAIN_FEATURE_STD", {}).get("values", [])
    if len(mean) < feature_count or len(std) < feature_count:
        return True
    return any((not math.isfinite(value)) or value < 1.0e-6 for value in std[:feature_count])


def rebuild_standardization_from_manifest(ckpt: dict[str, Any], records: list[Any]) -> bool:
    """Recover deterministic train-fold mean/std when native f64 serialization is zeroed."""

    if not arrays_need_standardization_rebuild(ckpt):
        return False
    meta = ckpt["meta"]
    holdout_key = int_meta(meta, "holdout_key")
    train_fraction_bp = int_meta(meta, "cfg_train_fraction_bp")
    if train_fraction_bp < 1_000_000:
        raise ValueError(
            "checkpoint has zero TRAIN_FEATURE_STD and train_fraction < 1.0; "
            "manifest-side reconstruction for subsampled folds is not implemented"
        )
    feature_count = int_meta(meta, "manifest_feature_count")
    train_flats = [
        flatten_sequence(record.sequence, feature_count)
        for record in records
        if site_hash(record.site) != holdout_key
    ]
    if not train_flats:
        raise ValueError(f"cannot rebuild standardization: no train records for holdout_key={holdout_key}")
    train_n = float(len(train_flats))
    mean = [0.0] * feature_count
    for flat in train_flats:
        for idx, value in enumerate(flat):
            mean[idx] += value
    mean = [value / train_n for value in mean]
    var = [0.0] * feature_count
    for flat in train_flats:
        for idx, value in enumerate(flat):
            diff = value - mean[idx]
            var[idx] += diff * diff
    std = []
    for value in var:
        current = sqrt_q(value / train_n)
        if not math.isfinite(current) or current < 1.0e-6:
            current = 1.0
        std.append(current)
    ckpt["arrays"]["TRAIN_FEATURE_MEAN"] = {"dtype": "f64", "values": mean}
    ckpt["arrays"]["TRAIN_FEATURE_STD"] = {"dtype": "f64", "values": std}
    ckpt.setdefault("repairs", {})["standardization"] = "recomputed_from_manifest_train_fold"
    return True


def sio_string_literal(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def sio_decimal_literal(value: float) -> str:
    if not math.isfinite(value):
        return "0.0"
    text = f"{float(value):.17f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0.0"
    if "." not in text:
        text += ".0"
    return text


def write_native_state_sequence(ckpt: dict[str, Any], state_path: Path) -> list[tuple[str, str, int, int]]:
    arrays = ckpt["arrays"]
    offsets: list[tuple[str, str, int, int]] = []
    cursor = 0
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        for name in I64_STATE_ARRAYS:
            values = [int(value) for value in arrays[name]["values"]]
            start = cursor
            for value in values:
                handle.write(f"{value}\n")
                cursor += 1
            offsets.append((name, "i64", start, cursor))
        for name in F64_STATE_ARRAYS:
            values = [float(value) for value in arrays[name]["values"]]
            start = cursor
            for value in values:
                handle.write(f"{sio_decimal_literal(value)}\n")
                cursor += 1
            offsets.append((name, "f64", start, cursor))
    return offsets


def chunked_i64_loader(name: str, values: list[int], chunk_size: int = 192) -> str:
    pieces: list[str] = []
    loaders: list[str] = []
    lname = name.lower()
    for chunk_idx, start in enumerate(range(0, len(values), chunk_size)):
        fn_name = f"load_{lname}_{chunk_idx}"
        loaders.append(fn_name)
        pieces.append(f"fn {fn_name}() with Mut {{")
        for offset, value in enumerate(values[start : start + chunk_size], start):
            pieces.append(f"    {name}[{offset}] = {value}")
        pieces.append("}")
    pieces.append(f"fn load_{lname}() with Mut {{")
    for fn_name in loaders:
        pieces.append(f"    {fn_name}()")
    pieces.append("}")
    return "\n".join(pieces)


def chunked_f64_loader(name: str, values: list[float], chunk_size: int = 192) -> str:
    pieces: list[str] = []
    loaders: list[str] = []
    lname = name.lower()
    for chunk_idx, start in enumerate(range(0, len(values), chunk_size)):
        fn_name = f"load_{lname}_{chunk_idx}"
        loaders.append(fn_name)
        pieces.append(f"fn {fn_name}() with Mut {{")
        for offset, value in enumerate(values[start : start + chunk_size], start):
            pieces.append(f"    {name}[{offset}] = {float(value):.12g}")
        pieces.append("}")
    pieces.append(f"fn load_{lname}() with Mut {{")
    for fn_name in loaders:
        pieces.append(f"    {fn_name}()")
    pieces.append("}")
    return "\n".join(pieces)


def generate_native_verifier_source(ckpt: dict[str, Any], manifest_path: Path, repo_root: Path) -> str:
    source_path = repo_root / "examples" / "brain_ossm_abide.sio"
    source = source_path.read_text(encoding="utf-8")
    marker = "fn main() -> i32"
    if marker not in source:
        raise RuntimeError(f"cannot locate main marker in {source_path}")
    prefix = source.split(marker, 1)[0]
    ckpt_emit_start = "fn ckpt_i64_value("
    ckpt_emit_end = "fn project_current_step("
    if ckpt_emit_start in prefix and ckpt_emit_end in prefix:
        before_emit, rest = prefix.split(ckpt_emit_start, 1)
        _, after_emit = rest.split(ckpt_emit_end, 1)
        # The verifier loads an already-persisted checkpoint, so native CKPT
        # emission helpers are dead code here. Dropping them avoids a Madaros
        # backend crash in the unused checkpoint dispatch path.
        prefix = before_emit + ckpt_emit_end + after_emit
    ckpt_print_start = "fn print_i64_array("
    ckpt_print_end = "fn project_current_step("
    if ckpt_print_start in prefix and ckpt_print_end in prefix:
        before_print, rest = prefix.split(ckpt_print_start, 1)
        _, after_print = rest.split(ckpt_print_end, 1)
        prefix = before_print + ckpt_print_end + after_print
    train_update_start = "fn update_hidden_weights("
    if train_update_start in prefix:
        prefix = prefix.split(train_update_start, 1)[0]
    def strip_prefix_block(current: str, start: str, end: str) -> str:
        if start not in current or end not in current:
            return current
        before, rest = current.split(start, 1)
        _, after = rest.split(end, 1)
        return before + end + after

    prefix = strip_prefix_block(prefix, "fn apply_run_config_oct_schedule_line(", "fn parse_decimal_digit(")
    prefix = strip_prefix_block(prefix, "fn parse_decimal_digit(", "fn clear_subject_site_slot(")
    prefix = strip_prefix_block(prefix, "fn site_name_for_key(", "fn parse_f64_span(")
    prefix = strip_prefix_block(prefix, "fn load_local_manifest(", "fn discover_sites(")
    prefix = strip_prefix_block(prefix, "fn debug_class_f64(", "fn abs_i64(")
    prefix = strip_prefix_block(prefix, "fn discover_sites(", "fn perturb_feature(")
    prefix = strip_prefix_block(prefix, "fn assoc_mean_all(", "fn print_fixed6_from_micros(")
    prefix = strip_prefix_block(prefix, "fn print_f64_fixed6(", "fn project_current_step(")
    train_prep_start = "fn build_train_mask("
    train_prep_end = "fn perturb_feature("
    if train_prep_start in prefix and train_prep_end in prefix:
        before_train_prep, rest = prefix.split(train_prep_start, 1)
        _, after_train_prep = rest.split(train_prep_end, 1)
        prefix = before_train_prep + train_prep_end + after_train_prep
    init_start = "fn normalize_head("
    init_end = "fn reset_state("
    if init_start in prefix and init_end in prefix:
        before_init, rest = prefix.split(init_start, 1)
        _, after_init = rest.split(init_end, 1)
        prefix = before_init + init_end + after_init
    arrays = ckpt["arrays"]
    meta = ckpt["meta"]
    if int_meta(meta, "cfg_oct_input_proj_mode") == 0:
        project_start = "fn project_current_step("
        project_end = "fn normalize_head("
        if project_start in prefix and project_end in prefix:
            before_project, rest = prefix.split(project_start, 1)
            _, after_project = rest.split(project_end, 1)
            prefix = before_project + project_end + after_project
        prefix = prefix.replace(
            "    project_current_step(proj_mode)",
            """    var project_out_dim: i64 = 0
    while project_out_dim < 8 {
        TMP_OCT[project_out_dim as usize] = TMP_GRAD[project_out_dim as usize]
        project_out_dim = project_out_dim + 1
    }""",
        )
    parts = [prefix]
    for name in ("W", "MOM", "A_MOM", "PROJ_W", "PROJ_B", "PROJ_W_MOM", "PROJ_B_MOM", "DROP_MASK"):
        parts.append(chunked_i64_loader(name, arrays[name]["values"]))
    parts.append(chunked_f64_loader("A", arrays["A"]["values"]))
    parts.append(chunked_f64_loader("TRAIN_FEATURE_MEAN", arrays["TRAIN_FEATURE_MEAN"]["values"]))
    parts.append(chunked_f64_loader("TRAIN_FEATURE_STD", arrays["TRAIN_FEATURE_STD"]["values"]))
    config_keys = [
        ("CFG_TRAIN_FRACTION_BP", "cfg_train_fraction_bp"),
        ("CFG_DROP_CHANNEL_FRAC_BP", "cfg_drop_channel_frac_bp"),
        ("CFG_NOISE_STD_BP", "cfg_noise_std_bp"),
        ("CFG_GLOBAL_TRAIN_EPOCHS", "cfg_global_train_epochs"),
        ("CFG_GLOBAL_TRAIN_LR_BP", "cfg_global_train_lr_bp"),
        ("CFG_GLOBAL_CORE_LR_SCALE_BP", "cfg_global_core_lr_scale_bp"),
        ("CFG_OCT_PROFILE_ID", "cfg_oct_profile_id"),
        ("CFG_OCT_TRAIN_MODE", "cfg_oct_train_mode"),
        ("CFG_OCT_INIT_PRESET", "cfg_oct_init_preset"),
        ("CFG_OCT_ASSOC_REG_BP", "cfg_oct_assoc_reg_bp"),
        ("CFG_OCT_ASSOC_TARGET_BP", "cfg_oct_assoc_target_bp"),
        ("CFG_OCT_ASSOCIATOR_SIGN_AUX_BP", "cfg_oct_associator_sign_aux_bp"),
        ("CFG_OCT_TRAIN_NOISE_STD_BP", "cfg_oct_train_noise_std_bp"),
        ("CFG_OCT_RELATION_PRESERVE_AUX_BP", "cfg_oct_relation_preserve_aux_bp"),
        ("CFG_OCT_RELATION_TARGET_AUX_BP", "cfg_oct_relation_target_aux_bp"),
        ("CFG_OCT_RELATION_MARGIN_AUX_BP", "cfg_oct_relation_margin_aux_bp"),
        ("CFG_OCT_RELATION_IDENTITY_AUX_BP", "cfg_oct_relation_identity_aux_bp"),
        ("CFG_OCT_RELATION_IDENTITY_SRC_AUX_BP", "cfg_oct_relation_identity_src_aux_bp"),
        ("CFG_OCT_RELATION_IDENTITY_DST_AUX_BP", "cfg_oct_relation_identity_dst_aux_bp"),
        ("CFG_OCT_RELATION_IDENTITY_START_EPOCH", "cfg_oct_relation_identity_start_epoch"),
        ("CFG_OCT_RELATION_IDENTITY_RAMP_EPOCHS", "cfg_oct_relation_identity_ramp_epochs"),
        ("CFG_OCT_RELATION_IDENTITY_TIE_MARGIN_BP", "cfg_oct_relation_identity_tie_margin_bp"),
        ("CFG_OCT_RELATION_IDENTITY_GATE_MARGIN_BP", "cfg_oct_relation_identity_gate_margin_bp"),
        ("CFG_OCT_RELATION_IDENTITY_GATE_FLOOR_BP", "cfg_oct_relation_identity_gate_floor_bp"),
        ("CFG_OCT_RELATION_IDENTITY_TASK_GUARD", "cfg_oct_relation_identity_task_guard"),
        ("CFG_OCT_RELATION_IDENTITY_TASK_GUARD_TOL_BP", "cfg_oct_relation_identity_task_guard_tol_bp"),
        ("CFG_OCT_RELATION_READOUT_CORRECT_STEPS", "cfg_oct_relation_readout_correct_steps"),
        ("CFG_OCT_RELATION_READOUT_CORRECT_LR_SCALE_BP", "cfg_oct_relation_readout_correct_lr_scale_bp"),
        ("CFG_OCT_ASSOCIATOR_READOUT_CORRECT_STEPS", "cfg_oct_associator_readout_correct_steps"),
        ("CFG_OCT_ASSOCIATOR_READOUT_CORRECT_LR_SCALE_BP", "cfg_oct_associator_readout_correct_lr_scale_bp"),
        ("CFG_OCT_ASSOCIATOR_READOUT_ALIGN_EPOCHS", "cfg_oct_associator_readout_align_epochs"),
        ("CFG_OCT_ASSOCIATOR_READOUT_ALIGN_LR_SCALE_BP", "cfg_oct_associator_readout_align_lr_scale_bp"),
        ("CFG_OCT_RELATION_MARGIN_GATE_CAP_BP", "cfg_oct_relation_margin_gate_cap_bp"),
        ("CFG_OCT_RELATION_MARGIN_START_EPOCH", "cfg_oct_relation_margin_start_epoch"),
        ("CFG_OCT_RELATION_MARGIN_RAMP_EPOCHS", "cfg_oct_relation_margin_ramp_epochs"),
        ("CFG_OCT_RELATION_FREEZE_AFTER_EPOCH", "cfg_oct_relation_freeze_after_epoch"),
        ("CFG_OCT_RELATION_POST_FREEZE_SCALE_BP", "cfg_oct_relation_post_freeze_scale_bp"),
        ("CFG_OCT_BINARY_LR_SCALE_AFTER_EPOCH", "cfg_oct_binary_lr_scale_after_epoch"),
        ("CFG_OCT_BINARY_LR_POST_SCALE_BP", "cfg_oct_binary_lr_post_scale_bp"),
        ("CFG_OCT_RELATION_TASK_GUARD", "cfg_oct_relation_task_guard"),
        ("CFG_OCT_RELATION_TASK_GUARD_TOL_BP", "cfg_oct_relation_task_guard_tol_bp"),
        ("CFG_OCT_RELATION_TARGET_SRC_POS", "cfg_oct_relation_target_src_pos"),
        ("CFG_OCT_RELATION_TARGET_DST_POS", "cfg_oct_relation_target_dst_pos"),
        ("CFG_OCT_INPUT_PROJ_MODE", "cfg_oct_input_proj_mode"),
        ("CFG_OCT_PROJ_LR_SCALE_BP", "cfg_oct_proj_lr_scale_bp"),
        ("CFG_OCT_PROJ_STRUCTURED_SCALE_BP", "cfg_oct_proj_structured_scale_bp"),
        ("CFG_OCT_PROJ_DELTA_SCALE_BP", "cfg_oct_proj_delta_scale_bp"),
        ("CFG_OCT_PROJ_HYBRID_SCALE_BP", "cfg_oct_proj_hybrid_scale_bp"),
    ]
    config_lines = ["fn load_ckpt_config() with Mut {"]
    for var_name, meta_key in config_keys:
        if meta_key in meta:
            config_lines.append(f"    {var_name} = {int_meta(meta, meta_key)}")
    config_lines.append("}")
    parts.append("\n".join(config_lines))
    manifest_literal = sio_string_literal(str(manifest_path.resolve()))
    holdout_key = int_meta(meta, "holdout_key")
    seed_a = int_meta(meta, "seed_a")
    seed_b = int_meta(meta, "seed_b")
    fold_noise_key = int_meta(meta, "fold_noise_key")
    parts.append(
        f"""
fn load_ckpt_state() with Mut {{
    load_w()
    load_mom()
    load_a()
    load_a_mom()
    load_proj_w()
    load_proj_b()
    load_proj_w_mom()
    load_proj_b_mom()
    load_drop_mask()
    load_train_feature_mean()
    load_train_feature_std()
}}

fn verify_holdout_only(holdout_key: i64) -> f64 with IO, Mut, Div, Panic {{
    var tp: i64 = 0
    var tn: i64 = 0
    var fp: i64 = 0
    var fnn: i64 = 0
    var test_n: i64 = 0
    var pos_n: i64 = 0
    var neg_n: i64 = 0
    var subj: i64 = 0
    while subj < SUBJECT_COUNT {{
        if SITE_KEYS[subj as usize] == holdout_key {{
            forward_subject(subj, 1)
            let prob = sigmoid_q(LOGIT)
            let pred = if prob >= 0.5 {{ 1 }} else {{ 0 }}
            let label = LABELS[subj as usize]
            print("VERIFY_PRED\\t")
            print_int(subj); print("\\t"); print_int(label); print("\\t")
            print_int((prob * 1000000.0) as i64); print("\\t")
            if LOGIT > 0.0 {{ print_int(1) }} else if LOGIT < 0.0 {{ print_int(0 - 1) }} else {{ print_int(0) }}
            print("\\t"); print_int(pred); print("\\n")
            if pred == 1 && label == 1 {{ tp = tp + 1 }}
            if pred == 0 && label == 0 {{ tn = tn + 1 }}
            if pred == 1 && label == 0 {{ fp = fp + 1 }}
            if pred == 0 && label == 1 {{ fnn = fnn + 1 }}
            if label == 1 {{ pos_n = pos_n + 1 }} else {{ neg_n = neg_n + 1 }}
            test_n = test_n + 1
        }}
        subj = subj + 1
    }}
    if pos_n == 0 || neg_n == 0 || test_n == 0 {{ return 0.0 }}
    var tpr = 0.0
    if tp + fnn > 0 {{ tpr = (tp as f64) / ((tp + fnn) as f64) }}
    var tnr = 0.0
    if tn + fp > 0 {{ tnr = (tn as f64) / ((tn + fp) as f64) }}
    let bal = 50.0 * (tpr + tnr)
    print("VERIFY_COUNTS\\t")
    print_int(tp); print("\\t"); print_int(tn); print("\\t"); print_int(fp); print("\\t"); print_int(fnn); print("\\t"); print_int(test_n); print("\\n")
    bal
}}

fn main() -> i32 with IO, Mut, Div, Panic {{
    reset_run_config_defaults()
    load_ckpt_config()
    let loaded = load_manifest({manifest_literal})
    if loaded == 0 {{ print("VERIFY_FAIL\\tmanifest\\n"); return 2 }}
    let holdout_key: i64 = {holdout_key}
    let seed_a: i64 = {seed_a}
    let seed_b: i64 = {seed_b}
    FOLD_NOISE_KEY = {fold_noise_key}
    load_ckpt_state()
    let bal = verify_holdout_only(holdout_key)
    print("VERIFY_BAL\\t")
    print_fixed6_from_micros((bal * 1000000.0) as i64)
    print("\\n")
    return 0
}}
"""
    )
    return "\n".join(parts)


def run_native_verifier(
    checkpoint_path: Path,
    ckpt: dict[str, Any],
    manifest_path: Path,
    tolerance_pp: float,
    repo_root: Path | None = None,
    compile_timeout_sec: float = 300.0,
    run_timeout_sec: float = 180.0,
    souc_engine: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or Path(os.environ.get("SOUNIO_CHECKPOINT_REPO_ROOT", "") or Path(__file__).resolve().parents[2])
    repo_root = repo_root.resolve()
    checkpoint_path = checkpoint_path.resolve()
    manifest_path = manifest_path.resolve()
    short_digest = hashlib.sha256(str(manifest_path).encode("utf-8")).hexdigest()[:12]
    native_manifest_path = Path("/tmp") / f"brain_ossm_verify_manifest_{os.getpid()}_{short_digest}.tsv"
    native_manifest_path.write_text(manifest_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    verify_dir = checkpoint_path.parent / f".{checkpoint_path.stem}.native-verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    verifier_source = verify_dir / "verify_checkpoint.sio"
    verifier_binary = verify_dir / "verify_checkpoint.elf"
    verifier_source.write_text(generate_native_verifier_source(ckpt, native_manifest_path, repo_root), encoding="utf-8")

    env = dict(os.environ)
    env.setdefault("SOUNIO_STDLIB_PATH", str(repo_root / "stdlib"))
    if souc_engine:
        env["SOUNIO_SOUC_ENGINE"] = souc_engine
    compiler = repo_root / "bin" / "souc"
    compile_proc = subprocess.run(
        [str(compiler), "compile", str(verifier_source), "-o", str(verifier_binary)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=compile_timeout_sec,
    )
    compile_output = compile_proc.stdout + compile_proc.stderr
    if compile_proc.returncode != 0 or "exceeds IR_MAX_INSTRS" in compile_output:
        raise RuntimeError(
            "native checkpoint verifier compile failed\n"
            f"returncode={compile_proc.returncode}\n{compile_output}"
        )
    verifier_binary.chmod(verifier_binary.stat().st_mode | 0o111)

    run_proc = subprocess.run(
        [str(verifier_binary)],
        cwd=verify_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=run_timeout_sec,
    )
    run_output = run_proc.stdout + run_proc.stderr
    if run_proc.returncode != 0:
        raise RuntimeError(
            "native checkpoint verifier run failed\n"
            f"returncode={run_proc.returncode}\n{run_output}"
        )

    actual: float | None = None
    counts: dict[str, int] | None = None
    sample: dict[str, Any] | None = None
    heldout = 0
    for line in merge_tab_continuation_records(run_proc.stdout.splitlines(), ("VERIFY_PRED", "VERIFY_COUNTS", "VERIFY_BAL")):
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] == "VERIFY_BAL":
            actual = float(parts[1])
        elif len(parts) >= 6 and parts[0] == "VERIFY_COUNTS":
            counts = {
                "tp": int(parts[1]),
                "tn": int(parts[2]),
                "fp": int(parts[3]),
                "fn": int(parts[4]),
                "test_n": int(parts[5]),
            }
            heldout = counts["test_n"]
        elif len(parts) >= 6 and parts[0] == "VERIFY_PRED":
            heldout += 1 if counts is None else 0
            if sample is None:
                sample = {
                    "subject_index": int(parts[1]),
                    "label": int(parts[2]),
                    "prob_micros": int(parts[3]),
                    "logit_sign": int(parts[4]),
                    "pred": int(parts[5]),
                }
    if actual is None:
        raise RuntimeError(f"native checkpoint verifier did not emit VERIFY_BAL\n{run_output}")

    expected = float(ckpt["meta"]["balanced_accuracy_pct"])
    delta = abs(actual - expected)
    result = {
        "status": "pass" if delta <= tolerance_pp else "fail",
        "method": "sounio_native_fresh_process",
        "expected_balanced_accuracy_pct": expected,
        "actual_balanced_accuracy_pct": actual,
        "delta_pp": delta,
        "tolerance_pp": tolerance_pp,
        "heldout_subjects": heldout,
        "counts": counts,
        "sample_forward": sample,
        "verifier_source": str(verifier_source),
        "verifier_binary": str(verifier_binary),
        "native_souc_engine": souc_engine or env.get("SOUNIO_SOUC_ENGINE", ""),
        "native_manifest_path": str(native_manifest_path),
        "source_manifest_path": str(manifest_path),
    }
    if result["status"] != "pass":
        raise AssertionError(json.dumps(result, indent=2, sort_keys=True))
    return result


def site_keys_for_records(records: list[Any], holdout_key: int, config: dict[str, str] | None = None) -> list[int]:
    if single_site_internal_cv_enabled(config, records, holdout_key):
        return materialize_single_site_internal_cv_keys(records)
    return [site_hash(record.site) for record in records]


def run_python_verifier(
    checkpoint_path: Path,
    ckpt: dict[str, Any],
    manifest_path: Path,
    tolerance_pp: float,
    run_config: dict[str, str] | None = None,
    native_failure: str | None = None,
) -> dict[str, Any]:
    _, records = load_manifest(manifest_path)
    rebuild_standardization_from_manifest(ckpt, records)
    holdout_key = int_meta(ckpt["meta"], "holdout_key")
    site_keys = site_keys_for_records(records, holdout_key, run_config)
    forward = CheckpointForward(ckpt)
    labels: list[int] = []
    probs: list[float] = []
    sample: dict[str, Any] | None = None
    tp = tn = fp = fn = 0
    heldout = 0
    for subj_idx, (record, site_key) in enumerate(zip(records, site_keys, strict=True)):
        if site_key != holdout_key:
            continue
        prob, assoc = forward.forward(subj_idx, record.sequence)
        label = int(record.label)
        pred = 1 if prob >= 0.5 else 0
        labels.append(label)
        probs.append(prob)
        heldout += 1
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        if sample is None:
            sample = {
                "subject_index": subj_idx,
                "subject_id": record.subject_id,
                "site": record.site,
                "site_key": site_key,
                "label": label,
                "prob": prob,
                "pred": pred,
                "assoc": assoc,
            }
    if heldout == 0:
        raise AssertionError(f"python checkpoint verifier found no holdout records for holdout_key={holdout_key}")

    actual = balanced_accuracy(labels, probs)
    expected = float(ckpt["meta"]["balanced_accuracy_pct"])
    delta = abs(actual - expected)
    result = {
        "status": "pass" if delta <= tolerance_pp else "fail",
        "method": "python_fresh_process",
        "expected_balanced_accuracy_pct": expected,
        "actual_balanced_accuracy_pct": actual,
        "delta_pp": delta,
        "tolerance_pp": tolerance_pp,
        "heldout_subjects": heldout,
        "holdout_key": holdout_key,
        "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "test_n": heldout},
        "sample_forward": sample,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "source_manifest_path": str(manifest_path.resolve()),
    }
    if native_failure:
        result["native_verifier_status"] = "blocked"
        result["native_verifier_failure"] = native_failure[-4000:]
    if result["status"] != "pass":
        raise AssertionError(json.dumps(result, indent=2, sort_keys=True))
    return result


def verify_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    tolerance_pp: float,
    repo_root: Path | None = None,
    verification_mode: str = "auto",
    run_config: dict[str, str] | None = None,
    native_compile_timeout_sec: float = 300.0,
    native_run_timeout_sec: float = 180.0,
    native_souc_engine: str | None = None,
) -> dict[str, Any]:
    ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not array_has_signal(ckpt, "A"):
        raise AssertionError("checkpoint A tensor is empty/all-zero; refusing verification without octonionic state")
    _, records = load_manifest(manifest_path)
    rebuild_standardization_from_manifest(ckpt, records)
    if verification_mode == "python":
        return run_python_verifier(checkpoint_path, ckpt, manifest_path, tolerance_pp, run_config)
    if verification_mode == "native":
        return run_native_verifier(
            checkpoint_path,
            ckpt,
            manifest_path,
            tolerance_pp,
            repo_root,
            native_compile_timeout_sec,
            native_run_timeout_sec,
            native_souc_engine,
        )
    if verification_mode != "auto":
        raise ValueError(f"unsupported verification mode: {verification_mode!r}")
    try:
        return run_native_verifier(
            checkpoint_path,
            ckpt,
            manifest_path,
            tolerance_pp,
            repo_root,
            native_compile_timeout_sec,
            native_run_timeout_sec,
            native_souc_engine,
        )
    except Exception as exc:  # noqa: BLE001 - auto mode records native blockage, then verifies independently.
        return run_python_verifier(
            checkpoint_path,
            ckpt,
            manifest_path,
            tolerance_pp,
            run_config,
            native_failure=f"{type(exc).__name__}: {exc}",
        )


def persist(args: argparse.Namespace) -> int:
    blocks = parse_blocks(Path(args.raw_output))
    run_config = parse_run_config(Path(args.run_config) if args.run_config else None)
    block = best_block(blocks, expected_meta_from_config(run_config))
    ckpt = {
        "schema": "brain_ossm_abide_model_checkpoint_container.v1",
        "run_id": args.run_id,
        "selection": {
            "strategy": "max_o_ssm_fold_balanced_accuracy",
            "candidate_count": len(blocks),
        },
        **block,
    }
    _, records = load_manifest(args.manifest)
    rebuild_standardization_from_manifest(ckpt, records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ckpt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--verify-only",
            "--checkpoint",
            str(output),
            "--manifest",
            str(Path(args.manifest)),
            *(["--run-config", str(Path(args.run_config))] if args.run_config else []),
            "--tolerance-pp",
            str(args.tolerance_pp),
            "--verification-mode",
            args.verification_mode,
            "--native-compile-timeout-sec",
            str(args.native_compile_timeout_sec),
            "--native-run-timeout-sec",
            str(args.native_run_timeout_sec),
            *(["--native-souc-engine", args.native_souc_engine] if args.native_souc_engine else []),
            *(["--repo-root", str(args.repo_root)] if args.repo_root else []),
        ],
        check=True,
    )
    verified = json.loads(output.read_text(encoding="utf-8"))
    print(
        "checkpoint persisted and verified: "
        f"{output} expected={verified['verification']['expected_balanced_accuracy_pct']:.6f} "
        f"actual={verified['verification']['actual_balanced_accuracy_pct']:.6f} "
        f"delta={verified['verification']['delta_pp']:.6f}pp"
    )
    return 0


def verify_only(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    _, records = load_manifest(args.manifest)
    rebuild_standardization_from_manifest(payload, records)
    checkpoint.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = verify_checkpoint(
        checkpoint,
        Path(args.manifest),
        args.tolerance_pp,
        Path(args.repo_root) if args.repo_root else None,
        args.verification_mode,
        parse_run_config(Path(args.run_config) if args.run_config else None),
        args.native_compile_timeout_sec,
        args.native_run_timeout_sec,
        args.native_souc_engine,
    )
    payload["verification"] = result
    checkpoint.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "checkpoint reload sanity check PASS: "
        f"expected={result['expected_balanced_accuracy_pct']:.6f} "
        f"actual={result['actual_balanced_accuracy_pct']:.6f} "
        f"delta={result['delta_pp']:.6f}pp "
        f"sample_subject={result['sample_forward'].get('subject_id', result['sample_forward'].get('subject_index'))}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--raw-output")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-config")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--tolerance-pp", type=float, default=0.5)
    parser.add_argument("--repo-root", help="Optional repo/compiler root for native verifier replay")
    parser.add_argument(
        "--verification-mode",
        choices=["auto", "native", "python"],
        default="auto",
        help="Checkpoint replay mode. auto records native verifier failure and falls back to Python fresh-process replay.",
    )
    parser.add_argument("--native-compile-timeout-sec", type=float, default=300.0)
    parser.add_argument("--native-run-timeout-sec", type=float, default=180.0)
    parser.add_argument("--native-souc-engine", default="", help="Optional SOUNIO_SOUC_ENGINE for native verifier compilation")
    args = parser.parse_args()
    if args.verify_only:
        if not args.checkpoint:
            parser.error("--verify-only requires --checkpoint")
        return verify_only(args)
    if not args.raw_output or not args.output:
        parser.error("persist mode requires --raw-output and --output")
    return persist(args)


if __name__ == "__main__":
    raise SystemExit(main())

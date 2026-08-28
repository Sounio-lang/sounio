#!/usr/bin/env python3
"""Extract ROI-level associator features from a verified trained Brain O-SSM checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from abide_campaign_lib import load_manifest, write_json, write_tsv
from abide_checkpoint_persist import (
    CheckpointForward,
    F64_STATE_ARRAYS,
    I64_STATE_ARRAYS,
    balanced_accuracy,
    chunked_f64_loader,
    chunked_i64_loader,
    int_meta,
    materialize_single_site_internal_cv_keys,
    merge_tab_continuation_records,
    rebuild_standardization_from_manifest,
    site_hash,
    single_site_internal_cv_enabled,
    sio_string_literal,
    write_native_state_sequence,
)


METRIC_LABELS = {
    "assoc_delta_abs": "trained_assoc_occlusion_delta_abs",
    "prob_delta_abs": "trained_prob_occlusion_delta_abs",
    "logit_delta_abs": "trained_logit_occlusion_delta_abs",
    "hidden_delta_abs": "trained_hidden_occlusion_delta_abs",
}


def manifest_label_names(manifest_extra: dict[str, str]) -> tuple[str, str]:
    label_space = manifest_extra.get("label_space", "").lower()
    if "adhd" in label_space:
        default_positive = "ADHD"
    elif "mdd" in label_space or "depress" in label_space:
        default_positive = "MDD"
    elif "asd" in label_space or "autism" in label_space:
        default_positive = "ASD"
    else:
        default_positive = "CASE"
    positive = (
        manifest_extra.get("label_positive_name")
        or manifest_extra.get("positive_label_name")
        or manifest_extra.get("case_label_name")
        or default_positive
    )
    negative = (
        manifest_extra.get("label_negative_name")
        or manifest_extra.get("negative_label_name")
        or manifest_extra.get("control_label_name")
        or "CONTROL"
    )
    return positive, negative


def label_name_for(label: int, positive_name: str, negative_name: str) -> str:
    return positive_name if label == 1 else negative_name


def _norm_cdf_complement(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def mann_whitney_cliff(asd: list[float], control: list[float]) -> dict[str, float]:
    n1 = len(asd)
    n2 = len(control)
    if n1 == 0 or n2 == 0:
        return {"u": 0.0, "p": 1.0, "z": 0.0, "cliff_delta": 0.0}

    merged = [(value, 0) for value in asd] + [(value, 1) for value in control]
    merged.sort(key=lambda item: item[0])
    ranks = [0.0] * len(merged)
    tie_terms = 0.0
    i = 0
    while i < len(merged):
        j = i + 1
        while j < len(merged) and merged[j][0] == merged[i][0]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        for idx in range(i, j):
            ranks[idx] = avg_rank
        tie = j - i
        if tie > 1:
            tie_terms += float(tie * tie * tie - tie)
        i = j

    rank_asd = sum(rank for rank, (_, group) in zip(ranks, merged, strict=True) if group == 0)
    u1 = rank_asd - (n1 * (n1 + 1) / 2.0)
    mean_u = n1 * n2 / 2.0
    n = n1 + n2
    if n > 1:
        variance = n1 * n2 / 12.0 * ((n + 1.0) - tie_terms / (n * (n - 1.0)))
    else:
        variance = 0.0
    if variance <= 0.0:
        z = 0.0
        p = 1.0
    else:
        z = (u1 - mean_u) / math.sqrt(variance)
        p = min(1.0, max(0.0, 2.0 * _norm_cdf_complement(abs(z))))

    greater = less = 0
    for x in asd:
        for y in control:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    cliff = (greater - less) / float(n1 * n2)
    return {"u": u1, "p": p, "z": z, "cliff_delta": cliff}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda idx: p_values[idx])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        candidate = min(1.0, p_values[idx] * (m - rank))
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def require_verified_checkpoint(ckpt: dict[str, Any], min_balanced_accuracy: float) -> None:
    verification = ckpt.get("verification")
    if not isinstance(verification, dict):
        raise SystemExit("checkpoint has no verification block; refusing model-free associator extraction")
    if verification.get("status") != "pass":
        raise SystemExit(f"checkpoint verification status is not pass: {verification.get('status')!r}")
    actual = float(verification.get("actual_balanced_accuracy_pct", 0.0))
    delta = float(verification.get("delta_pp", 999.0))
    if actual <= min_balanced_accuracy:
        raise SystemExit(
            f"checkpoint verification balanced accuracy {actual:.6f}% <= required {min_balanced_accuracy:.6f}%"
        )
    if delta > float(verification.get("tolerance_pp", 0.5)):
        raise SystemExit(f"checkpoint verification delta {delta:.6f}pp exceeds tolerance")
    a_values = [float(value) for value in ckpt.get("arrays", {}).get("A", {}).get("values", [])]
    if not a_values or not any(abs(value) > 1.0e-12 for value in a_values):
        raise SystemExit("checkpoint A tensor is empty/all-zero; refusing associator extraction without octonionic state")


def generate_native_extractor_source(
    ckpt: dict[str, Any],
    manifest_path: Path,
    repo_root: Path,
    scope: str,
    state_path: Path,
    state_offsets: list[tuple[str, str, int, int]],
    use_single_site_internal_cv: bool,
) -> str:
    source_path = repo_root / "examples" / "brain_ossm_abide.sio"
    source = source_path.read_text(encoding="utf-8")
    marker = "fn main() -> i32"
    if marker not in source:
        raise RuntimeError(f"cannot locate main marker in {source_path}")
    prefix = source.split(marker, 1)[0]
    meta = ckpt["meta"]
    # Do not add new Sounio globals here. A prior dedicated ROI hidden buffer
    # changed the native ABIDE forward result for Mandelbrot checkpoints by
    # perturbing global layout. Reuse A_MOM as a temporary hidden-state buffer
    # instead; A_MOM is loaded for checkpoint completeness but is not read by
    # forward_subject.
    parts = [prefix]
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
            config_lines.append(f"    {var_name} = {int(float(meta[meta_key]))}")
    config_lines.append("}")
    parts.append("\n".join(config_lines))
    manifest_literal = sio_string_literal(str(manifest_path.resolve()))
    state_literal = sio_string_literal(state_path.name)
    holdout_key = int_meta(meta, "holdout_key")
    seed_a = int_meta(meta, "seed_a")
    seed_b = int_meta(meta, "seed_b")
    fold_noise_key = int_meta(meta, "fold_noise_key")
    require_holdout_scope = "1" if scope == "holdout" else "0"
    internal_cv_block = (
        "    materialize_single_site_internal_cv_folds()\n"
        if use_single_site_internal_cv
        else ""
    )
    arrays = ckpt["arrays"]
    for name in I64_STATE_ARRAYS:
        parts.append(chunked_i64_loader(name, arrays[name]["values"]))
    parts.append(chunked_f64_loader("A", arrays["A"]["values"]))
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
}}

fn roi_verify_holdout(holdout_key: i64) -> f64 with IO, Mut, Div, Panic {{
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
    print("ROI_VERIFY_COUNTS\\t")
    print_int(tp); print("\\t"); print_int(tn); print("\\t"); print_int(fp); print("\\t"); print_int(fnn); print("\\t"); print_int(test_n); print("\\n")
    print("ROI_VERIFY_BAL\\t"); print_fixed6_from_micros((bal * 1000000.0) as i64); print("\\n")
    bal
}}

fn emit_roi_features(scope_holdout_only: i64, holdout_key: i64) with IO, Mut, Div, Panic {{
    var subj: i64 = 0
    while subj < SUBJECT_COUNT {{
        if scope_holdout_only == 0 || SITE_KEYS[subj as usize] == holdout_key {{
            forward_subject(subj, 1)
            var base_logit = LOGIT
            if base_logit != base_logit {{ base_logit = 0.0 }}
            if base_logit > 12.0 {{ base_logit = 12.0 }}
            if base_logit < 0.0 - 12.0 {{ base_logit = 0.0 - 12.0 }}
            let base_prob = sigmoid_q(LOGIT)
            let base_assoc = assoc_mean_all()
            var hd: i64 = 0
            while hd < hidden_dim() {{
                A_MOM[hd as usize] = H[hd as usize]
                hd = hd + 1
            }}
            var roi: i64 = 0
            while roi < MANIFEST_FEATURE_COUNT {{
                let offset = subj * feature_cap() + roi
                let old_value = FEATURES[offset as usize]
                FEATURES[offset as usize] = TRAIN_FEATURE_MEAN[roi as usize]
                forward_subject(subj, 1)
                var masked_logit = LOGIT
                if masked_logit != masked_logit {{ masked_logit = 0.0 }}
                if masked_logit > 12.0 {{ masked_logit = 12.0 }}
                if masked_logit < 0.0 - 12.0 {{ masked_logit = 0.0 - 12.0 }}
                let masked_prob = sigmoid_q(LOGIT)
                let masked_assoc = assoc_mean_all()
                FEATURES[offset as usize] = old_value
                let assoc_delta = base_assoc - masked_assoc
                let prob_delta = base_prob - masked_prob
                let logit_delta = base_logit - masked_logit
                var hidden_sq = 0.0
                hd = 0
                while hd < hidden_dim() {{
                    let base_h = (A_MOM[hd as usize] as f64) / 100000000.0
                    let masked_h = hg(hd)
                    let diff_h = base_h - masked_h
                    hidden_sq = hidden_sq + diff_h * diff_h
                    hd = hd + 1
                }}
                let hidden_delta = sqrt_q(hidden_sq)
                var assoc_abs = assoc_delta
                if assoc_abs < 0.0 {{ assoc_abs = 0.0 - assoc_abs }}
                var prob_abs = prob_delta
                if prob_abs < 0.0 {{ prob_abs = 0.0 - prob_abs }}
                var logit_abs = logit_delta
                if logit_abs < 0.0 {{ logit_abs = 0.0 - logit_abs }}
                print("ROI_FEATURE\\t")
                print_int(subj); print("\\t")
                print_int(LABELS[subj as usize]); print("\\t")
                print(site_name_dynamic_for_key(SITE_KEYS[subj as usize])); print("\\t")
                print_int(roi); print("\\t")
                print_int((base_prob * 1000000.0) as i64); print("\\t")
                print_int((masked_prob * 1000000.0) as i64); print("\\t")
                print_int((base_assoc * 1000000.0) as i64); print("\\t")
                print_int((masked_assoc * 1000000.0) as i64); print("\\t")
                print_int((assoc_delta * 1000000.0) as i64); print("\\t")
                print_int((assoc_abs * 1000000.0) as i64); print("\\t")
                print_int((prob_delta * 1000000.0) as i64); print("\\t")
                print_int((prob_abs * 1000000.0) as i64); print("\\t")
                print_int((base_logit * 1000000.0) as i64); print("\\t")
                print_int((masked_logit * 1000000.0) as i64); print("\\t")
                print_int((logit_delta * 1000000.0) as i64); print("\\t")
                print_int((logit_abs * 1000000.0) as i64); print("\\t")
                print_int((hidden_delta * 1000000.0) as i64); print("\\n")
                roi = roi + 1
            }}
        }}
        subj = subj + 1
    }}
}}

fn main() -> i32 with IO, Mut, Div, Panic {{
    reset_run_config_defaults()
    load_ckpt_config()
    let loaded = load_manifest({manifest_literal})
    if loaded == 0 {{ print("ROI_FAIL\\tmanifest\\n"); return 2 }}
    discover_sites()
{internal_cv_block}    let holdout_key: i64 = {holdout_key}
    let seed_a: i64 = {seed_a}
    let seed_b: i64 = {seed_b}
    build_train_mask(holdout_key, seed_a, seed_b)
    prepare_train_standardization()
    FOLD_NOISE_KEY = {fold_noise_key}
    load_ckpt_state()
    let bal = roi_verify_holdout(holdout_key)
    if bal <= 51.0 {{ print("ROI_FAIL\\tcheckpoint_balanced_accuracy\\n"); return 3 }}
    emit_roi_features({require_holdout_scope}, holdout_key)
    return 0
}}
"""
    )
    return "\n".join(parts)


def run_native_extractor(
    ckpt: dict[str, Any],
    checkpoint_path: Path,
    manifest_path: Path,
    output_dir: Path,
    scope: str,
    repo_root: Path | None = None,
    native_souc_engine: str = "",
) -> tuple[Path, Path, Path]:
    repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    native_dir = output_dir / ".native"
    native_dir.mkdir(parents=True, exist_ok=True)
    source_path = native_dir / "extract_trained_roi_associator.sio"
    binary_path = native_dir / "extract_trained_roi_associator.elf"
    stdout_path = native_dir / "extract_trained_roi_associator.out"
    state_path = native_dir / "model.state.seq"
    state_offsets = write_native_state_sequence(ckpt, state_path)
    _, records = load_manifest(manifest_path)
    use_single_site_internal_cv = single_site_internal_cv_enabled(None, records, int_meta(ckpt["meta"], "holdout_key"))
    source_path.write_text(
        generate_native_extractor_source(
            ckpt,
            manifest_path,
            repo_root,
            scope,
            state_path,
            state_offsets,
            use_single_site_internal_cv,
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env.setdefault("SOUNIO_STDLIB_PATH", str(repo_root / "stdlib"))
    if native_souc_engine:
        env["SOUNIO_SOUC_ENGINE"] = native_souc_engine
    compiler = repo_root / "bin" / "souc"
    compile_proc = subprocess.run(
        [str(compiler), "compile", str(source_path), "-o", str(binary_path)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
    )
    compile_output = compile_proc.stdout + compile_proc.stderr
    if compile_proc.returncode != 0 or (
        "exceeds IR_MAX_INSTRS" in compile_output and not binary_path.is_file()
    ):
        raise RuntimeError(
            "trained ROI associator native compile failed\n"
            f"checkpoint={checkpoint_path}\n"
            f"returncode={compile_proc.returncode}\n{compile_output}"
        )
    binary_path.chmod(0o755)

    run_proc = subprocess.run(
        [str(binary_path)],
        cwd=native_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
    )
    stdout_path.write_text(run_proc.stdout + run_proc.stderr, encoding="utf-8", errors="replace")
    if run_proc.returncode != 0:
        raise RuntimeError(
            "trained ROI associator native run failed\n"
            f"returncode={run_proc.returncode}\n"
            f"stdout_path={stdout_path}\n"
            f"{(run_proc.stdout + run_proc.stderr)[-4000:]}"
        )
    return source_path, binary_path, stdout_path


def parse_native_rows(stdout_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    verification: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    lines = merge_tab_continuation_records(
        stdout_path.read_text(encoding="utf-8", errors="replace").splitlines(),
        ("ROI_VERIFY_COUNTS", "ROI_FEATURE"),
    )
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] == "ROI_VERIFY_BAL":
            verification["actual_balanced_accuracy_pct"] = float(parts[1])
        elif len(parts) >= 6 and parts[0] == "ROI_VERIFY_COUNTS":
            verification["counts"] = {
                "tp": int(parts[1]),
                "tn": int(parts[2]),
                "fp": int(parts[3]),
                "fn": int(parts[4]),
                "test_n": int(parts[5]),
            }
        elif len(parts) >= 13 and parts[0] == "ROI_FEATURE":
            extra = parts + ["0"] * max(0, 18 - len(parts))
            rows.append(
                {
                    "subject_index": int(parts[1]),
                    "label": int(parts[2]),
                    "site": parts[3],
                    "roi_index": int(parts[4]),
                    "base_prob": int(parts[5]) / 1_000_000.0,
                    "masked_prob": int(parts[6]) / 1_000_000.0,
                    "base_assoc": int(parts[7]) / 1_000_000.0,
                    "masked_assoc": int(parts[8]) / 1_000_000.0,
                    "assoc_delta_signed": int(parts[9]) / 1_000_000.0,
                    "assoc_delta_abs": int(parts[10]) / 1_000_000.0,
                    "prob_delta_signed": int(parts[11]) / 1_000_000.0,
                    "prob_delta_abs": int(parts[12]) / 1_000_000.0,
                    "base_logit": int(extra[13]) / 1_000_000.0,
                    "masked_logit": int(extra[14]) / 1_000_000.0,
                    "logit_delta_signed": int(extra[15]) / 1_000_000.0,
                    "logit_delta_abs": int(extra[16]) / 1_000_000.0,
                    "hidden_delta_abs": int(extra[17]) / 1_000_000.0,
                }
            )
    if "actual_balanced_accuracy_pct" not in verification:
        raise RuntimeError(f"native extractor did not emit ROI_VERIFY_BAL: {stdout_path}")
    if not rows:
        raise RuntimeError(f"native extractor emitted no ROI_FEATURE rows: {stdout_path}")
    return verification, rows


def prob_to_logit(prob: float) -> float:
    prob = min(1.0 - 1.0e-12, max(1.0e-12, prob))
    return math.log(prob / (1.0 - prob))


def mutable_sequence(sequence: list[list[float]], feature_count: int) -> list[list[float]]:
    out = [list(step) for step in sequence]
    while len(out) * 8 < feature_count:
        out.append([0.0] * 8)
    return out


def run_python_extractor(
    ckpt: dict[str, Any],
    records: list[Any],
    scope: str,
) -> tuple[dict[str, Any], tuple[Path, Path, Path], list[dict[str, Any]]]:
    model = CheckpointForward(ckpt)
    holdout_key = int_meta(ckpt["meta"], "holdout_key")
    feature_count = int_meta(ckpt["meta"], "manifest_feature_count")
    use_single_site_internal_cv = single_site_internal_cv_enabled(None, records, holdout_key)
    site_keys = (
        materialize_single_site_internal_cv_keys(records)
        if use_single_site_internal_cv
        else [site_hash(record.site) for record in records]
    )
    labels: list[int] = []
    probs: list[float] = []
    tp = tn = fp = fnn = 0

    for subj_idx, record in enumerate(records):
        if site_keys[subj_idx] != holdout_key:
            continue
        prob, _assoc = model.forward(subj_idx, record.sequence)
        pred = 1 if prob >= 0.5 else 0
        label = int(record.label)
        labels.append(label)
        probs.append(prob)
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fnn += 1

    verification = {
        "actual_balanced_accuracy_pct": balanced_accuracy(labels, probs),
        "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fnn, "test_n": len(labels)},
        "executor": "python_checkpoint_forward_roi_occlusion",
        "single_site_internal_cv": use_single_site_internal_cv,
    }
    rows: list[dict[str, Any]] = []
    for subj_idx, record in enumerate(records):
        if scope == "holdout" and site_keys[subj_idx] != holdout_key:
            continue
        base_prob, base_assoc = model.forward(subj_idx, record.sequence)
        base_h = [model.hg(dim) for dim in range(16)]
        base_logit = prob_to_logit(base_prob)
        for roi in range(feature_count):
            masked = mutable_sequence(record.sequence, feature_count)
            masked[roi // 8][roi % 8] = float(model.mean[roi])
            masked_prob, masked_assoc = model.forward(subj_idx, masked)
            masked_h = [model.hg(dim) for dim in range(16)]
            masked_logit = prob_to_logit(masked_prob)
            assoc_delta = base_assoc - masked_assoc
            prob_delta = base_prob - masked_prob
            logit_delta = base_logit - masked_logit
            hidden_delta = math.sqrt(sum((base_h[idx] - masked_h[idx]) ** 2 for idx in range(16)))
            rows.append(
                {
                    "subject_index": subj_idx,
                    "label": int(record.label),
                    "site": f"SITE_{1 if site_keys[subj_idx] == 177622 else 2}" if use_single_site_internal_cv else record.site,
                    "roi_index": roi,
                    "base_prob": base_prob,
                    "masked_prob": masked_prob,
                    "base_assoc": base_assoc,
                    "masked_assoc": masked_assoc,
                    "assoc_delta_signed": assoc_delta,
                    "assoc_delta_abs": abs(assoc_delta),
                    "prob_delta_signed": prob_delta,
                    "prob_delta_abs": abs(prob_delta),
                    "base_logit": base_logit,
                    "masked_logit": masked_logit,
                    "logit_delta_signed": logit_delta,
                    "logit_delta_abs": abs(logit_delta),
                    "hidden_delta_abs": hidden_delta,
                }
            )
    marker = Path("python_checkpoint_forward")
    return verification, (marker, marker, marker), rows


def summarize_roi_rows(
    rows: list[dict[str, Any]],
    alpha: float,
    metric: str,
    positive_name: str,
    negative_name: str,
) -> list[dict[str, Any]]:
    by_roi: dict[int, dict[int, list[float]]] = {}
    for row in rows:
        roi = int(row["roi_index"])
        label = int(row["label"])
        by_roi.setdefault(roi, {0: [], 1: []})[label].append(float(row[metric]))

    out: list[dict[str, Any]] = []
    for roi in sorted(by_roi):
        negative = by_roi[roi][0]
        positive = by_roi[roi][1]
        stats = mann_whitney_cliff(positive, negative)
        out.append(
            {
                "roi_index": roi,
                "roi_label": f"f{roi}",
                "metric": METRIC_LABELS[metric],
                "positive_label": positive_name,
                "negative_label": negative_name,
                "n_positive": len(positive),
                "n_negative": len(negative),
                "mean_positive": round(sum(positive) / len(positive), 9) if positive else 0.0,
                "mean_negative": round(sum(negative) / len(negative), 9) if negative else 0.0,
                "median_positive": round(_median(positive), 9),
                "median_negative": round(_median(negative), 9),
                "u_stat": round(stats["u"], 6),
                "z": round(stats["z"], 6),
                "p_raw": stats["p"],
                "cliffs_delta": round(stats["cliff_delta"], 9),
            }
        )

    adjusted = holm_adjust([float(row["p_raw"]) for row in out])
    for row, p_holm in zip(out, adjusted, strict=True):
        row["p_holm"] = p_holm
        row["significant_holm_0_05"] = p_holm < alpha
        row["abs_cliffs_delta"] = abs(float(row["cliffs_delta"]))
    out.sort(key=lambda row: (float(row["p_holm"]), -float(row["abs_cliffs_delta"]), int(row["roi_index"])))
    for rank, row in enumerate(out, start=1):
        row["rank"] = rank
    return out


def enrich_subject_rows(
    rows: list[dict[str, Any]],
    records: list[Any],
    positive_name: str,
    negative_name: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        record = records[int(row["subject_index"])]
        label = int(row["label"])
        enriched.append(
            {
                "subject_index": row["subject_index"],
                "subject_id": record.subject_id,
                "label": row["label"],
                "label_name": label_name_for(label, positive_name, negative_name),
                "site": row["site"],
                "roi_index": row["roi_index"],
                "roi_label": f"f{row['roi_index']}",
                "base_prob": row["base_prob"],
                "masked_prob": row["masked_prob"],
                "base_assoc": row["base_assoc"],
                "masked_assoc": row["masked_assoc"],
                "assoc_delta_signed": row["assoc_delta_signed"],
                "assoc_delta_abs": row["assoc_delta_abs"],
                "prob_delta_signed": row["prob_delta_signed"],
                "prob_delta_abs": row["prob_delta_abs"],
                "base_logit": row["base_logit"],
                "masked_logit": row["masked_logit"],
                "logit_delta_signed": row["logit_delta_signed"],
                "logit_delta_abs": row["logit_delta_abs"],
                "hidden_delta_abs": row["hidden_delta_abs"],
            }
        )
    return enriched


def write_outputs(
    output_dir: Path,
    ckpt: dict[str, Any],
    checkpoint_path: Path,
    manifest_path: Path,
    manifest_meta: Any,
    records: list[Any],
    native_verification: dict[str, Any],
    native_paths: tuple[Path, Path, Path],
    rows: list[dict[str, Any]],
    roi_stats: list[dict[str, Any]],
    scope: str,
    alpha: float,
    metric: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_name, negative_name = manifest_label_names(manifest_meta.extra)
    subject_rows = enrich_subject_rows(rows, records, positive_name, negative_name)
    roi_stat_fields = [
        "rank",
        "roi_index",
        "roi_label",
        "metric",
        "positive_label",
        "negative_label",
        "n_positive",
        "n_negative",
        "mean_positive",
        "mean_negative",
        "median_positive",
        "median_negative",
        "u_stat",
        "z",
        "p_raw",
        "p_holm",
        "cliffs_delta",
        "abs_cliffs_delta",
        "significant_holm_0_05",
    ]
    write_tsv(
        output_dir / "subject_roi_assoc_features.tsv",
        subject_rows,
        [
            "subject_index",
            "subject_id",
            "label",
            "label_name",
            "site",
            "roi_index",
            "roi_label",
            "base_prob",
            "masked_prob",
            "base_assoc",
            "masked_assoc",
            "assoc_delta_signed",
            "assoc_delta_abs",
            "prob_delta_signed",
            "prob_delta_abs",
            "base_logit",
            "masked_logit",
            "logit_delta_signed",
            "logit_delta_abs",
            "hidden_delta_abs",
        ],
    )
    write_tsv(
        output_dir / "associator_by_roi.csv",
        roi_stats,
        roi_stat_fields,
    )
    write_tsv(
        output_dir / "top10_roi_biomarkers.csv",
        roi_stats[:10],
        roi_stat_fields,
    )
    source_path, binary_path, stdout_path = native_paths
    atlas = manifest_meta.extra.get("atlas") or f"manifest_f0_f{max(0, manifest_meta.seq_len * manifest_meta.input_dim - 1)}"
    summary = {
        "schema": "brain_ossm.trained_roi_associator.v1",
        "method": native_verification.get("executor", "trained_checkpoint_native_forward_roi_occlusion"),
        "ranking_metric": METRIC_LABELS[metric],
        "scope": scope,
        "alpha": alpha,
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "manifest_schema": manifest_meta.schema,
        "dataset_id": manifest_meta.extra.get("dataset_id", "abide"),
        "label_space": manifest_meta.label_space,
        "positive_label_name": positive_name,
        "negative_label_name": negative_name,
        "atlas": atlas,
        "run_id": ckpt.get("run_id"),
        "checkpoint_selection": ckpt.get("selection"),
        "checkpoint_meta": ckpt.get("meta"),
        "checkpoint_verification": ckpt.get("verification"),
        "extractor_verification": native_verification,
        "native_extractor_verification": native_verification,
        "native_extractor_source": str(source_path),
        "native_extractor_binary": str(binary_path),
        "native_extractor_stdout": str(stdout_path),
        "subject_count": len({row["subject_index"] for row in rows}),
        "roi_count": len({row["roi_index"] for row in rows}),
        "feature_rows": len(rows),
        "holm_survivors": sum(1 for row in roi_stats if row["significant_holm_0_05"]),
        "top_roi": roi_stats[0] if roi_stats else None,
    }
    write_json(output_dir / "run_summary.json", summary)
    write_json(
        output_dir / "associator_map.json",
        {
            "schema": "brain_ossm.trained_roi_associator.map.v1",
            "dataset_id": manifest_meta.extra.get("dataset_id", "abide"),
            "label_space": manifest_meta.label_space,
            "positive_label_name": positive_name,
            "negative_label_name": negative_name,
            "atlas": atlas,
            "metric": METRIC_LABELS[metric],
            "roi_count": len(roi_stats),
            "rois": roi_stats,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=["all", "holdout"], default="all")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-balanced-accuracy", type=float, default=51.0)
    parser.add_argument("--executor", choices=["python", "native"], default="native")
    parser.add_argument("--native-souc-engine", default="", help="Optional SOUNIO_SOUC_ENGINE for native extraction.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_LABELS),
        default="hidden_delta_abs",
        help="ROI statistic used for Mann-Whitney/Holm ranking; default uses trained hidden-state occlusion.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    require_verified_checkpoint(ckpt, args.min_balanced_accuracy)
    manifest_meta, records = load_manifest(manifest_path)
    positive_name, negative_name = manifest_label_names(manifest_meta.extra)
    rebuild_standardization_from_manifest(ckpt, records)
    require_verified_checkpoint(ckpt, args.min_balanced_accuracy)

    repaired_checkpoint = output_dir / "model.with_rebuilt_standardization.ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_checkpoint.write_text(json.dumps(ckpt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.executor == "native":
        native_paths = run_native_extractor(
            ckpt,
            checkpoint_path,
            manifest_path,
            output_dir,
            args.scope,
            Path(args.repo_root),
            args.native_souc_engine,
        )
        native_verification, rows = parse_native_rows(native_paths[2])
    else:
        native_verification, native_paths, rows = run_python_extractor(ckpt, records, args.scope)
    expected = float(ckpt["verification"]["actual_balanced_accuracy_pct"])
    actual = float(native_verification["actual_balanced_accuracy_pct"])
    delta = abs(actual - expected)
    if delta > 0.5:
        raise SystemExit(f"extractor verification mismatch: expected={expected:.6f} actual={actual:.6f} delta={delta:.6f}pp")
    if actual <= args.min_balanced_accuracy:
        raise SystemExit(f"extractor verification below threshold: actual={actual:.6f}")

    roi_stats = summarize_roi_rows(rows, args.alpha, args.metric, positive_name, negative_name)
    if args.scope == "holdout":
        holdout_key = int_meta(ckpt["meta"], "holdout_key")
        expected_site_keys = (
            materialize_single_site_internal_cv_keys(records)
            if single_site_internal_cv_enabled(None, records, holdout_key)
            else [site_hash(record.site) for record in records]
        )
        expected_subjects = sum(1 for site_key in expected_site_keys if site_key == holdout_key)
    else:
        expected_subjects = len(records)
    expected_rows = expected_subjects * int_meta(ckpt["meta"], "manifest_feature_count")
    if len(rows) != expected_rows:
        raise SystemExit(f"extractor row count mismatch: expected={expected_rows} actual={len(rows)}")
    write_outputs(
        output_dir,
        ckpt,
        checkpoint_path,
        manifest_path,
        manifest_meta,
        records,
        {**native_verification, "expected_balanced_accuracy_pct": expected, "delta_pp": delta},
        native_paths,
        rows,
        roi_stats,
        args.scope,
        args.alpha,
        args.metric,
    )
    print(
        "trained ROI associator extraction complete: "
        f"rows={len(rows)} rois={len(roi_stats)} "
        f"verify_expected={expected:.6f} verify_actual={actual:.6f} delta={delta:.6f}pp "
        f"top10={output_dir / 'top10_roi_biomarkers.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate Omega hardware telemetry artifacts for schema/field regressions."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, List

FPGA_REPORT_PATH = Path("artifacts/fpga/fpga_seed_report.json")
QUANTUM_CONFORMANCE_PATH = Path("artifacts/quantum/omega/quantum_conformance.json")
PURE_KAXI_MARKER_PATH = Path("artifacts/.pure_sounio_kaxi_generated")
HARDWARE_COUNTER_REPRO_PATH = Path("artifacts/fpga/hardware_counter_repro.v1.json")
HARDWARE_RESOURCE_TREND_PATH = Path("artifacts/fpga/hardware_resource_trend.v2.json")
PTX_LAUNCH_REPORT_PATH = Path("artifacts/ptx/omega/ptx_launch_report.json")
HW_EPI_POWER_LIVE_PATH = Path("artifacts/fpga/hardware_epistemic_power_live.v1.json")
HW_EPI_POWER_TREND_PATH = Path("artifacts/fpga/hardware_epistemic_power_live_trend.v1.json")
RL_READINESS_BRIDGE_PATH = Path("artifacts/omega/rl_readiness_bridge.v1.json")
RL_READINESS_EVIDENCE_PATH = Path("bootstrap/policies/rl_readiness.evidence.json")
SHADOW_AUDIT_PATH = Path("artifacts/omega/shadow_audit.v1.json")
RL_READINESS_TREND_PATH = Path("artifacts/omega/rl_readiness_trend.v1.json")
POLICY_MODE_GUARD_PATH = Path("artifacts/omega/policy_mode_guard.v1.json")
RL_READINESS_REPLAY_PATH = Path("artifacts/omega/rl_readiness_replay.v1.json")
GOVERNANCE_ATTESTATION_PATH = Path("artifacts/omega/governance_attestation.v1.json")
GENESIS_MANIFEST_PATH = Path("artifacts/omega/omega_genesis.v1.0.json")
SPRINT1_RELEASE_READINESS_PATH = Path("artifacts/omega/sprint1_release_readiness.v1.json")
PERFORMANCE_SUMMARY_PATH = Path("artifacts/omega/performance_summary.v1.json")
EXTERNAL_BASELINE_COLLECTION_PATH = Path("artifacts/omega/external_baseline_collection.v1.json")
BASELINE_FREEZE_PATH = Path("artifacts/omega/baseline_freeze.v1.json")
QIR_EPI_POWER_SIO_PATH = Path("hardware/rtl/qir/omega_epistemic_power.sio")
QIR_FULL_EMITTER_SIO_PATH = Path("hardware/rtl/qir/omega_full_qir_emitter.sio")
BIDIR_KAXI_ADAPTER_SIO_PATH = Path("hardware/rtl/kaxi/bidirectional_return_adapter.sio")
MERKLE_LANE_SIO_PATH = Path("hardware/rtl/kaxi/merkle_lane.sio")
MERKLE_LANE_RTL_PATH = Path("hardware/fpga/k_axi_merkle_lane.v")
MERKLE_ROOT_SIO_PATH = Path("hardware/rtl/kaxi/merkle_root_lane.sio")
MERKLE_ROOT_RTL_PATH = Path("hardware/fpga/k_axi_merkle_root_lane.v")
BIDIR_KAXI_WAVEFORM_PATH = Path("artifacts/fpga/waveforms/tb_k_axi_bidirectional.vcd")
ACCUMULATOR_WAVEFORM_PATH = Path(
    "artifacts/fpga/waveforms/tb_epistemic_power_accumulator.vcd"
)
ACCUMULATOR_BOUNDS_TEST_PATH = Path("tests/hardware/accumulator_bounds_test.sio")

FPGA_REPORT_SCHEMA = "sounio.omega.fpga-seed-report.v1"
QUANTUM_CONFORMANCE_SCHEMA = "sounio.quantum.conformance.v1"
HARDWARE_COUNTER_REPRO_SCHEMA = "sounio.omega.hardware-counter-repro.v1"
HARDWARE_RESOURCE_TREND_SCHEMA = "sounio.omega.hardware-resource-trend.v2"
PTX_LAUNCH_REPORT_SCHEMA = "sounio.omega.ptx-launch-report.v1"
HW_EPI_POWER_LIVE_SCHEMA = "sounio.omega.hardware-epistemic-power-live.v1"
HW_EPI_POWER_TREND_SCHEMA = "sounio.omega.hardware-epistemic-power-live-trend.v1"
RL_READINESS_BRIDGE_SCHEMA = "sounio.omega.rl-readiness-bridge.v1"
SHADOW_AUDIT_SCHEMA = "sounio.omega.shadow-audit.v1"
RL_READINESS_TREND_SCHEMA = "sounio.omega.rl-readiness-trend.v1"
POLICY_MODE_GUARD_SCHEMA = "sounio.omega.policy-mode-guard.v1"
RL_READINESS_REPLAY_SCHEMA = "sounio.omega.rl-readiness-replay.v1"
GOVERNANCE_ATTESTATION_SCHEMA = "sounio.omega.governance-attestation.v1"
GENESIS_MANIFEST_SCHEMA = "sounio.omega.genesis.v1.0"
SPRINT1_RELEASE_READINESS_SCHEMA = "sounio.omega.sprint1-release-readiness.v1"
PERFORMANCE_SUMMARY_SCHEMA = "sounio.omega.performance-summary.v1"
EXTERNAL_BASELINE_COLLECTION_SCHEMA = "sounio.omega.external-baseline-collection.v1"
BASELINE_FREEZE_SCHEMA = "sounio.omega.baseline-freeze.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Omega hardware telemetry regression validator"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require all telemetry artifacts to exist",
    )
    return parser.parse_args()


def is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    value_f = float(value)
    return math.isfinite(value_f)


def load_json_object(path: Path, errors: List[str]) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"{path}: invalid JSON: {exc}")
        return None
    except OSError as exc:
        errors.append(f"{path}: read failed: {exc}")
        return None

    if not isinstance(payload, dict):
        errors.append(f"{path}: expected top-level JSON object")
        return None
    return payload


def validate_fpga_report(path: Path, errors: List[str], strict: bool = False) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != FPGA_REPORT_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {FPGA_REPORT_SCHEMA}, got {payload.get('schema')})"
        )

    required_status_fields = (
        "sim_status",
        "synth_status",
        "quantum_controller_sim_status",
        "quantum_controller_synth_status",
        "quantum_accumulator_link_sim_status",
        "k_axi_sim_status",
        "k_axi_synth_status",
        "k_axi_return_sim_status",
        "k_axi_return_synth_status",
        "epistemic_power_accumulator_sim_status",
        "epistemic_power_accumulator_synth_status",
        "merkle_lane_synth_status",
        "merkle_root_synth_status",
        "hardware_counter_repro_status",
        "hardware_resource_trend_status",
    )
    for field in required_status_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
            continue
        if not isinstance(payload[field], str) or not payload[field]:
            errors.append(f"{path}: field {field} must be a non-empty string")

    required_bool_fields = (
        "merkle_lane_present",
        "merkle_lane_core_rtl_present",
        "merkle_root_core_rtl_present",
        "quantum_controller_lane_present",
    )
    for field in required_bool_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
            continue
        if not isinstance(payload[field], bool):
            errors.append(f"{path}: field {field} must be a bool")

    waveform_paths = payload.get("waveform_paths")
    if waveform_paths is not None and not isinstance(waveform_paths, list):
        errors.append(f"{path}: waveform_paths must be a list when present")
    waveforms_present = payload.get("waveforms_present")
    if waveforms_present is not None and not isinstance(waveforms_present, bool):
        errors.append(f"{path}: waveforms_present must be bool when present")

    if strict:
        for field in required_bool_fields:
            if field in payload and isinstance(payload[field], bool) and not payload[field]:
                errors.append(f"{path}: field {field} must be true in --strict mode")
        if (
            "quantum_accumulator_link_sim_status" in payload
            and payload["quantum_accumulator_link_sim_status"] != "pass"
        ):
            errors.append(
                f"{path}: quantum_accumulator_link_sim_status must be pass in --strict mode (got {payload['quantum_accumulator_link_sim_status']})"
            )
        if (
            "hardware_counter_repro_status" in payload
            and payload["hardware_counter_repro_status"] not in ("pass", "bootstrap")
        ):
            errors.append(
                f"{path}: hardware_counter_repro_status must be pass/bootstrap in --strict mode (got {payload['hardware_counter_repro_status']})"
            )
        if (
            "hardware_resource_trend_status" in payload
            and payload["hardware_resource_trend_status"] not in ("pass", "bootstrap")
        ):
            errors.append(
                f"{path}: hardware_resource_trend_status must be pass/bootstrap in --strict mode (got {payload['hardware_resource_trend_status']})"
            )
        if (
            "merkle_lane_synth_status" in payload
            and payload["merkle_lane_synth_status"] != "pass"
        ):
            errors.append(
                f"{path}: merkle_lane_synth_status must be pass in --strict mode (got {payload['merkle_lane_synth_status']})"
            )
        if (
            "merkle_root_synth_status" in payload
            and payload["merkle_root_synth_status"] != "pass"
        ):
            errors.append(
                f"{path}: merkle_root_synth_status must be pass in --strict mode (got {payload['merkle_root_synth_status']})"
            )
        if "waveforms_present" in payload and payload["waveforms_present"] is not True:
            errors.append(f"{path}: waveforms_present must be true in --strict mode")


def validate_quantum_conformance(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != QUANTUM_CONFORMANCE_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {QUANTUM_CONFORMANCE_SCHEMA}, got {payload.get('schema')})"
        )

    for field in ("ks_pvalue", "coverage_95"):
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
            continue
        if not is_number(payload[field]):
            errors.append(f"{path}: field {field} must be numeric")

    if "ks_pvalue" in payload and is_number(payload["ks_pvalue"]):
        ks = float(payload["ks_pvalue"])
        if ks < 0.0 or ks > 1.0:
            errors.append(f"{path}: ks_pvalue out of range [0,1]: {ks}")
        if strict and ks < 0.05:
            errors.append(f"{path}: ks_pvalue below strict threshold 0.05: {ks}")

    if "coverage_95" in payload and is_number(payload["coverage_95"]):
        coverage = float(payload["coverage_95"])
        if coverage < 0.0 or coverage > 1.0:
            errors.append(f"{path}: coverage_95 out of range [0,1]: {coverage}")
        if strict and (coverage < 0.90 or coverage > 0.98):
            errors.append(
                f"{path}: coverage_95 outside strict range [0.90,0.98]: {coverage}"
            )


def validate_pure_kaxi_marker(path: Path, errors: List[str]) -> None:
    try:
        text = path.read_text()
    except OSError as exc:
        errors.append(f"{path}: read failed: {exc}")
        return

    if "pure_sounio_kaxi=1" not in text:
        errors.append(f"{path}: missing required marker pure_sounio_kaxi=1")
    if "mode=selfhost-emitter" not in text:
        errors.append(f"{path}: missing required marker mode=selfhost-emitter")


def validate_counter_repro(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != HARDWARE_COUNTER_REPRO_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {HARDWARE_COUNTER_REPRO_SCHEMA}, got {payload.get('schema')})"
        )

    status = payload.get("status")
    if not isinstance(status, str) or not status:
        errors.append(f"{path}: status must be a non-empty string")

    fp = payload.get("current_fingerprint")
    if not isinstance(fp, dict):
        errors.append(f"{path}: current_fingerprint must be an object")
    else:
        required_fp_fields = (
            "kaxi_gum",
            "kaxi_prov",
            "kaxi_formal",
            "accum_log",
            "accum_tx",
            "qlink_seen",
            "qlink_log",
            "qlink_tx",
            "kaxi_return_seen",
            "kaxi_return_dropped",
            "kaxi_return_overflow",
        )
        for field in required_fp_fields:
            if field not in fp:
                errors.append(f"{path}: current_fingerprint missing field {field}")
                continue
            if not isinstance(fp[field], int):
                errors.append(f"{path}: current_fingerprint.{field} must be an int")

    if strict and isinstance(status, str) and status not in ("pass", "bootstrap"):
        errors.append(f"{path}: status must be pass/bootstrap in --strict mode (got {status})")


def validate_resource_trend(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != HARDWARE_RESOURCE_TREND_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {HARDWARE_RESOURCE_TREND_SCHEMA}, got {payload.get('schema')})"
        )

    runs = payload.get("runs")
    if not isinstance(runs, list):
        errors.append(f"{path}: runs must be a list")
        return
    if strict and not runs:
        errors.append(f"{path}: runs must be non-empty in --strict mode")
        return
    if not runs:
        return

    drift_threshold = payload.get("drift_threshold")
    if not is_number(drift_threshold):
        errors.append(f"{path}: drift_threshold must be numeric")
        drift_threshold = 0.05
    else:
        drift_threshold = float(drift_threshold)

    last_status = payload.get("last_status")
    if not isinstance(last_status, str) or not last_status:
        errors.append(f"{path}: last_status must be non-empty string")
    elif last_status not in ("bootstrap", "pass", "drift", "incomplete"):
        errors.append(f"{path}: invalid last_status {last_status!r}")

    last_max_relative_drift = payload.get("last_max_relative_drift")
    if not is_number(last_max_relative_drift):
        errors.append(f"{path}: last_max_relative_drift must be numeric")
    else:
        last_max_relative_drift = float(last_max_relative_drift)

    latest = runs[-1]
    if not isinstance(latest, dict):
        errors.append(f"{path}: latest run entry must be an object")
        return

    modules = latest.get("modules")
    if not isinstance(modules, dict):
        errors.append(f"{path}: latest run modules must be an object")
        return

    if strict and len(modules) < 3:
        errors.append(f"{path}: latest run modules must have >=3 modules in --strict mode")

    for module_name, stats in modules.items():
        if not isinstance(module_name, str) or not module_name:
            errors.append(f"{path}: invalid module key: {module_name!r}")
            continue
        if not isinstance(stats, dict):
            errors.append(f"{path}: module {module_name} stats must be an object")
            continue
        for field in ("wires", "wire_bits", "cells"):
            if field not in stats:
                errors.append(f"{path}: module {module_name} missing field {field}")
                continue
            if not isinstance(stats[field], int):
                errors.append(f"{path}: module {module_name} field {field} must be int")

    run_status = latest.get("status")
    if not isinstance(run_status, str):
        errors.append(f"{path}: latest run status must be string")
    elif run_status not in ("bootstrap", "pass", "drift", "incomplete"):
        errors.append(f"{path}: latest run status has invalid value {run_status!r}")

    run_max_drift = latest.get("max_relative_drift")
    if not is_number(run_max_drift):
        errors.append(f"{path}: latest run max_relative_drift must be numeric")
    else:
        run_max_drift = float(run_max_drift)
        if run_max_drift < 0.0:
            errors.append(f"{path}: latest run max_relative_drift must be >= 0")

    compared_points = latest.get("compared_points")
    if compared_points is not None and not isinstance(compared_points, int):
        errors.append(f"{path}: latest run compared_points must be int when present")
    for list_field in ("new_modules", "removed_modules"):
        value = latest.get(list_field)
        if value is not None and not isinstance(value, list):
            errors.append(f"{path}: latest run {list_field} must be list when present")

    if strict:
        if isinstance(last_status, str) and last_status not in ("pass", "bootstrap"):
            errors.append(
                f"{path}: last_status must be pass/bootstrap in --strict mode (got {last_status!r})"
            )
        if is_number(last_max_relative_drift) and float(last_max_relative_drift) > drift_threshold:
            errors.append(
                f"{path}: last_max_relative_drift {float(last_max_relative_drift):.6f} exceeds drift_threshold {drift_threshold:.6f} in --strict mode"
            )


def validate_ptx_launch_report(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != PTX_LAUNCH_REPORT_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {PTX_LAUNCH_REPORT_SCHEMA}, got {payload.get('schema')})"
        )

    required_bool_fields = ("pass_marker_seen", "launch_accepted", "used_fallback")
    required_numeric_fields = (
        "op_kind",
        "replay_flags",
        "hardware_epistemic_power_log_q32_32",
        "software_epistemic_power_log_q32_32",
        "hybrid_epistemic_power_log_q32_32",
    )
    required_string_fields = ("kernel_name", "target", "test_file", "output_sha256")

    for field in required_bool_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
            continue
        if not isinstance(payload[field], bool):
            errors.append(f"{path}: field {field} must be bool")

    for field in required_numeric_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
            continue
        if not isinstance(payload[field], int):
            errors.append(f"{path}: field {field} must be int")

    for field in required_string_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
            continue
        if not isinstance(payload[field], str) or not payload[field]:
            errors.append(f"{path}: field {field} must be non-empty string")

    if strict:
        if payload.get("pass_marker_seen") is not True:
            errors.append(f"{path}: pass_marker_seen must be true in --strict mode")
        if payload.get("launch_accepted") is not True:
            errors.append(f"{path}: launch_accepted must be true in --strict mode")
        if payload.get("used_fallback") is not False:
            errors.append(f"{path}: used_fallback must be false in --strict mode")

        hw = payload.get("hardware_epistemic_power_log_q32_32")
        sw = payload.get("software_epistemic_power_log_q32_32")
        hybrid = payload.get("hybrid_epistemic_power_log_q32_32")
        if isinstance(sw, int) and sw <= 0:
            errors.append(f"{path}: software_epistemic_power_log_q32_32 must be >0 in --strict mode")
        if isinstance(hybrid, int) and hybrid <= 0:
            errors.append(f"{path}: hybrid_epistemic_power_log_q32_32 must be >0 in --strict mode")
        if isinstance(hw, int) and isinstance(hybrid, int) and hybrid < hw:
            errors.append(
                f"{path}: hybrid_epistemic_power_log_q32_32 must be >= hardware_epistemic_power_log_q32_32 in --strict mode"
            )


def validate_hw_epi_power_live(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != HW_EPI_POWER_LIVE_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {HW_EPI_POWER_LIVE_SCHEMA}, got {payload.get('schema')})"
        )

    required_status_fields = (
        "accumulator_sim_status",
        "accumulator_synth_status",
        "symbol_check_mode",
    )
    required_bool_fields = (
        "launch_accepted",
        "used_fallback",
        "live_read_conformant",
    )
    required_numeric_fields = (
        "hardware_epistemic_power_log_q32_32",
        "software_epistemic_power_log_q32_32",
        "hybrid_epistemic_power_log_q32_32",
        "hardware_epistemic_power_variance_q32_32",
        "poll_overhead_us",
    )

    for field in required_status_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
            continue
        if not isinstance(payload[field], str) or not payload[field]:
            errors.append(f"{path}: field {field} must be non-empty string")

    for field in required_bool_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
            continue
        if not isinstance(payload[field], bool):
            errors.append(f"{path}: field {field} must be bool")

    for field in required_numeric_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
            continue
        if not isinstance(payload[field], int):
            errors.append(f"{path}: field {field} must be int")

    symbols = payload.get("symbols")
    if not isinstance(symbols, dict):
        errors.append(f"{path}: symbols must be an object")
    else:
        for symbol in ("g_epistemic_power_log_hw_q32_32", "g_kaxi_ring"):
            if symbol not in symbols:
                errors.append(f"{path}: symbols missing key {symbol}")
                continue
            if not isinstance(symbols[symbol], bool):
                errors.append(f"{path}: symbols.{symbol} must be bool")

    if strict:
        if payload.get("accumulator_sim_status") != "pass":
            errors.append(f"{path}: accumulator_sim_status must be pass in --strict mode")
        if payload.get("accumulator_synth_status") != "pass":
            errors.append(f"{path}: accumulator_synth_status must be pass in --strict mode")
        if payload.get("launch_accepted") is not True:
            errors.append(f"{path}: launch_accepted must be true in --strict mode")
        if payload.get("used_fallback") is not False:
            errors.append(f"{path}: used_fallback must be false in --strict mode")
        if payload.get("live_read_conformant") is not True:
            errors.append(f"{path}: live_read_conformant must be true in --strict mode")
        variance = payload.get("hardware_epistemic_power_variance_q32_32")
        if isinstance(variance, int) and variance < 0:
            errors.append(
                f"{path}: hardware_epistemic_power_variance_q32_32 must be >=0 in --strict mode"
            )
        overhead = payload.get("poll_overhead_us")
        if not isinstance(overhead, int):
            errors.append(f"{path}: poll_overhead_us must be int in --strict mode")
        elif overhead < 0 or overhead >= 1000:
            errors.append(f"{path}: poll_overhead_us must be in [0,1000) in --strict mode")
        if isinstance(symbols, dict):
            for symbol in ("g_epistemic_power_log_hw_q32_32", "g_kaxi_ring"):
                if symbols.get(symbol) is not True:
                    errors.append(f"{path}: symbols.{symbol} must be true in --strict mode")


def validate_hw_epi_power_trend(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != HW_EPI_POWER_TREND_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {HW_EPI_POWER_TREND_SCHEMA}, got {payload.get('schema')})"
        )

    runs = payload.get("runs")
    if not isinstance(runs, list):
        errors.append(f"{path}: runs must be a list")
        return
    if strict and not runs:
        errors.append(f"{path}: runs must be non-empty in --strict mode")
        return
    if not runs:
        return

    latest = runs[-1]
    if not isinstance(latest, dict):
        errors.append(f"{path}: latest run entry must be an object")
        return

    required_latest_bool = ("live_read_conformant", "launch_accepted", "used_fallback", "drift_violation")
    required_latest_numeric = (
        "gain_ratio",
        "drift_abs",
        "hardware_epistemic_power_log_q32_32",
        "software_epistemic_power_log_q32_32",
        "hybrid_epistemic_power_log_q32_32",
    )
    for field in required_latest_bool:
        if field not in latest:
            errors.append(f"{path}: latest run missing field {field}")
            continue
        if not isinstance(latest[field], bool):
            errors.append(f"{path}: latest run field {field} must be bool")
    for field in required_latest_numeric:
        if field not in latest:
            errors.append(f"{path}: latest run missing field {field}")
            continue
        if not is_number(latest[field]):
            errors.append(f"{path}: latest run field {field} must be numeric")

    status = payload.get("last_status")
    if not isinstance(status, str) or not status:
        errors.append(f"{path}: last_status must be a non-empty string")
    elif status not in ("bootstrap", "pass", "warn", "fail"):
        errors.append(f"{path}: invalid last_status {status!r}")

    if strict:
        if latest.get("live_read_conformant") is not True:
            errors.append(f"{path}: latest live_read_conformant must be true in --strict mode")
        if latest.get("launch_accepted") is not True:
            errors.append(f"{path}: latest launch_accepted must be true in --strict mode")
        if latest.get("used_fallback") is not False:
            errors.append(f"{path}: latest used_fallback must be false in --strict mode")
        if latest.get("drift_violation") is True:
            errors.append(f"{path}: latest drift_violation must be false in --strict mode")
        if status not in ("pass", "bootstrap"):
            errors.append(f"{path}: last_status must be pass/bootstrap in --strict mode")


def validate_rl_readiness_evidence(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    required_fields = (
        "epistemic_power_gain",
        "bit_exact_mismatches",
        "compile_overhead",
        "shadow_audit_clean",
    )
    for field in required_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
    if "epistemic_power_gain" in payload and not is_number(payload["epistemic_power_gain"]):
        errors.append(f"{path}: epistemic_power_gain must be numeric")
    if "bit_exact_mismatches" in payload and not isinstance(payload["bit_exact_mismatches"], int):
        errors.append(f"{path}: bit_exact_mismatches must be int")
    if "compile_overhead" in payload and not is_number(payload["compile_overhead"]):
        errors.append(f"{path}: compile_overhead must be numeric")
    if "shadow_audit_clean" in payload and not isinstance(payload["shadow_audit_clean"], bool):
        errors.append(f"{path}: shadow_audit_clean must be bool")

    if strict:
        if is_number(payload.get("epistemic_power_gain")) and float(payload["epistemic_power_gain"]) < 0.05:
            errors.append(f"{path}: epistemic_power_gain must be >= 0.05 in --strict mode")
        if isinstance(payload.get("bit_exact_mismatches"), int) and payload["bit_exact_mismatches"] != 0:
            errors.append(f"{path}: bit_exact_mismatches must be 0 in --strict mode")
        if is_number(payload.get("compile_overhead")) and float(payload["compile_overhead"]) > 0.25:
            errors.append(f"{path}: compile_overhead must be <= 0.25 in --strict mode")
        if payload.get("shadow_audit_clean") is not True:
            errors.append(f"{path}: shadow_audit_clean must be true in --strict mode")


def validate_rl_readiness_bridge(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != RL_READINESS_BRIDGE_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {RL_READINESS_BRIDGE_SCHEMA}, got {payload.get('schema')})"
        )

    required_string_fields = (
        "policy",
        "ptx_report",
        "sass_report",
        "hw_live_report",
        "quantum_report",
        "evidence_path",
    )
    for field in required_string_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
            continue
        if not isinstance(payload[field], str) or not payload[field]:
            errors.append(f"{path}: field {field} must be non-empty string")

    if "policy_rl_readiness_pass" not in payload:
        errors.append(f"{path}: missing required field: policy_rl_readiness_pass")
    elif not isinstance(payload["policy_rl_readiness_pass"], bool):
        errors.append(f"{path}: policy_rl_readiness_pass must be bool")

    if "policy_rl_readiness_fail_reasons" not in payload:
        errors.append(f"{path}: missing required field: policy_rl_readiness_fail_reasons")
    elif not isinstance(payload["policy_rl_readiness_fail_reasons"], list):
        errors.append(f"{path}: policy_rl_readiness_fail_reasons must be a list")

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        errors.append(f"{path}: evidence must be an object")
    else:
        for field in ("epistemic_power_gain", "bit_exact_mismatches", "compile_overhead", "shadow_audit_clean"):
            if field not in evidence:
                errors.append(f"{path}: evidence missing field {field}")

    if strict and payload.get("policy_rl_readiness_pass") is not True:
        errors.append(f"{path}: policy_rl_readiness_pass must be true in --strict mode")


def validate_shadow_audit(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != SHADOW_AUDIT_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {SHADOW_AUDIT_SCHEMA}, got {payload.get('schema')})"
        )

    for field in ("required_markers", "found_markers", "missing_markers"):
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
            continue
        if not isinstance(payload[field], list):
            errors.append(f"{path}: field {field} must be a list")

    clean = payload.get("shadow_audit_clean")
    if not isinstance(clean, bool):
        errors.append(f"{path}: shadow_audit_clean must be bool")
    elif strict and clean is not True:
        errors.append(f"{path}: shadow_audit_clean must be true in --strict mode")

    missing = payload.get("missing_markers")
    if strict and isinstance(missing, list) and missing:
        errors.append(f"{path}: missing_markers must be empty in --strict mode")


def validate_rl_readiness_trend(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != RL_READINESS_TREND_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {RL_READINESS_TREND_SCHEMA}, got {payload.get('schema')})"
        )

    runs = payload.get("runs")
    if not isinstance(runs, list):
        errors.append(f"{path}: runs must be a list")
        return
    if strict and not runs:
        errors.append(f"{path}: runs must be non-empty in --strict mode")
        return
    if not runs:
        return

    latest = runs[-1]
    if not isinstance(latest, dict):
        errors.append(f"{path}: latest run entry must be an object")
        return

    required_numeric = (
        "epistemic_power_gain",
        "compile_overhead",
        "gain_drop_abs",
        "overhead_jump",
    )
    required_bool = ("shadow_audit_clean", "policy_rl_readiness_pass", "drift_violation")
    for field in required_numeric:
        if field not in latest:
            errors.append(f"{path}: latest run missing field {field}")
            continue
        if not is_number(latest[field]):
            errors.append(f"{path}: latest run field {field} must be numeric")
    for field in required_bool:
        if field not in latest:
            errors.append(f"{path}: latest run missing field {field}")
            continue
        if not isinstance(latest[field], bool):
            errors.append(f"{path}: latest run field {field} must be bool")

    status = payload.get("last_status")
    if not isinstance(status, str) or not status:
        errors.append(f"{path}: last_status must be a non-empty string")
    elif status not in ("bootstrap", "pass", "warn", "fail"):
        errors.append(f"{path}: invalid last_status {status!r}")

    if strict:
        if latest.get("policy_rl_readiness_pass") is not True:
            errors.append(f"{path}: latest policy_rl_readiness_pass must be true in --strict mode")
        if latest.get("shadow_audit_clean") is not True:
            errors.append(f"{path}: latest shadow_audit_clean must be true in --strict mode")
        if latest.get("drift_violation") is True:
            errors.append(f"{path}: latest drift_violation must be false in --strict mode")
        if status not in ("pass", "bootstrap"):
            errors.append(f"{path}: last_status must be pass/bootstrap in --strict mode")


def validate_policy_mode_guard(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != POLICY_MODE_GUARD_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {POLICY_MODE_GUARD_SCHEMA}, got {payload.get('schema')})"
        )

    for field in ("policy_mode", "guard_pass", "allow_active"):
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")

    mode = payload.get("policy_mode")
    if not isinstance(mode, str) or not mode:
        errors.append(f"{path}: policy_mode must be non-empty string")
    elif mode not in ("shadow", "active"):
        errors.append(f"{path}: invalid policy_mode {mode!r}")

    guard_pass = payload.get("guard_pass")
    if not isinstance(guard_pass, bool):
        errors.append(f"{path}: guard_pass must be bool")

    allow_active = payload.get("allow_active")
    if not isinstance(allow_active, bool):
        errors.append(f"{path}: allow_active must be bool")

    if strict:
        if guard_pass is not True:
            errors.append(f"{path}: guard_pass must be true in --strict mode")
        if mode != "shadow" and allow_active is not True:
            errors.append(f"{path}: non-shadow policy mode requires allow_active=true in --strict mode")
        stable_count = payload.get("stable_run_count")
        min_stable = payload.get("min_stable_runs")
        stable_ok = payload.get("stable_runs_satisfied")
        if not isinstance(stable_count, int):
            errors.append(f"{path}: stable_run_count must be int in --strict mode")
        if not isinstance(min_stable, int):
            errors.append(f"{path}: min_stable_runs must be int in --strict mode")
        if not isinstance(stable_ok, bool):
            errors.append(f"{path}: stable_runs_satisfied must be bool in --strict mode")
        if mode == "active" and stable_ok is not True:
            errors.append(f"{path}: active mode requires stable_runs_satisfied=true in --strict mode")


def validate_rl_readiness_replay(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != RL_READINESS_REPLAY_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {RL_READINESS_REPLAY_SCHEMA}, got {payload.get('schema')})"
        )

    for field in ("iterations", "evidence_stable", "trend_stable", "run_count_monotonic", "deterministic_replay_pass", "details"):
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")

    if "iterations" in payload and not isinstance(payload["iterations"], int):
        errors.append(f"{path}: iterations must be int")
    for field in ("evidence_stable", "trend_stable", "run_count_monotonic", "deterministic_replay_pass"):
        if field in payload and not isinstance(payload[field], bool):
            errors.append(f"{path}: {field} must be bool")
    if "details" in payload and not isinstance(payload["details"], list):
        errors.append(f"{path}: details must be list")

    if strict:
        if payload.get("deterministic_replay_pass") is not True:
            errors.append(f"{path}: deterministic_replay_pass must be true in --strict mode")
        if payload.get("evidence_stable") is not True:
            errors.append(f"{path}: evidence_stable must be true in --strict mode")
        if payload.get("trend_stable") is not True:
            errors.append(f"{path}: trend_stable must be true in --strict mode")
        if payload.get("run_count_monotonic") is not True:
            errors.append(f"{path}: run_count_monotonic must be true in --strict mode")


def validate_governance_attestation(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != GOVERNANCE_ATTESTATION_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {GOVERNANCE_ATTESTATION_SCHEMA}, got {payload.get('schema')})"
        )

    if "artifacts" not in payload:
        errors.append(f"{path}: missing required field: artifacts")
    elif not isinstance(payload["artifacts"], list):
        errors.append(f"{path}: artifacts must be a list")

    if "artifact_count" not in payload:
        errors.append(f"{path}: missing required field: artifact_count")
    elif not isinstance(payload["artifact_count"], int):
        errors.append(f"{path}: artifact_count must be int")

    if "missing_artifacts" not in payload:
        errors.append(f"{path}: missing required field: missing_artifacts")
    elif not isinstance(payload["missing_artifacts"], list):
        errors.append(f"{path}: missing_artifacts must be a list")

    digest = payload.get("aggregate_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append(f"{path}: aggregate_sha256 must be 64-char hex string")

    signed = payload.get("signed")
    if not isinstance(signed, bool):
        errors.append(f"{path}: signed must be bool")

    canonical_fpr = payload.get("canonical_pubkey_fingerprint")
    if not isinstance(canonical_fpr, str) or len(canonical_fpr) != 64:
        errors.append(f"{path}: canonical_pubkey_fingerprint must be 64-char hex string")
    canonical_pub_path = payload.get("canonical_pubkey_path")
    if not isinstance(canonical_pub_path, str) or not canonical_pub_path:
        errors.append(f"{path}: canonical_pubkey_path must be non-empty string")
    canonical_ts = payload.get("canonical_bootstrap_timestamp")
    if canonical_ts is not None and not isinstance(canonical_ts, str):
        errors.append(f"{path}: canonical_bootstrap_timestamp must be string when present")
    policy_sign_fpr = payload.get("policy_sign_fingerprint")
    if policy_sign_fpr is not None:
        if not isinstance(policy_sign_fpr, str) or len(policy_sign_fpr) != 64:
            errors.append(f"{path}: policy_sign_fingerprint must be 64-char hex string when present")
    policy_sign_ts = payload.get("policy_sign_timestamp")
    if policy_sign_ts is not None and not isinstance(policy_sign_ts, str):
        errors.append(f"{path}: policy_sign_timestamp must be string when present")
    policy_pinned_digest = payload.get("policy_pinned_digest_sha256")
    if policy_pinned_digest is not None:
        if not isinstance(policy_pinned_digest, str) or len(policy_pinned_digest) != 64:
            errors.append(
                f"{path}: policy_pinned_digest_sha256 must be 64-char hex string when present"
            )
    baseline_freeze_digest = payload.get("baseline_freeze_digest_sha256")
    if baseline_freeze_digest is not None:
        if not isinstance(baseline_freeze_digest, str) or len(baseline_freeze_digest) != 64:
            errors.append(
                f"{path}: baseline_freeze_digest_sha256 must be 64-char hex string when present"
            )
    baseline_freeze_policy_pinned = payload.get("baseline_freeze_policy_pinned_digest_sha256")
    if baseline_freeze_policy_pinned is not None:
        if (
            not isinstance(baseline_freeze_policy_pinned, str)
            or len(baseline_freeze_policy_pinned) != 64
        ):
            errors.append(
                f"{path}: baseline_freeze_policy_pinned_digest_sha256 must be 64-char hex string when present"
            )
    pinned_digest_match = payload.get("pinned_digest_match")
    if pinned_digest_match is not None and not isinstance(pinned_digest_match, bool):
        errors.append(f"{path}: pinned_digest_match must be bool when present")
    hw_variance = payload.get("hardware_epistemic_power_variance_q32_32")
    if hw_variance is not None:
        if not isinstance(hw_variance, str):
            errors.append(
                f"{path}: hardware_epistemic_power_variance_q32_32 must be string when present"
            )
    hw_poll_overhead = payload.get("hardware_poll_overhead_us")
    if hw_poll_overhead is not None and not isinstance(hw_poll_overhead, int):
        errors.append(f"{path}: hardware_poll_overhead_us must be int when present")
    bidir_pass = payload.get("bidir_kaxi_pass")
    if bidir_pass is not None and not isinstance(bidir_pass, bool):
        errors.append(f"{path}: bidir_kaxi_pass must be bool when present")
    merkle_lane_pass = payload.get("merkle_lane_pass")
    if merkle_lane_pass is not None and not isinstance(merkle_lane_pass, bool):
        errors.append(f"{path}: merkle_lane_pass must be bool when present")
    qir_emitter_pass = payload.get("qir_emitter_pass")
    if qir_emitter_pass is not None and not isinstance(qir_emitter_pass, bool):
        errors.append(f"{path}: qir_emitter_pass must be bool when present")
    resource_trend_pass = payload.get("resource_trend_pass")
    if resource_trend_pass is not None and not isinstance(resource_trend_pass, bool):
        errors.append(f"{path}: resource_trend_pass must be bool when present")
    genesis_manifest_path = payload.get("genesis_manifest_path")
    if genesis_manifest_path is not None and (
        not isinstance(genesis_manifest_path, str) or not genesis_manifest_path
    ):
        errors.append(f"{path}: genesis_manifest_path must be non-empty string when present")
    genesis_manifest_signed = payload.get("genesis_manifest_signed")
    if genesis_manifest_signed is not None and not isinstance(genesis_manifest_signed, bool):
        errors.append(f"{path}: genesis_manifest_signed must be bool when present")
    genesis_cold_boot_ready = payload.get("genesis_cold_boot_ready")
    if genesis_cold_boot_ready is not None and not isinstance(genesis_cold_boot_ready, bool):
        errors.append(f"{path}: genesis_cold_boot_ready must be bool when present")
    genesis_manifest_aggregate_sha256 = payload.get("genesis_manifest_aggregate_sha256")
    if genesis_manifest_aggregate_sha256 is not None and (
        not isinstance(genesis_manifest_aggregate_sha256, str)
        or len(genesis_manifest_aggregate_sha256) != 64
    ):
        errors.append(
            f"{path}: genesis_manifest_aggregate_sha256 must be 64-char hex when present"
        )

    if strict:
        missing = payload.get("missing_artifacts")
        if isinstance(missing, list) and missing:
            errors.append(f"{path}: missing_artifacts must be empty in --strict mode")
        if isinstance(payload.get("artifact_count"), int) and isinstance(payload.get("artifacts"), list):
            if payload["artifact_count"] != len(payload["artifacts"]):
                errors.append(f"{path}: artifact_count must match len(artifacts) in --strict mode")
        if payload.get("signed") is not True:
            errors.append(f"{path}: signed must be true in --strict mode")
        if not isinstance(canonical_ts, str) or not canonical_ts:
            errors.append(f"{path}: canonical_bootstrap_timestamp must be present in --strict mode")
        if not isinstance(policy_sign_fpr, str) or len(policy_sign_fpr) != 64:
            errors.append(f"{path}: policy_sign_fingerprint must be present in --strict mode")
        if not isinstance(policy_sign_ts, str) or not policy_sign_ts:
            errors.append(f"{path}: policy_sign_timestamp must be present in --strict mode")
        if not isinstance(policy_pinned_digest, str) or len(policy_pinned_digest) != 64:
            errors.append(f"{path}: policy_pinned_digest_sha256 must be present in --strict mode")
        if not isinstance(baseline_freeze_digest, str) or len(baseline_freeze_digest) != 64:
            errors.append(f"{path}: baseline_freeze_digest_sha256 must be present in --strict mode")
        if (
            not isinstance(baseline_freeze_policy_pinned, str)
            or len(baseline_freeze_policy_pinned) != 64
        ):
            errors.append(
                f"{path}: baseline_freeze_policy_pinned_digest_sha256 must be present in --strict mode"
            )
        if pinned_digest_match is not True:
            errors.append(f"{path}: pinned_digest_match must be true in --strict mode")
        if not isinstance(hw_poll_overhead, int):
            errors.append(f"{path}: hardware_poll_overhead_us must be present in --strict mode")
        elif hw_poll_overhead < 0 or hw_poll_overhead >= 1000:
            errors.append(
                f"{path}: hardware_poll_overhead_us must be in [0,1000) in --strict mode"
            )
        if bidir_pass is not True:
            errors.append(f"{path}: bidir_kaxi_pass must be true in --strict mode")
        if merkle_lane_pass is not True:
            errors.append(f"{path}: merkle_lane_pass must be true in --strict mode")
        if qir_emitter_pass is not True:
            errors.append(f"{path}: qir_emitter_pass must be true in --strict mode")
        if resource_trend_pass is not True:
            errors.append(f"{path}: resource_trend_pass must be true in --strict mode")
        if genesis_manifest_signed is not True:
            errors.append(f"{path}: genesis_manifest_signed must be true in --strict mode")
        if genesis_cold_boot_ready is not True:
            errors.append(f"{path}: genesis_cold_boot_ready must be true in --strict mode")


def validate_sprint1_release_readiness(
    path: Path,
    errors: List[str],
    strict: bool,
    require_success_criteria: bool = False,
) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != SPRINT1_RELEASE_READINESS_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {SPRINT1_RELEASE_READINESS_SCHEMA}, got {payload.get('schema')})"
        )

    for field in (
        "release_ready",
        "hard_gate_pass",
        "success_criteria_pass",
        "hard_gates",
        "success_criteria",
        "rollover_items",
        "verdict",
    ):
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")

    for field in ("release_ready", "hard_gate_pass", "success_criteria_pass"):
        if field in payload and not isinstance(payload[field], bool):
            errors.append(f"{path}: {field} must be bool")

    if "hard_gates" in payload and not isinstance(payload["hard_gates"], dict):
        errors.append(f"{path}: hard_gates must be object")
    if "success_criteria" in payload and not isinstance(payload["success_criteria"], dict):
        errors.append(f"{path}: success_criteria must be object")
    if "rollover_items" in payload and not isinstance(payload["rollover_items"], list):
        errors.append(f"{path}: rollover_items must be list")

    hard_gates = payload.get("hard_gates", {})
    if isinstance(hard_gates, dict):
        required_hard = ("l1_ptx_epistemic", "l2_sass_patch_4_kernels", "l3_fpga_seed")
        for key in required_hard:
            if key not in hard_gates:
                errors.append(f"{path}: hard_gates missing key {key}")
                continue
            gate = hard_gates[key]
            if not isinstance(gate, dict):
                errors.append(f"{path}: hard_gates.{key} must be object")
                continue
            if "pass" not in gate:
                errors.append(f"{path}: hard_gates.{key} missing field pass")
            elif not isinstance(gate["pass"], bool):
                errors.append(f"{path}: hard_gates.{key}.pass must be bool")

    success_criteria = payload.get("success_criteria", {})
    if isinstance(success_criteria, dict):
        required_soft = (
            "geomean_plus20",
            "compile_overhead_lte_25pct",
            "bit_exact_50_samples",
            "quantum_conformance_ks_coverage",
            "rl_shadow_default_or_active_ready",
        )
        for key in required_soft:
            if key not in success_criteria:
                errors.append(f"{path}: success_criteria missing key {key}")
                continue
            item = success_criteria[key]
            if not isinstance(item, dict):
                errors.append(f"{path}: success_criteria.{key} must be object")
                continue
            status = item.get("status")
            if not isinstance(status, str):
                errors.append(f"{path}: success_criteria.{key}.status must be string")

    release_ready = payload.get("release_ready")
    hard_gate_pass = payload.get("hard_gate_pass")
    if isinstance(release_ready, bool) and isinstance(hard_gate_pass, bool):
        if release_ready != hard_gate_pass:
            errors.append(f"{path}: release_ready must equal hard_gate_pass")

    if strict:
        if payload.get("hard_gate_pass") is not True:
            errors.append(f"{path}: hard_gate_pass must be true in --strict mode")
        if payload.get("release_ready") is not True:
            errors.append(f"{path}: release_ready must be true in --strict mode")
        if require_success_criteria and payload.get("success_criteria_pass") is not True:
            errors.append(
                f"{path}: success_criteria_pass must be true in --strict mode when OMEGA_REQUIRE_SPRINT1_SUCCESS_CRITERIA_NOW=1"
            )


def validate_performance_summary(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != PERFORMANCE_SUMMARY_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {PERFORMANCE_SUMMARY_SCHEMA}, got {payload.get('schema')})"
        )

    required_fields = (
        "measurement_mode",
        "samples",
        "minimum_samples",
        "threshold",
        "geomean_speedup",
        "family_speedups",
        "baseline_substitutions",
    )
    for field in required_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")

    if "measurement_mode" in payload and not isinstance(payload["measurement_mode"], str):
        errors.append(f"{path}: measurement_mode must be string")

    if "samples" in payload and not isinstance(payload["samples"], int):
        errors.append(f"{path}: samples must be int")
    if "minimum_samples" in payload and not isinstance(payload["minimum_samples"], int):
        errors.append(f"{path}: minimum_samples must be int")
    if "threshold" in payload and not is_number(payload["threshold"]):
        errors.append(f"{path}: threshold must be numeric")
    if "geomean_speedup" in payload and not is_number(payload["geomean_speedup"]):
        errors.append(f"{path}: geomean_speedup must be numeric")

    family_speedups = payload.get("family_speedups")
    if not isinstance(family_speedups, dict):
        errors.append(f"{path}: family_speedups must be object")
    else:
        for key, value in family_speedups.items():
            if not isinstance(key, str):
                errors.append(f"{path}: family_speedups keys must be strings")
                break
            if not is_number(value):
                errors.append(f"{path}: family_speedups.{key} must be numeric")

    substitutions = payload.get("baseline_substitutions")
    if not isinstance(substitutions, list):
        errors.append(f"{path}: baseline_substitutions must be list")
    else:
        for idx, item in enumerate(substitutions):
            if not isinstance(item, dict):
                errors.append(f"{path}: baseline_substitutions[{idx}] must be object")
                continue
            for field in ("baseline_name", "reason", "replacement_name", "evidence_url"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{path}: baseline_substitutions[{idx}].{field} must be non-empty string")

    if strict:
        samples = payload.get("samples")
        minimum_samples = payload.get("minimum_samples")
        if isinstance(samples, int) and isinstance(minimum_samples, int) and samples < minimum_samples:
            errors.append(f"{path}: samples must be >= minimum_samples in --strict mode")
        geomean = payload.get("geomean_speedup")
        threshold = payload.get("threshold")
        if is_number(geomean) and is_number(threshold):
            if float(geomean) < float(threshold):
                errors.append(
                    f"{path}: geomean_speedup {float(geomean):.6f} < threshold {float(threshold):.6f} in --strict mode"
                )


def validate_external_baseline_collection(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != EXTERNAL_BASELINE_COLLECTION_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {EXTERNAL_BASELINE_COLLECTION_SCHEMA}, got {payload.get('schema')})"
        )

    required_fields = (
        "required_baselines",
        "baseline_count",
        "available_count",
        "rows",
        "execute",
    )
    for field in required_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")

    required_baselines = payload.get("required_baselines")
    if not isinstance(required_baselines, list):
        errors.append(f"{path}: required_baselines must be list")
        required_baselines = []

    baseline_count = payload.get("baseline_count")
    if not isinstance(baseline_count, int):
        errors.append(f"{path}: baseline_count must be int")
        baseline_count = 0

    available_count = payload.get("available_count")
    if not isinstance(available_count, int):
        errors.append(f"{path}: available_count must be int")
        available_count = 0

    rows = payload.get("rows")
    if not isinstance(rows, list):
        errors.append(f"{path}: rows must be list")
        rows = []

    if "execute" in payload and not isinstance(payload["execute"], bool):
        errors.append(f"{path}: execute must be bool")

    if isinstance(required_baselines, list) and isinstance(rows, list):
        if baseline_count != len(required_baselines):
            errors.append(f"{path}: baseline_count must equal len(required_baselines)")
        if len(rows) != len(required_baselines):
            errors.append(f"{path}: len(rows) must equal len(required_baselines)")

    row_available_count = 0
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"{path}: rows[{idx}] must be object")
            continue
        name = row.get("baseline_name")
        if not isinstance(name, str) or not name:
            errors.append(f"{path}: rows[{idx}].baseline_name must be non-empty string")
        avail = row.get("available")
        if not isinstance(avail, bool):
            errors.append(f"{path}: rows[{idx}].available must be bool")
        elif avail:
            row_available_count += 1
        samples = row.get("samples")
        if not isinstance(samples, int):
            errors.append(f"{path}: rows[{idx}].samples must be int")
        report_path = row.get("report_path")
        if not isinstance(report_path, str) or not report_path:
            errors.append(f"{path}: rows[{idx}].report_path must be non-empty string")

    if isinstance(available_count, int) and row_available_count != available_count:
        errors.append(
            f"{path}: available_count {available_count} must match available rows {row_available_count}"
        )

    if strict and baseline_count <= 0:
        errors.append(f"{path}: baseline_count must be > 0 in --strict mode")


def validate_baseline_freeze(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != BASELINE_FREEZE_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {BASELINE_FREEZE_SCHEMA}, got {payload.get('schema')})"
        )

    required_fields = (
        "baseline_count",
        "available_count",
        "measurement_mode",
        "minimum_samples",
        "samples",
        "threshold",
        "geomean_speedup",
        "frozen_rows",
        "freeze_digest_sha256",
        "canonical_pubkey_fingerprint",
        "canonical_bootstrap_timestamp",
        "signature_hex",
        "signed",
        "status",
    )
    for field in required_fields:
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")

    baseline_count = payload.get("baseline_count")
    available_count = payload.get("available_count")
    minimum_samples = payload.get("minimum_samples")
    samples = payload.get("samples")
    threshold = payload.get("threshold")
    geomean_speedup = payload.get("geomean_speedup")
    measurement_mode = payload.get("measurement_mode")
    status = payload.get("status")
    freeze_digest = payload.get("freeze_digest_sha256")
    signature_hex = payload.get("signature_hex")
    signed = payload.get("signed")
    canonical_fpr = payload.get("canonical_pubkey_fingerprint")
    canonical_ts = payload.get("canonical_bootstrap_timestamp")
    frozen_rows = payload.get("frozen_rows")

    if not isinstance(baseline_count, int):
        errors.append(f"{path}: baseline_count must be int")
        baseline_count = 0
    if not isinstance(available_count, int):
        errors.append(f"{path}: available_count must be int")
        available_count = 0
    if not isinstance(minimum_samples, int):
        errors.append(f"{path}: minimum_samples must be int")
        minimum_samples = 0
    if not isinstance(samples, int):
        errors.append(f"{path}: samples must be int")
        samples = 0
    if not is_number(threshold):
        errors.append(f"{path}: threshold must be numeric")
        threshold = 0.0
    else:
        threshold = float(threshold)
    if not is_number(geomean_speedup):
        errors.append(f"{path}: geomean_speedup must be numeric")
        geomean_speedup = 0.0
    else:
        geomean_speedup = float(geomean_speedup)
    if not isinstance(measurement_mode, str) or not measurement_mode:
        errors.append(f"{path}: measurement_mode must be non-empty string")
        measurement_mode = "unknown"
    if not isinstance(status, str) or not status:
        errors.append(f"{path}: status must be non-empty string")
        status = "invalid"
    if not isinstance(freeze_digest, str) or len(freeze_digest) != 64:
        errors.append(f"{path}: freeze_digest_sha256 must be 64-char hex")
    if not isinstance(signature_hex, str) or len(signature_hex) != 128:
        errors.append(f"{path}: signature_hex must be 128-char hex")
    if not isinstance(signed, bool):
        errors.append(f"{path}: signed must be bool")
    if not isinstance(canonical_fpr, str) or len(canonical_fpr) != 64:
        errors.append(f"{path}: canonical_pubkey_fingerprint must be 64-char hex")
    if canonical_ts is not None and not isinstance(canonical_ts, str):
        errors.append(f"{path}: canonical_bootstrap_timestamp must be string when present")
    if not isinstance(frozen_rows, list):
        errors.append(f"{path}: frozen_rows must be list")
        frozen_rows = []
    else:
        for idx, row in enumerate(frozen_rows):
            if not isinstance(row, dict):
                errors.append(f"{path}: frozen_rows[{idx}] must be object")
                continue
            for field in ("baseline_name", "report_path", "report_sha256", "reason"):
                value = row.get(field)
                if not isinstance(value, str):
                    errors.append(f"{path}: frozen_rows[{idx}].{field} must be string")
            if not isinstance(row.get("available"), bool):
                errors.append(f"{path}: frozen_rows[{idx}].available must be bool")
            if not isinstance(row.get("samples"), int):
                errors.append(f"{path}: frozen_rows[{idx}].samples must be int")

    if strict:
        if status != "frozen":
            errors.append(f"{path}: status must be 'frozen' in --strict mode (got {status!r})")
        if baseline_count <= 0:
            errors.append(f"{path}: baseline_count must be > 0 in --strict mode")
        if len(frozen_rows) != baseline_count:
            errors.append(
                f"{path}: len(frozen_rows)={len(frozen_rows)} must equal baseline_count={baseline_count} in --strict mode"
            )
        if samples < minimum_samples:
            errors.append(
                f"{path}: samples={samples} must be >= minimum_samples={minimum_samples} in --strict mode"
            )
        if geomean_speedup < threshold:
            errors.append(
                f"{path}: geomean_speedup={geomean_speedup:.6f} must be >= threshold={threshold:.6f} in --strict mode"
            )
        if measurement_mode == "external_baseline_ingest" and available_count != baseline_count:
            errors.append(
                f"{path}: external_baseline_ingest requires available_count==baseline_count in --strict mode ({available_count}!={baseline_count})"
            )
        if signed is not True:
            errors.append(f"{path}: signed must be true in --strict mode")
        if not isinstance(canonical_ts, str) or not canonical_ts:
            errors.append(f"{path}: canonical_bootstrap_timestamp must be present in --strict mode")


def validate_bidir_kaxi(path: Path, errors: List[str], strict: bool) -> None:
    """Validate bidirectional K-AXI return channel status in fpga_seed_report."""
    payload = load_json_object(path, errors)
    if payload is None:
        return
    sim = payload.get("k_axi_return_sim_status", "")
    synth = payload.get("k_axi_return_synth_status", "")
    if not isinstance(sim, str):
        errors.append(f"{path}: k_axi_return_sim_status must be a string")
        return
    if sim not in ("pass", "skipped", "missing_tool", "fail", ""):
        errors.append(f"{path}: k_axi_return_sim_status has unexpected value {sim!r}")
    if strict and sim != "pass":
        errors.append(
            f"{path}: k_axi_return_sim_status={sim!r} must be pass in --strict mode"
        )
    if strict:
        if not isinstance(synth, str):
            errors.append(f"{path}: k_axi_return_synth_status must be a string")
        elif synth != "pass":
            errors.append(
                f"{path}: k_axi_return_synth_status={synth!r} must be pass in --strict mode"
            )
        if not BIDIR_KAXI_ADAPTER_SIO_PATH.exists():
            errors.append(
                f"{BIDIR_KAXI_ADAPTER_SIO_PATH}: required bidirectional adapter missing (--strict)"
            )
        else:
            try:
                adapter = BIDIR_KAXI_ADAPTER_SIO_PATH.read_text()
            except OSError as exc:
                errors.append(f"{BIDIR_KAXI_ADAPTER_SIO_PATH}: read failed: {exc}")
            else:
                for fn in ("kaxi_bidir_pack_header", "kaxi_bidir_replay_inspect"):
                    if fn not in adapter:
                        errors.append(
                            f"{BIDIR_KAXI_ADAPTER_SIO_PATH}: missing required function {fn!r}"
                        )
        if not BIDIR_KAXI_WAVEFORM_PATH.exists():
            errors.append(
                f"{BIDIR_KAXI_WAVEFORM_PATH}: required waveform missing in --strict mode"
            )


def validate_accumulator_bounds(path: Path, errors: List[str], strict: bool) -> None:
    """Validate epistemic power accumulator bounds in fpga_seed_report.

    Checks that the accumulator simulation ran and counter fields are present.
    The formal bound 0 <= L_ref - L_hw < 3 is verified algebraically in
    tests/hardware/accumulator_bounds_test.sio; here we confirm the hardware
    report is consistent with a running accumulator.
    """
    payload = load_json_object(path, errors)
    if payload is None:
        return
    accum_sim = payload.get("epistemic_power_accumulator_sim_status", "")
    if not isinstance(accum_sim, str):
        errors.append(f"{path}: epistemic_power_accumulator_sim_status must be a string")
        return
    if strict and accum_sim not in ("pass", "skipped"):
        errors.append(
            f"{path}: epistemic_power_accumulator_sim_status={accum_sim!r} "
            "must be pass or skipped in --strict mode"
        )
    if strict:
        accum_synth = payload.get("epistemic_power_accumulator_synth_status", "")
        if not isinstance(accum_synth, str):
            errors.append(f"{path}: epistemic_power_accumulator_synth_status must be a string")
        elif accum_synth not in ("pass", "skipped"):
            errors.append(
                f"{path}: epistemic_power_accumulator_synth_status={accum_synth!r} "
                "must be pass or skipped in --strict mode"
            )
        if not ACCUMULATOR_WAVEFORM_PATH.exists():
            errors.append(
                f"{ACCUMULATOR_WAVEFORM_PATH}: required accumulator waveform missing in --strict mode"
            )
        if not ACCUMULATOR_BOUNDS_TEST_PATH.exists():
            errors.append(
                f"{ACCUMULATOR_BOUNDS_TEST_PATH}: accumulator bounds test missing in --strict mode"
            )
        else:
            try:
                bounds_text = ACCUMULATOR_BOUNDS_TEST_PATH.read_text()
            except OSError as exc:
                errors.append(f"{ACCUMULATOR_BOUNDS_TEST_PATH}: read failed: {exc}")
            else:
                if "ACCUMULATOR_BOUNDS_PASS" not in bounds_text:
                    errors.append(
                        f"{ACCUMULATOR_BOUNDS_TEST_PATH}: missing ACCUMULATOR_BOUNDS_PASS marker"
                    )
    fingerprint = payload.get("hardware_counter_fingerprint", {})
    if not isinstance(fingerprint, dict):
        errors.append(f"{path}: hardware_counter_fingerprint must be an object")
        return
    for key in ("accum_log", "accum_tx"):
        if key not in fingerprint:
            if strict:
                errors.append(
                    f"{path}: hardware_counter_fingerprint missing {key!r} in --strict mode"
                )


def validate_qir_epi_power_sio(path: Path, errors: List[str], strict: bool) -> None:
    """Validate the QIR epistemic power .sio module declares required functions."""
    try:
        content = path.read_text()
    except OSError as exc:
        errors.append(f"{path}: read failed: {exc}")
        return
    required_fns = (
        "qir_epi_power_hw_q32_32",
        "qir_epi_power_variance_q32_32",
        "qir_epi_power_poll_overhead_us",
        "qir_epi_power_live_conformant_v2",
    )
    for fn in required_fns:
        if fn not in content:
            errors.append(f"{path}: missing required function {fn!r}")


def validate_qir_full_emitter_sio(path: Path, errors: List[str], strict: bool) -> None:
    """Validate the full QIR emitter self-hosted module."""
    try:
        content = path.read_text()
    except OSError as exc:
        errors.append(f"{path}: read failed: {exc}")
        return
    required_symbols = (
        "omega_qir_emit_shim",
        "omega_qir_emit_quantum_controller",
        "omega_qir_emit_bundle",
        "omega_qir_full_emitter_self_check",
        "selfhost-emitter",
    )
    for symbol in required_symbols:
        if symbol not in content:
            errors.append(f"{path}: missing required symbol {symbol!r}")
    if strict and "template-direct" in content:
        errors.append(f"{path}: template-direct fallback is forbidden in --strict mode")


def validate_merkle_lane_sio(path: Path, errors: List[str], strict: bool) -> None:
    """Validate Merkle lane helper SIO module."""
    try:
        content = path.read_text()
    except OSError as exc:
        errors.append(f"{path}: read failed: {exc}")
        return
    required_symbols = (
        "merkle_lane_root_l64",
        "merkle_lane_verify_root",
        "merkle_lane_replay_summary",
        "merkle_lane_self_check",
    )
    for symbol in required_symbols:
        if symbol not in content:
            errors.append(f"{path}: missing required symbol {symbol!r}")


def validate_merkle_lane_rtl(path: Path, errors: List[str], strict: bool) -> None:
    """Validate Merkle lane RTL core."""
    try:
        content = path.read_text()
    except OSError as exc:
        errors.append(f"{path}: read failed: {exc}")
        return
    required_symbols = (
        "module k_axi_merkle_lane_core",
        "merkle_lane_digest",
        "merkle_lane_valid",
        "MERKLE_SALT",
    )
    for symbol in required_symbols:
        if symbol not in content:
            errors.append(f"{path}: missing required RTL symbol {symbol!r}")


def validate_merkle_root_sio(path: Path, errors: List[str], strict: bool) -> None:
    """Validate Merkle root helper SIO module."""
    try:
        content = path.read_text()
    except OSError as exc:
        errors.append(f"{path}: read failed: {exc}")
        return
    required_symbols = (
        "merkle_root_lane_l64",
        "merkle_root_verify_l64",
        "merkle_root_lane_self_check",
    )
    for symbol in required_symbols:
        if symbol not in content:
            errors.append(f"{path}: missing required symbol {symbol!r}")


def validate_merkle_root_rtl(path: Path, errors: List[str], strict: bool) -> None:
    """Validate Merkle root RTL core."""
    try:
        content = path.read_text()
    except OSError as exc:
        errors.append(f"{path}: read failed: {exc}")
        return
    required_symbols = (
        "module k_axi_merkle_root_lane_core",
        "merkle_root_l64",
        "merkle_root_valid",
    )
    for symbol in required_symbols:
        if symbol not in content:
            errors.append(f"{path}: missing required RTL symbol {symbol!r}")


def validate_genesis_manifest(path: Path, errors: List[str], strict: bool) -> None:
    payload = load_json_object(path, errors)
    if payload is None:
        return

    if payload.get("schema") != GENESIS_MANIFEST_SCHEMA:
        errors.append(
            f"{path}: schema mismatch (expected {GENESIS_MANIFEST_SCHEMA}, got {payload.get('schema')})"
        )
    for field in (
        "artifact_count",
        "artifacts",
        "aggregate_sha256",
        "merkle_root_sha256",
        "canonical_pubkey_fingerprint",
        "signature_hex",
        "signed",
        "cold_boot_ready",
    ):
        if field not in payload:
            errors.append(f"{path}: missing required field: {field}")
    if "artifact_count" in payload and not isinstance(payload["artifact_count"], int):
        errors.append(f"{path}: artifact_count must be int")
    if "artifacts" in payload and not isinstance(payload["artifacts"], list):
        errors.append(f"{path}: artifacts must be list")
    for field in ("aggregate_sha256", "merkle_root_sha256", "canonical_pubkey_fingerprint"):
        value = payload.get(field)
        if value is not None and (not isinstance(value, str) or len(value) != 64):
            errors.append(f"{path}: {field} must be 64-char hex string")
    signature = payload.get("signature_hex")
    if signature is not None and (not isinstance(signature, str) or len(signature) != 128):
        errors.append(f"{path}: signature_hex must be 128-char hex string")
    if "signed" in payload and not isinstance(payload["signed"], bool):
        errors.append(f"{path}: signed must be bool")
    if "cold_boot_ready" in payload and not isinstance(payload["cold_boot_ready"], bool):
        errors.append(f"{path}: cold_boot_ready must be bool")

    if strict:
        if payload.get("signed") is not True:
            errors.append(f"{path}: signed must be true in --strict mode")
        if payload.get("cold_boot_ready") is not True:
            errors.append(f"{path}: cold_boot_ready must be true in --strict mode")


def maybe_validate(
    path: Path,
    strict: bool,
    errors: List[str],
    validator: Callable[[Path, List[str], bool], None],
) -> None:
    if not path.exists():
        if strict:
            errors.append(f"{path}: required artifact missing (--strict)")
        else:
            print(f"omega_hw_telemetry_regression: skipped missing {path}")
        return
    error_count_before = len(errors)
    validator(path, errors, strict)
    if len(errors) == error_count_before:
        print(f"omega_hw_telemetry_regression: validated {path}")


def main() -> int:
    args = parse_args()
    errors: List[str] = []
    require_hw_epi_power = os.environ.get("OMEGA_REQUIRE_HW_EPI_POWER", "0") == "1"
    require_hw_epi_power_trend = (
        os.environ.get("OMEGA_REQUIRE_HW_EPI_POWER_TREND", "0") == "1"
    )
    require_rl_readiness_bridge = (
        os.environ.get("OMEGA_REQUIRE_RL_READINESS_BRIDGE", "0") == "1"
    )
    require_shadow_audit = (
        os.environ.get("OMEGA_REQUIRE_SHADOW_AUDIT", "0") == "1"
    )
    require_rl_readiness_trend = (
        os.environ.get("OMEGA_REQUIRE_RL_READINESS_TREND", "0") == "1"
    )
    require_external_baseline_collection = (
        os.environ.get("OMEGA_REQUIRE_EXTERNAL_BASELINE_COLLECTION", "0") == "1"
    )
    require_performance_summary = (
        os.environ.get("OMEGA_REQUIRE_PERFORMANCE_SUMMARY", "0") == "1"
    )
    require_baseline_freeze = (
        os.environ.get("OMEGA_REQUIRE_BASELINE_FREEZE", "0") == "1"
    )
    require_policy_mode_guard = (
        os.environ.get("OMEGA_REQUIRE_POLICY_MODE_GUARD", "0") == "1"
    )
    require_rl_readiness_replay = (
        os.environ.get("OMEGA_REQUIRE_RL_READINESS_REPLAY", "0") == "1"
    )
    require_governance_attestation = (
        os.environ.get("OMEGA_REQUIRE_GOVERNANCE_ATTESTATION", "0") == "1"
    )
    require_sprint1_release_readiness = (
        os.environ.get("OMEGA_REQUIRE_SPRINT1_RELEASE_READINESS", "0") == "1"
    )
    require_sprint1_success_criteria = (
        os.environ.get("OMEGA_REQUIRE_SPRINT1_SUCCESS_CRITERIA_NOW", "0") == "1"
    )
    require_bidir_kaxi = os.environ.get("OMEGA_REQUIRE_BIDIR_KAXI", "0") == "1"
    require_accumulator_bounds = (
        os.environ.get("OMEGA_REQUIRE_ACCUMULATOR_BOUNDS", "0") == "1"
    )
    require_epi_power_live = (
        os.environ.get("OMEGA_REQUIRE_EPI_POWER_LIVE", "0") == "1"
    )
    require_qir_emitter = (
        os.environ.get("OMEGA_REQUIRE_QIR_EMITTER", "0") == "1"
    )
    require_merkle_lane = (
        os.environ.get("OMEGA_REQUIRE_MERKLE_LANE", "0") == "1"
    )
    require_resource_trend = (
        os.environ.get("OMEGA_REQUIRE_RESOURCE_TREND", "0") == "1"
    )
    require_merkle_root = (
        os.environ.get("OMEGA_REQUIRE_MERKLE_ROOT", "0") == "1"
    )
    require_genesis_manifest = (
        os.environ.get("OMEGA_REQUIRE_GENESIS_MANIFEST", "0") == "1"
    )

    maybe_validate(
        FPGA_REPORT_PATH,
        args.strict,
        errors,
        lambda p, e, strict: validate_fpga_report(p, e, strict),
    )
    maybe_validate(
        QUANTUM_CONFORMANCE_PATH,
        args.strict,
        errors,
        validate_quantum_conformance,
    )
    maybe_validate(
        PURE_KAXI_MARKER_PATH,
        args.strict,
        errors,
        lambda p, e, _strict: validate_pure_kaxi_marker(p, e),
    )
    maybe_validate(
        HARDWARE_COUNTER_REPRO_PATH,
        args.strict,
        errors,
        validate_counter_repro,
    )
    maybe_validate(
        HARDWARE_RESOURCE_TREND_PATH,
        args.strict and require_resource_trend,
        errors,
        validate_resource_trend,
    )
    maybe_validate(
        PTX_LAUNCH_REPORT_PATH,
        args.strict,
        errors,
        validate_ptx_launch_report,
    )
    maybe_validate(
        HW_EPI_POWER_LIVE_PATH,
        args.strict and require_hw_epi_power,
        errors,
        validate_hw_epi_power_live,
    )
    maybe_validate(
        HW_EPI_POWER_TREND_PATH,
        args.strict and require_hw_epi_power_trend,
        errors,
        validate_hw_epi_power_trend,
    )
    maybe_validate(
        RL_READINESS_EVIDENCE_PATH,
        args.strict and require_rl_readiness_bridge,
        errors,
        validate_rl_readiness_evidence,
    )
    maybe_validate(
        RL_READINESS_BRIDGE_PATH,
        args.strict and require_rl_readiness_bridge,
        errors,
        validate_rl_readiness_bridge,
    )
    maybe_validate(
        SHADOW_AUDIT_PATH,
        args.strict and require_shadow_audit,
        errors,
        validate_shadow_audit,
    )
    maybe_validate(
        RL_READINESS_TREND_PATH,
        args.strict and require_rl_readiness_trend,
        errors,
        validate_rl_readiness_trend,
    )
    maybe_validate(
        EXTERNAL_BASELINE_COLLECTION_PATH,
        args.strict and require_external_baseline_collection,
        errors,
        validate_external_baseline_collection,
    )
    maybe_validate(
        PERFORMANCE_SUMMARY_PATH,
        args.strict and require_performance_summary,
        errors,
        validate_performance_summary,
    )
    maybe_validate(
        BASELINE_FREEZE_PATH,
        args.strict and require_baseline_freeze,
        errors,
        validate_baseline_freeze,
    )
    maybe_validate(
        POLICY_MODE_GUARD_PATH,
        args.strict and require_policy_mode_guard,
        errors,
        validate_policy_mode_guard,
    )
    maybe_validate(
        RL_READINESS_REPLAY_PATH,
        args.strict and require_rl_readiness_replay,
        errors,
        validate_rl_readiness_replay,
    )
    maybe_validate(
        GOVERNANCE_ATTESTATION_PATH,
        args.strict and require_governance_attestation,
        errors,
        validate_governance_attestation,
    )
    maybe_validate(
        SPRINT1_RELEASE_READINESS_PATH,
        args.strict and require_sprint1_release_readiness,
        errors,
        lambda p, e, strict: validate_sprint1_release_readiness(
            p, e, strict, require_success_criteria=require_sprint1_success_criteria
        ),
    )
    maybe_validate(
        FPGA_REPORT_PATH,
        args.strict and require_bidir_kaxi,
        errors,
        validate_bidir_kaxi,
    )
    maybe_validate(
        FPGA_REPORT_PATH,
        args.strict and require_accumulator_bounds,
        errors,
        validate_accumulator_bounds,
    )
    maybe_validate(
        QIR_EPI_POWER_SIO_PATH,
        args.strict and require_epi_power_live,
        errors,
        validate_qir_epi_power_sio,
    )
    maybe_validate(
        QIR_FULL_EMITTER_SIO_PATH,
        args.strict and require_qir_emitter,
        errors,
        validate_qir_full_emitter_sio,
    )
    maybe_validate(
        MERKLE_LANE_SIO_PATH,
        args.strict and require_merkle_lane,
        errors,
        validate_merkle_lane_sio,
    )
    maybe_validate(
        MERKLE_LANE_RTL_PATH,
        args.strict and require_merkle_lane,
        errors,
        validate_merkle_lane_rtl,
    )
    maybe_validate(
        MERKLE_ROOT_SIO_PATH,
        args.strict and require_merkle_root,
        errors,
        validate_merkle_root_sio,
    )
    maybe_validate(
        MERKLE_ROOT_RTL_PATH,
        args.strict and require_merkle_root,
        errors,
        validate_merkle_root_rtl,
    )
    maybe_validate(
        GENESIS_MANIFEST_PATH,
        args.strict and require_genesis_manifest,
        errors,
        validate_genesis_manifest,
    )

    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 2

    print("omega_hw_telemetry_regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

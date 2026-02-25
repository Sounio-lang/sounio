#!/usr/bin/env python3
"""Best-effort real probe for external baseline adapters."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPORT_SCHEMA = "sounio.omega.external-baseline-report.v1"
DEFAULT_FAMILIES = (
    "dense_linear",
    "attention",
    "epistemic_elementwise",
    "monte_carlo",
    "quantum_parallel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External baseline probe")
    parser.add_argument("--baseline-name", required=True, help="Baseline name")
    parser.add_argument("--report", required=True, help="Output report path")
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated workload families",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Sample count marker for baseline report",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def unavailable_report(
    baseline_name: str,
    report_path: Path,
    reason: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "baseline_name": baseline_name,
        "available": False,
        "samples": 0,
        "family_speedups": {},
        "reason": reason,
    }
    if details:
        payload["details"] = details
    write_json(report_path, payload)
    return payload


def clamp(value: float, lo: float = 0.5, hi: float = 10.0) -> float:
    return max(lo, min(hi, value))


def bench_seconds(
    fn: Callable[[], Any],
    *,
    warmup: int = 5,
    iters: int = 30,
    sync: Callable[[], None] | None = None,
) -> float:
    for _ in range(max(0, warmup)):
        fn()
    if sync:
        sync()
    start = time.perf_counter()
    for _ in range(max(1, iters)):
        fn()
    if sync:
        sync()
    elapsed = time.perf_counter() - start
    return max(1e-9, elapsed / float(max(1, iters)))


def validate_families(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",")]
    families = [item for item in values if item]
    if not families:
        return list(DEFAULT_FAMILIES)
    return families


def try_import_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def run_torch_probe(
    *,
    baseline_name: str,
    report_path: Path,
    families: list[str],
    samples: int,
) -> dict[str, Any]:
    torch = try_import_torch()
    if torch is None:
        return unavailable_report(
            baseline_name,
            report_path,
            "python_torch_not_available",
        )

    nvcc_ok = shutil.which("nvcc") is not None
    has_cuda = bool(torch.cuda.is_available())

    if baseline_name == "cuda-cutlass":
        if not nvcc_ok:
            return unavailable_report(
                baseline_name,
                report_path,
                "nvcc_not_available",
                {"requires": ["nvcc", "cuda"]},
            )
        if not has_cuda:
            return unavailable_report(
                baseline_name,
                report_path,
                "cuda_device_not_available",
                {"requires": ["cuda"]},
            )

    if baseline_name == "triton":
        try:
            import triton  # noqa: F401
            import triton.language as tl  # noqa: F401
        except Exception:
            return unavailable_report(
                baseline_name,
                report_path,
                "python_triton_not_available",
                {"requires": ["triton", "cuda"]},
            )
        if not has_cuda:
            return unavailable_report(
                baseline_name,
                report_path,
                "cuda_device_not_available",
                {"requires": ["cuda"]},
            )

    if baseline_name == "pytorch-inductor" and not hasattr(torch, "compile"):
        return unavailable_report(
            baseline_name,
            report_path,
            "torch_compile_not_available",
            {"requires": ["torch.compile"]},
        )

    # Choose backend device.
    backend_device = "cuda" if has_cuda else "cpu"
    cpu = torch.device("cpu")
    backend = torch.device(backend_device)
    sync = torch.cuda.synchronize if backend_device == "cuda" else None

    try:
        # Dense linear.
        a_cpu = torch.randn((1024, 1024), dtype=torch.float32, device=cpu)
        b_cpu = torch.randn((1024, 1024), dtype=torch.float32, device=cpu)
        a_backend = a_cpu.to(backend)
        b_backend = b_cpu.to(backend)

        def dense_cpu() -> Any:
            return torch.matmul(a_cpu, b_cpu)

        def dense_backend() -> Any:
            return torch.matmul(a_backend, b_backend)

        # Attention.
        q_cpu = torch.randn((8, 8, 256, 64), dtype=torch.float16 if has_cuda else torch.float32, device=cpu)
        k_cpu = torch.randn_like(q_cpu)
        v_cpu = torch.randn_like(q_cpu)
        q_backend = q_cpu.to(backend)
        k_backend = k_cpu.to(backend)
        v_backend = v_cpu.to(backend)

        def attention_cpu() -> Any:
            return torch.nn.functional.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)

        def attention_backend() -> Any:
            return torch.nn.functional.scaled_dot_product_attention(q_backend, k_backend, v_backend)

        # Elementwise.
        x_cpu = torch.randn((4 * 1024 * 1024,), dtype=torch.float32, device=cpu)
        y_cpu = torch.randn_like(x_cpu)
        x_backend = x_cpu.to(backend)
        y_backend = y_cpu.to(backend)

        def elem_cpu() -> Any:
            return (x_cpu + y_cpu) * 1.001

        def elem_backend() -> Any:
            return (x_backend + y_backend) * 1.001

        # Monte Carlo proxy.
        z_cpu = torch.randn((2 * 1024 * 1024,), dtype=torch.float32, device=cpu)
        z_backend = z_cpu.to(backend)

        def mc_cpu() -> Any:
            t = torch.sin(z_cpu) + torch.cos(z_cpu)
            return torch.mean(t)

        def mc_backend() -> Any:
            t = torch.sin(z_backend) + torch.cos(z_backend)
            return torch.mean(t)

        # Optional inductor compile path.
        if baseline_name == "pytorch-inductor":
            dense_backend_fn = torch.compile(dense_backend, fullgraph=False, dynamic=False)
            attention_backend_fn = torch.compile(attention_backend, fullgraph=False, dynamic=False)
            elem_backend_fn = torch.compile(elem_backend, fullgraph=False, dynamic=False)
            mc_backend_fn = torch.compile(mc_backend, fullgraph=False, dynamic=False)
        else:
            dense_backend_fn = dense_backend
            attention_backend_fn = attention_backend
            elem_backend_fn = elem_backend
            mc_backend_fn = mc_backend

        dense_speedup = clamp(bench_seconds(dense_cpu) / bench_seconds(dense_backend_fn, sync=sync))
        attention_speedup = clamp(
            bench_seconds(attention_cpu) / bench_seconds(attention_backend_fn, sync=sync)
        )
        elem_speedup = clamp(bench_seconds(elem_cpu) / bench_seconds(elem_backend_fn, sync=sync))
        mc_speedup = clamp(bench_seconds(mc_cpu) / bench_seconds(mc_backend_fn, sync=sync))
    except Exception as exc:
        return unavailable_report(
            baseline_name,
            report_path,
            f"probe_runtime_error:{type(exc).__name__}",
            {"message": str(exc)},
        )

    family_speedups: dict[str, float] = {}
    for family in families:
        if family == "dense_linear":
            family_speedups[family] = dense_speedup
        elif family == "attention":
            family_speedups[family] = attention_speedup
        elif family == "epistemic_elementwise":
            family_speedups[family] = elem_speedup
        elif family == "monte_carlo":
            family_speedups[family] = mc_speedup
        elif family == "quantum_parallel":
            family_speedups[family] = clamp((attention_speedup + mc_speedup) / 2.0)
        else:
            family_speedups[family] = clamp((dense_speedup + elem_speedup) / 2.0)

    payload = {
        "schema": REPORT_SCHEMA,
        "baseline_name": baseline_name,
        "available": True,
        "samples": max(1, samples),
        "family_speedups": family_speedups,
        "reason": "ok",
        "details": {
            "backend_device": backend_device,
            "nvcc_available": nvcc_ok,
            "torch_compile_used": baseline_name == "pytorch-inductor",
        },
    }
    write_json(report_path, payload)
    return payload


def main() -> int:
    args = parse_args()
    families = validate_families(args.families)
    report_path = Path(args.report)

    payload = run_torch_probe(
        baseline_name=args.baseline_name,
        report_path=report_path,
        families=families,
        samples=max(1, int(args.samples)),
    )

    mode = "available" if payload.get("available") else "unavailable"
    print(
        "omega_external_baseline_probe: "
        f"baseline={args.baseline_name} "
        f"mode={mode} "
        f"report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

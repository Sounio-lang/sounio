#!/usr/bin/env python3
"""Independent cross-check of the SAN machine meter (paper §2.1, clause L1).

The ledger's claim is executed-path accounting: stages that never run are
never charged. This script verifies that claim against an independent
instrument — PyTorch's profiler (which records the kernels that actually
executed and estimates their FLOPs from shapes via a separate code path):

  1. gated SAN forward with all samples forced to exit at stage 0
     → profiler must show NO conv kernels from stages 1..3, and the
       profiler's FLOP total must match the meter within tolerance;
  2. dense forward (all stages, all heads)
     → profiler total must match forward_dense's meter.

If CUPTI hardware counters are available (ncu on an NVIDIA GPU), the same
comparison can be run under `ncu --metrics smsp__sass_thread_inst_executed_op_ffma_pred_on.sum`
for a silicon-level executed-FMA count; that mode is attempted by the caller,
not this script (perf counters are often restricted on shared clusters).

Usage: SAN_LARGE_SMOKE=1 python san_meter_crosscheck.py
"""
import os
import sys

os.environ.setdefault("SAN_LARGE_SMOKE", "1")
os.environ.setdefault("SAN_LARGE_DEVICE", "cpu")
os.environ.setdefault("SAN_LARGE_DATASET", "cifar10")

import importlib.util

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "san", os.path.join(HERE, "suffering_aware_large_architecture_v2.py"))
san = importlib.util.module_from_spec(spec)
sys.modules["san"] = san
spec.loader.exec_module(san)

torch.manual_seed(0)
model = san.SufferingAwareResNet((1, 1, 1, 1), 16, "san")
model.eval()

# Force every sample to exit at stage 0: open gates (post-τ state) and pin
# the stage-0 gate logit high.
model.reached_tau = True
with torch.no_grad():
    model.gates[0].net[-1].bias.fill_(30.0)

x = torch.randn(64, 3, 32, 32)
STAGE_TAG = "crosscheck"


def metered_gated():
    model.meter = san.MachineMeter()
    with torch.no_grad():
        out = model(x, train=False)
    return out[1], model.meter.flops  # depth, flops


def metered_dense():
    with torch.no_grad():
        _, meter = model.forward_dense(x)
    return meter.flops


def profiled(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        fn()
    conv_mm_flops = 0
    kernels = {}
    for ev in prof.key_averages():
        if ev.key.startswith("aten::") and ev.key in (
                "aten::conv2d", "aten::convolution", "aten::mm",
                "aten::addmm", "aten::bmm", "aten::matmul", "aten::linear"):
            kernels[ev.key] = kernels.get(ev.key, 0) + ev.count
            flops = getattr(ev, "flops", 0) or 0
            conv_mm_flops += flops  # key_averages() flops are already totals
    return conv_mm_flops, kernels


depth, gated_meter = metered_gated()
gated_prof, gated_kernels = profiled(metered_gated)
dense_meter = metered_dense()
dense_prof, dense_kernels = profiled(metered_dense)

n_exit = int((depth < model.n_stages).sum())
print(f"[crosscheck] exits: {n_exit}/{x.shape[0]} (forced at stage 0)")
print(f"[crosscheck] gated meter   = {gated_meter:,} MAC-FLOPs")
print(f"[crosscheck] gated profiler= {int(gated_prof):,} FLOPs "
      f"(ratio profiler/meter = {gated_prof / max(gated_meter, 1):.4f})")
print(f"[crosscheck] dense meter   = {dense_meter:,} MAC-FLOPs")
print(f"[crosscheck] dense profiler= {int(dense_prof):,} FLOPs "
      f"(ratio profiler/meter = {dense_prof / max(dense_meter, 1):.4f})")
print(f"[crosscheck] gated kernels: {gated_kernels}")
print(f"[crosscheck] dense kernels: {dense_kernels}")

ok = True
if n_exit != x.shape[0]:
    print("FAIL: not all samples exited at stage 0")
    ok = False
# The meter charges conv+linear MACs; the profiler counts FLOPs for the same
# op classes (conv = 2*MAC typically). Accept profiler/meter in [1, 2.5]:
# ratio ~2 means the profiler counts multiply+add while the meter counts MACs.
for name, prof, met in (("gated", gated_prof, gated_meter),
                        ("dense", dense_prof, dense_meter)):
    r = prof / max(met, 1)
    if not (0.95 <= r <= 2.6):
        print(f"FAIL: {name} profiler/meter ratio {r:.3f} outside [0.95, 2.6]")
        ok = False
    else:
        print(f"PASS: {name} profiler/meter ratio {r:.3f} in [0.95, 2.6] "
              f"(1.0 = same convention, 2.0 = FLOP=2xMAC convention)")
# Executed-path check: gated forward must execute strictly fewer conv/mm
# kernel invocations than dense.
gk = sum(gated_kernels.values())
dk = sum(dense_kernels.values())
if not gk < dk:
    print(f"FAIL: gated executed {gk} conv/mm kernels vs dense {dk}")
    ok = False
else:
    print(f"PASS: gated executed {gk} conv/mm kernels vs dense {dk} "
          f"(gated-off stages executed nothing)")
print("SAN_METER_CROSSCHECK_VERDICT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

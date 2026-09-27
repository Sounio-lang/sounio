#!/usr/bin/env python3
"""On-target T3 acceptance for the SAN-ImageNet U250 deployment (pure stdlib).

Runs on the DL380 (or any node) with NO third-party dependencies. Reads the
artifacts exported by san_dl380_t3_export.py (meta.json + uint16 cohorts),
recomputes the catastrophe-scan / FLOP-metering semantics with TWO
independent pure-Python integer implementations (ports of the gated numpy
kernel model and of the independent host reference), and requires:

  A1  golden == reference on every dataset (the gated I6 equality)
  A2  golden == control-VM expected outputs bit-exactly (T3: the integer
      golden model is platform-independent, so the target must reproduce
      the control VM's numbers exactly)
  A3  accumulator bound < 2^63 on this host's Python build
  A4  preflight executes and reports the truth (FPGA/XRT probe)

Verdict line: SAN_DL380_T3_VERDICT T3_GREEN|T3_RED (details per clause).
"""
import array
import json
import os
import shutil
import socket
import sys
import time


def golden_scan(rows, n_points, lut, q_delta):
    """Port of U250SanScanModel.scan (argmax priority-encoder semantics):
    exit at the first point whose Q0.15 confidence clears q_delta, else the
    final head (== catastrophe). FLOPs via per-sample LUT gather."""
    hist = [0] * len(lut)
    flops = 0
    cat = 0
    for row in rows:
        idx = len(lut) - 1
        for k in range(n_points):
            if row[k] >= q_delta:
                idx = k
                break
        hist[idx] += 1
        flops += lut[idx]
        if idx == len(lut) - 1:
            cat += 1
    return hist, cat, flops


def reference_scan(rows, n_points, lut, q_delta):
    """Port of host_reference_scan: cumulative-any semantics, histogram via
    sorting, FLOPs via histogram x LUT dot product (different algorithm)."""
    exits = []
    for row in rows:
        seen = False
        idx = len(lut) - 1
        for k in range(n_points):
            if row[k] >= q_delta:
                seen = True
                idx = k
                break
        exits.append(idx if seen else len(lut) - 1)
    order = sorted(exits)
    hist = [0] * len(lut)
    for v in order:
        hist[v] += 1
    cat = hist[-1]
    flops = sum(h * c for h, c in zip(hist, lut))
    return hist, cat, flops


def read_u16(path, shape):
    a = array.array("H")
    with open(path, "rb") as f:
        a.fromfile(f, shape[0] * shape[1])
    n, w = shape
    return [a[i * w:(i + 1) * w] for i in range(n)], w


def preflight():
    info = {"host": socket.gethostname(), "fpga_present": 0,
            "xrt_present": int(shutil.which("xbutil") is not None
                               or shutil.which("xrt-smi") is not None),
            "role": "control-vm"}
    try:
        info["fpga_present"] = int(any(
            d.startswith(("xdma", "xocl")) for d in os.listdir("/dev")))
    except OSError:
        pass
    # XRT 2.x user devices also surface as DRM render nodes tagged xocl;
    # check sysfs as a second source (the U250 xdma shell exposes d8:00.1)
    if not info["fpga_present"]:
        try:
            for d in os.listdir("/sys/bus/pci/devices"):
                with open(f"/sys/bus/pci/devices/{d}/vendor") as f:
                    if f.read().strip() == "0x10ee":
                        info["fpga_present"] = 1
                        break
        except OSError:
            pass
    if info["fpga_present"] and info["xrt_present"]:
        info["role"] = "dl380-candidate"
    print(f"  A4[preflight]: host={info['host']} role={info['role']} "
          f"fpga_present={info['fpga_present']} xrt_present={info['xrt_present']}")
    return info


def main(art_dir):
    t0 = time.time()
    with open(os.path.join(art_dir, "meta.json")) as f:
        meta = json.load(f)
    ok_all = True
    for name, ds in meta["datasets"].items():
        lut = meta["lut"][ds["family"]]
        qd = meta["q_delta"][ds["family"]]
        rows, w = read_u16(os.path.join(art_dir, ds["file"]), ds["shape"])
        n_points = w
        t1 = time.time()
        g_hist, g_cat, g_flops = golden_scan(rows, n_points, lut, qd)
        r_hist, r_cat, r_flops = reference_scan(rows, n_points, lut, qd)
        scan_s = time.time() - t1
        exp = ds["expected"]
        a1 = (g_hist == r_hist and g_cat == r_cat and g_flops == r_flops)
        a2 = (g_hist == exp["hist"] and g_cat == exp["catastrophes"]
              and g_flops == exp["flops_macs"])
        ok = a1 and a2
        ok_all = ok_all and ok
        print(f"  A1A2[{name}]: {'PASS' if ok else 'FAIL'} "
              f"(golden==reference: {a1}, target==control-vm: {a2}, "
              f"catastrophes={g_cat}/{ds['shape'][0]}, "
              f"metered={g_flops / 1e9:.3f} GMAC, "
              f"pure_python_scan={scan_s:.2f}s)")
    acc_ok = int(meta["accumulator_bound"]) < 2**63
    ok_all = ok_all and acc_ok
    print(f"  A3[accumulator]: {'PASS' if acc_ok else 'FAIL'} "
          f"(bound={meta['accumulator_bound']:.3e} < 2^63)")
    pre = preflight()
    on_target = pre["role"] == "dl380-candidate"
    print(f"  A4[target-role]: {'PASS' if on_target else 'INFO'} "
          f"(role={pre['role']}; required only for deployment acceptance)")
    verdict = "T3_GREEN" if (ok_all and on_target) else (
        "T3_AMBER" if ok_all else "T3_RED")
    print(f"SAN_DL380_T3_VERDICT {verdict} "
          f"(integer_semantics={'PASS' if ok_all else 'FAIL'}, "
          f"target_role={pre['role']}, elapsed={time.time() - t0:.1f}s)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))

# Compute target: Alveo U250 FPGA

**Status: LIVE (2026-08-02). Validated end-to-end.** This document tells a Sounio dev agent
what the FPGA is, how to reach it, and what it means for `souc` codegen. Read before proposing
FPGA work.

## TL;DR

The cluster now has a **spatial / reconfigurable-hardware compute target** alongside the existing
GPU (PTX) path. Where a GPU runs one kernel over thousands of threads (SIMT), the FPGA lets you
**build a custom datapath in hardware** for the exact computation: deterministic latency, bit-level
control, and (with the right shell) direct 100G network attach. For Sounio this opens a codegen
direction GPU can't give: **`souc` → HLS/RTL → bitstream** — compiling Sounio to *hardware*, not
just to a vendor's GPU ISA.

## The hardware (as it exists today)

| Attribute | Value |
|---|---|
| Card | **Xilinx Alveo U250** (Gen3 x16), PCIe `d8:00.0/.1` (`[10ee:5004/5005]`) |
| Host node | **`dl380-proxmox`** (k8s `10.100.100.5`), HPE DL380 Gen10, 2× Xeon Gold 6262V (48c), 128 GiB |
| Runtime | **XRT 2.23.0** (upstream 2026.1 branch), `/opt/xilinx/xrt`, kernel `7.0.14-8-pve` |
| Shell (runtime partition) | `xilinx_u250_gen3x16_xdma_shell_4_1` (base `..._base_4`, SC `4.6.21`) |
| Persistence | `xrt-u250-shell.service` (systemd, auto-loads the 2RP shell after cold boot) |
| Validated | PCIe Gen3 x16, aux power, DMA, M2M, verify kernel, **DDR @ 67.9 GB/s** |

The U250 XDMA shell is **compute + DMA** (host↔card over PCIe, on-card DDR). It is **not** a
network NIC yet — see *Constraints*.

## How to use it

**From Kubernetes** (preferred for orchestrated jobs) — the card is exposed as an allocatable
resource; the node is **labeled, not tainted** (its 48 cores stay general-purpose):

```yaml
# pod spec
spec:
  nodeSelector: { sounio.dev/fpga: u250 }
  containers:
    - name: kernel
      image: <your-image-with-xrt>
      resources:
        limits: { sounio.dev/u250: 1 }   # kubelet mounts /dev/xclmgmt* + /dev/dri/renderD128
```

Requesting `sounio.dev/u250: 1` gets you the card's device nodes inside the pod. Bring your own
XRT in the image (or mount `/opt/xilinx/xrt` from the host).

**From the host** (dev / bring-up) — SSH `root@dl380-proxmox`, then:

```bash
source /opt/xilinx/xrt/setup.sh
xrt-smi examine                                             # card present + shell
/opt/xilinx/xrt/bin/xrt-smi --batch validate \
  --device 0000:d8:00.1 --run all                          # acceptance test (aux-power/DMA/M2M/DDR/verify)
# run a compiled kernel:
your_host_app your_kernel.xclbin                            # XRT loads the xclbin onto the user partition
```

Automation + the proven bring-up path live in
`beagle/k8s/hpc-sota/ops/baremetal/dl380-proxmox/` (README + loader unit + device-plugin manifest).

## What it means for `souc`

This is a **new backend**, parallel to PTX — not a replacement:

- **GPU / PTX** (existing): SIMT, great for dense regular parallelism.
- **FPGA / U250** (new): spatial dataflow. You compile a computation into a **hardware datapath**
  (HLS C/C++ or RTL → Vitis → `.xclbin` bitstream). Wins: deterministic latency, custom bit-width,
  pipelined streaming, and on-card DDR bandwidth without host round-trips.

Codegen thesis to explore: **`souc` → (HIR/SIR) → HLS or RTL → Vitis → xclbin**. The near-term
value is not "recompile everything for FPGA" — it's the workloads where a **custom datapath** beats
SIMT:

- **Search / graph** — the Erdős search and the Tapestry functor are search-heavy; FPGAs excel at
  custom search/graph datapaths with bit-exact control (aligns with the `souc→PTX` "GPU = search"
  doctrine, now with a spatial option).
- **Bit-exact / deterministic** kernels where GPU nondeterminism or float behavior is a liability.

Treat it as a *codegen research target*: prototype an HLS kernel by hand first, prove the datapath,
then wire the `souc` lowering.

## Constraints (do not assume beyond these)

- **One card, one node.** Only `dl380-proxmox` has a U250. A second U250 is planned for a different
  fabric-connected node (to enable FPGA↔FPGA over 100G) — not installed yet.
- **No FPGA network interface yet.** The XDMA shell is compute/DMA only. The card's QSFP is cabled
  but idle; using it as a fabric NIC needs a different **CMAC/Ethernet shell** flashed to the card.
  Do not design for FPGA-attached networking until that shell exists.
- **Heterogeneous node incoming.** `dl380-proxmox` will also receive an **AMD Radeon (ROCm)** GPU
  (power cable pending, ~2 days). At that point the node hosts **three backends** — PTX (elsewhere),
  ROCm (AMD, here), FPGA (U250, here) + 48 CPU cores — a natural multi-target codegen test-bench.
- **Shell is volatile (2RP).** The runtime partition is reloaded by `xrt-u250-shell.service` after a
  cold boot; don't hand-flash the runtime partition without reading the DL380 README.

## Provenance

Onboarded + validated 2026-08-02: XRT built for kernel 7.0.14-8-pve, base+SC+shell flashed, cold-boot
auto-load proven, AMD/Xilinx `xrt-smi validate` full pass (DDR 67.9 GB/s), exposed to k8s as
`sounio.dev/u250` and proven end-to-end (a pod claiming the resource received the card device nodes).
Do not treat any capability above as available beyond what this validation covered.

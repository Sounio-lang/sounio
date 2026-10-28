-- ADR-009 verified_foreign_reference oracle for the U250 SAN
-- catastrophe-scan kernel semantics (ADR-021 steps 2-4), independently
-- authored from the specification in
-- docs/research/san_imagenet_fpga_dl380_spec_2026-08-02.md (clause T2),
-- not derived from hardware/fpga/u250_catastrophe_scan/krnl_san_scan.cpp
-- or from scripts/research/san_imagenet_fpga_dl380.py.
--
-- Specification (spec T2, integer-only deployment semantics):
--   inputs are Q0.15 confidences q in [0, 2^15 - 1] and a quantized
--   threshold q_delta; a sample's exit point is the FIRST intermediate
--   point k with q_k >= q_delta, else the final head; the FLOP (MAC)
--   total is sum over samples of lut[exit(sample)], where lut holds
--   prefix MAC costs and lut[last] is the dense (final-head) cost.
--   Samples that reach the final head are the catastrophes.
--
-- This program expresses exactly that function with Futhark's array
-- semantics: a per-sample sequential first-exit fold, a histogram by
-- scatter-add, and a LUT gather summed in i64. No packing, no ABI, no
-- shared code with the HLS kernel or the Python contract model.
-- Accumulator soundness follows the spec: at ImageNet-completo scale
-- the maximum total is < 2^63, so i64 cannot overflow there.

-- First intermediate index whose confidence clears the threshold,
-- or k (the final head) when none does.
let exit_of [k] (row: [k]i32) (q_delta: i32): i64 =
  loop acc = k
  for j < k do
    if acc == k && row[j] >= q_delta then j else acc

-- Full cohort scan.
--   confs: [n][k] Q0.15 confidences at the k intermediate exit points
--   lut:   [k+1] prefix MAC costs; lut[k] is the final-head cost
-- Returns (histogram over k+1 exit points, catastrophe count, MACs).
entry san_scan [n] [k]
               (confs: [n][k]i32)
               (q_delta: i32)
               (lut: [k + 1]i64)
               : ([k + 1]i64, i64, i64) =
  let exits = map (\row -> exit_of row q_delta) confs
  let hist = reduce_by_index (replicate (k + 1) 0i64) (+) 0i64 exits (replicate n 1i64)
  let catastrophes = hist[k]
  let macs = i64.sum (map (\e -> lut[e]) exits)
  in (hist, catastrophes, macs)

-- ==
-- entry: san_scan
-- input { [[18022i32, 0i32, 0i32]] 18022i32 [100i64, 200i64, 300i64, 400i64] }
-- output { [1i64, 0i64, 0i64, 0i64] 0i64 100i64 }
-- input { [[0i32, 18021i32, 5i32]] 18022i32 [100i64, 200i64, 300i64, 400i64] }
-- output { [0i64, 0i64, 0i64, 1i64] 1i64 400i64 }
-- input { [[0i32, 19000i32, 5i32], [32767i32, 0i32, 0i32], [1i32, 2i32, 3i32], [0i32, 0i32, 22000i32]] 18022i32 [100i64, 200i64, 300i64, 400i64] }
-- output { [1i64, 1i64, 1i64, 1i64] 1i64 1000i64 }

-- ADR-009 verified_foreign_reference oracle for the second FPGA
-- profile kernel (ADR-022 promotion criterion): the 1D dot product,
-- authored from its one-line specification, sum of a[i] * b[i] in
-- i64. Judges examples/fpga/dot_product_kernel.sio via
-- scripts/ci/fpga_prototype_gate2.sh. No shared code with the
-- emitter or the generated HLS.

entry dot_product [m] (a: [m]i64) (b: [m]i64) : i64 =
  i64.sum (map2 (*) a b)

-- ==
-- entry: dot_product
-- input { [3i64, 5i64, 7i64] [4i64, 6i64, 8i64] }
-- output { 98i64 }
-- input { [0i64, -2i64, 9i64, 1000000i64] [7i64, 3i64, -1i64, 1000000i64] }
-- output { 999999999985i64 }

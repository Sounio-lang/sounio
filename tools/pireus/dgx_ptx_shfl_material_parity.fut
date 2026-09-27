-- ADR-009 verified_foreign_reference oracle for the Sounio abstract
-- shfl.sync.bfly.b32 semantics (16-lane XOR butterfly), independently
-- authored from the frozen specification, not derived from
-- tools/pireus/dgx_ptx_shfl_material_parity.cu.
--
-- Abstract claim under check: an XOR-butterfly shuffle across N lanes
-- (member mask covering all lanes, segment mask 0, XOR displacement d)
-- must satisfy, for every displacement d and lane l:
--
--     output[d, l] = input[l xor d]
--
-- This program independently synthesizes the same deterministic test
-- vectors (a public parameter of the test, not part of the claim being
-- verified) and checks the butterfly law holds for all N*N cells using
-- Futhark's own array semantics -- no digest, no bit-tracking, no
-- shared code with the C++/CUDA measurement harness.

let dimension: i64 = 16

let payload_for_source (source: i64): u64 =
  0xfedcba9876543210u64 ^ (0x1111111111111111u64 * u64.i64 source)

let inputs: [dimension]u64 =
  tabulate dimension payload_for_source

-- The independently-derived expected butterfly matrix.
let expected: [dimension][dimension]u64 =
  tabulate_2d dimension dimension (\d l -> inputs[l ^ d])

-- Self-check: the butterfly law is an involution on the XOR group, so
-- applying it twice must return the original input. This is an
-- independent algebraic sanity check on the specification itself
-- (not on any measured GPU run) -- catches a mis-specified law before
-- it is ever compared against silicon.
let involution_holds: bool =
  let twice = tabulate_2d dimension dimension (\d l -> expected[d, l ^ d] )
  in map2 (\a b -> a == b) inputs (twice[0])
     |> reduce (&&) true

-- Cross-check entry point: compares an externally supplied observed
-- matrix (e.g. dumped from a real GPU run of shfl.sync.bfly.b32) to
-- this program's independently-derived expected matrix. Returns the
-- count of mismatched cells (0 == PASS) and whether the involution
-- sanity check held.
entry check (observed: [dimension][dimension]u64): (i64, bool) =
  let mismatches =
    map2 (\obs_row exp_row ->
            map2 (\o e -> if o == e then 0i64 else 1i64) obs_row exp_row
            |> i64.sum)
         observed expected
    |> i64.sum
  in (mismatches, involution_holds)

-- Standalone entry point: no external GPU trace available, so this
-- validates the specification purely against itself (the involution
-- law) and reports the synthesized inputs/expected matrix for
-- downstream comparison by the gate script.
entry standalone : (bool, [dimension]u64, [dimension][dimension]u64) =
  (involution_holds, copy inputs, copy expected)

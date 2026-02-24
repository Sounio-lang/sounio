// Omega Sprint 4.0 Genesis - K-AXI Merkle root lane core.
// Deterministic root fold over policy/baseline/telemetry/bitstream lanes.

`timescale 1ns/1ps

module k_axi_merkle_root_lane_core (
    input  wire [63:0] policy_hash_l64,
    input  wire [63:0] baseline_hash_l64,
    input  wire [63:0] telemetry_hash_l64,
    input  wire [63:0] bitstream_hash_l64,
    input  wire [63:0] manifest_hash_l64,
    output wire        merkle_root_valid,
    output wire [63:0] merkle_root_l64
);
    localparam [63:0] ROOT_SALT_A = 64'h4F4D45474147454E;
    localparam [63:0] ROOT_SALT_B = 64'h4553495376312E30;

    wire [63:0] h0 = ((policy_hash_l64 ^ baseline_hash_l64) + telemetry_hash_l64 + ROOT_SALT_A) ^
                     (policy_hash_l64 << 5) ^ (baseline_hash_l64 >> 7);
    wire [63:0] h1 = ((bitstream_hash_l64 ^ manifest_hash_l64) + h0 + ROOT_SALT_B) ^
                     (bitstream_hash_l64 << 3) ^ (manifest_hash_l64 >> 11);

    assign merkle_root_l64 = (h0 ^ h1) + (policy_hash_l64 ^ telemetry_hash_l64);
    assign merkle_root_valid = 1'b1;
endmodule

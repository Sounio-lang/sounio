// Omega Sprint 4.0 - K-AXI Merkle lane core.
// Deterministic provenance root fold for FPGA pipeline.

`timescale 1ns/1ps

module k_axi_merkle_lane_core #(
    parameter PROV_WIDTH = 256
) (
    input  wire [PROV_WIDTH-1:0] merkle_lane_seed,
    input  wire [63:0]           tx_value,
    input  wire [31:0]           tx_flags,
    input  wire [31:0]           tx_seq,
    output wire                  merkle_lane_valid,
    output wire [PROV_WIDTH-1:0] merkle_lane_digest
);
    localparam [255:0] MERKLE_SALT = 256'h4B5F4158495F4D45524B4C455F4C414E455F76315F434F52455FA5A5A5A5A5A5;

    wire [255:0] tx_seq_wide = {224'd0, tx_seq};

    assign merkle_lane_digest =
        {merkle_lane_seed[191:0], merkle_lane_seed[255:192]} ^
        {192'd0, tx_value} ^
        {224'd0, tx_flags} ^
        tx_seq_wide ^
        MERKLE_SALT;

    assign merkle_lane_valid = 1'b1;
endmodule

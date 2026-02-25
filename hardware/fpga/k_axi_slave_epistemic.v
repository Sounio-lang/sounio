// GENERATED_FROM_STDLIB: stdlib/hardware/kaxi.sio
// GENERATED_BY: self-hosted/compiler/codegen/hardware/kaxi_emitter.sio
// Sounio Omega - K-AXI epistemic slave/propagation engine
// Consumes AXI-Stream value + epistemic sideband and applies hardware-side updates.

`timescale 1ns/1ps

module k_axi_merkle_core #(
    parameter PROV_WIDTH = 256
) (
    input  wire [PROV_WIDTH-1:0] in_prov,
    input  wire [63:0]           in_value,
    input  wire [31:0]           in_flags,
    output wire [PROV_WIDTH-1:0] digest
);
    localparam [255:0] PROV_SALT = 256'h4B5F4158495F4D45524B4C455F4550495354454D49435F4255535F7631A5A5A5;

    // Dedicated deterministic Merkle-seed fold (replace with hash core in v2).
    assign digest = {in_prov[247:0], in_prov[255:248]}
        ^ {192'd0, in_value}
        ^ {224'd0, in_flags}
        ^ PROV_SALT;
endmodule

module k_axi_slave_epistemic #(
    parameter DATA_WIDTH = 64,
    parameter VAR_WIDTH = 64,
    parameter PROV_WIDTH = 256,
    parameter CONF_WIDTH = 128,
    parameter FLAGS_WIDTH = 32,
    parameter SCALE_SHIFT = 30
) (
    input  wire                         aclk,
    input  wire                         aresetn,

    // AXI-Stream input.
    input  wire [DATA_WIDTH-1:0]        s_tdata,
    input  wire [VAR_WIDTH+PROV_WIDTH+CONF_WIDTH+FLAGS_WIDTH-1:0] s_tuser,
    input  wire                         s_tvalid,
    output wire                         s_tready,

    // Downstream interface.
    output reg  [DATA_WIDTH-1:0]        out_value,
    output reg  [VAR_WIDTH-1:0]         out_variance,
    output reg  [PROV_WIDTH-1:0]        out_prov,
    output reg  [CONF_WIDTH-1:0]        out_conf,
    output reg  [FLAGS_WIDTH-1:0]       out_flags,
    output reg                          out_valid,
    input  wire                         downstream_ready,

    // Merkle lane interface (optional external provenance core).
    input  wire                         merkle_lane_valid,
    input  wire [PROV_WIDTH-1:0]        merkle_lane_digest,
    output reg                          merkle_lane_req,
    output reg  [PROV_WIDTH-1:0]        merkle_lane_seed,

    // Epistemic Power accounting lanes.
    output reg  [31:0]                  gum_fidelity_inc,
    output reg  [31:0]                  provenance_depth_inc,
    output reg  [31:0]                  formal_coverage_inc
);

    localparam KUSER_WIDTH = VAR_WIDTH + PROV_WIDTH + CONF_WIDTH + FLAGS_WIDTH;

    localparam OP_PASS = 4'h0;
    localparam OP_ADD = 4'h1;
    localparam OP_MUL = 4'h2;
    localparam OP_DIV = 4'h3;
    localparam OP_FMA = 4'h4;

    wire accept;
    assign s_tready = !out_valid || downstream_ready;
    assign accept = s_tvalid && s_tready;

    wire [63:0] in_variance = s_tuser[63:0];
    wire [255:0] in_prov = s_tuser[319:64];
    wire [127:0] in_conf = s_tuser[447:320];
    wire [31:0] in_flags = s_tuser[479:448];

    wire [3:0] op_kind = in_flags[3:0];
    wire [7:0] failure_inc = in_flags[15:8];
    wire [7:0] success_inc = in_flags[23:16];
    wire [7:0] alpha_q8 = in_flags[31:24];

    wire [63:0] in_alpha = in_conf[127:64];
    wire [63:0] in_beta = in_conf[63:0];
    wire [63:0] alpha_next_wire = in_alpha + {56'd0, success_inc};
    wire [63:0] beta_next_wire = in_beta + {56'd0, failure_inc};
    wire [255:0] merkle_core_digest;

    k_axi_merkle_core #(
        .PROV_WIDTH(PROV_WIDTH)
    ) u_merkle_core (
        .in_prov(in_prov),
        .in_value(s_tdata),
        .in_flags(in_flags),
        .digest(merkle_core_digest)
    );

    reg [63:0] op_var;
    reg [127:0] degradation_full;
    reg [63:0] degradation_var;

    always @(*) begin
        // Richer op-sensitive variance profile (Q32.32-oriented approximation).
        // This keeps deterministic hardware semantics while differentiating
        // additive vs multiplicative/divisive uncertainty growth.
        case (op_kind)
            OP_ADD: op_var = in_variance + (in_variance >> 3);                         // +12.5%
            OP_MUL: op_var = in_variance + (in_variance >> 1) + (in_variance >> 3);   // +62.5%
            OP_DIV: op_var = in_variance + (in_variance >> 1) + (in_variance >> 2);   // +75.0%
            OP_FMA: op_var = in_variance + (in_variance >> 1) + (in_variance >> 2) + (in_variance >> 4); // +81.25%
            default: op_var = in_variance;
        endcase

        // Epistemic degradation term from alpha_q8 (Q0.8), now applied after
        // op-profile shaping so downstream sees profile+degradation.
        degradation_full = op_var * alpha_q8;
        degradation_var = degradation_full[71:8];
    end

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            out_value <= {DATA_WIDTH{1'b0}};
            out_variance <= {VAR_WIDTH{1'b0}};
            out_prov <= {PROV_WIDTH{1'b0}};
            out_conf <= {CONF_WIDTH{1'b0}};
            out_flags <= {FLAGS_WIDTH{1'b0}};
            out_valid <= 1'b0;
            merkle_lane_req <= 1'b0;
            merkle_lane_seed <= {PROV_WIDTH{1'b0}};
            gum_fidelity_inc <= 32'd0;
            provenance_depth_inc <= 32'd0;
            formal_coverage_inc <= 32'd0;
        end else begin
            // One-cycle request pulse unless a transaction is accepted.
            merkle_lane_req <= 1'b0;

            if (out_valid && downstream_ready) begin
                out_valid <= 1'b0;
            end

            if (accept) begin
                out_value <= s_tdata;
                out_variance <= op_var + degradation_var;
                out_flags <= in_flags;

                // Main path consumes dedicated Merkle-core digest.
                merkle_lane_seed <= merkle_core_digest;
                merkle_lane_req <= 1'b1;
                if (merkle_lane_valid) begin
                    out_prov <= merkle_lane_digest;
                end else begin
                    out_prov <= merkle_core_digest;
                end
                out_conf <= {alpha_next_wire, beta_next_wire};

                gum_fidelity_inc <= gum_fidelity_inc + 32'd1;
                provenance_depth_inc <= provenance_depth_inc + 32'd1;
                formal_coverage_inc <= formal_coverage_inc + 32'd1;

                out_valid <= 1'b1;
            end
        end
    end

endmodule

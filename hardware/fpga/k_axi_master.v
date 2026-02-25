// Sounio Omega - K-AXI master wrapper
// Streams Knowledge<T> payloads over AXI-Stream with epistemic metadata in TUSER.

`timescale 1ns/1ps

module k_axi_master #(
    parameter DATA_WIDTH = 64,
    parameter VAR_WIDTH = 64,
    parameter PROV_WIDTH = 256,
    parameter CONF_WIDTH = 128,
    parameter FLAGS_WIDTH = 32
) (
    input  wire                         s_axi_aclk,
    input  wire                         s_axi_aresetn,

    output reg  [DATA_WIDTH-1:0]        s_axi_tdata,
    output reg  [VAR_WIDTH+PROV_WIDTH+CONF_WIDTH+FLAGS_WIDTH-1:0] s_axi_tuser,
    output reg                          s_axi_tvalid,
    input  wire                         s_axi_tready,

    // Knowledge payload from PTX/runtime side.
    input  wire [63:0]                  k_value,
    input  wire [63:0]                  k_variance,
    input  wire [255:0]                 k_prov,
    input  wire [127:0]                 k_conf,
    input  wire [31:0]                  k_flags,

    input  wire                         start_transaction,
    output reg                          transaction_done,

    // Increment signal consumed by EpistemicPower accounting.
    output reg  [31:0]                  fidelity_increment
);

    localparam KUSER_WIDTH = VAR_WIDTH + PROV_WIDTH + CONF_WIDTH + FLAGS_WIDTH;
    reg pending;

    always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_tdata <= {DATA_WIDTH{1'b0}};
            s_axi_tuser <= {KUSER_WIDTH{1'b0}};
            s_axi_tvalid <= 1'b0;
            transaction_done <= 1'b0;
            fidelity_increment <= 32'd0;
            pending <= 1'b0;
        end else begin
            transaction_done <= 1'b0;
            fidelity_increment <= 32'd0;

            // Launch a new transaction if link is idle.
            if (start_transaction && !pending && !s_axi_tvalid) begin
                s_axi_tdata <= k_value;
                s_axi_tuser <= {k_flags, k_conf, k_prov, k_variance};
                s_axi_tvalid <= 1'b1;
                pending <= 1'b1;
            end

            // Complete handshake and emit one-shot accounting increment.
            if (s_axi_tvalid && s_axi_tready) begin
                s_axi_tvalid <= 1'b0;
                pending <= 1'b0;
                transaction_done <= 1'b1;
                fidelity_increment <= 32'd1;
            end
        end
    end

endmodule

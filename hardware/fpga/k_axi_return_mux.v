// GENERATED_FROM_STDLIB: stdlib/hardware/kaxi.sio
// GENERATED_BY: self-hosted/compiler/codegen/hardware/kaxi_emitter.sio
// K-AXI bidirectional return mux: fabric -> return FIFO -> host runtime.

`timescale 1ns/1ps

module k_axi_return_mux #(
    parameter PACKET_WIDTH = 544,
    parameter FIFO_DEPTH = 8,
    parameter FIFO_ADDR_BITS = 3
) (
    input  wire                  aclk,
    input  wire                  aresetn,

    // Inbound return lane from epistemic fabric.
    input  wire                  fabric_valid,
    output wire                  fabric_ready,
    input  wire [7:0]            kind,
    input  wire [7:0]            status,
    input  wire [7:0]            op_kind,
    input  wire [7:0]            flags,
    input  wire [31:0]           tx_seq,
    input  wire [31:0]           fidelity_inc,
    input  wire [31:0]           prov_depth_inc,
    input  wire [31:0]           formal_cov_inc,
    input  wire [63:0]           epi_log_q32_32,
    input  wire [255:0]          digest,
    input  wire [63:0]           timestamp_cycles,

    // Outbound lane to host/runtime consumer.
    output wire                  ret_valid,
    input  wire                  ret_ready,
    output wire [PACKET_WIDTH-1:0] ret_data,

    // Telemetry counters.
    output reg  [31:0]           accepted_packets,
    output reg  [31:0]           dropped_packets,
    output reg  [31:0]           overflow_events
);
    wire [PACKET_WIDTH-1:0] packet_in;
    assign packet_in = {
        timestamp_cycles,
        digest,
        epi_log_q32_32,
        formal_cov_inc,
        prov_depth_inc,
        fidelity_inc,
        tx_seq,
        flags,
        op_kind,
        status,
        kind
    };

    wire fifo_full;
    wire fifo_empty;
    wire fifo_overflow;
    wire fifo_underflow;
    wire [FIFO_ADDR_BITS:0] fifo_count;
    wire [PACKET_WIDTH-1:0] fifo_rd_data;

    assign fabric_ready = !fifo_full;

    wire wr_en = fabric_valid && fabric_ready;
    wire rd_en = ret_ready && !fifo_empty;

    assign ret_valid = !fifo_empty;
    assign ret_data = fifo_rd_data;

    k_axi_return_fifo #(
        .WIDTH(PACKET_WIDTH),
        .DEPTH(FIFO_DEPTH),
        .ADDR_BITS(FIFO_ADDR_BITS)
    ) u_return_fifo (
        .clk(aclk),
        .rst_n(aresetn),
        .wr_en(wr_en),
        .wr_data(packet_in),
        .full(fifo_full),
        .overflow(fifo_overflow),
        .rd_en(rd_en),
        .rd_data(fifo_rd_data),
        .empty(fifo_empty),
        .underflow(fifo_underflow),
        .count(fifo_count)
    );

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            accepted_packets <= 32'd0;
            dropped_packets <= 32'd0;
            overflow_events <= 32'd0;
        end else begin
            if (wr_en) begin
                accepted_packets <= accepted_packets + 32'd1;
            end
            if (fabric_valid && !fabric_ready) begin
                dropped_packets <= dropped_packets + 32'd1;
            end
            if (fifo_overflow) begin
                overflow_events <= overflow_events + 32'd1;
            end
            if (fifo_underflow) begin
                overflow_events <= overflow_events + 32'd1;
            end
        end
    end
endmodule

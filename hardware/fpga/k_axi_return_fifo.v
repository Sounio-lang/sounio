// GENERATED_FROM_STDLIB: stdlib/hardware/kaxi.sio
// GENERATED_BY: self-hosted/compiler/codegen/hardware/kaxi_emitter.sio
// K-AXI bidirectional return FIFO (deterministic fixed-width queue).

`timescale 1ns/1ps

module k_axi_return_fifo #(
    parameter WIDTH = 544,
    parameter DEPTH = 8,
    parameter ADDR_BITS = 3
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  wr_en,
    input  wire [WIDTH-1:0]      wr_data,
    output wire                  full,
    output reg                   overflow,
    input  wire                  rd_en,
    output wire [WIDTH-1:0]      rd_data,
    output wire                  empty,
    output reg                   underflow,
    output reg  [ADDR_BITS:0]    count
);
    localparam integer DEPTH_INT = DEPTH;

    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [ADDR_BITS-1:0] wptr;
    reg [ADDR_BITS-1:0] rptr;

    assign full = (count == DEPTH_INT);
    assign empty = (count == 0);
    assign rd_data = mem[rptr];

    wire do_write = wr_en && !full;
    wire do_read = rd_en && !empty;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wptr <= {ADDR_BITS{1'b0}};
            rptr <= {ADDR_BITS{1'b0}};
            count <= {(ADDR_BITS + 1){1'b0}};
            overflow <= 1'b0;
            underflow <= 1'b0;
        end else begin
            overflow <= 1'b0;
            underflow <= 1'b0;

            if (wr_en && full) begin
                overflow <= 1'b1;
            end
            if (rd_en && empty) begin
                underflow <= 1'b1;
            end

            if (do_write) begin
                mem[wptr] <= wr_data;
                wptr <= wptr + 1'b1;
            end

            if (do_read) begin
                rptr <= rptr + 1'b1;
            end

            case ({do_write, do_read})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end
endmodule

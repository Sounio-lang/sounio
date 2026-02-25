`timescale 1ns/1ps

module tb_k_axi_bidirectional;
    reg aclk = 1'b0;
    reg aresetn = 1'b0;
    always #5 aclk = ~aclk;

    reg fabric_valid;
    wire fabric_ready;
    reg [7:0] kind;
    reg [7:0] status;
    reg [7:0] op_kind;
    reg [7:0] flags;
    reg [31:0] tx_seq;
    reg [31:0] fidelity_inc;
    reg [31:0] prov_depth_inc;
    reg [31:0] formal_cov_inc;
    reg [63:0] epi_log_q32_32;
    reg [255:0] digest;
    reg [63:0] timestamp_cycles;

    wire ret_valid;
    reg ret_ready;
    wire [543:0] ret_data;

    wire [31:0] accepted_packets;
    wire [31:0] dropped_packets;
    wire [31:0] overflow_events;

    integer failures;
    integer drained;
    reg [31:0] first_seq;

    k_axi_return_mux #(
        .PACKET_WIDTH(544),
        .FIFO_DEPTH(4),
        .FIFO_ADDR_BITS(2)
    ) dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .fabric_valid(fabric_valid),
        .fabric_ready(fabric_ready),
        .kind(kind),
        .status(status),
        .op_kind(op_kind),
        .flags(flags),
        .tx_seq(tx_seq),
        .fidelity_inc(fidelity_inc),
        .prov_depth_inc(prov_depth_inc),
        .formal_cov_inc(formal_cov_inc),
        .epi_log_q32_32(epi_log_q32_32),
        .digest(digest),
        .timestamp_cycles(timestamp_cycles),
        .ret_valid(ret_valid),
        .ret_ready(ret_ready),
        .ret_data(ret_data),
        .accepted_packets(accepted_packets),
        .dropped_packets(dropped_packets),
        .overflow_events(overflow_events)
    );

    task send_packet(input [31:0] seq, input [7:0] op);
    begin
        @(posedge aclk);
        fabric_valid <= 1'b1;
        kind <= 8'hA1;
        status <= 8'h01;
        op_kind <= op;
        flags <= 8'h5A;
        tx_seq <= seq;
        fidelity_inc <= 32'd1;
        prov_depth_inc <= 32'd1;
        formal_cov_inc <= 32'd1;
        epi_log_q32_32 <= {32'd0, seq};
        digest <= {224'd0, seq};
        timestamp_cycles <= 64'h1000 + seq;

        @(posedge aclk);
        fabric_valid <= 1'b0;
    end
    endtask

    initial begin
        failures = 0;
        drained = 0;
        first_seq = 32'd0;
        fabric_valid = 1'b0;
        kind = 8'd0;
        status = 8'd0;
        op_kind = 8'd0;
        flags = 8'd0;
        tx_seq = 32'd0;
        fidelity_inc = 32'd0;
        prov_depth_inc = 32'd0;
        formal_cov_inc = 32'd0;
        epi_log_q32_32 = 64'd0;
        digest = 256'd0;
        timestamp_cycles = 64'd0;
        ret_ready = 1'b0;

        $dumpfile("artifacts/fpga/waveforms/tb_k_axi_bidirectional.vcd");
        $dumpvars(0, tb_k_axi_bidirectional);

        repeat (4) @(posedge aclk);
        aresetn = 1'b1;
        repeat (2) @(posedge aclk);

        // Fill FIFO and force one overflow/drop packet.
        send_packet(32'd1, 8'h11);
        send_packet(32'd2, 8'h12);
        send_packet(32'd3, 8'h13);
        send_packet(32'd4, 8'h14);
        send_packet(32'd5, 8'h15);

        repeat (2) @(posedge aclk);
        if (accepted_packets !== 32'd4) begin
            $display("ASSERT FAIL: accepted_packets expected 4 got %0d", accepted_packets);
            failures = failures + 1;
        end
        if (dropped_packets !== 32'd1) begin
            $display("ASSERT FAIL: dropped_packets expected 1 got %0d", dropped_packets);
            failures = failures + 1;
        end

        // Drain queue.
        ret_ready = 1'b1;
        while (drained < 4) begin
            @(posedge aclk);
            if (ret_valid && ret_ready) begin
                if (drained == 0) begin
                    first_seq = ret_data[63:32];
                end
                drained = drained + 1;
            end
        end

        if (first_seq !== 32'd1) begin
            $display("ASSERT FAIL: first packet tx_seq expected 1 got %0d", first_seq);
            failures = failures + 1;
        end
        if (overflow_events == 0) begin
            $display("ASSERT FAIL: overflow_events should be > 0");
            failures = failures + 1;
        end

        $display(
            "OMEGA_COUNTERS kaxi_return_seen=%0d kaxi_return_dropped=%0d kaxi_return_overflow=%0d",
            accepted_packets,
            dropped_packets,
            overflow_events
        );

        if (failures > 0) begin
            $display("FAIL tb_k_axi_bidirectional cases=%0d", failures);
            $finish(1);
        end

        $display("PASS tb_k_axi_bidirectional");
        $finish(0);
    end
endmodule

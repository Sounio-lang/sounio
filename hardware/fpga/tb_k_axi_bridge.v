`timescale 1ns/1ps

module tb_k_axi_bridge;
    reg clk;
    reg rst_n;

    reg [63:0] k_value;
    reg [63:0] k_variance;
    reg [255:0] k_prov;
    reg [127:0] k_conf;
    reg [31:0] k_flags;
    reg start_transaction;

    wire [63:0] s_tdata;
    wire [479:0] s_tuser;
    wire s_tvalid;
    wire s_tready;
    wire transaction_done;
    wire [31:0] fidelity_increment;

    wire [63:0] out_value;
    wire [63:0] out_variance;
    wire [255:0] out_prov;
    wire [127:0] out_conf;
    wire [31:0] out_flags;
    wire out_valid;
    reg downstream_ready;
    wire [31:0] gum_fidelity_inc;
    wire [31:0] provenance_depth_inc;
    wire [31:0] formal_coverage_inc;
    reg merkle_lane_valid;
    reg [255:0] merkle_lane_digest;
    wire merkle_lane_req;
    wire [255:0] merkle_lane_seed;

    integer failures;
    reg [63:0] conf_alpha;
    reg [63:0] conf_beta;
    reg [63:0] expected_var;
    reg [255:0] expected_merkle_seed;
    reg seen_transaction_done;
    reg seen_fidelity_pulse;
    reg seen_merkle_req;
    reg [1023:0] dumpfile_path;

    localparam [255:0] PROV_SALT = 256'h4B5F4158495F4D45524B4C455F4550495354454D49435F4255535F7631A5A5A5;

    always @(posedge clk) begin
        if (!rst_n) begin
            seen_transaction_done <= 1'b0;
            seen_fidelity_pulse <= 1'b0;
            seen_merkle_req <= 1'b0;
        end else begin
            if (transaction_done) begin
                seen_transaction_done <= 1'b1;
            end
            if (fidelity_increment != 32'd0) begin
                seen_fidelity_pulse <= 1'b1;
            end
            if (merkle_lane_req) begin
                seen_merkle_req <= 1'b1;
            end
        end
    end

    k_axi_master u_master (
        .s_axi_aclk(clk),
        .s_axi_aresetn(rst_n),
        .s_axi_tdata(s_tdata),
        .s_axi_tuser(s_tuser),
        .s_axi_tvalid(s_tvalid),
        .s_axi_tready(s_tready),
        .k_value(k_value),
        .k_variance(k_variance),
        .k_prov(k_prov),
        .k_conf(k_conf),
        .k_flags(k_flags),
        .start_transaction(start_transaction),
        .transaction_done(transaction_done),
        .fidelity_increment(fidelity_increment)
    );

    k_axi_slave_epistemic u_slave (
        .aclk(clk),
        .aresetn(rst_n),
        .s_tdata(s_tdata),
        .s_tuser(s_tuser),
        .s_tvalid(s_tvalid),
        .s_tready(s_tready),
        .out_value(out_value),
        .out_variance(out_variance),
        .out_prov(out_prov),
        .out_conf(out_conf),
        .out_flags(out_flags),
        .out_valid(out_valid),
        .downstream_ready(downstream_ready),
        .merkle_lane_valid(merkle_lane_valid),
        .merkle_lane_digest(merkle_lane_digest),
        .merkle_lane_req(merkle_lane_req),
        .merkle_lane_seed(merkle_lane_seed),
        .gum_fidelity_inc(gum_fidelity_inc),
        .provenance_depth_inc(provenance_depth_inc),
        .formal_coverage_inc(formal_coverage_inc)
    );

    always #5 clk = ~clk;

    task expect_true;
        input cond;
        input [255:0] msg;
        begin
            if (!cond) begin
                $display("FAIL %0s", msg);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        if (!$value$plusargs("dumpfile=%s", dumpfile_path)) begin
            dumpfile_path = "artifacts/fpga/waveforms/tb_k_axi_bridge.vcd";
        end
        $dumpfile(dumpfile_path);
        $dumpvars(0, tb_k_axi_bridge);

        clk = 1'b0;
        rst_n = 1'b0;
        k_value = 64'd0;
        k_variance = 64'd0;
        k_prov = 256'd0;
        k_conf = 128'd0;
        k_flags = 32'd0;
        start_transaction = 1'b0;
        downstream_ready = 1'b0;
        merkle_lane_valid = 1'b0;
        merkle_lane_digest = 256'd0;
        failures = 0;
        seen_transaction_done = 1'b0;
        seen_fidelity_pulse = 1'b0;
        seen_merkle_req = 1'b0;

        #20;
        rst_n = 1'b1;
        #10;

        // TX1: OP_ADD with alpha_q8=8, success=2, failure=1
        k_value = 64'd4096;
        k_variance = 64'd64;
        k_prov = 256'h1234_5678_9ABC_DEF0_1111_2222_3333_4444_5555_6666_7777_8888_9999_AAAA_BBBB_CCCC;
        k_conf = {64'd40, 64'd70};
        k_flags = {8'd8, 8'd2, 8'd1, 4'd0, 4'd1};
        start_transaction = 1'b1;
        #10;
        start_transaction = 1'b0;

        #20;
        expected_var = (k_variance + (k_variance >> 3));
        expected_var = expected_var + ((expected_var * 8) >> 8);
        expect_true(out_valid == 1'b1, "tx1 out_valid asserted");
        expect_true(seen_transaction_done == 1'b1, "tx1 transaction_done pulse");
        expect_true(seen_fidelity_pulse == 1'b1, "tx1 master fidelity increment");
        expect_true(out_value == k_value, "tx1 value forwarded");
        expect_true(out_variance == expected_var, "tx1 variance exact");
        expect_true(out_prov != k_prov, "tx1 provenance merged");
        expect_true(out_flags == k_flags, "tx1 flags forwarded");
        conf_alpha = out_conf[127:64];
        conf_beta = out_conf[63:0];
        expect_true(conf_alpha == 64'd42, "tx1 alpha incremented");
        expect_true(conf_beta == 64'd71, "tx1 beta incremented");
        expect_true(gum_fidelity_inc == 32'd1, "tx1 gum counter");
        expect_true(provenance_depth_inc == 32'd1, "tx1 provenance counter");
        expect_true(formal_coverage_inc == 32'd1, "tx1 formal counter");

        // Drain first packet so next transfer can proceed.
        downstream_ready = 1'b1;
        #10;
        downstream_ready = 1'b0;
        seen_transaction_done = 1'b0;
        seen_fidelity_pulse = 1'b0;
        seen_merkle_req = 1'b0;

        // TX2: OP_FMA with larger alpha_q8 degradation.
        k_value = 64'd68719476736; // 2^36
        k_variance = 64'd256;
        k_conf = {64'd100, 64'd200};
        k_flags = {8'd16, 8'd3, 8'd0, 4'd0, 4'd4};
        start_transaction = 1'b1;
        #10;
        start_transaction = 1'b0;

        #20;
        expected_var = (k_variance + (k_variance >> 1) + (k_variance >> 2) + (k_variance >> 4));
        expected_var = expected_var + ((expected_var * 16) >> 8);
        expect_true(out_valid == 1'b1, "slave out_valid asserted");
        expect_true(seen_transaction_done == 1'b1, "tx2 transaction_done pulse");
        expect_true(seen_fidelity_pulse == 1'b1, "tx2 master fidelity increment");
        expect_true(out_value == k_value, "tx2 value forwarded");
        expect_true(out_variance == expected_var, "tx2 variance exact");
        expect_true(out_flags == k_flags, "tx2 flags forwarded");
        conf_alpha = out_conf[127:64];
        conf_beta = out_conf[63:0];
        expect_true(conf_alpha == 64'd103, "tx2 alpha incremented");
        expect_true(conf_beta == 64'd200, "tx2 beta unchanged");

        expect_true(gum_fidelity_inc == 32'd2, "tx2 gum counter incremented");
        expect_true(provenance_depth_inc == 32'd2, "tx2 provenance counter incremented");
        expect_true(formal_coverage_inc == 32'd2, "tx2 formal counter incremented");

        // Drain second packet so Merkle-lane TX can proceed.
        downstream_ready = 1'b1;
        #10;
        downstream_ready = 1'b0;
        seen_transaction_done = 1'b0;
        seen_fidelity_pulse = 1'b0;
        seen_merkle_req = 1'b0;

        // TX3: Inject provenance from Merkle lane when valid.
        k_value = 64'd123456;
        k_variance = 64'd32;
        k_prov = 256'hDEAD_BEEF_0001_0002_0003_0004_0005_0006_0007_0008_0009_000A_000B_000C_000D_000E;
        k_conf = {64'd55, 64'd89};
        k_flags = {8'd4, 8'd1, 8'd1, 4'd0, 4'd2};
        merkle_lane_digest = 256'hCAFE_BABE_1111_2222_3333_4444_5555_6666_7777_8888_9999_AAAA_BBBB_CCCC_DDDD_EEEE;
        merkle_lane_valid = 1'b1;
        start_transaction = 1'b1;
        #10;
        start_transaction = 1'b0;
        #20;
        merkle_lane_valid = 1'b0;

        #10;
        expected_merkle_seed = {k_prov[247:0], k_prov[255:248]}
            ^ {192'd0, k_value}
            ^ {224'd0, k_flags}
            ^ PROV_SALT;
        expect_true(out_valid == 1'b1, "tx3 out_valid asserted");
        expect_true(seen_transaction_done == 1'b1, "tx3 transaction_done pulse");
        expect_true(seen_fidelity_pulse == 1'b1, "tx3 master fidelity increment");
        expect_true(seen_merkle_req == 1'b1, "tx3 merkle request pulse");
        expect_true(merkle_lane_seed == expected_merkle_seed, "tx3 merkle seed fallback fold");
        expect_true(out_prov == merkle_lane_digest, "tx3 provenance sourced from merkle lane");
        expected_var = (k_variance + (k_variance >> 1) + (k_variance >> 3));
        expected_var = expected_var + ((expected_var * 4) >> 8);
        expect_true(out_variance == expected_var, "tx3 mul variance profile");
        expect_true(gum_fidelity_inc == 32'd3, "tx3 gum counter incremented");
        expect_true(provenance_depth_inc == 32'd3, "tx3 provenance counter incremented");
        expect_true(formal_coverage_inc == 32'd3, "tx3 formal counter incremented");

        if (failures != 0) begin
            $display("FAIL tb_k_axi_bridge cases=%0d", failures);
            $finish(1);
        end

        $display("OMEGA_COUNTERS kaxi_gum=%0d kaxi_prov=%0d kaxi_formal=%0d", gum_fidelity_inc, provenance_depth_inc, formal_coverage_inc);
        $display("PASS tb_k_axi_bridge");
        $finish(0);
    end
endmodule

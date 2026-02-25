`timescale 1ns/1ps

module tb_epistemic_power_accumulator;
    reg clk;
    reg rst_n;
    reg [31:0] fidelity_inc;
    reg [31:0] prov_depth_inc;
    reg [31:0] quantum_fidelity_inc;
    reg [31:0] quantum_controller_inc;
    reg clear_counters;

    wire [63:0] epistemic_power_log;
    wire [31:0] total_transactions;

    integer failures;
    reg [63:0] expected_log;
    reg [1023:0] dumpfile_path;

    epistemic_power_accumulator dut (
        .aclk(clk),
        .aresetn(rst_n),
        .fidelity_inc(fidelity_inc),
        .prov_depth_inc(prov_depth_inc),
        .quantum_fidelity_inc(quantum_fidelity_inc),
        .quantum_controller_inc(quantum_controller_inc),
        .clear_counters(clear_counters),
        .epistemic_power_log(epistemic_power_log),
        .total_transactions(total_transactions)
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
            dumpfile_path = "artifacts/fpga/waveforms/tb_epistemic_power_accumulator.vcd";
        end
        $dumpfile(dumpfile_path);
        $dumpvars(0, tb_epistemic_power_accumulator);

        clk = 1'b0;
        rst_n = 1'b0;
        fidelity_inc = 32'd0;
        prov_depth_inc = 32'd0;
        quantum_fidelity_inc = 32'd0;
        quantum_controller_inc = 32'd0;
        clear_counters = 1'b0;
        failures = 0;
        expected_log = 64'd0;

        #20;
        rst_n = 1'b1;
        #10;

        // TX1
        fidelity_inc = 32'd8;
        prov_depth_inc = 32'd4;
        quantum_fidelity_inc = 32'd2;
        quantum_controller_inc = 32'd30;
        #10;
        expected_log = ((64'd8 >> 3) + (64'd4 >> 4) + ((64'd2 + 64'd30) >> 5));
        expect_true(epistemic_power_log == expected_log, "tx1 weighted log");
        expect_true(total_transactions == 32'd1, "tx1 transaction count");

        // TX2
        fidelity_inc = 32'd16;
        prov_depth_inc = 32'd16;
        quantum_fidelity_inc = 32'd32;
        quantum_controller_inc = 32'd32;
        #10;
        expected_log = ((64'd24 >> 3) + (64'd20 >> 4) + (64'd96 >> 5));
        expect_true(epistemic_power_log == expected_log, "tx2 weighted log");
        expect_true(total_transactions == 32'd2, "tx2 transaction count");

        // clear path
        clear_counters = 1'b1;
        fidelity_inc = 32'd0;
        prov_depth_inc = 32'd0;
        quantum_fidelity_inc = 32'd0;
        quantum_controller_inc = 32'd0;
        #10;
        clear_counters = 1'b0;
        #10;
        expect_true(epistemic_power_log == 64'd0, "clear resets log");
        expect_true(total_transactions == 32'd0, "clear resets transactions");

        if (failures != 0) begin
            $display("FAIL tb_epistemic_power_accumulator cases=%0d", failures);
            $finish(1);
        end

        $display("OMEGA_COUNTERS accum_log=%0d accum_tx=%0d", epistemic_power_log, total_transactions);
        $display("PASS tb_epistemic_power_accumulator");
        $finish(0);
    end
endmodule

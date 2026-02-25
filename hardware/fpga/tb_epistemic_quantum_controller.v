`timescale 1ns/1ps

module tb_epistemic_quantum_controller;
    reg clk;
    reg rst_n;
    reg [63:0] classical_value;
    reg [63:0] classical_var;
    reg [255:0] classical_prov;
    reg [127:0] classical_conf;
    reg [1:0] qubit_meas;
    reg [31:0] shots;
    reg [63:0] calib_t1;
    reg [63:0] calib_t2;
    reg [31:0] gate_error_rate_ppm;
    reg enable_fusion;

    wire [63:0] updated_value;
    wire [63:0] updated_var;
    wire [255:0] updated_prov;
    wire [127:0] updated_conf;
    wire [31:0] quantum_controller_inc;
    wire fusion_done;

    integer failures;
    reg [1023:0] dumpfile_path;

    epistemic_quantum_controller dut (
        .clk(clk),
        .rst_n(rst_n),
        .classical_value(classical_value),
        .classical_var(classical_var),
        .classical_prov(classical_prov),
        .classical_conf(classical_conf),
        .qubit_meas(qubit_meas),
        .shots(shots),
        .calib_t1(calib_t1),
        .calib_t2(calib_t2),
        .gate_error_rate_ppm(gate_error_rate_ppm),
        .updated_value(updated_value),
        .updated_var(updated_var),
        .updated_prov(updated_prov),
        .updated_conf(updated_conf),
        .quantum_controller_inc(quantum_controller_inc),
        .enable_fusion(enable_fusion),
        .fusion_done(fusion_done)
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
            dumpfile_path = "artifacts/fpga/waveforms/tb_epistemic_quantum_controller.vcd";
        end
        $dumpfile(dumpfile_path);
        $dumpvars(0, tb_epistemic_quantum_controller);

        clk = 0;
        rst_n = 0;
        classical_value = 64'd0;
        classical_var = 64'd0;
        classical_prov = 256'd0;
        classical_conf = 128'd0;
        qubit_meas = 2'b00;
        shots = 32'd0;
        calib_t1 = 64'd0;
        calib_t2 = 64'd0;
        gate_error_rate_ppm = 32'd0;
        enable_fusion = 0;
        failures = 0;

        #20;
        rst_n = 1;

        // Case 1: deterministic |1> measurement increases alpha.
        classical_value = 64'd100;
        classical_var = 64'd10;
        classical_prov = 256'hA5A5;
        classical_conf = {64'd10, 64'd20};
        qubit_meas = 2'b01;
        shots = 32'd16;
        calib_t1 = 64'd50;
        calib_t2 = 64'd100;
        gate_error_rate_ppm = 32'd2000;
        enable_fusion = 1'b1;

        #20;
        expect_true(fusion_done == 1'b1, "fusion_done asserted");
        expect_true(updated_value == classical_value, "value forwarded");
        expect_true(updated_var >= classical_var, "variance increased");
        expect_true(updated_prov != classical_prov, "provenance updated");
        expect_true(updated_conf[127:64] >= 64'd10, "alpha updated");
        expect_true(updated_conf[63:0] >= 64'd20, "beta updated");
        expect_true(quantum_controller_inc == 32'd32, "quantum controller lane pulse");

        // Case 2: disable fusion returns fusion_done low.
        enable_fusion = 1'b0;
        #10;
        expect_true(fusion_done == 1'b0, "fusion_done deasserted when disabled");
        expect_true(quantum_controller_inc == 32'd0, "quantum controller lane clears");

        if (failures != 0) begin
            $display("FAIL tb_epistemic_quantum_controller cases=%0d", failures);
            $finish(1);
        end

        $display("PASS tb_epistemic_quantum_controller");
        $finish(0);
    end
endmodule

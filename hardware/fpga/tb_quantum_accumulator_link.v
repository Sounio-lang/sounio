`timescale 1ns/1ps

module tb_quantum_accumulator_link;
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

    reg [31:0] fidelity_inc;
    reg [31:0] prov_depth_inc;
    reg [31:0] quantum_fidelity_inc;
    reg clear_counters;

    wire [63:0] epistemic_power_log;
    wire [31:0] total_transactions;

    integer failures;
    reg seen_fusion_pulse;
    reg seen_quantum_lane_pulse;
    reg [1023:0] dumpfile_path;

    epistemic_quantum_controller qctrl (
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

    epistemic_power_accumulator accum (
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

    always @(posedge clk) begin
        if (!rst_n) begin
            seen_fusion_pulse <= 1'b0;
            seen_quantum_lane_pulse <= 1'b0;
        end else begin
            if (fusion_done) begin
                seen_fusion_pulse <= 1'b1;
            end
            if (quantum_controller_inc != 32'd0) begin
                seen_quantum_lane_pulse <= 1'b1;
            end
        end
    end

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
            dumpfile_path = "artifacts/fpga/waveforms/tb_quantum_accumulator_link.vcd";
        end
        $dumpfile(dumpfile_path);
        $dumpvars(0, tb_quantum_accumulator_link);

        clk = 1'b0;
        rst_n = 1'b0;

        classical_value = 64'd0;
        classical_var = 64'd0;
        classical_prov = 256'd0;
        classical_conf = 128'd0;
        qubit_meas = 2'b00;
        shots = 32'd0;
        calib_t1 = 64'd0;
        calib_t2 = 64'd0;
        gate_error_rate_ppm = 32'd0;
        enable_fusion = 1'b0;

        fidelity_inc = 32'd0;
        prov_depth_inc = 32'd0;
        quantum_fidelity_inc = 32'd0;
        clear_counters = 1'b0;

        failures = 0;
        seen_fusion_pulse = 1'b0;
        seen_quantum_lane_pulse = 1'b0;

        #20;
        rst_n = 1'b1;
        #10;

        // Drive one controller fusion pulse into the accumulator quantum lane.
        classical_value = 64'd100;
        classical_var = 64'd8;
        classical_prov = 256'h1111;
        classical_conf = {64'd5, 64'd9};
        qubit_meas = 2'b10;
        shots = 32'd32;
        calib_t1 = 64'd7;
        calib_t2 = 64'd9;
        gate_error_rate_ppm = 32'd500;
        enable_fusion = 1'b1;

        #10;
        enable_fusion = 1'b0;

        #30;
        expect_true(seen_fusion_pulse == 1'b1, "fusion pulse observed");
        expect_true(seen_quantum_lane_pulse == 1'b1, "quantum controller lane observed");
        // quantum_controller_inc=32 contributes one unit via accumulator >>5 lane.
        expect_true(epistemic_power_log == 64'd1, "quantum lane contributes to accumulator log");
        expect_true(total_transactions == 32'd0, "no fidelity lane transaction counted");

        if (failures != 0) begin
            $display("FAIL tb_quantum_accumulator_link cases=%0d", failures);
            $finish(1);
        end

        $display("OMEGA_COUNTERS qlink_seen=%0d qlink_log=%0d qlink_tx=%0d", seen_quantum_lane_pulse, epistemic_power_log, total_transactions);
        $display("OMEGA_COUNTERS qlink_seen=%0d qlink_log=%0d qlink_tx=%0d", seen_quantum_lane_pulse, epistemic_power_log, total_transactions);
        $display("PASS tb_quantum_accumulator_link");
        $finish(0);
    end
endmodule

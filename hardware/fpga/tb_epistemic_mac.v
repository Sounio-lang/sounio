`timescale 1ns/1ps

module tb_epistemic_mac;
    reg signed [63:0] a_value;
    reg signed [63:0] a_var;
    reg signed [63:0] b_value;
    reg signed [63:0] b_var;
    reg signed [63:0] c_value;
    reg signed [63:0] c_var;
    reg [255:0] a_prov;
    reg [255:0] b_prov;
    reg [255:0] c_prov;

    wire signed [63:0] out_value;
    wire signed [63:0] out_var;
    wire [255:0] out_prov;
    integer failures;
    reg [1023:0] dumpfile_path;

    epistemic_mac dut (
        .a_value(a_value),
        .a_var(a_var),
        .b_value(b_value),
        .b_var(b_var),
        .c_value(c_value),
        .c_var(c_var),
        .a_prov(a_prov),
        .b_prov(b_prov),
        .c_prov(c_prov),
        .out_value(out_value),
        .out_var(out_var),
        .out_prov(out_prov)
    );

    task run_case;
        input signed [63:0] in_a_value;
        input signed [63:0] in_a_var;
        input signed [63:0] in_b_value;
        input signed [63:0] in_b_var;
        input signed [63:0] in_c_value;
        input signed [63:0] in_c_var;
        input [255:0] in_a_prov;
        input [255:0] in_b_prov;
        input [255:0] in_c_prov;
        input signed [63:0] expected_value;
        input signed [63:0] expected_var;
        input [255:0] expected_prov;
        begin
            a_value = in_a_value;
            a_var = in_a_var;
            b_value = in_b_value;
            b_var = in_b_var;
            c_value = in_c_value;
            c_var = in_c_var;
            a_prov = in_a_prov;
            b_prov = in_b_prov;
            c_prov = in_c_prov;

            #1;
            if (out_value !== expected_value) begin
                $display("FAIL out_value expected %0d got %0d", expected_value, out_value);
                failures = failures + 1;
            end

            if (out_var !== expected_var) begin
                $display("FAIL out_var expected %0d got %0d", expected_var, out_var);
                failures = failures + 1;
            end

            if (out_prov !== expected_prov) begin
                $display("FAIL out_prov expected %h got %h", expected_prov, out_prov);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        if (!$value$plusargs("dumpfile=%s", dumpfile_path)) begin
            dumpfile_path = "artifacts/fpga/waveforms/tb_epistemic_mac.vcd";
        end
        $dumpfile(dumpfile_path);
        $dumpvars(0, tb_epistemic_mac);

        failures = 0;

        // Case 1: positive values
        // out_value = 3*4 + 5 = 17
        // out_var = |4|^2*2 + |3|^2*1 + 1 = 42
        run_case(
            64'sd3, 64'sd2, 64'sd4, 64'sd1, 64'sd5, 64'sd1,
            256'h0001, 256'h0002, 256'h0004,
            64'sd17, 64'sd42, 256'h0007
        );

        // Case 2: signed values
        // out_value = (-2)*5 + (-1) = -11
        // out_var = |5|^2*3 + |2|^2*4 + 2 = 93
        run_case(
            -64'sd2, 64'sd3, 64'sd5, 64'sd4, -64'sd1, 64'sd2,
            256'h0010, 256'h0003, 256'h0001,
            -64'sd11, 64'sd93, 256'h0012
        );

        if (failures != 0) begin
            $display("FAIL tb_epistemic_mac cases=%0d", failures);
            $finish(1);
        end

        $display("PASS tb_epistemic_mac");
        $finish(0);
    end
endmodule

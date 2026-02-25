// Sounio Omega - Epistemic Quantum Controller (hardware seed)
// Synthesizable fixed-point/integer approximation of GUM + decoherence update.

`timescale 1ns/1ps

module epistemic_quantum_controller #(
    parameter WIDTH_VALUE = 64,
    parameter WIDTH_VAR = 64,
    parameter WIDTH_PROV = 256,
    parameter WIDTH_CONF = 128
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Classical Knowledge state
    input  wire [WIDTH_VALUE-1:0]  classical_value,
    input  wire [WIDTH_VAR-1:0]    classical_var,
    input  wire [WIDTH_PROV-1:0]   classical_prov,
    input  wire [WIDTH_CONF-1:0]   classical_conf,

    // Quantum calibration + measurement metadata
    input  wire [1:0]              qubit_meas,
    input  wire [31:0]             shots,
    input  wire [63:0]             calib_t1,
    input  wire [63:0]             calib_t2,
    input  wire [31:0]             gate_error_rate_ppm,

    // Updated epistemic state
    output reg  [WIDTH_VALUE-1:0]  updated_value,
    output reg  [WIDTH_VAR-1:0]    updated_var,
    output reg  [WIDTH_PROV-1:0]   updated_prov,
    output reg  [WIDTH_CONF-1:0]   updated_conf,
    output reg  [31:0]             quantum_controller_inc,

    input  wire                    enable_fusion,
    output reg                     fusion_done
);

    reg [127:0] value_sq;
    reg [63:0] t1_term;
    reg [63:0] t2_term;
    reg [63:0] shot_term;
    reg [4:0] t1_shift;
    reg [4:0] t2_shift;
    reg [4:0] shot_shift;
    reg [95:0] gate_term_full;
    reg [63:0] gate_term;
    reg [63:0] decoherence_var;

    reg [63:0] alpha;
    reg [63:0] beta;
    reg [63:0] success_shots;
    reg [63:0] failure_shots;
    reg [63:0] alpha_next;
    reg [63:0] beta_next;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            updated_value <= {WIDTH_VALUE{1'b0}};
            updated_var <= {WIDTH_VAR{1'b0}};
            updated_prov <= {WIDTH_PROV{1'b0}};
            updated_conf <= {WIDTH_CONF{1'b0}};
            quantum_controller_inc <= 32'd0;
            fusion_done <= 1'b0;

            value_sq <= 128'd0;
            t1_term <= 64'd0;
            t2_term <= 64'd0;
            shot_term <= 64'd0;
            t1_shift <= 5'd0;
            t2_shift <= 5'd0;
            shot_shift <= 5'd0;
            gate_term_full <= 96'd0;
            gate_term <= 64'd0;
            decoherence_var <= 64'd0;
            alpha <= 64'd0;
            beta <= 64'd0;
            success_shots <= 64'd0;
            failure_shots <= 64'd0;
            alpha_next <= 64'd0;
            beta_next <= 64'd0;
        end else if (enable_fusion) begin
            updated_value <= classical_value;

            // value_sq uses 128-bit lane to avoid truncation before scaling.
            value_sq <= classical_value * classical_value;

            // Integer approximations for decoherence terms.
            // Avoiding runtime division keeps synthesis fast for Sprint 1.
            t1_shift <= calib_t1[4:0];
            t2_shift <= calib_t2[4:0];

            if (t1_shift != 5'd0) begin
                t1_term <= value_sq[95:32] >> t1_shift;
            end else begin
                t1_term <= value_sq[95:32];
            end

            if (t2_shift != 5'd0) begin
                t2_term <= value_sq[95:32] >> t2_shift;
            end else begin
                t2_term <= value_sq[95:32];
            end

            // Shot-noise proxy: reciprocal approximation with shift buckets.
            shot_shift <= shots[4:0];
            if (shots != 32'd0) begin
                shot_term <= (64'd1 << 24) >> shot_shift;
            end else begin
                shot_term <= 64'd0;
            end

            // Gate error in ppm, scaled down by ~2^20 for fixed-point stability.
            gate_term_full <= classical_value * gate_error_rate_ppm;
            gate_term <= gate_term_full[95:20];

            decoherence_var <= classical_var + t1_term + t2_term + shot_term + gate_term;
            updated_var <= decoherence_var;

            // Provenance folding; Merkle core can replace this path in v2.
            updated_prov <= classical_prov ^ {
                32'h0,
                calib_t1,
                calib_t2,
                gate_error_rate_ppm,
                shots,
                qubit_meas,
                30'h0
            };

            // Beta(alpha,beta) update with measurement-derived successes/failures.
            alpha <= classical_conf[127:64];
            beta <= classical_conf[63:0];

            if (qubit_meas == 2'b01) begin
                success_shots <= {32'd0, shots};
            end else if (qubit_meas == 2'b10) begin
                success_shots <= {33'd0, shots[31:1]};
            end else begin
                success_shots <= 64'd0;
            end

            failure_shots <= {32'd0, shots} - success_shots;
            alpha_next <= alpha + success_shots;
            beta_next <= beta + failure_shots;
            updated_conf <= {alpha_next, beta_next};

            // Export a real hardware lane for downstream EpistemicPower accumulation.
            // 32 aligns with accumulator's >>5 lane weighting to produce a visible unit step.
            quantum_controller_inc <= 32'd32;
            fusion_done <= 1'b1;
        end else begin
            quantum_controller_inc <= 32'd0;
            fusion_done <= 1'b0;
        end
    end

endmodule

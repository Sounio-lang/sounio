// GENERATED_FROM_STDLIB: stdlib/hardware/kaxi.sio
// GENERATED_BY: self-hosted/compiler/codegen/hardware/kaxi_emitter.sio
// Sounio Omega - Hardware Epistemic Power accumulator
// Aggregates hardware counter increments and exposes live log-space score (Q32.32).

`timescale 1ns/1ps

module epistemic_power_accumulator #(
    parameter SCALE_SHIFT = 30
) (
    input  wire         aclk,
    input  wire         aresetn,

    // Counter deltas from epistemic fabric.
    input  wire [31:0]  fidelity_inc,
    input  wire [31:0]  prov_depth_inc,
    input  wire [31:0]  quantum_fidelity_inc,
    input  wire [31:0]  quantum_controller_inc,

    input  wire         clear_counters,

    // Q32.32 log-space approximation.
    output reg  [63:0]  epistemic_power_log,
    output reg  [31:0]  total_transactions
);

    reg [63:0] fidelity_accum;
    reg [63:0] prov_accum;
    reg [63:0] quantum_accum;

    wire [63:0] fidelity_inc64 = {32'd0, fidelity_inc};
    wire [63:0] prov_inc64 = {32'd0, prov_depth_inc};
    wire [63:0] quantum_inc64 = {32'd0, quantum_fidelity_inc};
    wire [63:0] quantum_controller_inc64 = {32'd0, quantum_controller_inc};

    wire [63:0] fidelity_next = fidelity_accum + fidelity_inc64;
    wire [63:0] prov_next = prov_accum + prov_inc64;
    wire [63:0] quantum_next = quantum_accum + quantum_inc64 + quantum_controller_inc64;

    wire [63:0] weighted_log_next =
        (fidelity_next >> 3) + // runtime+gum dominant lane
        (prov_next >> 4) +     // provenance lane
        (quantum_next >> 5);   // quantum lane

    wire tx_seen = |fidelity_inc;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            fidelity_accum <= 64'd0;
            prov_accum <= 64'd0;
            quantum_accum <= 64'd0;
            total_transactions <= 32'd0;
            epistemic_power_log <= 64'd0;
        end else if (clear_counters) begin
            fidelity_accum <= 64'd0;
            prov_accum <= 64'd0;
            quantum_accum <= 64'd0;
            total_transactions <= 32'd0;
            epistemic_power_log <= 64'd0;
        end else begin
            fidelity_accum <= fidelity_next;
            prov_accum <= prov_next;
            quantum_accum <= quantum_next;
            total_transactions <= total_transactions + {31'd0, tx_seen};
            epistemic_power_log <= weighted_log_next;
        end
    end

endmodule

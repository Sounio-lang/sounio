// Sounio Omega Sprint 1 FPGA seed: epistemic MAC lane
// Value/variance paths are modeled in signed fixed-point lanes.

module epistemic_mac (
    input  signed [63:0] a_value,
    input  signed [63:0] a_var,
    input  signed [63:0] b_value,
    input  signed [63:0] b_var,
    input  signed [63:0] c_value,
    input  signed [63:0] c_var,
    input         [255:0] a_prov,
    input         [255:0] b_prov,
    input         [255:0] c_prov,
    output signed [63:0] out_value,
    output signed [63:0] out_var,
    output        [255:0] out_prov
);

    wire signed [127:0] mul_full = a_value * b_value;
    wire signed [63:0] mul_lo = mul_full[63:0];

    assign out_value = mul_lo + c_value;

    // Delta-method proxy in fixed-point lanes:
    // var = |b|^2 * var(a) + |a|^2 * var(b) + var(c)
    wire signed [63:0] abs_a = a_value[63] ? -a_value : a_value;
    wire signed [63:0] abs_b = b_value[63] ? -b_value : b_value;

    wire signed [127:0] bb_full = abs_b * abs_b;
    wire signed [127:0] aa_full = abs_a * abs_a;

    wire signed [127:0] term1_full = bb_full[63:0] * a_var;
    wire signed [127:0] term2_full = aa_full[63:0] * b_var;

    wire signed [63:0] term1 = term1_full[63:0];
    wire signed [63:0] term2 = term2_full[63:0];

    assign out_var = term1 + term2 + c_var;

    // Provenance merge seed (placeholder for future Merkle core).
    assign out_prov = a_prov ^ b_prov ^ c_prov;

endmodule

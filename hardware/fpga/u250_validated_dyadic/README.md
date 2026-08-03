# U250 validated dyadic arithmetic nucleus

This experimental HLS kernel implements signed S1.I31.F96 interval addition,
subtraction, multiplication with exact 256-bit intermediates, and division by a
small exact positive integer. It is an arithmetic KAT, not an ODE certificate.

Generate vectors with `scripts/research/cs6_u250_validated_dyadic_generate.py`,
then run Vitis HLS C simulation with `run_hls_csim.tcl`. The frozen contract is
`scripts/research/cs6_u250_validated_dyadic_contract_v1.txt`.

# Pre-execution review disposition

XAI classified both the frozen arithmetic contract and novelty-window report as
containing no standalone derivation. Z.AI verified the 128-by-128-to-256-bit
product width and the directed multiplication formulas. Its contract review
spent most of its response budget resolving whether division meant general
fixed-point division or division by an exact integer scalar. The contract now
states the intended operation explicitly as `A_RAW / D`; general Q-by-Q
division is forbidden in this stage.

Z.AI also requested an explicit fixed-point naming convention. The final
contract uses `S1_I31_F96`: one separate sign bit, 31 integer bits, and 96
fraction bits. No review enabled an ODE, Picard, FPGA-execution, novelty,
priority, promotion, or open-problem claim.

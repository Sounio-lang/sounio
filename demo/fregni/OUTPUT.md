=== Sounio x PPCR demo for Prof. Felipe Fregni ===

--- 1. Reference arithmetic (Python) ---
Scenario 1 reference dose = 32000.00 mg
Scenario 2 confidence 700 permille < threshold 950 permille -> REJECT
Scenario 3 CV = 30.00% > threshold 25.00% -> REJECT
REFERENCE: all checks match demo expectations

--- 2. Confidence-gated dosing pipeline (Sounio/Madares) ---
Madares v0.80.0 -- the Sounio self-hosted compiler
the bare highland that does not negotiate with ill-formed code -- Sfakia, Crete
Horizon 3: self-hosted primary compiler.

native_v2_compile: emitted path=/tmp/madaros-run.vvAz5e/main.elf
=== PPCR dosing gate demo ===
Scenario 1: measured clearance (980 permille confidence)
PASS
32000.0
Scenario 2
FAIL
Scenario 3
FAIL

--- 3. Compile-time provenance guard (expected to FAIL type check) ---
Madares v0.80.0 -- the Sounio self-hosted compiler
the bare highland that does not negotiate with ill-formed code -- Sfakia, Crete
Horizon 3: self-hosted primary compiler.

error[E009
] at 0
..863
: argument type does not match parameter
   |
   = expected MeasuredGUMI64
   = found SimulationI64
   |
   = help: check the function signature and provide an argument of the correct type
   = note: function arguments are checked against the declared parameter types
(Expected failure: simulated value cannot be passed to measured-only extractor.)

=== Demo complete ===

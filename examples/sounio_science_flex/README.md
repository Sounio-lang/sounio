# Sounio Science Flex

One native binary. Five pillars. No Python.

```bash
export SOUNIO_STDLIB_PATH=$PWD/stdlib
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/sounio_science_flex/main.sio
# expect: SOUNIO_SCIENCE_FLEX_OK
```

| Tag | Domain | What it proves |
|---|---|---|
| 3101 | HEP GUM | PDG α_EM → σ(e⁺e⁻→μμ) ≈ 868.54 pb with ppb-class std |
| 3102 | EW GUM + amp | Γ(Z→ee) + NonUnitary pole \|M\|² |
| 3103 | QCD running | α_s(M_Z) > α_s(1 TeV) with PDG uncertainty |
| 3104 | Clinical Knightian | pre-TDM REFUSE / post-TDM PRESCRIBE on vancomycin trough |
| 3105 | Non-associative algebra | octonion Fano associator GUM var = 0.64 exact |

Gate: `bash scripts/ci/sounio_science_flex_gate.sh`

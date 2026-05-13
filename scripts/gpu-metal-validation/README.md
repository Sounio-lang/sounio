# Metal F₃ Validation

End-to-end test: sedenion F₃ L¹-associator mass on Apple GPU via Metal.

## What this proves

The MSL kernel in `stdlib/gpu/sedenion_f3.metal` (Sounio Metal backend output)
produces the correct numerical result on real Apple Silicon hardware.

Reference: `rng(20260421).standard_normal(16)` → F₃ ≈ 85.47

## Run on Mac (requires Xcode Command Line Tools)

```bash
cd /path/to/sounio
cp stdlib/gpu/sedenion_f3.metal scripts/gpu-metal-validation/
cd scripts/gpu-metal-validation
bash run_sed_f3_metal.sh
```

Expected output:
```
F3 = 85.47XXXX  (expected 85.47±0.01)
PASS: gpu_sedenion_f3 Metal
```

## Files

- `sedenion_f3.metal` — MSL kernel (Sounio Metal backend output, f32)
- `sed_f3_metal_runner.swift` — Metal dispatch harness
- `run_sed_f3_metal.sh` — build + run script

## Sounio CPU validation (any platform)

```bash
./bin/souc run tests/run-pass/gpu_sedenion_f3.sio
# PASS: gpu_sedenion_f3
```

# Golden expected vectors — PBPK 2-compartment ODE

Pre-computed expected outputs for the f32 / f32-epistemic 2-compartment
4-step PBPK ODE matrix kernels in `slurm-jobs/kretikos/submit-kaxi-ptx-matrix.sh`.

These are bit-exact f32-rounded reference values consumed by the matrix
gate to validate GPU kernel outputs. The matrix uses fixed parameters
k_in=1.0, k12=0.5, k21=0.25, dt=0.5, and N=1024 threads with C1_0=tid+1,
C2_0=0, σ²(C1)=1, σ²(C2)=0. Every operation is forced through f32
rounding (single-precision pack/unpack) so the host-side reference
matches the GPU kernel byte-for-byte.

| File                  | Pattern                              | Lane          |
|-----------------------|--------------------------------------|---------------|
| `emem_2c4mb.txt`      | `pbpk2c4_mb` (32×32, f32_2c)         | mem (C1)      |
| `evar_2c4mb.txt`      | `pbpk2c4_mb` (32×32, f32_2c)         | var (C2)      |
| `emem_2ce4_1024.txt`  | `pbpk2c_e4_1024` (1×1024, f32e)      | mem (C1‖C2)   |
| `evar_2ce4_1024.txt`  | `pbpk2c_e4_1024` (1×1024, f32e)      | var (vC1‖vC2) |

Captured from the four `python3 -c "..."` heredocs that previously lived
inline in `submit-kaxi-ptx-matrix.sh`. To regenerate, see the docstring
at the top of each file (the original Python expression is checked into
git history at the heredoc-removal commit).

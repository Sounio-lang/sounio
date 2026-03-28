# Door 3: Octonion State Space Model + Epistemic KAN

## Mission

Implement two novel architectures in Sounio that no other language or framework can express:

1. **Octonion State Space Model (O-SSM)**: Mamba-style selective state space with octonion multiplication, where the non-associative associator field is a learned geometric signal
2. **Epistemic KAN (E-KAN)**: Kolmogorov-Arnold Network with learnable activations on edges, where each activation carries GUM uncertainty

Both architectures exist as working examples in `examples/`, with tests that verify correctness, uncertainty propagation, and algebraic properties.

## Context

### What Sounio has that nobody else does

1. **`algebra Octonion over f64 { mul: alternative, non_commutative; reassociate: fano_selective }`** — compiler-enforced octonion algebra with Fano-plane-selective reassociation
2. **GUM uncertainty** through every operation (JCGM 100:2008)
3. **Effect system** tracking what each function can do (`with GPU, Mut, NonAssoc`)
4. **Door 1**: 1024 locals, BSS spill for 65K+ arrays, dynamic Box
5. **Working epistemic transformer** (`examples/epistemic_transformer.sio`, 2096 params, 6/6 PASS)
6. **`stdlib/algebra/octonion.sio`**: g2_3form, su3_branching_quality, associator_deviation_field, fano_line operations

### Why these architectures matter

**O-SSM**: State space models (Mamba/S4) use `h_t = A·h_{t-1} + B·x_t` in ℝⁿ. In 𝕆ⁿ, one octonion multiplication encodes 7 coupled rotations. The associator `[A,B,C] = (AB)C - A(BC)` is a FREE geometric signal — it measures how far the state dynamics deviate from associativity. This is information that real-valued SSMs cannot see.

**E-KAN**: Kolmogorov-Arnold Networks replace fixed activations (ReLU, sigmoid) with learnable functions on edges. Adding GUM uncertainty to each learnable activation means the network knows WHERE its function approximation is confident and where it's uncertain. No existing KAN implementation tracks this.

### References

- Gu & Dao (2023): "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- Liu et al. (2024): "KAN: Kolmogorov-Arnold Networks"
- Baez (2002): "The Octonions" (Bull. AMS)
- JCGM 100:2008: Guide to the Expression of Uncertainty in Measurement

## Architecture: O-SSM

### State transition in 𝕆ⁿ

```
h_t = σ(A ⊗ h_{t-1} + B ⊗ x_t)
y_t = C ⊗ h_t

where:
  h_t ∈ 𝕆^d_state    (octonion-valued hidden state, d_state components)
  x_t ∈ ℝ^d_input     (real-valued input)
  A ∈ 𝕆^(d_state)     (state transition matrix, diagonal in octonion space)
  B ∈ 𝕆^(d_state × d_input)  (input projection)
  C ∈ 𝕆^(d_output × d_state) (output projection)
  ⊗ = octonion multiplication (non-associative, non-commutative)
  σ = component-wise sigmoid (applied to each of 8 real components)
```

### Associator field as learned representation

The associator `[A, B, C]_ijk = (e_i · e_j) · e_k - e_i · (e_j · e_k)` for octonion basis elements encodes which triples are Fano (associative) and which are non-Fano. During sequence processing:

```
assoc_t = norm(associator(A, h_{t-1}, B ⊗ x_t))
```

This scalar measures "how non-associative the current state dynamics are." It can be:
- Concatenated to the output as a geometric feature
- Used as a gating signal (like Mamba's selectivity mechanism)
- Tracked as a time series for interpretability

### Octonion multiplication (Cayley-Dickson)

An octonion `o = (a, b)` where `a, b ∈ ℍ` (quaternions). Multiplication:
```
(a, b) · (c, d) = (ac - d̄b, da + bc̄)
```

In component form (8 reals), this is 120 FLOPs per product. The sign pattern follows the Fano plane.

### Dimensions for the example

```
d_input = 4        (real-valued input per timestep)
d_state = 2        (2 octonion-valued hidden units = 16 real values)
d_output = 4       (real-valued output per timestep)
seq_len = 8        (sequence length)

Parameters:
  A: 2 octonions = 16 f64 (diagonal state transition)
  B: 2 × 4 = 8 octonions = 64 f64 (input projection)
  C: 4 × 2 = 8 octonions = 64 f64 (output projection)
  Total: 144 f64 values + 144 uncertainties = 288 f64
```

### GUM through octonion multiplication

For `z = x ⊗ y` where `x = (x0..x7)` and `y = (y0..y7)`:
```
σ_z_k² = Σ_i Σ_j (∂z_k/∂x_i)² · σ_xi² + (∂z_k/∂y_j)² · σ_yj²
```

The Jacobian `∂z/∂x` for octonion multiplication is a 8×8 matrix determined by the Fano plane structure table. For the Cayley-Dickson convention:
- Fano triples: `∂z_k/∂x_i = ±y_j` (sign from structure constants)
- All 8×8 entries are populated (dense Jacobian)

First-order GUM approximation:
```
σ_z² ≈ Σ_i (y_fano(i))² · σ_xi² + Σ_j (x_fano(j))² · σ_yj²
```

## Architecture: E-KAN

### Learnable edge activations with uncertainty

Standard KAN: `f(x) = Σ_i φ_i(x_i)` where each `φ_i` is a learnable univariate function (B-spline).

E-KAN: Each `φ_i` carries uncertainty `σ_φ(x)` that depends on the input region:
```
φ_i(x) = Σ_k c_k · B_k(x)     (B-spline with coefficients c_k)
σ_φ(x) = Σ_k σ_ck · |B_k(x)|  (uncertainty from coefficient uncertainty)
```

Where `B_k(x)` is the k-th B-spline basis function and `c_k ± σ_ck` are the learnable coefficients with GUM uncertainty.

### Dimensions for the example

```
d_input = 4
d_hidden = 8
d_output = 2
n_knots = 6 (B-spline order 3 with 6 knots)

Parameters per edge:
  6 B-spline coefficients + 6 uncertainties = 12 f64
Edges:
  Layer 1: 4 × 8 = 32 edges → 384 f64
  Layer 2: 8 × 2 = 16 edges → 192 f64
  Total: 576 f64 values
```

### B-spline evaluation

For order-3 B-splines with uniform knots on [0, 1]:
```
B_k(x) = cubic basis function centered at knot k
φ(x) = Σ_k c_k · B_k(x)
φ'(x) = Σ_k c_k · B_k'(x)  (for gradient computation)
```

Simplified for the demo: use piecewise linear basis functions (order 1) instead of cubic B-splines. Each basis function is a hat function centered at a knot.

## Required Files

### `examples/octonion_ssm.sio` (~300 lines)

```sio
// Octonion State Space Model
// h_t = sigmoid(A ⊗ h_{t-1} + B ⊗ x_t)
// y_t = Re(C ⊗ h_t)  (project to real output)

struct Oct { e: [f64; 8] }  // octonion with 8 components

fn oct_mul(a: &Oct, b: &Oct) -> Oct { ... }  // Cayley-Dickson
fn oct_add(a: &Oct, b: &Oct) -> Oct { ... }
fn oct_norm(a: &Oct) -> f64 { ... }
fn oct_sigmoid(a: &Oct) -> Oct { ... }  // component-wise
fn oct_associator(a: &Oct, b: &Oct, c: &Oct) -> Oct { ... }  // (ab)c - a(bc)

// GUM through octonion ops
fn oct_mul_gum(a: &Oct, b: &Oct, a_u: &Oct, b_u: &Oct) -> (Oct, Oct) { ... }

// SSM step
fn ossm_step(h: &Oct, x: &[f64; 4], A: &Oct, B: &[Oct; 4],
             h_u: &Oct, A_u: &Oct, B_u: &[Oct; 4]) -> (Oct, Oct) { ... }

// Full sequence
fn ossm_forward(seq: &[f64; 32], A: &Oct, B: &[Oct; 4], C: &[Oct; 4],
                uncertainties...) -> ([f64; 32], [f64; 32]) { ... }
```

Tests:
- T1: Octonion multiplication follows Cayley-Dickson (verify e1·e2 = e4)
- T2: Associator is non-zero for non-Fano triples
- T3: SSM processes 8-step sequence producing non-trivial output
- T4: GUM uncertainty propagates through all timesteps
- T5: Associator norm varies across timesteps (geometric signal present)
- T6: Output uncertainty bounded (not exploding)

### `examples/epistemic_kan.sio` (~250 lines)

```sio
// Epistemic Kolmogorov-Arnold Network
// Each edge has a learnable activation φ(x) = Σ c_k · basis_k(x)
// with uncertainty σ_φ(x) = Σ σ_ck · |basis_k(x)|

struct KANEdge { coeffs: [f64; 6], coeffs_unc: [f64; 6], knots: [f64; 6] }

fn kan_edge_eval(edge: &KANEdge, x: f64) -> (f64, f64) { ... }  // (value, uncertainty)

struct KANLayer { edges: [KANEdge; 32] }  // max 32 edges per layer

fn kan_layer_forward(layer: &KANLayer, input: &[f64; 8], input_unc: &[f64; 8],
                     n_in: i64, n_out: i64) -> ([f64; 8], [f64; 8]) { ... }

fn kan_forward(input: &[f64; 4], input_unc: &[f64; 4],
               layer1: &KANLayer, layer2: &KANLayer) -> ([f64; 2], [f64; 2]) { ... }
```

Tests:
- T1: Edge activation is non-trivial (not constant)
- T2: Edge uncertainty varies with input region (higher near knot boundaries)
- T3: Forward pass produces non-zero output
- T4: GUM uncertainty in all outputs
- T5: Network can approximate sin(x) (universal approximation check)
- T6: Uncertainty is higher far from training data

## Hard Constraints

- **Self-host preservation**: gen2==gen3 (examples are unreachable during bootstrap)
- **No regressions**: All existing run-pass tests must pass
- **Pure Sounio syntax**: `var` not `let mut`, `&!` not `&mut`, no semicolons
- **Self-contained math**: No stdlib calls for exp/sin/sqrt — inline Taylor/Padé
- **Stack-friendly**: Use globals for large arrays, keep functions under 1024 locals
- **Struct returns**: Use named structs with fixed-size arrays (not tuples with `.0` access)
- **Sounio idioms**: Effects (`with Mut, Div, Panic, NonAssoc`), struct wrappers for arrays

## Verification

```bash
./bin/souc run examples/octonion_ssm.sio        # expect: ALL PASS (6+ tests)
./bin/souc run examples/epistemic_kan.sio        # expect: ALL PASS (6+ tests)
./bin/souc run examples/epistemic_transformer.sio # regression: 6/6 PASS
./bin/souc run tests/run-pass/algebra_g2_invariants.sio  # regression
```

## Expected Impact

| Architecture | What's novel | Why it matters |
|---|---|---|
| O-SSM | Non-associative state dynamics with associator as feature | 7 coupled rotations per multiply, geometric signal no real SSM can see |
| E-KAN | Per-edge learnable activation with GUM uncertainty | Knows WHERE its function approximation is uncertain |
| Both | Compile-time algebra constraints + effect tracking | Compiler prevents algebraic mistakes (NonAssoc effect) |

## What This Enables

- **Paper**: "Epistemic State Space Models with Non-Associative Dynamics" — first O-SSM
- **Paper**: "Uncertainty-Aware Kolmogorov-Arnold Networks" — first E-KAN
- **Dissertation**: O-SSM for PBPK drug propagation (octonion state = drug compartments)
- **Harvard PPCR**: PPCR-governed O-SSM experiment with pre-registered hypothesis

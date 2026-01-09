# Matrix Types API Reference

The `linalg` module provides fixed-size matrix types with row-major storage, optimized for scientific computing and cache efficiency.

## Type Overview

| Type | Dimensions | Use Cases |
|------|------------|-----------|
| `Mat2` | 2x2 | 2D transformations, rotation |
| `Mat3` | 3x3 | 3D transformations, Jacobians |
| `Mat4` | 4x4 | Homogeneous coordinates, projections |
| `Mat14Diag` | 14x14 diagonal | PBPK flow rates, scaling |
| `Mat14Tridiag` | 14x14 tridiagonal | Diffusion, implicit solvers |

---

## Mat2

2x2 matrix in row-major storage.

### Type Definition

```sio
struct Mat2 {
    m00: f64, m01: f64,   // Row 0
    m10: f64, m11: f64    // Row 1
}
```

### Layout

```
[ m00  m01 ]
[ m10  m11 ]
```

### Constructors

#### `mat2_new`

Create matrix with explicit elements.

```sio
fn mat2_new(m00: f64, m01: f64, m10: f64, m11: f64) -> Mat2
```

**Example:**
```sio
let m = mat2_new(1.0, 2.0, 3.0, 4.0)
// [ 1  2 ]
// [ 3  4 ]
```

#### `mat2_zero`

Zero matrix.

```sio
fn mat2_zero() -> Mat2
```

#### `mat2_identity`

Identity matrix.

```sio
fn mat2_identity() -> Mat2
```

**Returns:**
```
[ 1  0 ]
[ 0  1 ]
```

#### `mat2_diag`

Diagonal matrix from values.

```sio
fn mat2_diag(d0: f64, d1: f64) -> Mat2
```

### Arithmetic Operations

#### `mat2_add`

Element-wise addition.

```sio
fn mat2_add(a: Mat2, b: Mat2) -> Mat2
```

#### `mat2_sub`

Element-wise subtraction.

```sio
fn mat2_sub(a: Mat2, b: Mat2) -> Mat2
```

#### `mat2_scale`

Scalar multiplication.

```sio
fn mat2_scale(m: Mat2, s: f64) -> Mat2
```

#### `mat2_neg`

Negate all elements.

```sio
fn mat2_neg(m: Mat2) -> Mat2
```

### Matrix Operations

#### `mat2_transpose`

Transpose matrix.

```sio
fn mat2_transpose(m: Mat2) -> Mat2
```

#### `mat2_mul`

Matrix multiplication (A * B).

```sio
fn mat2_mul(a: Mat2, b: Mat2) -> Mat2
```

**Formula:** `C[i,j] = sum(A[i,k] * B[k,j])`

#### `mat2_vec_mul`

Matrix-vector multiplication (M * v).

```sio
fn mat2_vec_mul(m: Mat2, v: Vec2) -> Vec2
```

**Example:**
```sio
let m = mat2_new(1.0, 2.0, 3.0, 4.0)
let v = Vec2 { x: 1.0, y: 1.0 }
let result = mat2_vec_mul(m, v)  // Vec2 { x: 3.0, y: 7.0 }
```

### Matrix Properties

#### `mat2_det`

Determinant.

```sio
fn mat2_det(m: Mat2) -> f64
```

**Formula:** `m00 * m11 - m01 * m10`

#### `mat2_trace`

Trace (sum of diagonal elements).

```sio
fn mat2_trace(m: Mat2) -> f64
```

**Formula:** `m00 + m11`

#### `mat2_frobenius_norm`

Frobenius norm.

```sio
fn mat2_frobenius_norm(m: Mat2) -> f64
```

**Formula:** `sqrt(sum(m[i,j]^2))`

### Matrix Inverse

#### `mat2_inverse`

Compute inverse matrix.

```sio
fn mat2_inverse(m: Mat2) -> Mat2
```

**Notes:**
- Returns identity matrix if determinant < 1e-10 (singular)
- Caller should check determinant before use

**Formula:**
```
[ m11/det  -m01/det ]
[ -m10/det  m00/det ]
```

---

## Mat3

3x3 matrix in row-major storage.

### Type Definition

```sio
struct Mat3 {
    m00: f64, m01: f64, m02: f64,   // Row 0
    m10: f64, m11: f64, m12: f64,   // Row 1
    m20: f64, m21: f64, m22: f64    // Row 2
}
```

### Constructors

#### `mat3_new`

```sio
fn mat3_new(
    m00: f64, m01: f64, m02: f64,
    m10: f64, m11: f64, m12: f64,
    m20: f64, m21: f64, m22: f64
) -> Mat3
```

#### `mat3_zero`

```sio
fn mat3_zero() -> Mat3
```

#### `mat3_identity`

```sio
fn mat3_identity() -> Mat3
```

#### `mat3_diag`

```sio
fn mat3_diag(d0: f64, d1: f64, d2: f64) -> Mat3
```

### Arithmetic Operations

#### `mat3_add`

```sio
fn mat3_add(a: Mat3, b: Mat3) -> Mat3
```

#### `mat3_sub`

```sio
fn mat3_sub(a: Mat3, b: Mat3) -> Mat3
```

#### `mat3_scale`

```sio
fn mat3_scale(m: Mat3, s: f64) -> Mat3
```

#### `mat3_neg`

```sio
fn mat3_neg(m: Mat3) -> Mat3
```

### Matrix Operations

#### `mat3_transpose`

```sio
fn mat3_transpose(m: Mat3) -> Mat3
```

#### `mat3_mul`

```sio
fn mat3_mul(a: Mat3, b: Mat3) -> Mat3
```

#### `mat3_vec_mul`

```sio
fn mat3_vec_mul(m: Mat3, v: Vec3) -> Vec3
```

**Example:**
```sio
let m = mat3_new(
    1.0, 2.0, 3.0,
    0.0, 1.0, 4.0,
    5.0, 6.0, 0.0
)
let v = Vec3 { x: 1.0, y: 2.0, z: 3.0 }
let result = mat3_vec_mul(m, v)  // Vec3 { x: 14.0, y: 14.0, z: 17.0 }
```

### Matrix Properties

#### `mat3_det`

Determinant using cofactor expansion.

```sio
fn mat3_det(m: Mat3) -> f64
```

**Formula (Sarrus rule):**
```
det = m00 * (m11*m22 - m12*m21)
    - m01 * (m10*m22 - m12*m20)
    + m02 * (m10*m21 - m11*m20)
```

#### `mat3_trace`

```sio
fn mat3_trace(m: Mat3) -> f64
```

#### `mat3_frobenius_norm`

```sio
fn mat3_frobenius_norm(m: Mat3) -> f64
```

### Matrix Inverse

#### `mat3_inverse`

```sio
fn mat3_inverse(m: Mat3) -> Mat3
```

**Notes:**
- Uses adjugate matrix method
- Returns identity if singular (det < 1e-10)

### Linear Solver

#### `mat3_solve`

Solve linear system Ax = b.

```sio
fn mat3_solve(a: Mat3, b: Vec3) -> Vec3
```

**Example:**
```sio
// Solve [1 2 3] [x]   [14]
//       [0 1 4] [y] = [14]
//       [5 6 0] [z]   [17]
let A = mat3_new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0)
let b = Vec3 { x: 14.0, y: 14.0, z: 17.0 }
let x = mat3_solve(A, b)  // Vec3 { x: 1.0, y: 2.0, z: 3.0 }
```

---

## Mat4

4x4 matrix for transformations and projections.

### Type Definition

```sio
struct Mat4 {
    m00: f64, m01: f64, m02: f64, m03: f64,   // Row 0
    m10: f64, m11: f64, m12: f64, m13: f64,   // Row 1
    m20: f64, m21: f64, m22: f64, m23: f64,   // Row 2
    m30: f64, m31: f64, m32: f64, m33: f64    // Row 3
}
```

### Constructors

#### `mat4_zero`

```sio
fn mat4_zero() -> Mat4
```

#### `mat4_identity`

```sio
fn mat4_identity() -> Mat4
```

#### `mat4_diag`

```sio
fn mat4_diag(d0: f64, d1: f64, d2: f64, d3: f64) -> Mat4
```

### Arithmetic Operations

#### `mat4_add`

```sio
fn mat4_add(a: Mat4, b: Mat4) -> Mat4
```

#### `mat4_scale`

```sio
fn mat4_scale(m: Mat4, s: f64) -> Mat4
```

### Matrix Operations

#### `mat4_transpose`

```sio
fn mat4_transpose(m: Mat4) -> Mat4
```

#### `mat4_mul`

```sio
fn mat4_mul(a: Mat4, b: Mat4) -> Mat4
```

#### `mat4_vec_mul`

```sio
fn mat4_vec_mul(m: Mat4, v: Vec4) -> Vec4
```

### Matrix Properties

#### `mat4_det`

Determinant using cofactor expansion.

```sio
fn mat4_det(m: Mat4) -> f64
```

**Complexity:** O(n!) via recursive cofactor expansion, but optimized for 4x4.

#### `mat4_trace`

```sio
fn mat4_trace(m: Mat4) -> f64
```

### Matrix Inverse

#### `mat4_inverse`

Inverse using cofactor matrix.

```sio
fn mat4_inverse(m: Mat4) -> Mat4
```

### Linear Solver

#### `mat4_solve`

Solve linear system Ax = b.

```sio
fn mat4_solve(a: Mat4, b: Vec4) -> Vec4
```

---

## Mat14Diag

Diagonal 14x14 matrix for PBPK applications. Only stores diagonal elements.

### Type Definition

```sio
struct Mat14Diag {
    d0: f64, d1: f64, d2: f64, d3: f64, d4: f64,
    d5: f64, d6: f64, d7: f64, d8: f64, d9: f64,
    d10: f64, d11: f64, d12: f64, d13: f64
}
```

### Storage Efficiency

| Storage | Dense 14x14 | Mat14Diag |
|---------|-------------|-----------|
| Elements | 196 | 14 |
| Bytes (f64) | 1568 | 112 |

### Constructors

#### `mat14_diag_new`

```sio
fn mat14_diag_new(
    d0: f64, d1: f64, d2: f64, d3: f64, d4: f64,
    d5: f64, d6: f64, d7: f64, d8: f64, d9: f64,
    d10: f64, d11: f64, d12: f64, d13: f64
) -> Mat14Diag
```

#### `mat14_diag_identity`

```sio
fn mat14_diag_identity() -> Mat14Diag
```

### Arithmetic Operations

#### `mat14_diag_scale`

```sio
fn mat14_diag_scale(m: Mat14Diag, s: f64) -> Mat14Diag
```

#### `mat14_diag_add`

```sio
fn mat14_diag_add(a: Mat14Diag, b: Mat14Diag) -> Mat14Diag
```

### Matrix Operations

#### `mat14_diag_vec_mul`

Matrix-vector multiplication: O(n) instead of O(n^2).

```sio
fn mat14_diag_vec_mul(m: Mat14Diag, v: Vec14) -> Vec14
```

**Formula:** `result[i] = m.d[i] * v.c[i]`

#### `mat14_diag_inverse`

Diagonal inverse: 1/d[i] for each element.

```sio
fn mat14_diag_inverse(m: Mat14Diag) -> Mat14Diag
```

**Note:** Caller must ensure no zero diagonal elements.

### Properties

#### `mat14_diag_trace`

```sio
fn mat14_diag_trace(m: Mat14Diag) -> f64
```

### Linear Solver

#### `mat14_diag_solve`

Solve Dx = b in O(n) time.

```sio
fn mat14_diag_solve(m: Mat14Diag, b: Vec14) -> Vec14
```

**Formula:** `x[i] = b[i] / d[i]`

**Example:**
```sio
let D = mat14_diag_new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                       8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)
let b = vec14_new(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
                  16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0)
let x = mat14_diag_solve(D, b)  // All components = 2.0
```

---

## Mat14Tridiag

Tridiagonal 14x14 matrix for diffusion and implicit ODE solvers.

### Type Definition

```sio
struct Mat14Tridiag {
    // Lower diagonal: l[1..13] (l[0] unused)
    l1: f64, l2: f64, l3: f64, l4: f64, l5: f64, l6: f64,
    l7: f64, l8: f64, l9: f64, l10: f64, l11: f64, l12: f64, l13: f64,

    // Main diagonal: d[0..13]
    d0: f64, d1: f64, d2: f64, d3: f64, d4: f64, d5: f64, d6: f64,
    d7: f64, d8: f64, d9: f64, d10: f64, d11: f64, d12: f64, d13: f64,

    // Upper diagonal: u[0..12] (u[13] unused)
    u0: f64, u1: f64, u2: f64, u3: f64, u4: f64, u5: f64,
    u6: f64, u7: f64, u8: f64, u9: f64, u10: f64, u11: f64, u12: f64
}
```

### Matrix Structure

```
[ d0  u0   0   0  ...  0   0   0  ]
[ l1  d1  u1   0  ...  0   0   0  ]
[  0  l2  d2  u2  ...  0   0   0  ]
[  .   .   .   .   .   .   .   .  ]
[  0   0   0   0  ... l12 d12 u12 ]
[  0   0   0   0  ...  0  l13 d13 ]
```

### Storage Efficiency

| Storage | Dense 14x14 | Mat14Tridiag |
|---------|-------------|--------------|
| Elements | 196 | 40 |
| Bytes (f64) | 1568 | 320 |

### Constructors

#### `mat14_tridiag_identity`

```sio
fn mat14_tridiag_identity() -> Mat14Tridiag
```

### Matrix Operations

#### `mat14_tridiag_vec_mul`

Matrix-vector multiplication: O(n).

```sio
fn mat14_tridiag_vec_mul(m: Mat14Tridiag, v: Vec14) -> Vec14
```

### Linear Solver (Thomas Algorithm)

#### `mat14_tridiag_solve`

Solve Ax = b using the Thomas algorithm in O(n) time.

```sio
fn mat14_tridiag_solve(m: Mat14Tridiag, b: Vec14) -> Vec14
```

**Algorithm:**

1. **Forward elimination:** Modify diagonal and RHS
   ```
   c'[i] = u[i] / (d[i] - l[i] * c'[i-1])
   d'[i] = (b[i] - l[i] * d'[i-1]) / (d[i] - l[i] * c'[i-1])
   ```

2. **Back substitution:**
   ```
   x[n-1] = d'[n-1]
   x[i] = d'[i] - c'[i] * x[i+1]
   ```

**Complexity:** O(n) vs O(n^3) for general LU decomposition

**Example:**
```sio
// 1D Laplacian: -1 on sub/super diagonal, 2 on main
let laplacian = Mat14Tridiag {
    l1: -1.0, l2: -1.0, ..., l13: -1.0,
    d0: 2.0, d1: 2.0, ..., d13: 2.0,
    u0: -1.0, u1: -1.0, ..., u12: -1.0
}
let b = vec14_ones()
let x = mat14_tridiag_solve(laplacian, b)
```

---

## Common Applications

### 2D Rotation

```sio
fn rotation_2d(angle: f64) -> Mat2 {
    let c = cos(angle)
    let s = sin(angle)
    return mat2_new(c, -s, s, c)
}

let rot = rotation_2d(PI / 4.0)  // 45 degree rotation
let v = Vec2 { x: 1.0, y: 0.0 }
let rotated = mat2_vec_mul(rot, v)  // Vec2 { x: 0.707, y: 0.707 }
```

### 3D Rotation (around Z-axis)

```sio
fn rotation_z(angle: f64) -> Mat3 {
    let c = cos(angle)
    let s = sin(angle)
    return mat3_new(
        c, -s, 0.0,
        s,  c, 0.0,
        0.0, 0.0, 1.0
    )
}
```

### Jacobian for ODE Systems

```sio
// Jacobian of 3-compartment PK model
fn pk_jacobian(ka: f64, ke: f64, k12: f64, k21: f64) -> Mat3 {
    return mat3_new(
        -ka,   0.0,   0.0,      // dGut'/d(Gut,Central,Periph)
         ka, -ke-k12, k21,      // dCentral'/d(...)
         0.0,  k12,  -k21       // dPeriph'/d(...)
    )
}
```

### PBPK Implicit Solver

```sio
// Build (I - gamma*h*J) for implicit step
fn build_newton_matrix(J: Mat14Diag, gamma_h: f64) -> Mat14Diag {
    let I = mat14_diag_identity()
    let scaled_J = mat14_diag_scale(J, gamma_h)
    return mat14_diag_add(I, mat14_diag_neg(scaled_J))
}
```

---

## Performance Notes

1. **Row-major storage**: Optimized for row-wise traversal
2. **Fixed size**: No heap allocation, deterministic performance
3. **Specialized solvers**: Diagonal and tridiagonal use O(n) algorithms
4. **Inverse via formula**: Small matrices use closed-form formulas

## Numerical Considerations

1. **Singularity detection**: Operations return identity for det < 1e-10
2. **Condition number**: Not computed; caller responsible for well-conditioning
3. **Pivoting**: Thomas algorithm assumes diagonal dominance

---

## See Also

- [Vector Types](vectors.md)
- [Matrix Decompositions](decompositions.md)
- [ODE Solvers](../ode/solvers.md)

# Vector Types API Reference

The `linalg` module provides fixed-size vector types optimized for scientific computing. These stack-allocated types are designed for SIMD optimization and cache efficiency.

## Type Overview

| Type | Dimensions | Use Cases |
|------|------------|-----------|
| `Vec2` | 2 | 2D coordinates, complex numbers, polar coordinates |
| `Vec3` | 3 | 3D graphics, physics, RGB colors |
| `Vec4` | 4 | Homogeneous coordinates, quaternions, RGBA |
| `Vec14` | 14 | PBPK compartment models, domain-specific |

---

## Vec2

2-element vector for 2D operations.

### Type Definition

```sio
struct Vec2 {
    x: f64,
    y: f64
}
```

### Constructors

#### `vec2_new`

Create a new Vec2 with specified components.

```sio
fn vec2_new(x: f64, y: f64) -> Vec2
```

**Example:**
```sio
let v = vec2_new(3.0, 4.0)
```

#### `vec2_zero`

Create a zero vector.

```sio
fn vec2_zero() -> Vec2
```

**Returns:** `Vec2 { x: 0.0, y: 0.0 }`

#### `vec2_ones`

Create a vector of ones.

```sio
fn vec2_ones() -> Vec2
```

**Returns:** `Vec2 { x: 1.0, y: 1.0 }`

### Arithmetic Operations

#### `vec2_add`

Element-wise addition.

```sio
fn vec2_add(a: Vec2, b: Vec2) -> Vec2
```

**Example:**
```sio
let a = vec2_new(1.0, 2.0)
let b = vec2_new(3.0, 4.0)
let c = vec2_add(a, b)  // Vec2 { x: 4.0, y: 6.0 }
```

#### `vec2_sub`

Element-wise subtraction.

```sio
fn vec2_sub(a: Vec2, b: Vec2) -> Vec2
```

#### `vec2_scale`

Scalar multiplication.

```sio
fn vec2_scale(v: Vec2, s: f64) -> Vec2
```

**Example:**
```sio
let v = vec2_new(2.0, 3.0)
let scaled = vec2_scale(v, 2.0)  // Vec2 { x: 4.0, y: 6.0 }
```

#### `vec2_neg`

Negate all components.

```sio
fn vec2_neg(v: Vec2) -> Vec2
```

#### `vec2_hadamard`

Element-wise (Hadamard) product.

```sio
fn vec2_hadamard(a: Vec2, b: Vec2) -> Vec2
```

**Example:**
```sio
let a = vec2_new(2.0, 3.0)
let b = vec2_new(4.0, 5.0)
let c = vec2_hadamard(a, b)  // Vec2 { x: 8.0, y: 15.0 }
```

### Inner Product and Norms

#### `vec2_dot`

Dot (inner) product.

```sio
fn vec2_dot(a: Vec2, b: Vec2) -> f64
```

**Formula:** `a.x * b.x + a.y * b.y`

**Example:**
```sio
let a = vec2_new(1.0, 2.0)
let b = vec2_new(3.0, 4.0)
let dot = vec2_dot(a, b)  // 11.0
```

#### `vec2_norm_sq`

Squared Euclidean norm (avoids square root).

```sio
fn vec2_norm_sq(v: Vec2) -> f64
```

**Formula:** `x^2 + y^2`

#### `vec2_norm`

Euclidean (L2) norm.

```sio
fn vec2_norm(v: Vec2) -> f64
```

**Formula:** `sqrt(x^2 + y^2)`

**Example:**
```sio
let v = vec2_new(3.0, 4.0)
let n = vec2_norm(v)  // 5.0
```

#### `vec2_normalize`

Return unit vector in same direction.

```sio
fn vec2_normalize(v: Vec2) -> Vec2
```

**Notes:**
- Returns zero vector if input norm < 1e-10
- Result has norm = 1.0 (within numerical precision)

### Component Operations

#### `vec2_max`

Element-wise maximum.

```sio
fn vec2_max(a: Vec2, b: Vec2) -> Vec2
```

#### `vec2_min`

Element-wise minimum.

```sio
fn vec2_min(a: Vec2, b: Vec2) -> Vec2
```

#### `vec2_abs`

Element-wise absolute value.

```sio
fn vec2_abs(v: Vec2) -> Vec2
```

#### `vec2_max_component`

Maximum component value.

```sio
fn vec2_max_component(v: Vec2) -> f64
```

#### `vec2_sum`

Sum of all components.

```sio
fn vec2_sum(v: Vec2) -> f64
```

---

## Vec3

3-element vector for 3D operations. Includes cross product.

### Type Definition

```sio
struct Vec3 {
    x: f64,
    y: f64,
    z: f64
}
```

### Constructors

#### `vec3_new`

```sio
fn vec3_new(x: f64, y: f64, z: f64) -> Vec3
```

#### `vec3_zero`

```sio
fn vec3_zero() -> Vec3
```

#### `vec3_ones`

```sio
fn vec3_ones() -> Vec3
```

### Arithmetic Operations

#### `vec3_add`

```sio
fn vec3_add(a: Vec3, b: Vec3) -> Vec3
```

#### `vec3_sub`

```sio
fn vec3_sub(a: Vec3, b: Vec3) -> Vec3
```

#### `vec3_scale`

```sio
fn vec3_scale(v: Vec3, s: f64) -> Vec3
```

#### `vec3_neg`

```sio
fn vec3_neg(v: Vec3) -> Vec3
```

#### `vec3_hadamard`

```sio
fn vec3_hadamard(a: Vec3, b: Vec3) -> Vec3
```

### Products and Norms

#### `vec3_dot`

Dot product.

```sio
fn vec3_dot(a: Vec3, b: Vec3) -> f64
```

**Formula:** `a.x * b.x + a.y * b.y + a.z * b.z`

#### `vec3_cross`

Cross (vector) product.

```sio
fn vec3_cross(a: Vec3, b: Vec3) -> Vec3
```

**Formula:**
```
result.x = a.y * b.z - a.z * b.y
result.y = a.z * b.x - a.x * b.z
result.z = a.x * b.y - a.y * b.x
```

**Example:**
```sio
let i = vec3_new(1.0, 0.0, 0.0)
let j = vec3_new(0.0, 1.0, 0.0)
let k = vec3_cross(i, j)  // Vec3 { x: 0.0, y: 0.0, z: 1.0 }
```

**Properties:**
- `cross(a, b) = -cross(b, a)` (anti-commutative)
- `cross(a, a) = zero`
- Result is perpendicular to both inputs
- Magnitude = `|a| * |b| * sin(theta)`

#### `vec3_norm_sq`

```sio
fn vec3_norm_sq(v: Vec3) -> f64
```

#### `vec3_norm`

```sio
fn vec3_norm(v: Vec3) -> f64
```

#### `vec3_normalize`

```sio
fn vec3_normalize(v: Vec3) -> Vec3
```

### Component Operations

#### `vec3_max`

```sio
fn vec3_max(a: Vec3, b: Vec3) -> Vec3
```

#### `vec3_min`

```sio
fn vec3_min(a: Vec3, b: Vec3) -> Vec3
```

#### `vec3_abs`

```sio
fn vec3_abs(v: Vec3) -> Vec3
```

#### `vec3_max_component`

```sio
fn vec3_max_component(v: Vec3) -> f64
```

#### `vec3_sum`

```sio
fn vec3_sum(v: Vec3) -> f64
```

---

## Vec4

4-element vector for homogeneous coordinates and quaternions.

### Type Definition

```sio
struct Vec4 {
    x: f64,
    y: f64,
    z: f64,
    w: f64
}
```

### Constructors

#### `vec4_new`

```sio
fn vec4_new(x: f64, y: f64, z: f64, w: f64) -> Vec4
```

#### `vec4_zero`

```sio
fn vec4_zero() -> Vec4
```

#### `vec4_ones`

```sio
fn vec4_ones() -> Vec4
```

### Arithmetic Operations

#### `vec4_add`

```sio
fn vec4_add(a: Vec4, b: Vec4) -> Vec4
```

#### `vec4_sub`

```sio
fn vec4_sub(a: Vec4, b: Vec4) -> Vec4
```

#### `vec4_scale`

```sio
fn vec4_scale(v: Vec4, s: f64) -> Vec4
```

#### `vec4_neg`

```sio
fn vec4_neg(v: Vec4) -> Vec4
```

#### `vec4_hadamard`

```sio
fn vec4_hadamard(a: Vec4, b: Vec4) -> Vec4
```

### Products and Norms

#### `vec4_dot`

```sio
fn vec4_dot(a: Vec4, b: Vec4) -> f64
```

#### `vec4_norm_sq`

```sio
fn vec4_norm_sq(v: Vec4) -> f64
```

#### `vec4_norm`

```sio
fn vec4_norm(v: Vec4) -> f64
```

#### `vec4_normalize`

```sio
fn vec4_normalize(v: Vec4) -> Vec4
```

### Component Operations

#### `vec4_max`

```sio
fn vec4_max(a: Vec4, b: Vec4) -> Vec4
```

#### `vec4_min`

```sio
fn vec4_min(a: Vec4, b: Vec4) -> Vec4
```

#### `vec4_abs`

```sio
fn vec4_abs(v: Vec4) -> Vec4
```

#### `vec4_max_component`

```sio
fn vec4_max_component(v: Vec4) -> f64
```

#### `vec4_sum`

```sio
fn vec4_sum(v: Vec4) -> f64
```

---

## Vec14

14-element vector for PBPK (Physiologically Based Pharmacokinetic) models. Designed for compartment-based pharmacokinetic simulations.

### Type Definition

```sio
struct Vec14 {
    c0: f64, c1: f64, c2: f64, c3: f64, c4: f64,
    c5: f64, c6: f64, c7: f64, c8: f64, c9: f64,
    c10: f64, c11: f64, c12: f64, c13: f64
}
```

### Constructors

#### `vec14_new`

```sio
fn vec14_new(
    v0: f64, v1: f64, v2: f64, v3: f64, v4: f64,
    v5: f64, v6: f64, v7: f64, v8: f64, v9: f64,
    v10: f64, v11: f64, v12: f64, v13: f64
) -> Vec14
```

#### `vec14_zero`

```sio
fn vec14_zero() -> Vec14
```

#### `vec14_ones`

```sio
fn vec14_ones() -> Vec14
```

### Arithmetic Operations

#### `vec14_add`

```sio
fn vec14_add(a: Vec14, b: Vec14) -> Vec14
```

#### `vec14_sub`

```sio
fn vec14_sub(a: Vec14, b: Vec14) -> Vec14
```

#### `vec14_scale`

```sio
fn vec14_scale(v: Vec14, s: f64) -> Vec14
```

#### `vec14_neg`

```sio
fn vec14_neg(v: Vec14) -> Vec14
```

#### `vec14_hadamard`

```sio
fn vec14_hadamard(a: Vec14, b: Vec14) -> Vec14
```

### Products and Norms

#### `vec14_dot`

```sio
fn vec14_dot(a: Vec14, b: Vec14) -> f64
```

#### `vec14_norm_sq`

```sio
fn vec14_norm_sq(v: Vec14) -> f64
```

#### `vec14_norm`

```sio
fn vec14_norm(v: Vec14) -> f64
```

#### `vec14_sum`

```sio
fn vec14_sum(v: Vec14) -> f64
```

### ODE-Specific Operations

#### `vec14_rms_weighted`

Weighted RMS norm for ODE error estimation.

```sio
fn vec14_rms_weighted(err: Vec14, scale: Vec14) -> f64
```

**Formula:** `sqrt(sum((err[i] / scale[i])^2) / 14)`

**Use:** Adaptive step size control in ODE solvers.

**Example:**
```sio
let err = vec14_new(0.001, 0.002, ...)   // Error vector
let scale = vec14_ones()                  // Scale factors
let rms = vec14_rms_weighted(err, scale)  // RMS error
```

### Component Operations

#### `vec14_max`

```sio
fn vec14_max(a: Vec14, b: Vec14) -> Vec14
```

#### `vec14_abs`

```sio
fn vec14_abs(v: Vec14) -> Vec14
```

---

## Usage Examples

### 2D Physics

```sio
// Position and velocity
let pos = vec2_new(0.0, 0.0)
let vel = vec2_new(10.0, 20.0)

// Update position: pos = pos + vel * dt
let dt = 0.01
let new_pos = vec2_add(pos, vec2_scale(vel, dt))

// Distance between two points
let p1 = vec2_new(1.0, 2.0)
let p2 = vec2_new(4.0, 6.0)
let dist = vec2_norm(vec2_sub(p2, p1))  // 5.0
```

### 3D Graphics

```sio
// Surface normal from two edge vectors
let edge1 = vec3_new(1.0, 0.0, 0.0)
let edge2 = vec3_new(0.0, 1.0, 0.0)
let normal = vec3_normalize(vec3_cross(edge1, edge2))

// Angle between vectors using dot product
let a = vec3_new(1.0, 0.0, 0.0)
let b = vec3_new(0.707, 0.707, 0.0)
let cos_angle = vec3_dot(a, b) / (vec3_norm(a) * vec3_norm(b))
// angle = acos(cos_angle) = 45 degrees
```

### PBPK Simulation

```sio
// 14-compartment PBPK state
let state = vec14_new(
    100.0,  // Blood
    50.0,   // Liver
    25.0,   // Kidney
    10.0,   // Brain
    5.0,    // Heart
    5.0,    // Lung
    5.0,    // Muscle
    5.0,    // Fat
    5.0,    // Skin
    5.0,    // Bone
    5.0,    // Gut
    5.0,    // Spleen
    5.0,    // Rest
    0.0     // Eliminated
)

// Total drug mass
let total_mass = vec14_sum(state)
```

---

## Performance Notes

1. **Stack allocation**: All vector types are stack-allocated, avoiding heap overhead
2. **SIMD potential**: Layout designed for vectorization by the compiler
3. **Copy semantics**: Vectors are small enough to be efficiently passed by value
4. **No bounds checking**: Fixed-size types have no runtime bounds checks

## Numerical Considerations

1. **Normalization**: `normalize` returns zero vector for very small inputs (norm < 1e-10)
2. **Precision**: All operations use f64 for maximum precision
3. **Stability**: Square root uses Newton-Raphson with 10 iterations

---

## See Also

- [Matrix Types](matrices.md)
- [Matrix Decompositions](decompositions.md)
- [ODE Solvers](../ode/solvers.md) (uses Vec14 for state vectors)

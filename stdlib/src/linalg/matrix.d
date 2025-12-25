// matrix.d - Dense Matrix and Vector Library for Demetrios
//
// Provides basic linear algebra types and operations.
// Simplified implementation compatible with current Demetrios syntax.
//
// Module: linalg::matrix (for future module system)

// =============================================================================
// MATH HELPERS
// =============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    let mut y = x
    let mut i = 0
    while i < 15 {
        y = 0.5 * (y + x / y)
        i = i + 1
    }
    return y
}

// =============================================================================
// VECTOR TYPE (Fixed size for simplicity)
// =============================================================================

// 3D Vector
struct Vec3 {
    x: f64,
    y: f64,
    z: f64
}

fn vec3_new(x: f64, y: f64, z: f64) -> Vec3 {
    return Vec3 { x: x, y: y, z: z }
}

fn vec3_zero() -> Vec3 {
    return Vec3 { x: 0.0, y: 0.0, z: 0.0 }
}

fn vec3_add(a: Vec3, b: Vec3) -> Vec3 {
    return Vec3 { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
}

fn vec3_sub(a: Vec3, b: Vec3) -> Vec3 {
    return Vec3 { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }
}

fn vec3_scale(v: Vec3, s: f64) -> Vec3 {
    return Vec3 { x: v.x * s, y: v.y * s, z: v.z * s }
}

fn vec3_dot(a: Vec3, b: Vec3) -> f64 {
    return a.x * b.x + a.y * b.y + a.z * b.z
}

fn vec3_cross(a: Vec3, b: Vec3) -> Vec3 {
    return Vec3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x
    }
}

fn vec3_norm(v: Vec3) -> f64 {
    return sqrt_f64(v.x * v.x + v.y * v.y + v.z * v.z)
}

fn vec3_normalize(v: Vec3) -> Vec3 {
    let n = vec3_norm(v)
    if n < 0.0000001 { return vec3_zero() }
    return vec3_scale(v, 1.0 / n)
}

// =============================================================================
// 4D VECTOR (for state vectors)
// =============================================================================

struct Vec4 {
    x: f64,
    y: f64,
    z: f64,
    w: f64
}

fn vec4_new(x: f64, y: f64, z: f64, w: f64) -> Vec4 {
    return Vec4 { x: x, y: y, z: z, w: w }
}

fn vec4_zero() -> Vec4 {
    return Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }
}

fn vec4_add(a: Vec4, b: Vec4) -> Vec4 {
    return Vec4 { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z, w: a.w + b.w }
}

fn vec4_sub(a: Vec4, b: Vec4) -> Vec4 {
    return Vec4 { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z, w: a.w - b.w }
}

fn vec4_scale(v: Vec4, s: f64) -> Vec4 {
    return Vec4 { x: v.x * s, y: v.y * s, z: v.z * s, w: v.w * s }
}

fn vec4_dot(a: Vec4, b: Vec4) -> f64 {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
}

fn vec4_norm(v: Vec4) -> f64 {
    return sqrt_f64(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w)
}

// =============================================================================
// N-DIMENSIONAL VECTOR (using array of 16 elements max)
// =============================================================================

struct VecN {
    data0: f64, data1: f64, data2: f64, data3: f64,
    data4: f64, data5: f64, data6: f64, data7: f64,
    data8: f64, data9: f64, data10: f64, data11: f64,
    data12: f64, data13: f64, data14: f64, data15: f64,
    len: i32
}

fn vecn_zero(n: i32) -> VecN {
    return VecN {
        data0: 0.0, data1: 0.0, data2: 0.0, data3: 0.0,
        data4: 0.0, data5: 0.0, data6: 0.0, data7: 0.0,
        data8: 0.0, data9: 0.0, data10: 0.0, data11: 0.0,
        data12: 0.0, data13: 0.0, data14: 0.0, data15: 0.0,
        len: n
    }
}

fn vecn_get(v: VecN, i: i32) -> f64 {
    if i == 0 { return v.data0 }
    if i == 1 { return v.data1 }
    if i == 2 { return v.data2 }
    if i == 3 { return v.data3 }
    if i == 4 { return v.data4 }
    if i == 5 { return v.data5 }
    if i == 6 { return v.data6 }
    if i == 7 { return v.data7 }
    if i == 8 { return v.data8 }
    if i == 9 { return v.data9 }
    if i == 10 { return v.data10 }
    if i == 11 { return v.data11 }
    if i == 12 { return v.data12 }
    if i == 13 { return v.data13 }
    if i == 14 { return v.data14 }
    if i == 15 { return v.data15 }
    return 0.0
}

fn vecn_set(v: VecN, i: i32, val: f64) -> VecN {
    let mut r = v
    if i == 0 { r.data0 = val }
    if i == 1 { r.data1 = val }
    if i == 2 { r.data2 = val }
    if i == 3 { r.data3 = val }
    if i == 4 { r.data4 = val }
    if i == 5 { r.data5 = val }
    if i == 6 { r.data6 = val }
    if i == 7 { r.data7 = val }
    if i == 8 { r.data8 = val }
    if i == 9 { r.data9 = val }
    if i == 10 { r.data10 = val }
    if i == 11 { r.data11 = val }
    if i == 12 { r.data12 = val }
    if i == 13 { r.data13 = val }
    if i == 14 { r.data14 = val }
    if i == 15 { r.data15 = val }
    return r
}

fn vecn_add(a: VecN, b: VecN) -> VecN {
    let mut r = vecn_zero(a.len)
    let mut i = 0
    while i < a.len {
        r = vecn_set(r, i, vecn_get(a, i) + vecn_get(b, i))
        i = i + 1
    }
    return r
}

fn vecn_scale(v: VecN, s: f64) -> VecN {
    let mut r = vecn_zero(v.len)
    let mut i = 0
    while i < v.len {
        r = vecn_set(r, i, vecn_get(v, i) * s)
        i = i + 1
    }
    return r
}

fn vecn_dot(a: VecN, b: VecN) -> f64 {
    let mut sum = 0.0
    let mut i = 0
    while i < a.len {
        sum = sum + vecn_get(a, i) * vecn_get(b, i)
        i = i + 1
    }
    return sum
}

fn vecn_norm(v: VecN) -> f64 {
    return sqrt_f64(vecn_dot(v, v))
}

// =============================================================================
// 3x3 MATRIX
// =============================================================================

struct Mat3 {
    m00: f64, m01: f64, m02: f64,
    m10: f64, m11: f64, m12: f64,
    m20: f64, m21: f64, m22: f64
}

fn mat3_zero() -> Mat3 {
    return Mat3 {
        m00: 0.0, m01: 0.0, m02: 0.0,
        m10: 0.0, m11: 0.0, m12: 0.0,
        m20: 0.0, m21: 0.0, m22: 0.0
    }
}

fn mat3_identity() -> Mat3 {
    return Mat3 {
        m00: 1.0, m01: 0.0, m02: 0.0,
        m10: 0.0, m11: 1.0, m12: 0.0,
        m20: 0.0, m21: 0.0, m22: 1.0
    }
}

fn mat3_add(a: Mat3, b: Mat3) -> Mat3 {
    return Mat3 {
        m00: a.m00 + b.m00, m01: a.m01 + b.m01, m02: a.m02 + b.m02,
        m10: a.m10 + b.m10, m11: a.m11 + b.m11, m12: a.m12 + b.m12,
        m20: a.m20 + b.m20, m21: a.m21 + b.m21, m22: a.m22 + b.m22
    }
}

fn mat3_scale(m: Mat3, s: f64) -> Mat3 {
    return Mat3 {
        m00: m.m00 * s, m01: m.m01 * s, m02: m.m02 * s,
        m10: m.m10 * s, m11: m.m11 * s, m12: m.m12 * s,
        m20: m.m20 * s, m21: m.m21 * s, m22: m.m22 * s
    }
}

fn mat3_mul_vec(m: Mat3, v: Vec3) -> Vec3 {
    return Vec3 {
        x: m.m00 * v.x + m.m01 * v.y + m.m02 * v.z,
        y: m.m10 * v.x + m.m11 * v.y + m.m12 * v.z,
        z: m.m20 * v.x + m.m21 * v.y + m.m22 * v.z
    }
}

fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
    return Mat3 {
        m00: a.m00*b.m00 + a.m01*b.m10 + a.m02*b.m20,
        m01: a.m00*b.m01 + a.m01*b.m11 + a.m02*b.m21,
        m02: a.m00*b.m02 + a.m01*b.m12 + a.m02*b.m22,
        m10: a.m10*b.m00 + a.m11*b.m10 + a.m12*b.m20,
        m11: a.m10*b.m01 + a.m11*b.m11 + a.m12*b.m21,
        m12: a.m10*b.m02 + a.m11*b.m12 + a.m12*b.m22,
        m20: a.m20*b.m00 + a.m21*b.m10 + a.m22*b.m20,
        m21: a.m20*b.m01 + a.m21*b.m11 + a.m22*b.m21,
        m22: a.m20*b.m02 + a.m21*b.m12 + a.m22*b.m22
    }
}

fn mat3_transpose(m: Mat3) -> Mat3 {
    return Mat3 {
        m00: m.m00, m01: m.m10, m02: m.m20,
        m10: m.m01, m11: m.m11, m12: m.m21,
        m20: m.m02, m21: m.m12, m22: m.m22
    }
}

fn mat3_det(m: Mat3) -> f64 {
    return m.m00 * (m.m11 * m.m22 - m.m12 * m.m21)
         - m.m01 * (m.m10 * m.m22 - m.m12 * m.m20)
         + m.m02 * (m.m10 * m.m21 - m.m11 * m.m20)
}

fn mat3_trace(m: Mat3) -> f64 {
    return m.m00 + m.m11 + m.m22
}

fn mat3_frobenius_norm(m: Mat3) -> f64 {
    let sum = m.m00*m.m00 + m.m01*m.m01 + m.m02*m.m02 +
              m.m10*m.m10 + m.m11*m.m11 + m.m12*m.m12 +
              m.m20*m.m20 + m.m21*m.m21 + m.m22*m.m22
    return sqrt_f64(sum)
}

// =============================================================================
// TESTS
// =============================================================================

fn main() -> i32 {
    println("=== Linear Algebra Tests ===")
    println("")

    // Test 1: Vec3 operations
    println("Test 1: Vec3 operations")
    let v1 = vec3_new(1.0, 2.0, 3.0)
    let v2 = vec3_new(4.0, 5.0, 6.0)
    let v_add = vec3_add(v1, v2)
    let v_dot = vec3_dot(v1, v2)
    let v_norm = vec3_norm(v1)
    println("  v1 = (1, 2, 3)")
    println("  v2 = (4, 5, 6)")
    println("  v1 + v2 = ")
    println(v_add.x)
    println(v_add.y)
    println(v_add.z)
    println("  v1 . v2 = ")
    println(v_dot)
    println("  |v1| = ")
    println(v_norm)
    println("")

    // Test 2: Cross product
    println("Test 2: Cross product")
    let i = vec3_new(1.0, 0.0, 0.0)
    let j = vec3_new(0.0, 1.0, 0.0)
    let k = vec3_cross(i, j)
    println("  i x j = ")
    println(k.x)
    println(k.y)
    println(k.z)
    println("")

    // Test 3: Matrix operations
    println("Test 3: Mat3 operations")
    let eye = mat3_identity()
    let det_eye = mat3_det(eye)
    let trace_eye = mat3_trace(eye)
    println("  det(I) = ")
    println(det_eye)
    println("  trace(I) = ")
    println(trace_eye)
    println("")

    // Test 4: Matrix-vector multiply
    println("Test 4: Matrix-vector multiply")
    let m = Mat3 {
        m00: 2.0, m01: 0.0, m02: 0.0,
        m10: 0.0, m11: 3.0, m12: 0.0,
        m20: 0.0, m21: 0.0, m22: 4.0
    }
    let v = vec3_new(1.0, 1.0, 1.0)
    let mv = mat3_mul_vec(m, v)
    println("  diag(2,3,4) * (1,1,1) = ")
    println(mv.x)
    println(mv.y)
    println(mv.z)
    println("")

    // Validation
    let err1 = abs_f64(v_dot - 32.0)  // 1*4 + 2*5 + 3*6 = 32
    let err2 = abs_f64(v_norm - 3.7416573867739413)  // sqrt(14)
    let err3 = abs_f64(k.z - 1.0)  // i x j = k
    let err4 = abs_f64(det_eye - 1.0)
    let err5 = abs_f64(trace_eye - 3.0)

    if err1 < 0.001 {
        if err2 < 0.001 {
            if err3 < 0.001 {
                if err4 < 0.001 {
                    if err5 < 0.001 {
                        println("ALL TESTS PASSED")
                        return 0
                    }
                }
            }
        }
    }

    println("TESTS FAILED")
    return 1
}

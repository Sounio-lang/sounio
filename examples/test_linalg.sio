// Test linear algebra primitive types and operations

fn main() -> i32 {
    // Vector constructors
    let v1 = vec3(1.0, 0.0, 0.0);
    let v2 = vec3(0.0, 1.0, 0.0);

    // Vector operations
    let d = dot(v1, v2);           // Should be 0.0
    let c = cross(v1, v2);         // Should be (0, 0, 1)
    let n = normalize(v1);         // Should be (1, 0, 0)
    let len = length(v1);          // Should be 1.0

    // Quaternion operations
    let q1 = quat(0.0, 0.0, 0.0, 1.0);  // identity
    let q2 = quat_identity();           // also identity
    let qm = quat_mul(q1, q2);          // multiply
    let qc = quat_conj(q1);             // conjugate
    let qi = quat_inv(q1);              // inverse
    let qn = quat_normalize(q1);        // normalize

    // Matrix operations
    let m1 = mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    let mt = transpose(m1);
    let mi = inverse(m1);
    let det = determinant(m1);

    // Interpolation
    let v3 = vec3(2.0, 0.0, 0.0);
    let interp = lerp(v1, v3, 0.5);     // Should be (1.5, 0, 0)
    let sq = slerp(q1, q2, 0.5);        // Spherical interpolation

    // Conversions
    let euler = quat_to_euler(q1);
    let qe = euler_to_quat(euler);
    let m3 = quat_to_mat3(q1);
    let m4 = quat_to_mat4(q1);

    return 0;
}

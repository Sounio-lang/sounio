/// Approximate Symmetry Metric
///
/// Computation of d_min/L: minimum normalized Hamming distance
/// to any non-identity dihedral transform.

module approx_metric

use operators::{shift, reverse, reverse_complement, hamming_distance, Sequence};

/// Minimum distance to any non-identity transform in the dihedral group
/// d_min(w) = min_{g ∈ D_n \ {id}} H(w, g(w))
///
/// Transforms tested:
/// - S^k for k in 1..n (cyclic shifts, excluding identity)
/// - R∘S^k for k in 0..n (reverse then shift)
/// - RC∘S^k for k in 0..n (reverse complement then shift, if include_rc)
pub fn dmin(seq: &Sequence, include_rc: bool) -> usize {
    let n = seq.len()
    if n == 0 { return 0 }

    var min_dist = n  // Maximum possible

    // S^k transforms (k=1..n-1, excluding identity k=0)
    var k: usize = 1
    while k < n {
        let shifted = shift(seq, k)
        let d = hamming_distance(seq, &shifted)
        if d < min_dist { min_dist = d }
        k = k + 1
    }

    // R∘S^k transforms (k=0..n-1)
    let rev = reverse(seq)
    k = 0
    while k < n {
        let shifted_rev = shift(&rev, k)
        let d = hamming_distance(seq, &shifted_rev)
        if d < min_dist { min_dist = d }
        k = k + 1
    }

    // RC∘S^k transforms (optional)
    if include_rc {
        let rc = reverse_complement(seq)
        k = 0
        while k < n {
            let shifted_rc = shift(&rc, k)
            let d = hamming_distance(seq, &shifted_rc)
            if d < min_dist { min_dist = d }
            k = k + 1
        }
    }

    min_dist
}

/// Normalized d_min: d_min / L
/// Range: [0, 1] where 0 = exact symmetry, 1 = maximum asymmetry
pub fn dmin_normalized(seq: &Sequence, include_rc: bool) -> f64 {
    let n = seq.len()
    if n == 0 { return 0.0 }
    dmin(seq, include_rc) as f64 / n as f64
}

/// Transform type enum (simple variant without data)
pub enum TransformType {
    Shift,           // S^k
    ReverseShift,    // R∘S^k
    RCShift,         // RC∘S^k
}

/// Identify which transform achieves d_min
pub struct NearestTransform {
    pub transform_type: TransformType,
    pub k: usize,
}

pub fn nearest_transform(seq: &Sequence, include_rc: bool) -> (usize, NearestTransform) {
    let n = seq.len()
    if n == 0 {
        return (0, NearestTransform { transform_type: TransformType::Shift, k: 0 })
    }

    var min_dist = n
    var nearest_type = TransformType::Shift
    var nearest_k: usize = 1

    // S^k transforms
    var k: usize = 1
    while k < n {
        let d = hamming_distance(seq, &shift(seq, k))
        if d < min_dist {
            min_dist = d
            nearest_type = TransformType::Shift
            nearest_k = k
        }
        k = k + 1
    }

    // R∘S^k transforms
    let rev = reverse(seq)
    k = 0
    while k < n {
        let d = hamming_distance(seq, &shift(&rev, k))
        if d < min_dist {
            min_dist = d
            nearest_type = TransformType::ReverseShift
            nearest_k = k
        }
        k = k + 1
    }

    // RC∘S^k transforms
    if include_rc {
        let rc = reverse_complement(seq)
        k = 0
        while k < n {
            let d = hamming_distance(seq, &shift(&rc, k))
            if d < min_dist {
                min_dist = d
                nearest_type = TransformType::RCShift
                nearest_k = k
            }
            k = k + 1
        }
    }

    (min_dist, NearestTransform { transform_type: nearest_type, k: nearest_k })
}

// =============================================================================
// Test Functions
// =============================================================================

use operators::{A, C, G, T};

pub fn test_dmin_bounds() -> bool {
    let seq = [A, C, G, T]
    let d = dmin(&seq, true)
    d <= seq.len()
}

pub fn test_dmin_normalized_range() -> bool {
    let seq = [A, C, G, T, A, C, G, T]
    let dn = dmin_normalized(&seq, true)
    dn >= 0.0 && dn <= 1.0
}

pub fn test_periodic_sequence() -> bool {
    // ACGTACGT has shift symmetry S^4 = id
    let seq = [A, C, G, T, A, C, G, T]
    let d = dmin(&seq, false)
    d == 0  // S^4(seq) = seq
}

/// Run all tests
pub fn run_metric_tests() -> bool {
    test_dmin_bounds() && test_dmin_normalized_range() && test_periodic_sequence()
}

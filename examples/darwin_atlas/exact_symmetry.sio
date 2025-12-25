/// Exact Symmetry Analysis
///
/// Computation of orbit sizes, fixed points, and symmetry detection
/// under the dihedral group action.

module exact_symmetry

use operators::{shift, reverse, complement, reverse_complement, Sequence};

/// Helper: compare two sequences for equality
fn sequences_equal(a: &Sequence, b: &Sequence) -> bool {
    let len_a = a.len()
    let len_b = b.len()
    if len_a != len_b { return false }

    var i: usize = 0
    while i < len_a {
        if a[i] != b[i] { return false }
        i = i + 1
    }
    true
}

/// Check if a sequence is already in a list of sequences
fn is_in_orbit(seq: &Sequence, orbit: &[Sequence], count: usize) -> bool {
    var i: usize = 0
    let len = if count < orbit.len() { count } else { orbit.len() }
    while i < len {
        if sequences_equal(seq, &orbit[i]) {
            return true
        }
        i = i + 1
    }
    false
}

/// Size of the orbit under D_n action
/// Computes unique transforms: {S^k(w), R∘S^k(w)} for k in 0..n
pub fn orbit_size(seq: &Sequence) -> usize {
    let n = seq.len()
    if n == 0 { return 1 }

    // Maximum orbit size is 2n
    // We track unique sequences found
    var orbit: [Sequence] = []
    var count: usize = 0

    let rev = reverse(seq)

    var k: usize = 0
    while k < n {
        let shifted = shift(seq, k)
        if !is_in_orbit(&shifted, &orbit, count) {
            orbit.push(shifted)
            count = count + 1
        }

        let shifted_rev = shift(&rev, k)
        if !is_in_orbit(&shifted_rev, &orbit, count) {
            orbit.push(shifted_rev)
            count = count + 1
        }
        k = k + 1
    }

    count
}

/// Orbit ratio: |orbit| / (2n)
/// Range: [1/(2n), 1.0]
/// Lower values indicate higher symmetry
pub fn orbit_ratio(seq: &Sequence) -> f64 {
    let n = seq.len()
    if n == 0 { return 1.0 }
    orbit_size(seq) as f64 / (2 * n) as f64
}

/// Check if sequence is a palindrome (fixed under R)
/// R(w) = w
pub fn is_palindrome(seq: &Sequence) -> bool {
    let rev = reverse(seq)
    sequences_equal(seq, &rev)
}

/// Check if sequence is fixed under reverse complement
/// RC(w) = w
pub fn is_rc_fixed(seq: &Sequence) -> bool {
    let rc = reverse_complement(seq)
    sequences_equal(seq, &rc)
}

/// Check if sequence has k-fold rotational symmetry
/// S^k(w) = w for some k | n
pub fn rotational_period(seq: &Sequence) -> usize {
    let n = seq.len()
    if n == 0 { return 1 }

    var k: usize = 1
    while k <= n {
        if n % k == 0 {
            let shifted = shift(seq, k)
            if sequences_equal(seq, &shifted) {
                return k
            }
        }
        k = k + 1
    }
    n
}

/// Compute symmetry statistics for a sequence
pub struct SymmetryStats {
    pub length: usize,
    pub orbit_size: usize,
    pub orbit_ratio: f64,
    pub is_palindrome: bool,
    pub is_rc_fixed: bool,
    pub rotational_period: usize,
}

pub fn compute_symmetry_stats(seq: &Sequence) -> SymmetryStats {
    SymmetryStats {
        length: seq.len(),
        orbit_size: orbit_size(seq),
        orbit_ratio: orbit_ratio(seq),
        is_palindrome: is_palindrome(seq),
        is_rc_fixed: is_rc_fixed(seq),
        rotational_period: rotational_period(seq),
    }
}

// =============================================================================
// Test Functions
// =============================================================================

use operators::{A, C, G, T};

pub fn test_palindrome_detection() -> bool {
    // ACGT reversed is TGCA, not palindrome
    let not_pal = !is_palindrome(&[A, C, G, T])
    // ACCA reversed is ACCA, palindrome
    let is_pal = is_palindrome(&[A, C, C, A])
    not_pal && is_pal
}

pub fn test_rc_fixed() -> bool {
    // ACGT: RC = complement(TGCA) = ACGT, so RC-fixed
    is_rc_fixed(&[A, C, G, T])
}

pub fn test_orbit_bounds() -> bool {
    let seq = [A, C, G, T]
    let n = seq.len()
    let os = orbit_size(&seq)
    os >= 1 && os <= 2 * n
}

/// Run all tests
pub fn run_symmetry_tests() -> bool {
    test_palindrome_detection() && test_rc_fixed() && test_orbit_bounds()
}

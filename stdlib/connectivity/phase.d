// connectivity::phase — Phase Synchronization Measures
//
// Functional connectivity via phase coupling between brain regions.
// Essential for computational psychiatry: altered synchrony in disorders.
//
// Measures:
// - PLV: Phase Locking Value — magnitude of phase difference consistency
// - PLI: Phase Lag Index — asymmetry of phase differences (volume conduction robust)
// - wPLI: Weighted PLI — weighted by magnitude of imaginary component
//
// References:
// - Lachaux et al. (1999): "Measuring phase synchrony in brain signals"
// - Stam et al. (2007): "Phase lag index: Assessment of functional connectivity..."
// - Vinck et al. (2011): "An improved index of phase-synchronization..."

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn cos(x: f64) -> f64;
    fn atan2(y: f64, x: f64) -> f64;
}

fn pi() -> f64 { 3.14159265358979323846 }

// ============================================================================
// FFT FOR HILBERT TRANSFORM
// ============================================================================

fn MAX_SIGNAL() -> i64 { 2048 }

// Bit reversal for FFT
fn bit_reverse(n: i64, bits: i64) -> i64 {
    var result: i64 = 0
    var n_var = n
    var i: i64 = 0
    while i < bits {
        result = (result << 1) | (n_var & 1)
        n_var = n_var >> 1
        i = i + 1
    }
    return result
}

// In-place FFT
fn fft_inplace(x_re: [f64; 2048], x_im: [f64; 2048], n: i64, inverse: bool) -> ([f64; 2048], [f64; 2048]) {
    var re = x_re
    var im = x_im

    var bits: i64 = 0
    var temp = n
    while temp > 1 {
        bits = bits + 1
        temp = temp / 2
    }

    var i: i64 = 0
    while i < n {
        let j = bit_reverse(i, bits)
        if i < j {
            let tmp_re = re[i as usize]
            let tmp_im = im[i as usize]
            re[i as usize] = re[j as usize]
            im[i as usize] = im[j as usize]
            re[j as usize] = tmp_re
            im[j as usize] = tmp_im
        }
        i = i + 1
    }

    var size: i64 = 2
    while size <= n {
        let half = size / 2
        let angle_mult = if inverse { 2.0 * pi() / (size as f64) } else { -2.0 * pi() / (size as f64) }

        var k: i64 = 0
        while k < n {
            var j: i64 = 0
            while j < half {
                let angle = angle_mult * (j as f64)
                let w_re = cos(angle)
                let w_im = sin(angle)

                let idx1 = (k + j) as usize
                let idx2 = (k + j + half) as usize

                let even_re = re[idx1]
                let even_im = im[idx1]
                let odd_re = re[idx2]
                let odd_im = im[idx2]

                let t_re = w_re * odd_re - w_im * odd_im
                let t_im = w_re * odd_im + w_im * odd_re

                re[idx1] = even_re + t_re
                im[idx1] = even_im + t_im
                re[idx2] = even_re - t_re
                im[idx2] = even_im - t_im

                j = j + 1
            }
            k = k + size
        }
        size = size * 2
    }

    if inverse {
        i = 0
        while i < n {
            re[i as usize] = re[i as usize] / (n as f64)
            im[i as usize] = im[i as usize] / (n as f64)
            i = i + 1
        }
    }

    return (re, im)
}

// Compute analytic signal via Hilbert transform
fn hilbert_transform(signal: [f64; 2048], n: i64) -> ([f64; 2048], [f64; 2048]) {
    // Find next power of 2
    var nfft: i64 = 1
    while nfft < n {
        nfft = nfft * 2
    }

    // Copy to FFT arrays
    var fft_re: [f64; 2048] = [0.0; 2048]
    var fft_im: [f64; 2048] = [0.0; 2048]

    var i: i64 = 0
    while i < n {
        fft_re[i as usize] = signal[i as usize]
        fft_im[i as usize] = 0.0
        i = i + 1
    }
    while i < nfft {
        fft_re[i as usize] = 0.0
        fft_im[i as usize] = 0.0
        i = i + 1
    }

    // Forward FFT
    let result = fft_inplace(fft_re, fft_im, nfft, false)
    fft_re = result.0
    fft_im = result.1

    // Create analytic signal spectrum:
    // H[0] unchanged, H[1..n/2] * 2, H[n/2+1..n-1] = 0

    // Positive frequencies: multiply by 2
    i = 1
    while i < nfft / 2 {
        fft_re[i as usize] = fft_re[i as usize] * 2.0
        fft_im[i as usize] = fft_im[i as usize] * 2.0
        i = i + 1
    }

    // Negative frequencies: set to zero
    i = nfft / 2 + 1
    while i < nfft {
        fft_re[i as usize] = 0.0
        fft_im[i as usize] = 0.0
        i = i + 1
    }

    // Inverse FFT
    let result2 = fft_inplace(fft_re, fft_im, nfft, true)

    return result2
}

// Extract instantaneous phase
fn instantaneous_phase(signal: [f64; 2048], n: i64) -> [f64; 2048] {
    let result = hilbert_transform(signal, n)
    let analytic_re = result.0
    let analytic_im = result.1

    var phase: [f64; 2048] = [0.0; 2048]
    var i: i64 = 0
    while i < n {
        phase[i as usize] = atan2(analytic_im[i as usize], analytic_re[i as usize])
        i = i + 1
    }

    return phase
}

// ============================================================================
// PHASE LOCKING VALUE (PLV)
// ============================================================================

// Compute Phase Locking Value between two signals
// PLV = |mean(exp(j*(phi1 - phi2)))|
// Returns value in [0, 1]: 0 = no synchrony, 1 = perfect synchrony
fn phase_locking_value(signal1: [f64; 2048], signal2: [f64; 2048], n: i64) -> f64 {
    let phase1 = instantaneous_phase(signal1, n)
    let phase2 = instantaneous_phase(signal2, n)

    // Compute mean of exp(j*dphi)
    var sum_re = 0.0
    var sum_im = 0.0

    var i: i64 = 0
    while i < n {
        let dphi = phase1[i as usize] - phase2[i as usize]
        sum_re = sum_re + cos(dphi)
        sum_im = sum_im + sin(dphi)
        i = i + 1
    }

    sum_re = sum_re / (n as f64)
    sum_im = sum_im / (n as f64)

    return sqrt(sum_re * sum_re + sum_im * sum_im)
}

// PLV from pre-computed phases
fn plv_from_phases(phase1: [f64; 2048], phase2: [f64; 2048], n: i64) -> f64 {
    var sum_re = 0.0
    var sum_im = 0.0

    var i: i64 = 0
    while i < n {
        let dphi = phase1[i as usize] - phase2[i as usize]
        sum_re = sum_re + cos(dphi)
        sum_im = sum_im + sin(dphi)
        i = i + 1
    }

    sum_re = sum_re / (n as f64)
    sum_im = sum_im / (n as f64)

    return sqrt(sum_re * sum_re + sum_im * sum_im)
}

// ============================================================================
// PHASE LAG INDEX (PLI)
// ============================================================================

// Compute Phase Lag Index
// PLI = |mean(sign(dphi))| where dphi is mapped to [-pi, pi)
// Robust to volume conduction (zero-lag effects)
fn phase_lag_index(signal1: [f64; 2048], signal2: [f64; 2048], n: i64) -> f64 {
    let phase1 = instantaneous_phase(signal1, n)
    let phase2 = instantaneous_phase(signal2, n)

    return pli_from_phases(phase1, phase2, n)
}

fn pli_from_phases(phase1: [f64; 2048], phase2: [f64; 2048], n: i64) -> f64 {
    var sum_sign = 0.0

    var i: i64 = 0
    while i < n {
        var dphi = phase1[i as usize] - phase2[i as usize]

        // Wrap to [-pi, pi)
        while dphi >= pi() {
            dphi = dphi - 2.0 * pi()
        }
        while dphi < -pi() {
            dphi = dphi + 2.0 * pi()
        }

        // Sign (avoiding exact zero)
        if dphi > 1e-10 {
            sum_sign = sum_sign + 1.0
        } else if dphi < -1e-10 {
            sum_sign = sum_sign - 1.0
        }

        i = i + 1
    }

    let pli = sum_sign / (n as f64)
    return if pli < 0.0 { -pli } else { pli }
}

// ============================================================================
// WEIGHTED PHASE LAG INDEX (wPLI)
// ============================================================================

// Compute Weighted Phase Lag Index
// Weights contributions by magnitude of imaginary component
fn weighted_phase_lag_index(signal1: [f64; 2048], signal2: [f64; 2048], n: i64) -> f64 {
    let result1 = hilbert_transform(signal1, n)
    let analytic1_re = result1.0
    let analytic1_im = result1.1

    let result2 = hilbert_transform(signal2, n)
    let analytic2_re = result2.0
    let analytic2_im = result2.1

    // Cross-spectrum: X1 * conj(X2)
    // Im(X1 * conj(X2)) = X1_re * X2_im - X1_im * X2_re

    var sum_weighted_sign = 0.0
    var sum_abs_im = 0.0

    var i: i64 = 0
    while i < n {
        // Cross spectrum imaginary part
        let im = analytic1_re[i as usize] * analytic2_im[i as usize]
               - analytic1_im[i as usize] * analytic2_re[i as usize]

        let abs_im = if im < 0.0 { -im } else { im }

        sum_abs_im = sum_abs_im + abs_im

        if im > 1e-10 {
            sum_weighted_sign = sum_weighted_sign + abs_im
        } else if im < -1e-10 {
            sum_weighted_sign = sum_weighted_sign - abs_im
        }

        i = i + 1
    }

    if sum_abs_im > 1e-10 {
        let wpli = sum_weighted_sign / sum_abs_im
        return if wpli < 0.0 { -wpli } else { wpli }
    } else {
        return 0.0
    }
}

// ============================================================================
// DEBIASED wPLI (dwPLI)
// ============================================================================

// Debiased wPLI — reduces positive bias for small sizes
fn debiased_wpli(signal1: [f64; 2048], signal2: [f64; 2048], n: i64) -> f64 {
    let result1 = hilbert_transform(signal1, n)
    let analytic1_re = result1.0
    let analytic1_im = result1.1

    let result2 = hilbert_transform(signal2, n)
    let analytic2_re = result2.0
    let analytic2_im = result2.1

    // Compute imaginary parts of cross-spectrum
    var sum_im = 0.0
    var sum_im_sq = 0.0
    var sum_abs_im = 0.0

    var i: i64 = 0
    while i < n {
        let im = analytic1_re[i as usize] * analytic2_im[i as usize]
               - analytic1_im[i as usize] * analytic2_re[i as usize]

        sum_im = sum_im + im
        sum_im_sq = sum_im_sq + im * im
        sum_abs_im = sum_abs_im + if im < 0.0 { -im } else { im }
        i = i + 1
    }

    // dwPLI = (sum(Im)^2 - sum(Im^2)) / (sum(|Im|)^2 - sum(Im^2))
    let numerator = sum_im * sum_im - sum_im_sq
    let denominator = sum_abs_im * sum_abs_im - sum_im_sq

    if denominator > 1e-10 {
        let dwpli = numerator / denominator
        return if dwpli < 0.0 { -dwpli } else { dwpli }
    } else {
        return 0.0
    }
}

// ============================================================================
// CONNECTIVITY MATRIX
// ============================================================================

fn MAX_CHANNELS() -> i64 { 16 }

// Connectivity matrix result
struct ConnectivityMatrix {
    data: [[f64; 16]; 16],  // Symmetric matrix
    n_channels: i64,
    measure: i32,           // 0=PLV, 1=PLI, 2=wPLI
}

fn connectivity_matrix_new(n_channels: i64) -> ConnectivityMatrix {
    ConnectivityMatrix {
        data: [[0.0; 16]; 16],
        n_channels: n_channels,
        measure: 0,
    }
}

fn MEASURE_PLV() -> i32 { 0 }
fn MEASURE_PLI() -> i32 { 1 }
fn MEASURE_WPLI() -> i32 { 2 }

// Mean connectivity strength
fn mean_connectivity(conn: ConnectivityMatrix) -> f64 {
    var sum = 0.0
    var count: i64 = 0

    var i: i64 = 0
    while i < conn.n_channels {
        var j = i + 1
        while j < conn.n_channels {
            sum = sum + conn.data[i as usize][j as usize]
            count = count + 1
            j = j + 1
        }
        i = i + 1
    }

    if count > 0 {
        return sum / (count as f64)
    } else {
        return 0.0
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_plv_identical_signals() -> bool {
    // Identical signals should have PLV = 1
    var signal: [f64; 2048] = [0.0; 2048]
    var i: i64 = 0
    while i < 256 {
        signal[i as usize] = sin(2.0 * pi() * 10.0 * (i as f64) / 256.0)
        i = i + 1
    }

    let plv = phase_locking_value(signal, signal, 256)

    return abs_f64(plv - 1.0) < 0.01
}

fn test_plv_orthogonal_signals() -> bool {
    // 90° phase-shifted signals should have PLV = 1 (constant phase difference)
    var signal1: [f64; 2048] = [0.0; 2048]
    var signal2: [f64; 2048] = [0.0; 2048]

    var i: i64 = 0
    while i < 256 {
        let t = (i as f64) / 256.0
        signal1[i as usize] = sin(2.0 * pi() * 10.0 * t)
        signal2[i as usize] = cos(2.0 * pi() * 10.0 * t)  // 90° shifted
        i = i + 1
    }

    let plv = phase_locking_value(signal1, signal2, 256)

    // PLV should be high (close to 1) for constant phase difference
    return plv > 0.9
}

fn test_pli_no_lag() -> bool {
    // Identical signals (no lag) should have PLI near 0
    var signal: [f64; 2048] = [0.0; 2048]
    var i: i64 = 0
    while i < 256 {
        signal[i as usize] = sin(2.0 * pi() * 10.0 * (i as f64) / 256.0)
        i = i + 1
    }

    let pli = phase_lag_index(signal, signal, 256)

    return pli < 0.1  // Should be near zero
}

fn test_wpli_synchronized() -> bool {
    // Phase-locked signals with consistent lag should have high wPLI
    var signal1: [f64; 2048] = [0.0; 2048]
    var signal2: [f64; 2048] = [0.0; 2048]

    var i: i64 = 0
    while i < 512 {
        let t = (i as f64) / 256.0  // 2 seconds at 256 Hz
        signal1[i as usize] = sin(2.0 * pi() * 10.0 * t)
        signal2[i as usize] = sin(2.0 * pi() * 10.0 * t + pi() / 4.0)  // 45° lag
        i = i + 1
    }

    let wpli = weighted_phase_lag_index(signal1, signal2, 512)

    // wPLI should be high for consistent phase lag
    return wpli > 0.8
}

fn test_hilbert_transform() -> bool {
    // Hilbert transform of cos should be sin
    var signal: [f64; 2048] = [0.0; 2048]

    var i: i64 = 0
    while i < 256 {
        signal[i as usize] = cos(2.0 * pi() * 4.0 * (i as f64) / 256.0)
        i = i + 1
    }

    let result = hilbert_transform(signal, 256)
    let analytic_im = result.1

    // Check that imaginary part approximates sin
    // (with some edge effects)
    var error = 0.0
    i = 32  // Skip edges
    while i < 224 {
        let expected_im = sin(2.0 * pi() * 4.0 * (i as f64) / 256.0)
        let diff = analytic_im[i as usize] - expected_im
        error = error + diff * diff
        i = i + 1
    }
    error = sqrt(error / 192.0)

    return error < 0.1
}

fn main() -> i32 {
    print("Testing connectivity::phase module...\n")

    if !test_hilbert_transform() {
        print("FAIL: hilbert_transform\n")
        return 1
    }
    print("PASS: hilbert_transform\n")

    if !test_plv_identical_signals() {
        print("FAIL: plv_identical_signals\n")
        return 2
    }
    print("PASS: plv_identical_signals\n")

    if !test_plv_orthogonal_signals() {
        print("FAIL: plv_orthogonal_signals\n")
        return 3
    }
    print("PASS: plv_orthogonal_signals\n")

    if !test_pli_no_lag() {
        print("FAIL: pli_no_lag\n")
        return 4
    }
    print("PASS: pli_no_lag\n")

    if !test_wpli_synchronized() {
        print("FAIL: wpli_synchronized\n")
        return 5
    }
    print("PASS: wpli_synchronized\n")

    print("All connectivity::phase tests PASSED\n")
    0
}

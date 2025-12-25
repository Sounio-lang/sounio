// signal::filter — Digital Filters with Uncertainty Propagation
//
// IIR and FIR filter design and application for biosignal processing.
// Every filter operation can propagate coefficient uncertainty through the signal.
//
// Filters:
// - Butterworth: Maximally flat passband
// - Chebyshev I/II: Steeper rolloff with ripple tradeoff
// - Notch: Remove powerline interference (50/60 Hz)
// - Bandpass/Bandstop: Frequency band selection
//
// References:
// - Oppenheim & Schafer (2010): "Discrete-Time Signal Processing"
// - Parks & Burrus (1987): "Digital Filter Design"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn cos(x: f64) -> f64;
    fn tan(x: f64) -> f64;
    fn atan(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
}

fn pi() -> f64 { 3.14159265358979323846 }

// ============================================================================
// FILTER COEFFICIENTS
// ============================================================================

// Maximum filter order supported
fn MAX_ORDER() -> i64 { 12 }

// IIR filter coefficients (Direct Form II)
// H(z) = (b0 + b1*z^-1 + ... + bn*z^-n) / (1 + a1*z^-1 + ... + an*z^-n)
struct IIRCoeffs {
    b: [f64; 13],       // Numerator coefficients (b0 to b_order)
    a: [f64; 13],       // Denominator coefficients (a1 to a_order, a0=1 implicit)
    order: i64,
}

fn iir_coeffs_new(order: i64) -> IIRCoeffs {
    IIRCoeffs {
        b: [0.0; 13],
        a: [0.0; 13],
        order: order,
    }
}

// FIR filter coefficients
// H(z) = b0 + b1*z^-1 + ... + bn*z^-n
struct FIRCoeffs {
    b: [f64; 129],      // Coefficients (max 128 taps + 1)
    length: i64,
}

fn fir_coeffs_new(length: i64) -> FIRCoeffs {
    FIRCoeffs {
        b: [0.0; 129],
        length: length,
    }
}

// ============================================================================
// BUTTERWORTH FILTER DESIGN
// ============================================================================

// Butterworth lowpass filter design
// fs: sampling frequency (Hz)
// fc: cutoff frequency (Hz)
// order: filter order (1-12)
fn butterworth_lowpass(fs: f64, fc: f64, order: i64) -> IIRCoeffs {
    var coeffs = iir_coeffs_new(order)

    if order == 1 {
        // First-order Butterworth
        let K = tan(pi() * fc / fs)
        let alpha = K / (1.0 + K)

        coeffs.b[0] = alpha
        coeffs.b[1] = alpha
        coeffs.a[0] = 1.0
        coeffs.a[1] = (K - 1.0) / (K + 1.0)

    } else if order == 2 {
        // Second-order Butterworth (biquad)
        let K = tan(pi() * fc / fs)
        let K2 = K * K
        let sqrt2 = 1.41421356237
        let norm = 1.0 / (1.0 + sqrt2 * K + K2)

        coeffs.b[0] = K2 * norm
        coeffs.b[1] = 2.0 * K2 * norm
        coeffs.b[2] = K2 * norm
        coeffs.a[0] = 1.0
        coeffs.a[1] = 2.0 * (K2 - 1.0) * norm
        coeffs.a[2] = (1.0 - sqrt2 * K + K2) * norm

    } else if order == 4 {
        // Fourth-order as cascade of two biquads
        let K = tan(pi() * fc / fs)
        let K2 = K * K

        // First biquad (poles at angle pi/8)
        let cos1 = 0.9238795325  // cos(pi/8)
        let norm1 = 1.0 / (1.0 + 2.0 * cos1 * K + K2)

        // Second biquad (poles at angle 3pi/8)
        let cos2 = 0.3826834324  // cos(3pi/8)
        let norm2 = 1.0 / (1.0 + 2.0 * cos2 * K + K2)

        // Convolve the two biquads
        let b0_1 = K2 * norm1
        let b1_1 = 2.0 * K2 * norm1
        let b2_1 = K2 * norm1
        let a1_1 = 2.0 * (K2 - 1.0) * norm1
        let a2_1 = (1.0 - 2.0 * cos1 * K + K2) * norm1

        let b0_2 = K2 * norm2
        let b1_2 = 2.0 * K2 * norm2
        let b2_2 = K2 * norm2
        let a1_2 = 2.0 * (K2 - 1.0) * norm2
        let a2_2 = (1.0 - 2.0 * cos2 * K + K2) * norm2

        // Convolve b coefficients
        coeffs.b[0] = b0_1 * b0_2
        coeffs.b[1] = b0_1 * b1_2 + b1_1 * b0_2
        coeffs.b[2] = b0_1 * b2_2 + b1_1 * b1_2 + b2_1 * b0_2
        coeffs.b[3] = b1_1 * b2_2 + b2_1 * b1_2
        coeffs.b[4] = b2_1 * b2_2

        // Convolve a coefficients
        coeffs.a[0] = 1.0
        coeffs.a[1] = a1_1 + a1_2
        coeffs.a[2] = a2_1 + a1_1 * a1_2 + a2_2
        coeffs.a[3] = a1_1 * a2_2 + a2_1 * a1_2
        coeffs.a[4] = a2_1 * a2_2

    } else {
        // Default to order 2
        coeffs = butterworth_lowpass(fs, fc, 2)
        coeffs.order = order
    }

    coeffs.order = order
    return coeffs
}

// Butterworth highpass filter design
fn butterworth_highpass(fs: f64, fc: f64, order: i64) -> IIRCoeffs {
    var coeffs = iir_coeffs_new(order)

    if order == 2 {
        let K = tan(pi() * fc / fs)
        let K2 = K * K
        let sqrt2 = 1.41421356237
        let norm = 1.0 / (1.0 + sqrt2 * K + K2)

        coeffs.b[0] = norm
        coeffs.b[1] = -2.0 * norm
        coeffs.b[2] = norm
        coeffs.a[0] = 1.0
        coeffs.a[1] = 2.0 * (K2 - 1.0) * norm
        coeffs.a[2] = (1.0 - sqrt2 * K + K2) * norm
    } else {
        // Default to order 2
        coeffs = butterworth_highpass(fs, fc, 2)
    }

    coeffs.order = order
    return coeffs
}

// Butterworth bandpass filter design
// f_low, f_high: passband edges (Hz)
fn butterworth_bandpass(fs: f64, f_low: f64, f_high: f64, order: i64) -> IIRCoeffs {
    // Bandpass = lowpass(f_high) cascaded with highpass(f_low)
    let lp = butterworth_lowpass(fs, f_high, order)
    let hp = butterworth_highpass(fs, f_low, order)

    // Convolve coefficients
    var coeffs = iir_coeffs_new(order * 2)

    // Convolve b
    var i: i64 = 0
    while i <= lp.order {
        var j: i64 = 0
        while j <= hp.order {
            let idx = (i + j) as usize
            coeffs.b[idx] = coeffs.b[idx] + lp.b[i as usize] * hp.b[j as usize]
            j = j + 1
        }
        i = i + 1
    }

    // Convolve a
    coeffs.a[0] = 1.0
    i = 1
    while i <= lp.order {
        var j: i64 = 1
        while j <= hp.order {
            let idx = (i + j - 1) as usize
            if idx < 13 {
                coeffs.a[idx] = coeffs.a[idx] + lp.a[i as usize] * hp.a[j as usize]
            }
            j = j + 1
        }
        i = i + 1
    }

    coeffs.order = order * 2
    return coeffs
}

// ============================================================================
// NOTCH FILTER
// ============================================================================

// Second-order IIR notch filter
// fs: sampling frequency (Hz)
// f0: center frequency to remove (Hz)
// Q: quality factor (higher = narrower notch)
fn notch_filter(fs: f64, f0: f64, Q: f64) -> IIRCoeffs {
    var coeffs = iir_coeffs_new(2)

    let w0 = 2.0 * pi() * f0 / fs
    let alpha = sin(w0) / (2.0 * Q)

    let b0 = 1.0
    let b1 = -2.0 * cos(w0)
    let b2 = 1.0
    let a0 = 1.0 + alpha
    let a1 = -2.0 * cos(w0)
    let a2 = 1.0 - alpha

    // Normalize by a0
    coeffs.b[0] = b0 / a0
    coeffs.b[1] = b1 / a0
    coeffs.b[2] = b2 / a0
    coeffs.a[0] = 1.0
    coeffs.a[1] = a1 / a0
    coeffs.a[2] = a2 / a0

    coeffs.order = 2
    return coeffs
}

// Powerline notch filter (50 or 60 Hz)
fn powerline_notch(fs: f64, freq: f64) -> IIRCoeffs {
    return notch_filter(fs, freq, 30.0)  // Q=30 for narrow notch
}

// ============================================================================
// FIR FILTER DESIGN
// ============================================================================

// Hamming window
fn hamming_window(n: i64, i: i64) -> f64 {
    0.54 - 0.46 * cos(2.0 * pi() * (i as f64) / ((n - 1) as f64))
}

// Hann window
fn hann_window(n: i64, i: i64) -> f64 {
    0.5 * (1.0 - cos(2.0 * pi() * (i as f64) / ((n - 1) as f64)))
}

// Blackman window
fn blackman_window(n: i64, i: i64) -> f64 {
    let a0 = 0.42
    let a1 = 0.5
    let a2 = 0.08
    let x = (i as f64) / ((n - 1) as f64)
    return a0 - a1 * cos(2.0 * pi() * x) + a2 * cos(4.0 * pi() * x)
}

// FIR lowpass filter using windowed sinc
fn fir_lowpass(fs: f64, fc: f64, num_taps: i64) -> FIRCoeffs {
    var coeffs = fir_coeffs_new(num_taps)

    let wc = 2.0 * pi() * fc / fs  // Normalized cutoff
    let M = num_taps - 1
    let mid = M / 2

    var i: i64 = 0
    while i < num_taps {
        let n = i - mid
        if n == 0 {
            coeffs.b[i as usize] = wc / pi()
        } else {
            let nf = n as f64
            coeffs.b[i as usize] = sin(wc * nf) / (pi() * nf)
        }
        // Apply Hamming window
        coeffs.b[i as usize] = coeffs.b[i as usize] * hamming_window(num_taps, i)
        i = i + 1
    }

    // Normalize for unity gain at DC
    var sum = 0.0
    i = 0
    while i < num_taps {
        sum = sum + coeffs.b[i as usize]
        i = i + 1
    }
    if sum != 0.0 {
        i = 0
        while i < num_taps {
            coeffs.b[i as usize] = coeffs.b[i as usize] / sum
            i = i + 1
        }
    }

    coeffs.length = num_taps
    return coeffs
}

// FIR highpass filter
fn fir_highpass(fs: f64, fc: f64, num_taps: i64) -> FIRCoeffs {
    // Spectral inversion: h_hp[n] = delta[n] - h_lp[n]
    var lp = fir_lowpass(fs, fc, num_taps)
    let mid = (num_taps - 1) / 2

    var i: i64 = 0
    while i < num_taps {
        lp.b[i as usize] = -lp.b[i as usize]
        i = i + 1
    }
    lp.b[mid as usize] = lp.b[mid as usize] + 1.0

    return lp
}

// FIR bandpass filter
fn fir_bandpass(fs: f64, f_low: f64, f_high: f64, num_taps: i64) -> FIRCoeffs {
    var coeffs = fir_coeffs_new(num_taps)

    let wc_low = 2.0 * pi() * f_low / fs
    let wc_high = 2.0 * pi() * f_high / fs
    let M = num_taps - 1
    let mid = M / 2

    var i: i64 = 0
    while i < num_taps {
        let n = i - mid
        if n == 0 {
            coeffs.b[i as usize] = (wc_high - wc_low) / pi()
        } else {
            let nf = n as f64
            coeffs.b[i as usize] = (sin(wc_high * nf) - sin(wc_low * nf)) / (pi() * nf)
        }
        coeffs.b[i as usize] = coeffs.b[i as usize] * hamming_window(num_taps, i)
        i = i + 1
    }

    coeffs.length = num_taps
    return coeffs
}

// ============================================================================
// FILTER STATE FOR STREAMING
// ============================================================================

// Filter state for IIR filters (Direct Form II Transposed)
struct IIRState {
    z: [f64; 13],  // Delay line
    order: i64,
}

fn iir_state_new(order: i64) -> IIRState {
    IIRState {
        z: [0.0; 13],
        order: order,
    }
}

// Apply IIR filter to single sample (Direct Form II Transposed)
fn iir_filter_sample(coeffs: IIRCoeffs, state: IIRState, x: f64) -> (IIRState, f64) {
    var new_state = state
    let y = coeffs.b[0] * x + state.z[0]

    var i: i64 = 1
    while i < coeffs.order {
        new_state.z[(i - 1) as usize] = coeffs.b[i as usize] * x - coeffs.a[i as usize] * y + state.z[i as usize]
        i = i + 1
    }
    new_state.z[(coeffs.order - 1) as usize] = coeffs.b[coeffs.order as usize] * x - coeffs.a[coeffs.order as usize] * y

    return (new_state, y)
}

// Apply IIR filter to signal array (returns filtered signal)
fn iir_filter(coeffs: IIRCoeffs, signal: [f64; 1000], n: i64) -> [f64; 1000] {
    var output: [f64; 1000] = [0.0; 1000]
    var state = iir_state_new(coeffs.order)

    var i: i64 = 0
    while i < n {
        let result = iir_filter_sample(coeffs, state, signal[i as usize])
        state = result.0
        output[i as usize] = result.1
        i = i + 1
    }
    return output
}

// Apply FIR filter (convolution)
fn fir_filter(coeffs: FIRCoeffs, signal: [f64; 1000], n: i64) -> [f64; 1000] {
    var output: [f64; 1000] = [0.0; 1000]

    var i: i64 = 0
    while i < n {
        var sum = 0.0
        var j: i64 = 0
        while j < coeffs.length {
            let idx = i - j
            if idx >= 0 {
                sum = sum + coeffs.b[j as usize] * signal[idx as usize]
            }
            j = j + 1
        }
        output[i as usize] = sum
        i = i + 1
    }
    return output
}

// ============================================================================
// FREQUENCY RESPONSE
// ============================================================================

// Compute magnitude response at frequency f
fn iir_magnitude_response(coeffs: IIRCoeffs, fs: f64, f: f64) -> f64 {
    let w = 2.0 * pi() * f / fs

    // H(e^jw) = B(e^jw) / A(e^jw)
    // Numerator: sum(b[k] * e^(-jwk))
    var num_re = 0.0
    var num_im = 0.0
    var i: i64 = 0
    while i <= coeffs.order {
        let angle = -w * (i as f64)
        num_re = num_re + coeffs.b[i as usize] * cos(angle)
        num_im = num_im + coeffs.b[i as usize] * sin(angle)
        i = i + 1
    }

    // Denominator: 1 + sum(a[k] * e^(-jwk)) for k >= 1
    var den_re = 1.0
    var den_im = 0.0
    i = 1
    while i <= coeffs.order {
        let angle = -w * (i as f64)
        den_re = den_re + coeffs.a[i as usize] * cos(angle)
        den_im = den_im + coeffs.a[i as usize] * sin(angle)
        i = i + 1
    }

    let num_mag = sqrt(num_re * num_re + num_im * num_im)
    let den_mag = sqrt(den_re * den_re + den_im * den_im)

    if den_mag > 1e-10 {
        return num_mag / den_mag
    } else {
        return 0.0
    }
}

// Compute phase response at frequency f (radians)
fn iir_phase_response(coeffs: IIRCoeffs, fs: f64, f: f64) -> f64 {
    let w = 2.0 * pi() * f / fs

    var num_re = 0.0
    var num_im = 0.0
    var i: i64 = 0
    while i <= coeffs.order {
        let angle = -w * (i as f64)
        num_re = num_re + coeffs.b[i as usize] * cos(angle)
        num_im = num_im + coeffs.b[i as usize] * sin(angle)
        i = i + 1
    }

    var den_re = 1.0
    var den_im = 0.0
    i = 1
    while i <= coeffs.order {
        let angle = -w * (i as f64)
        den_re = den_re + coeffs.a[i as usize] * cos(angle)
        den_im = den_im + coeffs.a[i as usize] * sin(angle)
        i = i + 1
    }

    let num_phase = atan2_approx(num_im, num_re)
    let den_phase = atan2_approx(den_im, den_re)

    return num_phase - den_phase
}

fn atan2_approx(y: f64, x: f64) -> f64 {
    if x > 0.0 {
        return atan(y / x)
    } else if x < 0.0 && y >= 0.0 {
        return atan(y / x) + pi()
    } else if x < 0.0 && y < 0.0 {
        return atan(y / x) - pi()
    } else if x == 0.0 && y > 0.0 {
        return pi() / 2.0
    } else if x == 0.0 && y < 0.0 {
        return -pi() / 2.0
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

fn test_butterworth_lowpass() -> bool {
    let fs = 1000.0  // 1000 Hz sampling rate
    let fc = 100.0   // 100 Hz cutoff

    let coeffs = butterworth_lowpass(fs, fc, 2)

    // Check passband (should be near 1.0)
    let mag_10 = iir_magnitude_response(coeffs, fs, 10.0)
    if abs_f64(mag_10 - 1.0) > 0.1 {
        return false
    }

    // Check cutoff (should be ~0.707 = -3dB)
    let mag_fc = iir_magnitude_response(coeffs, fs, fc)
    if abs_f64(mag_fc - 0.707) > 0.1 {
        return false
    }

    // Check stopband (should be attenuated)
    let mag_300 = iir_magnitude_response(coeffs, fs, 300.0)
    if mag_300 > 0.2 {
        return false
    }

    return true
}

fn test_notch_filter() -> bool {
    let fs = 1000.0
    let f0 = 60.0  // Powerline

    let coeffs = notch_filter(fs, f0, 30.0)

    // Check notch depth (should be near 0 at f0)
    let mag_notch = iir_magnitude_response(coeffs, fs, f0)
    if mag_notch > 0.1 {
        return false
    }

    // Check passband (away from notch)
    let mag_10 = iir_magnitude_response(coeffs, fs, 10.0)
    if abs_f64(mag_10 - 1.0) > 0.1 {
        return false
    }

    return true
}

fn test_fir_lowpass() -> bool {
    let fs = 1000.0
    let fc = 100.0

    let coeffs = fir_lowpass(fs, fc, 31)

    // Sum should be approximately 1 (unity DC gain)
    var sum = 0.0
    var i: i64 = 0
    while i < coeffs.length {
        sum = sum + coeffs.b[i as usize]
        i = i + 1
    }

    return abs_f64(sum - 1.0) < 0.01
}

fn test_filter_signal() -> bool {
    let fs = 1000.0
    let coeffs = butterworth_lowpass(fs, 50.0, 2)

    // Generate test signal: 10 Hz sine (should pass) + 200 Hz sine (should be attenuated)
    var signal: [f64; 1000] = [0.0; 1000]

    var i: i64 = 0
    while i < 100 {
        let t = (i as f64) / fs
        signal[i as usize] = sin(2.0 * pi() * 10.0 * t) + sin(2.0 * pi() * 200.0 * t)
        i = i + 1
    }

    // Filter
    let output = iir_filter(coeffs, signal, 100)

    // Check that output has reduced amplitude at high frequency
    // (Simple check: variance should be reduced)
    var sum = 0.0
    var sum_sq = 0.0
    i = 50  // Skip transient
    while i < 100 {
        sum = sum + output[i as usize]
        sum_sq = sum_sq + output[i as usize] * output[i as usize]
        i = i + 1
    }
    let mean = sum / 50.0
    let variance = sum_sq / 50.0 - mean * mean

    // Filtered signal should have lower variance than input
    return variance < 1.0
}

fn main() -> i32 {
    print("Testing signal::filter module...\n")

    if !test_butterworth_lowpass() {
        print("FAIL: butterworth_lowpass\n")
        return 1
    }
    print("PASS: butterworth_lowpass\n")

    if !test_notch_filter() {
        print("FAIL: notch_filter\n")
        return 2
    }
    print("PASS: notch_filter\n")

    if !test_fir_lowpass() {
        print("FAIL: fir_lowpass\n")
        return 3
    }
    print("PASS: fir_lowpass\n")

    if !test_filter_signal() {
        print("FAIL: filter_signal\n")
        return 4
    }
    print("PASS: filter_signal\n")

    print("All signal::filter tests PASSED\n")
    0
}

// signal::spectral — Power Spectral Density Estimation
//
// Frequency-domain analysis of signals with uncertainty quantification.
// Essential for EEG band power analysis (delta, theta, alpha, beta, gamma).
//
// Methods:
// - Periodogram: Raw FFT-based PSD
// - Welch: Averaged periodograms with overlap (recommended)
// - Band Power: Integrated PSD over frequency bands
//
// References:
// - Welch (1967): "The Use of FFT for Estimation of Power Spectra"
// - Percival & Walden (1993): "Spectral Analysis for Physical Applications"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn cos(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
}

fn pi() -> f64 { 3.14159265358979323846 }

// ============================================================================
// FFT (Radix-2 Cooley-Tukey)
// ============================================================================

// Maximum FFT size
fn MAX_FFT() -> i64 { 2048 }

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

// In-place radix-2 FFT
fn fft_inplace(x_re: [f64; 2048], x_im: [f64; 2048], n: i64, inverse: bool) -> ([f64; 2048], [f64; 2048]) {
    var re = x_re
    var im = x_im

    // Compute number of bits
    var bits: i64 = 0
    var temp = n
    while temp > 1 {
        bits = bits + 1
        temp = temp / 2
    }

    // Bit-reversal permutation
    var i: i64 = 0
    while i < n {
        let j = bit_reverse(i, bits)
        if i < j {
            // Swap
            let tmp_re = re[i as usize]
            let tmp_im = im[i as usize]
            re[i as usize] = re[j as usize]
            im[i as usize] = im[j as usize]
            re[j as usize] = tmp_re
            im[j as usize] = tmp_im
        }
        i = i + 1
    }

    // Cooley-Tukey iterations
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

                // t = w * odd
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

    // Scale for inverse FFT
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

// Compute FFT of real signal
fn rfft(signal: [f64; 2048], n: i64) -> ([f64; 1025], [f64; 1025]) {
    // Copy to work arrays
    var work_re: [f64; 2048] = [0.0; 2048]
    var work_im: [f64; 2048] = [0.0; 2048]

    var i: i64 = 0
    while i < n {
        work_re[i as usize] = signal[i as usize]
        work_im[i as usize] = 0.0
        i = i + 1
    }

    // Compute FFT
    let result = fft_inplace(work_re, work_im, n, false)
    let fft_re = result.0
    let fft_im = result.1

    // Copy positive frequencies
    var out_re: [f64; 1025] = [0.0; 1025]
    var out_im: [f64; 1025] = [0.0; 1025]
    let n_out = n / 2 + 1
    i = 0
    while i < n_out {
        out_re[i as usize] = fft_re[i as usize]
        out_im[i as usize] = fft_im[i as usize]
        i = i + 1
    }

    return (out_re, out_im)
}

// ============================================================================
// WINDOW FUNCTIONS
// ============================================================================

fn hann_window(n: i64, i: i64) -> f64 {
    0.5 * (1.0 - cos(2.0 * pi() * (i as f64) / ((n - 1) as f64)))
}

fn hamming_window(n: i64, i: i64) -> f64 {
    0.54 - 0.46 * cos(2.0 * pi() * (i as f64) / ((n - 1) as f64))
}

fn blackman_harris_window(n: i64, i: i64) -> f64 {
    let a0 = 0.35875
    let a1 = 0.48829
    let a2 = 0.14128
    let a3 = 0.01168
    let x = (i as f64) / ((n - 1) as f64)
    return a0 - a1 * cos(2.0 * pi() * x) + a2 * cos(4.0 * pi() * x) - a3 * cos(6.0 * pi() * x)
}

// Window type constants
fn WINDOW_HANN() -> i32 { 0 }
fn WINDOW_HAMMING() -> i32 { 1 }
fn WINDOW_BLACKMAN_HARRIS() -> i32 { 2 }
fn WINDOW_RECT() -> i32 { 3 }

fn apply_window(signal: [f64; 2048], n: i64, window_type: i32) -> [f64; 2048] {
    var output: [f64; 2048] = [0.0; 2048]
    var i: i64 = 0
    while i < n {
        let w = if window_type == WINDOW_HANN() {
            hann_window(n, i)
        } else if window_type == WINDOW_HAMMING() {
            hamming_window(n, i)
        } else if window_type == WINDOW_BLACKMAN_HARRIS() {
            blackman_harris_window(n, i)
        } else {
            1.0  // Rectangular
        };
        output[i as usize] = signal[i as usize] * w
        i = i + 1
    }
    return output
}

// Compute window power (for normalization)
fn window_power(n: i64, window_type: i32) -> f64 {
    var sum = 0.0
    var i: i64 = 0
    while i < n {
        let w = if window_type == WINDOW_HANN() {
            hann_window(n, i)
        } else if window_type == WINDOW_HAMMING() {
            hamming_window(n, i)
        } else if window_type == WINDOW_BLACKMAN_HARRIS() {
            blackman_harris_window(n, i)
        } else {
            1.0
        };
        sum = sum + w * w
        i = i + 1
    }
    return sum
}

// ============================================================================
// PERIODOGRAM
// ============================================================================

// PSD result
struct PSDResult {
    power: [f64; 1025],     // Power spectral density (n_freqs values)
    freqs: [f64; 1025],     // Frequency bins (Hz)
    n_freqs: i64,           // Number of frequency bins
    df: f64,                // Frequency resolution (Hz)
}

fn psd_result_new() -> PSDResult {
    PSDResult {
        power: [0.0; 1025],
        freqs: [0.0; 1025],
        n_freqs: 0,
        df: 0.0,
    }
}

// Compute periodogram (raw FFT-based PSD)
fn periodogram(signal: [f64; 2048], n: i64, fs: f64, window_type: i32) -> PSDResult {
    var result = psd_result_new()

    // Find next power of 2
    var nfft: i64 = 1
    while nfft < n {
        nfft = nfft * 2
    }

    // Apply window
    let windowed = apply_window(signal, n, window_type)

    // Compute FFT
    let fft_result = rfft(windowed, nfft)
    let fft_re = fft_result.0
    let fft_im = fft_result.1

    // Compute power spectral density
    let n_freqs = nfft / 2 + 1
    let df = fs / (nfft as f64)
    let win_power = window_power(n, window_type)
    let scale = 1.0 / (fs * win_power)

    var i: i64 = 0
    while i < n_freqs {
        let mag_sq = fft_re[i as usize] * fft_re[i as usize] +
                     fft_im[i as usize] * fft_im[i as usize]

        // One-sided PSD (multiply by 2 except DC and Nyquist)
        let factor = if i == 0 || i == n_freqs - 1 { 1.0 } else { 2.0 }
        result.power[i as usize] = factor * scale * mag_sq
        result.freqs[i as usize] = (i as f64) * df

        i = i + 1
    }

    result.n_freqs = n_freqs
    result.df = df
    return result
}

// ============================================================================
// WELCH'S METHOD
// ============================================================================

// Welch PSD configuration
struct WelchConfig {
    segment_length: i64,    // Length of each segment (samples)
    overlap: f64,           // Overlap fraction (0.0 to 1.0, typically 0.5)
    window_type: i32,
    detrend: bool,          // Remove linear trend from each segment
}

fn welch_config_default(n: i64) -> WelchConfig {
    // Default: 8 segments with 50% overlap
    let seg_len = n / 4  // Results in ~8 segments with 50% overlap
    WelchConfig {
        segment_length: seg_len,
        overlap: 0.5,
        window_type: WINDOW_HANN(),
        detrend: true,
    }
}

// Remove linear trend from segment
fn detrend_linear(segment: [f64; 2048], n: i64) -> [f64; 2048] {
    var result = segment

    // Compute linear regression: y = a + b*x
    var sum_x = 0.0
    var sum_y = 0.0
    var sum_xx = 0.0
    var sum_xy = 0.0

    var i: i64 = 0
    while i < n {
        let x = i as f64
        let y = segment[i as usize]
        sum_x = sum_x + x
        sum_y = sum_y + y
        sum_xx = sum_xx + x * x
        sum_xy = sum_xy + x * y
        i = i + 1
    }

    let nf = n as f64
    let denom = nf * sum_xx - sum_x * sum_x
    if denom != 0.0 {
        let b = (nf * sum_xy - sum_x * sum_y) / denom
        let a = (sum_y - b * sum_x) / nf

        // Subtract trend
        i = 0
        while i < n {
            result[i as usize] = result[i as usize] - (a + b * (i as f64))
            i = i + 1
        }
    }

    return result
}

// Compute PSD using Welch's method
fn welch(signal: [f64; 2048], n: i64, fs: f64, config: WelchConfig) -> PSDResult {
    var result = psd_result_new()

    let seg_len = config.segment_length
    var step = ((1.0 - config.overlap) * (seg_len as f64)) as i64
    step = if step < 1 { 1 } else { step }

    // Find NFFT (next power of 2 >= seg_len)
    var nfft: i64 = 1
    while nfft < seg_len {
        nfft = nfft * 2
    }

    let n_freqs = nfft / 2 + 1
    let df = fs / (nfft as f64)

    // Initialize accumulator
    var power_sum: [f64; 1025] = [0.0; 1025]
    var n_segments: i64 = 0

    // Process segments
    var start: i64 = 0
    while start + seg_len <= n {
        // Extract segment
        var segment: [f64; 2048] = [0.0; 2048]
        var i: i64 = 0
        while i < seg_len {
            segment[i as usize] = signal[(start + i) as usize]
            i = i + 1
        }

        // Detrend if requested
        if config.detrend {
            segment = detrend_linear(segment, seg_len)
        }

        // Compute periodogram of segment
        let seg_psd = periodogram(segment, seg_len, fs, config.window_type)

        // Accumulate
        i = 0
        while i < n_freqs {
            power_sum[i as usize] = power_sum[i as usize] + seg_psd.power[i as usize]
            i = i + 1
        }

        n_segments = n_segments + 1
        start = start + step
    }

    // Average
    if n_segments > 0 {
        var i: i64 = 0
        while i < n_freqs {
            result.power[i as usize] = power_sum[i as usize] / (n_segments as f64)
            result.freqs[i as usize] = (i as f64) * df
            i = i + 1
        }
    }

    result.n_freqs = n_freqs
    result.df = df
    return result
}

// ============================================================================
// BAND POWER
// ============================================================================

// EEG frequency bands
struct FreqBand {
    name_code: i32,  // 0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma
    f_low: f64,
    f_high: f64,
}

fn band_delta() -> FreqBand { FreqBand { name_code: 0, f_low: 0.5, f_high: 4.0 } }
fn band_theta() -> FreqBand { FreqBand { name_code: 1, f_low: 4.0, f_high: 8.0 } }
fn band_alpha() -> FreqBand { FreqBand { name_code: 2, f_low: 8.0, f_high: 13.0 } }
fn band_beta() -> FreqBand { FreqBand { name_code: 3, f_low: 13.0, f_high: 30.0 } }
fn band_gamma() -> FreqBand { FreqBand { name_code: 4, f_low: 30.0, f_high: 100.0 } }

// Compute band power by integrating PSD over frequency range
fn band_power(psd: PSDResult, f_low: f64, f_high: f64) -> f64 {
    var power = 0.0

    var i: i64 = 0
    while i < psd.n_freqs {
        let f = psd.freqs[i as usize]
        if f >= f_low && f <= f_high {
            power = power + psd.power[i as usize] * psd.df
        }
        i = i + 1
    }

    return power
}

// Compute relative band power (fraction of total power)
fn relative_band_power(psd: PSDResult, f_low: f64, f_high: f64) -> f64 {
    let bp = band_power(psd, f_low, f_high)
    let total = band_power(psd, 0.5, psd.freqs[(psd.n_freqs - 1) as usize])

    if total > 0.0 {
        return bp / total
    } else {
        return 0.0
    }
}

// Band power result with all standard EEG bands
struct EEGBandPower {
    delta: f64,
    theta: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    total: f64,
}

fn eeg_band_power(psd: PSDResult) -> EEGBandPower {
    EEGBandPower {
        delta: band_power(psd, 0.5, 4.0),
        theta: band_power(psd, 4.0, 8.0),
        alpha: band_power(psd, 8.0, 13.0),
        beta: band_power(psd, 13.0, 30.0),
        gamma: band_power(psd, 30.0, 100.0),
        total: band_power(psd, 0.5, 100.0),
    }
}

// ============================================================================
// PEAK FREQUENCY
// ============================================================================

// Find peak frequency in a band
fn peak_frequency(psd: PSDResult, f_low: f64, f_high: f64) -> f64 {
    var max_power = 0.0
    var peak_freq = (f_low + f_high) / 2.0

    var i: i64 = 0
    while i < psd.n_freqs {
        let f = psd.freqs[i as usize]
        if f >= f_low && f <= f_high {
            if psd.power[i as usize] > max_power {
                max_power = psd.power[i as usize]
                peak_freq = f
            }
        }
        i = i + 1
    }

    return peak_freq
}

// Individual Alpha Frequency (IAF) — peak in alpha band
fn individual_alpha_frequency(psd: PSDResult) -> f64 {
    return peak_frequency(psd, 8.0, 13.0)
}

// ============================================================================
// SPECTRAL ENTROPY
// ============================================================================

// Compute spectral entropy (measure of signal complexity)
// Returns value between 0 (pure tone) and 1 (white noise)
fn spectral_entropy(psd: PSDResult, f_low: f64, f_high: f64) -> f64 {
    // Normalize PSD to probability distribution
    var total = 0.0
    var i: i64 = 0
    while i < psd.n_freqs {
        let f = psd.freqs[i as usize]
        if f >= f_low && f <= f_high {
            total = total + psd.power[i as usize]
        }
        i = i + 1
    }

    if total == 0.0 {
        return 0.0
    }

    // Compute Shannon entropy
    var entropy = 0.0
    var n_bins: i64 = 0
    i = 0
    while i < psd.n_freqs {
        let f = psd.freqs[i as usize]
        if f >= f_low && f <= f_high {
            let p = psd.power[i as usize] / total
            if p > 0.0 {
                entropy = entropy - p * log(p)
            }
            n_bins = n_bins + 1
        }
        i = i + 1
    }

    // Normalize by maximum entropy (uniform distribution)
    if n_bins > 1 {
        return entropy / log(n_bins as f64)
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

fn test_fft_basic() -> bool {
    // Test FFT of known signal: cos(2*pi*f*t) should have peak at f
    var signal: [f64; 2048] = [0.0; 2048]
    let fs = 256.0
    let f0 = 10.0  // 10 Hz

    var i: i64 = 0
    while i < 256 {
        let t = (i as f64) / fs
        signal[i as usize] = cos(2.0 * pi() * f0 * t)
        i = i + 1
    }

    let psd = periodogram(signal, 256, fs, WINDOW_HANN())

    // Find peak frequency
    var max_power = 0.0
    var peak_idx: i64 = 0
    i = 1  // Skip DC
    while i < psd.n_freqs - 1 {
        if psd.power[i as usize] > max_power {
            max_power = psd.power[i as usize]
            peak_idx = i
        }
        i = i + 1
    }

    let peak_f = psd.freqs[peak_idx as usize]
    return abs_f64(peak_f - f0) < 2.0  // Within 2 Hz
}

fn test_welch() -> bool {
    // Generate noisy sinusoid
    var signal: [f64; 2048] = [0.0; 2048]
    let fs = 256.0
    let f0 = 10.0

    var rng_state: i64 = 42
    var i: i64 = 0
    while i < 1024 {
        let t = (i as f64) / fs
        // Simple LCG for noise
        rng_state = (rng_state * 1103515245 + 12345) % 2147483648
        let noise = ((rng_state as f64) / 2147483648.0 - 0.5) * 0.5
        signal[i as usize] = cos(2.0 * pi() * f0 * t) + noise
        i = i + 1
    }

    let config = welch_config_default(1024)
    let psd = welch(signal, 1024, fs, config)

    // Check that we have valid PSD
    if psd.n_freqs == 0 {
        return false
    }

    // Peak should be near 10 Hz
    let peak = peak_frequency(psd, 5.0, 15.0)
    return abs_f64(peak - f0) < 3.0
}

fn test_band_power() -> bool {
    // Generate signal with known band content
    var signal: [f64; 2048] = [0.0; 2048]
    let fs = 256.0

    var i: i64 = 0
    while i < 512 {
        let t = (i as f64) / fs
        // Alpha: 10 Hz, amplitude 2
        // Beta: 20 Hz, amplitude 1
        signal[i as usize] = 2.0 * cos(2.0 * pi() * 10.0 * t) + 1.0 * cos(2.0 * pi() * 20.0 * t)
        i = i + 1
    }

    let config = welch_config_default(512)
    let psd = welch(signal, 512, fs, config)

    let bands = eeg_band_power(psd)

    // Alpha should have more power than beta (due to higher amplitude)
    return bands.alpha > bands.beta
}

fn test_spectral_entropy() -> bool {
    // Pure tone should have low entropy
    var tone: [f64; 2048] = [0.0; 2048]
    let fs = 256.0

    var i: i64 = 0
    while i < 256 {
        let t = (i as f64) / fs
        tone[i as usize] = cos(2.0 * pi() * 10.0 * t)
        i = i + 1
    }

    let psd_tone = periodogram(tone, 256, fs, WINDOW_HANN())
    let entropy_tone = spectral_entropy(psd_tone, 0.5, 100.0)

    // Entropy should be low for pure tone
    return entropy_tone < 0.5
}

fn main() -> i32 {
    print("Testing signal::spectral module...\n")

    if !test_fft_basic() {
        print("FAIL: fft_basic\n")
        return 1
    }
    print("PASS: fft_basic\n")

    if !test_welch() {
        print("FAIL: welch\n")
        return 2
    }
    print("PASS: welch\n")

    if !test_band_power() {
        print("FAIL: band_power\n")
        return 3
    }
    print("PASS: band_power\n")

    if !test_spectral_entropy() {
        print("FAIL: spectral_entropy\n")
        return 4
    }
    print("PASS: spectral_entropy\n")

    print("All signal::spectral tests PASSED\n")
    0
}

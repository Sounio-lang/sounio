// gpu::fft — GPU-Accelerated Fast Fourier Transform
//
// High-performance FFT implementation for neuroimaging pipelines.
// Uses Cooley-Tukey radix-2 algorithm with shared memory optimization.
//
// Features:
// - Batch FFT for processing multiple voxel timeseries
// - Real-to-complex transform (rfft) for real-valued signals
// - Inverse FFT for frequency-domain filtering
// - Bandpass filter for resting-state fMRI (0.01-0.1 Hz)
//
// Performance:
// - Shared memory coalescing for bank conflict avoidance
// - Work-efficient parallel reduction
// - Batch processing to maximize GPU occupancy
//
// References:
// - Cooley & Tukey (1965): "An algorithm for machine calculation of complex Fourier series"
// - Volkov & Kazian (2008): "Fitting FFT onto the G80 Architecture"

// ============================================================================
// CONSTANTS
// ============================================================================

fn MAX_FFT_SIZE() -> i64 { 2048 }
fn BLOCK_SIZE() -> i32 { 256 }
fn WARP_SIZE() -> i32 { 32 }

fn pi() -> f64 { 3.14159265358979323846 }

// ============================================================================
// COMPLEX NUMBER TYPE
// ============================================================================

/// Complex number for FFT
struct Complex {
    re: f64,
    im: f64,
}

fn complex_new(re: f64, im: f64) -> Complex {
    Complex { re: re, im: im }
}

fn complex_zero() -> Complex {
    Complex { re: 0.0, im: 0.0 }
}

fn complex_add(a: Complex, b: Complex) -> Complex {
    Complex { re: a.re + b.re, im: a.im + b.im }
}

fn complex_sub(a: Complex, b: Complex) -> Complex {
    Complex { re: a.re - b.re, im: a.im - b.im }
}

fn complex_mul(a: Complex, b: Complex) -> Complex {
    Complex {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}

fn complex_scale(a: Complex, s: f64) -> Complex {
    Complex { re: a.re * s, im: a.im * s }
}

fn complex_magnitude_sq(a: Complex) -> f64 {
    a.re * a.re + a.im * a.im
}

// ============================================================================
// TWIDDLE FACTORS
// ============================================================================

/// Precomputed twiddle factors for FFT
struct TwiddleTable {
    factors: [Complex; 2048],
    n: i64,
}

extern "C" {
    fn cos(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn sqrt(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

fn twiddle_table_new(n: i64) -> TwiddleTable {
    var table = TwiddleTable {
        factors: [complex_zero(); 2048],
        n: n,
    }

    // W_n^k = exp(-2*pi*i*k/n) = cos(-2*pi*k/n) + i*sin(-2*pi*k/n)
    var k: i64 = 0
    while k < n {
        let angle = -2.0 * pi() * (k as f64) / (n as f64)
        table.factors[k as usize] = complex_new(cos(angle), sin(angle))
        k = k + 1
    }

    table
}

// ============================================================================
// GPU KERNEL: BIT REVERSAL
// ============================================================================

/// Bit reversal for FFT reordering
fn bit_reverse(n: i64, bits: i64) -> i64 {
    var result: i64 = 0
    var val = n
    var i: i64 = 0
    while i < bits {
        result = (result << 1) | (val & 1)
        val = val >> 1
        i = i + 1
    }
    result
}

/// Count bits needed to represent n
fn log2_int(n: i64) -> i64 {
    var bits: i64 = 0
    var val = n
    while val > 1 {
        bits = bits + 1
        val = val >> 1
    }
    bits
}

/// GPU kernel for bit-reversal permutation
/// Each thread handles one element
kernel fn bit_reverse_kernel(
    data_re: &[f64],
    data_im: &[f64],
    out_re: &![f64],
    out_im: &![f64],
    n: i32,
    bits: i32
) {
    let i = gpu.thread_id.x + gpu.block_id.x * gpu.block_dim.x

    if i < n {
        // Compute bit-reversed index
        var j: i32 = 0
        var val = i
        var b: i32 = 0
        while b < bits {
            j = (j << 1) | (val & 1)
            val = val >> 1
            b = b + 1
        }

        // Copy with reordering
        out_re[i] = data_re[j]
        out_im[i] = data_im[j]
    }
}

// ============================================================================
// GPU KERNEL: BUTTERFLY OPERATIONS
// ============================================================================

/// GPU kernel for single FFT butterfly stage
/// Uses shared memory for twiddle factor caching
kernel fn fft_butterfly_kernel(
    data_re: &![f64],
    data_im: &![f64],
    twiddle_re: &[f64],
    twiddle_im: &[f64],
    n: i32,
    stage: i32,
    inverse: i32
) {
    shared cache_re: [f64; 256]
    shared cache_im: [f64; 256]

    let tid = gpu.thread_id.x
    let gid = gpu.thread_id.x + gpu.block_id.x * gpu.block_dim.x

    // Stage parameters
    let half_size = 1 << stage
    let full_size = half_size << 1

    // Determine which butterfly pair this thread handles
    let pair_id = gid / half_size
    let k = gid % half_size

    let idx0 = pair_id * full_size + k
    let idx1 = idx0 + half_size

    if idx1 < n {
        // Load twiddle factor
        let twiddle_idx = k * (n / full_size)
        var w_re = twiddle_re[twiddle_idx]
        var w_im = twiddle_im[twiddle_idx]

        // For inverse FFT, use conjugate twiddle
        if inverse != 0 {
            w_im = 0.0 - w_im
        }

        // Load data
        let a_re = data_re[idx0]
        let a_im = data_im[idx0]
        let b_re = data_re[idx1]
        let b_im = data_im[idx1]

        // Twiddle multiply: t = w * b
        let t_re = w_re * b_re - w_im * b_im
        let t_im = w_re * b_im + w_im * b_re

        // Butterfly: a' = a + t, b' = a - t
        data_re[idx0] = a_re + t_re
        data_im[idx0] = a_im + t_im
        data_re[idx1] = a_re - t_re
        data_im[idx1] = a_im - t_im
    }
}

/// GPU kernel for final scaling (inverse FFT normalization)
kernel fn fft_scale_kernel(
    data_re: &![f64],
    data_im: &![f64],
    n: i32,
    scale: f64
) {
    let i = gpu.thread_id.x + gpu.block_id.x * gpu.block_dim.x

    if i < n {
        data_re[i] = data_re[i] * scale
        data_im[i] = data_im[i] * scale
    }
}

// ============================================================================
// GPU KERNEL: BATCH FFT
// ============================================================================

/// GPU kernel for batch FFT - process multiple signals in parallel
/// Each block handles one signal, threads within block do butterflies
kernel fn batch_fft_kernel(
    data_re: &![f64],      // [batch_size * n]
    data_im: &![f64],
    twiddle_re: &[f64],
    twiddle_im: &[f64],
    n: i32,
    n_stages: i32,
    inverse: i32
) {
    shared local_re: [f64; 512]
    shared local_im: [f64; 512]

    let batch_id = gpu.block_id.x
    let tid = gpu.thread_id.x
    let offset = batch_id * n

    // Load to shared memory with bit-reversal
    if tid < n {
        // Compute bit-reversed index
        var j: i32 = 0
        var val = tid
        var bits = n_stages
        var b: i32 = 0
        while b < bits {
            j = (j << 1) | (val & 1)
            val = val >> 1
            b = b + 1
        }

        local_re[tid] = data_re[offset + j]
        local_im[tid] = data_im[offset + j]
    }
    gpu.sync()

    // FFT stages
    var stage: i32 = 0
    while stage < n_stages {
        let half_size = 1 << stage
        let full_size = half_size << 1

        if tid < n / 2 {
            let pair_id = tid / half_size
            let k = tid % half_size

            let idx0 = pair_id * full_size + k
            let idx1 = idx0 + half_size

            // Twiddle factor
            let twiddle_idx = k * (n / full_size)
            var w_re = twiddle_re[twiddle_idx]
            var w_im = twiddle_im[twiddle_idx]

            if inverse != 0 {
                w_im = 0.0 - w_im
            }

            // Load
            let a_re = local_re[idx0]
            let a_im = local_im[idx0]
            let b_re = local_re[idx1]
            let b_im = local_im[idx1]

            // Twiddle multiply
            let t_re = w_re * b_re - w_im * b_im
            let t_im = w_re * b_im + w_im * b_re

            // Butterfly
            local_re[idx0] = a_re + t_re
            local_im[idx0] = a_im + t_im
            local_re[idx1] = a_re - t_re
            local_im[idx1] = a_im - t_im
        }
        gpu.sync()

        stage = stage + 1
    }

    // Write back with scaling for inverse
    if tid < n {
        if inverse != 0 {
            let scale = 1.0 / n as f64
            data_re[offset + tid] = local_re[tid] * scale
            data_im[offset + tid] = local_im[tid] * scale
        } else {
            data_re[offset + tid] = local_re[tid]
            data_im[offset + tid] = local_im[tid]
        }
    }
}

// ============================================================================
// GPU KERNEL: POWER SPECTRUM
// ============================================================================

/// Compute power spectral density from complex FFT output
kernel fn power_spectrum_kernel(
    fft_re: &[f64],
    fft_im: &[f64],
    power: &![f64],
    n_freqs: i32,      // n/2 + 1 for one-sided
    scale: f64         // 1/(fs * window_power)
) {
    let i = gpu.thread_id.x + gpu.block_id.x * gpu.block_dim.x

    if i < n_freqs {
        let mag_sq = fft_re[i] * fft_re[i] + fft_im[i] * fft_im[i]

        // One-sided PSD: multiply by 2 except DC and Nyquist
        let factor = if i == 0 || i == n_freqs - 1 { 1.0 } else { 2.0 }
        power[i] = factor * scale * mag_sq
    }
}

// ============================================================================
// GPU KERNEL: BANDPASS FILTER
// ============================================================================

/// Apply bandpass filter in frequency domain
kernel fn bandpass_filter_kernel(
    fft_re: &![f64],
    fft_im: &![f64],
    freq_bins: &[f64],
    n_freqs: i32,
    low_cutoff: f64,
    high_cutoff: f64,
    order: i32          // Butterworth filter order
) {
    let i = gpu.thread_id.x + gpu.block_id.x * gpu.block_dim.x

    if i < n_freqs {
        let f = freq_bins[i]

        // Butterworth bandpass response
        var gain = 1.0

        if f < low_cutoff {
            // High-pass attenuation
            if f > 0.001 {
                let ratio = f / low_cutoff
                var denom = 1.0
                var p: i32 = 0
                while p < order {
                    denom = denom * (1.0 + 1.0 / (ratio * ratio))
                    p = p + 1
                }
                gain = 1.0 / denom
            } else {
                gain = 0.0  // DC block
            }
        } else if f > high_cutoff {
            // Low-pass attenuation
            let ratio = f / high_cutoff
            var denom = 1.0
            var p: i32 = 0
            while p < order {
                denom = denom * (1.0 + ratio * ratio)
                p = p + 1
            }
            gain = 1.0 / denom
        }

        // Apply gain
        fft_re[i] = fft_re[i] * gain
        fft_im[i] = fft_im[i] * gain
    }
}

// ============================================================================
// GPU KERNEL: BATCH BANDPASS FOR VOXELS
// ============================================================================

/// Batch bandpass filter for multiple voxel timeseries
/// Optimized for fMRI preprocessing where we filter all voxels
kernel fn batch_bandpass_kernel(
    data_re: &![f64],      // [n_voxels * n_times]
    data_im: &![f64],
    twiddle_re: &[f64],
    twiddle_im: &[f64],
    freq_bins: &[f64],
    n_times: i32,
    n_stages: i32,
    low_cutoff: f64,
    high_cutoff: f64
) {
    shared local_re: [f64; 512]
    shared local_im: [f64; 512]

    let voxel_id = gpu.block_id.x
    let tid = gpu.thread_id.x
    let offset = voxel_id * n_times

    // Load with bit-reversal
    if tid < n_times {
        var j: i32 = 0
        var val = tid
        var b: i32 = 0
        while b < n_stages {
            j = (j << 1) | (val & 1)
            val = val >> 1
            b = b + 1
        }

        local_re[tid] = data_re[offset + j]
        local_im[tid] = 0.0  // Real input
    }
    gpu.sync()

    // Forward FFT
    var stage: i32 = 0
    while stage < n_stages {
        let half_size = 1 << stage
        let full_size = half_size << 1

        if tid < n_times / 2 {
            let pair_id = tid / half_size
            let k = tid % half_size
            let idx0 = pair_id * full_size + k
            let idx1 = idx0 + half_size

            let twiddle_idx = k * (n_times / full_size)
            let w_re = twiddle_re[twiddle_idx]
            let w_im = twiddle_im[twiddle_idx]

            let a_re = local_re[idx0]
            let a_im = local_im[idx0]
            let b_re = local_re[idx1]
            let b_im = local_im[idx1]

            let t_re = w_re * b_re - w_im * b_im
            let t_im = w_re * b_im + w_im * b_re

            local_re[idx0] = a_re + t_re
            local_im[idx0] = a_im + t_im
            local_re[idx1] = a_re - t_re
            local_im[idx1] = a_im - t_im
        }
        gpu.sync()
        stage = stage + 1
    }

    // Apply bandpass filter
    if tid < n_times {
        let f = freq_bins[tid]
        var gain = 1.0

        if f < low_cutoff && f > 0.001 {
            let ratio = f / low_cutoff
            gain = ratio * ratio / (1.0 + ratio * ratio)
        } else if f < 0.001 {
            gain = 0.0
        } else if f > high_cutoff {
            let ratio = high_cutoff / f
            gain = ratio * ratio / (1.0 + ratio * ratio)
        }

        local_re[tid] = local_re[tid] * gain
        local_im[tid] = local_im[tid] * gain
    }
    gpu.sync()

    // Inverse FFT (same structure, conjugate twiddles)
    stage = 0
    while stage < n_stages {
        let half_size = 1 << stage
        let full_size = half_size << 1

        if tid < n_times / 2 {
            let pair_id = tid / half_size
            let k = tid % half_size
            let idx0 = pair_id * full_size + k
            let idx1 = idx0 + half_size

            let twiddle_idx = k * (n_times / full_size)
            let w_re = twiddle_re[twiddle_idx]
            let w_im = 0.0 - twiddle_im[twiddle_idx]  // Conjugate

            let a_re = local_re[idx0]
            let a_im = local_im[idx0]
            let b_re = local_re[idx1]
            let b_im = local_im[idx1]

            let t_re = w_re * b_re - w_im * b_im
            let t_im = w_re * b_im + w_im * b_re

            local_re[idx0] = a_re + t_re
            local_im[idx0] = a_im + t_im
            local_re[idx1] = a_re - t_re
            local_im[idx1] = a_im - t_im
        }
        gpu.sync()
        stage = stage + 1
    }

    // Write back with inverse FFT scaling
    if tid < n_times {
        let scale = 1.0 / n_times as f64
        data_re[offset + tid] = local_re[tid] * scale
        data_im[offset + tid] = local_im[tid] * scale
    }
}

// ============================================================================
// HOST-SIDE FFT INTERFACE
// ============================================================================

/// FFT configuration
struct FFTConfig {
    n: i64,
    n_stages: i64,
    twiddle: TwiddleTable,
}

fn fft_config_new(n: i64) -> FFTConfig {
    FFTConfig {
        n: n,
        n_stages: log2_int(n),
        twiddle: twiddle_table_new(n),
    }
}

/// Compute FFT on CPU (reference implementation)
fn fft_cpu(
    input_re: &[f64; 2048],
    input_im: &[f64; 2048],
    output_re: &![f64; 2048],
    output_im: &![f64; 2048],
    n: i64,
    inverse: bool
) {
    let bits = log2_int(n)

    // Bit-reversal permutation
    var i: i64 = 0
    while i < n {
        let j = bit_reverse(i, bits)
        output_re[i as usize] = input_re[j as usize]
        output_im[i as usize] = input_im[j as usize]
        i = i + 1
    }

    // Cooley-Tukey butterflies
    var stage: i64 = 0
    while stage < bits {
        let half_size = 1 << stage
        let full_size = half_size << 1

        let angle_mult = if inverse {
            2.0 * pi() / (full_size as f64)
        } else {
            -2.0 * pi() / (full_size as f64)
        }

        var pair: i64 = 0
        while pair < n / full_size {
            var k: i64 = 0
            while k < half_size {
                let idx0 = (pair * full_size + k) as usize
                let idx1 = (pair * full_size + k + half_size) as usize

                let angle = angle_mult * (k as f64)
                let w_re = cos(angle)
                let w_im = sin(angle)

                let a_re = output_re[idx0]
                let a_im = output_im[idx0]
                let b_re = output_re[idx1]
                let b_im = output_im[idx1]

                let t_re = w_re * b_re - w_im * b_im
                let t_im = w_re * b_im + w_im * b_re

                output_re[idx0] = a_re + t_re
                output_im[idx0] = a_im + t_im
                output_re[idx1] = a_re - t_re
                output_im[idx1] = a_im - t_im

                k = k + 1
            }
            pair = pair + 1
        }
        stage = stage + 1
    }

    // Scale for inverse
    if inverse {
        let scale = 1.0 / (n as f64)
        i = 0
        while i < n {
            output_re[i as usize] = output_re[i as usize] * scale
            output_im[i as usize] = output_im[i as usize] * scale
            i = i + 1
        }
    }
}

/// Compute real FFT (for real-valued signals)
fn rfft_cpu(
    input: &[f64; 2048],
    output_re: &![f64; 1025],
    output_im: &![f64; 1025],
    n: i64
) {
    // Prepare complex input
    var work_re: [f64; 2048] = [0.0; 2048]
    var work_im: [f64; 2048] = [0.0; 2048]

    var i: i64 = 0
    while i < n {
        work_re[i as usize] = input[i as usize]
        work_im[i as usize] = 0.0
        i = i + 1
    }

    // Full FFT
    var fft_re: [f64; 2048] = [0.0; 2048]
    var fft_im: [f64; 2048] = [0.0; 2048]
    fft_cpu(&work_re, &work_im, &!fft_re, &!fft_im, n, false)

    // Extract positive frequencies
    let n_out = n / 2 + 1
    i = 0
    while i < n_out {
        output_re[i as usize] = fft_re[i as usize]
        output_im[i as usize] = fft_im[i as usize]
        i = i + 1
    }
}

/// Apply bandpass filter to signal (CPU reference)
fn bandpass_filter_cpu(
    signal: &[f64; 2048],
    output: &![f64; 2048],
    n: i64,
    fs: f64,
    low_hz: f64,
    high_hz: f64
) {
    // FFT
    var work_re: [f64; 2048] = [0.0; 2048]
    var work_im: [f64; 2048] = [0.0; 2048]

    var i: i64 = 0
    while i < n {
        work_re[i as usize] = signal[i as usize]
        work_im[i as usize] = 0.0
        i = i + 1
    }

    var fft_re: [f64; 2048] = [0.0; 2048]
    var fft_im: [f64; 2048] = [0.0; 2048]
    fft_cpu(&work_re, &work_im, &!fft_re, &!fft_im, n, false)

    // Apply filter
    let df = fs / (n as f64)
    i = 0
    while i < n {
        let f = if i <= n / 2 {
            (i as f64) * df
        } else {
            ((n - i) as f64) * df
        }

        var gain = 1.0
        if f < low_hz {
            if f > 0.001 {
                let ratio = f / low_hz
                gain = ratio * ratio / (1.0 + ratio * ratio)
            } else {
                gain = 0.0
            }
        } else if f > high_hz {
            let ratio = high_hz / f
            gain = ratio * ratio / (1.0 + ratio * ratio)
        }

        fft_re[i as usize] = fft_re[i as usize] * gain
        fft_im[i as usize] = fft_im[i as usize] * gain
        i = i + 1
    }

    // Inverse FFT
    var ifft_re: [f64; 2048] = [0.0; 2048]
    var ifft_im: [f64; 2048] = [0.0; 2048]
    fft_cpu(&fft_re, &fft_im, &!ifft_re, &!ifft_im, n, true)

    // Copy real part
    i = 0
    while i < n {
        output[i as usize] = ifft_re[i as usize]
        i = i + 1
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_fft_roundtrip() -> bool {
    // Test: FFT then IFFT should recover original signal
    var input_re: [f64; 2048] = [0.0; 2048]
    var input_im: [f64; 2048] = [0.0; 2048]
    let n: i64 = 256

    // Create test signal: cosine
    var i: i64 = 0
    while i < n {
        input_re[i as usize] = cos(2.0 * pi() * 10.0 * (i as f64) / 256.0)
        i = i + 1
    }

    // Forward FFT
    var fft_re: [f64; 2048] = [0.0; 2048]
    var fft_im: [f64; 2048] = [0.0; 2048]
    fft_cpu(&input_re, &input_im, &!fft_re, &!fft_im, n, false)

    // Inverse FFT
    var result_re: [f64; 2048] = [0.0; 2048]
    var result_im: [f64; 2048] = [0.0; 2048]
    fft_cpu(&fft_re, &fft_im, &!result_re, &!result_im, n, true)

    // Check reconstruction
    var max_error: f64 = 0.0
    i = 0
    while i < n {
        let err = fabs(result_re[i as usize] - input_re[i as usize])
        if err > max_error {
            max_error = err
        }
        i = i + 1
    }

    max_error < 1e-10
}

fn test_fft_parseval() -> bool {
    // Test: Energy in time domain equals energy in freq domain
    var input_re: [f64; 2048] = [0.0; 2048]
    var input_im: [f64; 2048] = [0.0; 2048]
    let n: i64 = 128

    // Create test signal
    var time_energy: f64 = 0.0
    var i: i64 = 0
    while i < n {
        input_re[i as usize] = cos(2.0 * pi() * 5.0 * (i as f64) / 128.0)
        time_energy = time_energy + input_re[i as usize] * input_re[i as usize]
        i = i + 1
    }

    // FFT
    var fft_re: [f64; 2048] = [0.0; 2048]
    var fft_im: [f64; 2048] = [0.0; 2048]
    fft_cpu(&input_re, &input_im, &!fft_re, &!fft_im, n, false)

    // Freq energy (divided by N due to FFT scaling)
    var freq_energy: f64 = 0.0
    i = 0
    while i < n {
        freq_energy = freq_energy + fft_re[i as usize] * fft_re[i as usize] +
                      fft_im[i as usize] * fft_im[i as usize]
        i = i + 1
    }
    freq_energy = freq_energy / (n as f64)

    fabs(time_energy - freq_energy) < 1e-6
}

fn test_bandpass() -> bool {
    // Test: Bandpass should attenuate out-of-band frequencies
    var signal: [f64; 2048] = [0.0; 2048]
    let n: i64 = 256
    let fs = 100.0

    // Signal: 5 Hz (in-band) + 40 Hz (out-of-band for 0.01-0.1 Hz filter)
    var i: i64 = 0
    while i < n {
        let t = (i as f64) / fs
        signal[i as usize] = cos(2.0 * pi() * 0.05 * t) + cos(2.0 * pi() * 40.0 * t)
        i = i + 1
    }

    // Apply bandpass
    var filtered: [f64; 2048] = [0.0; 2048]
    bandpass_filter_cpu(&signal, &!filtered, n, fs, 0.01, 0.1)

    // Check that high frequency is attenuated
    // (simplified check: variance should decrease)
    var var_orig: f64 = 0.0
    var var_filt: f64 = 0.0
    i = 0
    while i < n {
        var_orig = var_orig + signal[i as usize] * signal[i as usize]
        var_filt = var_filt + filtered[i as usize] * filtered[i as usize]
        i = i + 1
    }

    // Filtered should have less energy (40 Hz removed)
    var_filt < var_orig * 0.9
}

fn main() -> i32 {
    print("Testing gpu::fft module...\n")

    if !test_fft_roundtrip() {
        print("FAIL: fft_roundtrip\n")
        return 1
    }
    print("PASS: fft_roundtrip\n")

    if !test_fft_parseval() {
        print("FAIL: fft_parseval\n")
        return 2
    }
    print("PASS: fft_parseval\n")

    if !test_bandpass() {
        print("FAIL: bandpass\n")
        return 3
    }
    print("PASS: bandpass\n")

    print("All gpu::fft tests PASSED\n")
    0
}

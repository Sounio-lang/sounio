// signal::epoch — Event-Locked Segmentation
//
// Extract time-locked segments (epochs) around events for ERP/ERF analysis.
// Handles baseline correction, artifact rejection, and trial averaging.
//
// Key Features:
// - Flexible event markers (numeric codes)
// - Pre/post-stimulus windows
// - Baseline correction
// - Artifact rejection (amplitude threshold)
// - Trial averaging with uncertainty (SEM)
//
// References:
// - Luck (2014): "An Introduction to the Event-Related Potential Technique"
// - Cohen (2014): "Analyzing Neural Time Series Data"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
}

// ============================================================================
// EVENT MARKERS
// ============================================================================

// Maximum constants
fn MAX_EVENTS() -> i64 { 100 }
fn MAX_EPOCH_LEN() -> i64 { 512 }
fn MAX_EPOCHS() -> i64 { 100 }

// Event marker
struct Event {
    idx: i64,           // Index in continuous signal
    code: i32,          // Event code (e.g., stimulus type)
}

fn event_new(idx: i64, code: i32) -> Event {
    Event {
        idx: idx,
        code: code,
    }
}

// Event list
struct EventList {
    indices: [i64; 100],
    codes: [i32; 100],
    n_events: i64,
}

fn event_list_new() -> EventList {
    EventList {
        indices: [0; 100],
        codes: [0; 100],
        n_events: 0,
    }
}

fn event_list_add(list: EventList, idx: i64, code: i32) -> EventList {
    var new_list = list
    if list.n_events < MAX_EVENTS() {
        new_list.indices[list.n_events as usize] = idx
        new_list.codes[list.n_events as usize] = code
        new_list.n_events = list.n_events + 1
    }
    return new_list
}

// ============================================================================
// EPOCH CONFIGURATION
// ============================================================================

// Epoch configuration
struct EpochConfig {
    pre_samp: i64,       // Samples before event (positive number)
    post_samp: i64,      // Samples after event (positive number)
    baseline_start: i64,    // Baseline start (samples from epoch start)
    baseline_end: i64,      // Baseline end (samples from epoch start)
    reject_threshold: f64,  // Amplitude threshold for rejection (0 = no rejection)
}

fn epoch_config_default(fs: f64) -> EpochConfig {
    // Default: -200ms to 800ms, baseline -200 to 0ms
    let pre = (0.2 * fs) as i64
    let post = (0.8 * fs) as i64
    EpochConfig {
        pre_samp: pre,
        post_samp: post,
        baseline_start: 0,
        baseline_end: pre,
        reject_threshold: 100.0,  // µV for EEG
    }
}

fn epoch_config_custom(pre_ms: f64, post_ms: f64, fs: f64) -> EpochConfig {
    let pre = (pre_ms * fs / 1000.0) as i64
    let post = (post_ms * fs / 1000.0) as i64
    EpochConfig {
        pre_samp: pre,
        post_samp: post,
        baseline_start: 0,
        baseline_end: pre,
        reject_threshold: 100.0,
    }
}

// ============================================================================
// SINGLE EPOCH
// ============================================================================

// Single epoch data
struct Epoch {
    data: [f64; 512],       // Time series data
    n_samp: i64,
    event_code: i32,
    event_idx: i64,      // Original sample in continuous signal
    rejected: bool,
    reject_reason: i32,     // 0=ok, 1=amplitude, 2=boundary
}

fn epoch_new() -> Epoch {
    Epoch {
        data: [0.0; 512],
        n_samp: 0,
        event_code: 0,
        event_idx: 0,
        rejected: false,
        reject_reason: 0,
    }
}

// ============================================================================
// EPOCHS COLLECTION
// ============================================================================

// Epochs collection
struct Epochs {
    data: [[f64; 512]; 100],    // All epoch data
    n_samp: [i64; 100],
    event_codes: [i32; 100],
    rejected: [bool; 100],
    n_epochs: i64,
    n_accepted: i64,
    epoch_len: i64,
    fs: f64,
    t_pre: f64,
    t_post: f64,
}

fn epochs_new() -> Epochs {
    Epochs {
        data: [[0.0; 512]; 100],
        n_samp: [0; 100],
        event_codes: [0; 100],
        rejected: [false; 100],
        n_epochs: 0,
        n_accepted: 0,
        epoch_len: 0,
        fs: 1.0,
        t_pre: 0.0,
        t_post: 0.0,
    }
}

// Extract epochs from continuous signal
fn extract_epochs(
    signal: [f64; 2048],
    signal_len: i64,
    events: EventList,
    event_code: i32,        // Which event code to extract (use -1 for all)
    config: EpochConfig,
    fs: f64
) -> Epochs {
    var result = epochs_new()
    result.fs = fs
    result.epoch_len = config.pre_samp + config.post_samp
    result.t_pre = -(config.pre_samp as f64) / fs
    result.t_post = (config.post_samp as f64) / fs

    var i: i64 = 0
    while i < events.n_events && result.n_epochs < MAX_EPOCHS() {
        let ev_idx = events.indices[i as usize]
        let ev_code = events.codes[i as usize]

        // Check event code filter
        if event_code >= 0 && ev_code != event_code {
            i = i + 1
            continue
        }

        let start = ev_idx - config.pre_samp
        let end = ev_idx + config.post_samp

        // Check boundaries
        if start < 0 || end > signal_len {
            result.rejected[result.n_epochs as usize] = true
            result.event_codes[result.n_epochs as usize] = ev_code
            result.n_epochs = result.n_epochs + 1
            i = i + 1
            continue
        }

        // Extract data
        var max_amp = 0.0
        var j: i64 = 0
        while j < result.epoch_len {
            let val = signal[(start + j) as usize]
            result.data[result.n_epochs as usize][j as usize] = val

            // Track amplitude
            let abs_val = if val < 0.0 { -val } else { val }
            if abs_val > max_amp {
                max_amp = abs_val
            }
            j = j + 1
        }

        result.n_samp[result.n_epochs as usize] = result.epoch_len
        result.event_codes[result.n_epochs as usize] = ev_code

        // Check rejection criteria
        if config.reject_threshold > 0.0 && max_amp > config.reject_threshold {
            result.rejected[result.n_epochs as usize] = true
        } else {
            result.rejected[result.n_epochs as usize] = false
            result.n_accepted = result.n_accepted + 1
        }

        result.n_epochs = result.n_epochs + 1
        i = i + 1
    }

    return result
}

// ============================================================================
// BASELINE CORRECTION
// ============================================================================

// Apply baseline correction to all epochs
fn baseline_correct(epochs: Epochs, baseline_start: i64, baseline_end: i64) -> Epochs {
    var result = epochs

    var i: i64 = 0
    while i < epochs.n_epochs {
        if epochs.rejected[i as usize] {
            i = i + 1
            continue
        }

        // Compute baseline mean
        var sum = 0.0
        var count: i64 = 0
        var j = baseline_start
        while j < baseline_end && j < epochs.epoch_len {
            sum = sum + epochs.data[i as usize][j as usize]
            count = count + 1
            j = j + 1
        }

        if count > 0 {
            let baseline = sum / (count as f64)

            // Subtract baseline
            j = 0
            while j < epochs.epoch_len {
                result.data[i as usize][j as usize] = epochs.data[i as usize][j as usize] - baseline
                j = j + 1
            }
        }

        i = i + 1
    }

    return result
}

// ============================================================================
// EPOCH AVERAGING (ERP/ERF)
// ============================================================================

// ERP result with uncertainty
struct ERPResult {
    mean: [f64; 512],           // Grand average
    std: [f64; 512],            // Standard deviation
    sem: [f64; 512],            // Standard error of mean
    ci_lower: [f64; 512],       // 95% CI lower bound
    ci_upper: [f64; 512],       // 95% CI upper bound
    n_samp: i64,
    n_trials: i64,
    fs: f64,
    t_pre: f64,
    t_post: f64,
}

fn erp_result_new() -> ERPResult {
    ERPResult {
        mean: [0.0; 512],
        std: [0.0; 512],
        sem: [0.0; 512],
        ci_lower: [0.0; 512],
        ci_upper: [0.0; 512],
        n_samp: 0,
        n_trials: 0,
        fs: 1.0,
        t_pre: 0.0,
        t_post: 0.0,
    }
}

// Compute average ERP with uncertainty
fn compute_erp(epochs: Epochs) -> ERPResult {
    var result = erp_result_new()
    result.n_samp = epochs.epoch_len
    result.fs = epochs.fs
    result.t_pre = epochs.t_pre
    result.t_post = epochs.t_post

    if epochs.n_accepted == 0 {
        return result
    }

    // First pass: compute mean
    var i: i64 = 0
    while i < epochs.n_epochs {
        if epochs.rejected[i as usize] {
            i = i + 1
            continue
        }

        var j: i64 = 0
        while j < epochs.epoch_len {
            result.mean[j as usize] = result.mean[j as usize] +
                epochs.data[i as usize][j as usize]
            j = j + 1
        }
        result.n_trials = result.n_trials + 1
        i = i + 1
    }

    let n = result.n_trials as f64
    var j: i64 = 0
    while j < epochs.epoch_len {
        result.mean[j as usize] = result.mean[j as usize] / n
        j = j + 1
    }

    // Second pass: compute variance
    i = 0
    while i < epochs.n_epochs {
        if epochs.rejected[i as usize] {
            i = i + 1
            continue
        }

        j = 0
        while j < epochs.epoch_len {
            let diff = epochs.data[i as usize][j as usize] - result.mean[j as usize]
            result.std[j as usize] = result.std[j as usize] + diff * diff
            j = j + 1
        }
        i = i + 1
    }

    // Finalize std, sem, CI
    j = 0
    while j < epochs.epoch_len {
        if n > 1.0 {
            result.std[j as usize] = sqrt(result.std[j as usize] / (n - 1.0))
            result.sem[j as usize] = result.std[j as usize] / sqrt(n)

            // 95% CI (t-distribution approximation, using 1.96 for large n)
            let t_crit = if n > 30.0 { 1.96 } else { 2.0 + 4.0 / n }
            result.ci_lower[j as usize] = result.mean[j as usize] - t_crit * result.sem[j as usize]
            result.ci_upper[j as usize] = result.mean[j as usize] + t_crit * result.sem[j as usize]
        }
        j = j + 1
    }

    return result
}

// ============================================================================
// ERP PEAK DETECTION
// ============================================================================

// ERP peak/component measurement
struct ERPComponent {
    latency: f64,           // Peak latency (seconds)
    amplitude: f64,         // Peak amplitude
    amplitude_sem: f64,     // SEM of amplitude
    window_start: f64,      // Search window start
    window_end: f64,        // Search window end
}

fn erp_component_new() -> ERPComponent {
    ERPComponent {
        latency: 0.0,
        amplitude: 0.0,
        amplitude_sem: 0.0,
        window_start: 0.0,
        window_end: 0.0,
    }
}

// Find positive peak in time window
fn find_positive_peak(erp: ERPResult, t_start: f64, t_end: f64) -> ERPComponent {
    var result = erp_component_new()
    result.window_start = t_start
    result.window_end = t_end

    let dt = 1.0 / erp.fs
    var max_amp = -1e308
    var max_idx: i64 = 0

    var i: i64 = 0
    while i < erp.n_samp {
        let t = erp.t_pre + (i as f64) * dt
        if t >= t_start && t <= t_end {
            if erp.mean[i as usize] > max_amp {
                max_amp = erp.mean[i as usize]
                max_idx = i
            }
        }
        i = i + 1
    }

    result.latency = erp.t_pre + (max_idx as f64) * dt
    result.amplitude = max_amp
    result.amplitude_sem = erp.sem[max_idx as usize]

    return result
}

// Find negative peak in time window
fn find_negative_peak(erp: ERPResult, t_start: f64, t_end: f64) -> ERPComponent {
    var result = erp_component_new()
    result.window_start = t_start
    result.window_end = t_end

    let dt = 1.0 / erp.fs
    var min_amp = 1e308
    var min_idx: i64 = 0

    var i: i64 = 0
    while i < erp.n_samp {
        let t = erp.t_pre + (i as f64) * dt
        if t >= t_start && t <= t_end {
            if erp.mean[i as usize] < min_amp {
                min_amp = erp.mean[i as usize]
                min_idx = i
            }
        }
        i = i + 1
    }

    result.latency = erp.t_pre + (min_idx as f64) * dt
    result.amplitude = min_amp
    result.amplitude_sem = erp.sem[min_idx as usize]

    return result
}

// Standard ERP components
fn find_N100(erp: ERPResult) -> ERPComponent { find_negative_peak(erp, 0.080, 0.150) }
fn find_P200(erp: ERPResult) -> ERPComponent { find_positive_peak(erp, 0.150, 0.250) }
fn find_N200(erp: ERPResult) -> ERPComponent { find_negative_peak(erp, 0.180, 0.300) }
fn find_P300(erp: ERPResult) -> ERPComponent { find_positive_peak(erp, 0.250, 0.500) }
fn find_N400(erp: ERPResult) -> ERPComponent { find_negative_peak(erp, 0.300, 0.500) }

// ============================================================================
// MEAN AMPLITUDE
// ============================================================================

// Compute mean amplitude in time window
fn mean_amplitude(erp: ERPResult, t_start: f64, t_end: f64) -> f64 {
    let dt = 1.0 / erp.fs
    var sum = 0.0
    var count: i64 = 0

    var i: i64 = 0
    while i < erp.n_samp {
        let t = erp.t_pre + (i as f64) * dt
        if t >= t_start && t <= t_end {
            sum = sum + erp.mean[i as usize]
            count = count + 1
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

fn test_event_list() -> bool {
    var events = event_list_new()
    events = event_list_add(events, 100, 1)
    events = event_list_add(events, 500, 2)
    events = event_list_add(events, 900, 1)

    return events.n_events == 3
}

fn test_epoch_extraction() -> bool {
    // Generate synthetic ERP signal
    let fs = 256.0
    let signal_len: i64 = 2048
    var signal: [f64; 2048] = [0.0; 2048]

    // Add events and responses
    var events = event_list_new()
    events = event_list_add(events, 256, 1)   // Event at 1 second
    events = event_list_add(events, 768, 1)   // Event at 3 seconds

    // Add synthetic ERP (P300-like at 300ms after each event)
    var ev: i64 = 0
    while ev < 2 {
        let ev_idx = if ev == 0 { 256 } else { 768 }
        var j: i64 = 0
        while j < 256 {
            let t = (j as f64) / fs
            // P300: Gaussian at 300ms, amplitude 10
            let p300 = 10.0 * exp(-0.5 * pow((t - 0.3) / 0.05, 2.0))
            let idx = ev_idx + j
            if idx < signal_len {
                signal[idx as usize] = signal[idx as usize] + p300
            }
            j = j + 1
        }
        ev = ev + 1
    }

    // Extract epochs
    var config = epoch_config_default(fs)
    config.reject_threshold = 0.0  // Disable rejection

    let epochs = extract_epochs(signal, signal_len, events, 1, config, fs)

    // Check number of epochs
    if epochs.n_epochs != 2 {
        return false
    }
    if epochs.n_accepted != 2 {
        return false
    }

    return true
}

fn test_erp_computation() -> bool {
    // Generate synthetic epochs
    let fs = 256.0
    var epochs = epochs_new()
    epochs.fs = fs
    epochs.epoch_len = 256  // 1 second
    epochs.t_pre = -0.2
    epochs.t_post = 0.8

    // Create 10 identical epochs (no noise for testing)
    var i: i64 = 0
    while i < 10 {
        var j: i64 = 0
        while j < 256 {
            let t = -0.2 + (j as f64) / fs
            // P300-like component at 300ms
            epochs.data[i as usize][j as usize] = 10.0 * exp(-0.5 * pow((t - 0.3) / 0.05, 2.0))
            j = j + 1
        }
        epochs.n_samp[i as usize] = 256
        epochs.rejected[i as usize] = false
        epochs.n_epochs = epochs.n_epochs + 1
        epochs.n_accepted = epochs.n_accepted + 1
        i = i + 1
    }

    // Compute ERP
    let erp = compute_erp(epochs)

    // Check number of trials
    if erp.n_trials != 10 {
        return false
    }

    // Find P300
    let p300 = find_P300(erp)

    // Latency should be near 300ms
    if abs_f64(p300.latency - 0.3) > 0.05 {
        return false
    }

    // Amplitude should be near 10
    if abs_f64(p300.amplitude - 10.0) > 1.0 {
        return false
    }

    return true
}

fn main() -> i32 {
    print("Testing signal::epoch module...\n")

    if !test_event_list() {
        print("FAIL: event_list\n")
        return 1
    }
    print("PASS: event_list\n")

    if !test_epoch_extraction() {
        print("FAIL: epoch_extraction\n")
        return 2
    }
    print("PASS: epoch_extraction\n")

    if !test_erp_computation() {
        print("FAIL: erp_computation\n")
        return 3
    }
    print("PASS: erp_computation\n")

    print("All signal::epoch tests PASSED\n")
    0
}

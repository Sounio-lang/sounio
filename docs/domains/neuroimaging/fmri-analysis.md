# fMRI Analysis in Sounio

This guide covers functional magnetic resonance imaging (fMRI) data handling and preprocessing in Sounio, with emphasis on epistemic uncertainty tracking throughout the analysis pipeline.

## Overview

The `fmri` module provides tools for:

- Loading and manipulating NIfTI volumetric data
- Preprocessing pipelines following fMRIPrep best practices
- Quality control metrics (tSNR, DVARS, framewise displacement)
- Motion artifact handling with scrubbing
- GLM-ready time series preparation

## NIfTI Data Handling

### The NiftiImage Structure

```sio
use fmri::nifti::{NiftiImage, nifti_create, get_voxel, voxel_to_world}

/// NIfTI image with spatial metadata
struct NiftiImage {
    nx: i64,              // X dimension (voxels)
    ny: i64,              // Y dimension
    nz: i64,              // Z dimension
    nvox: i64,            // Total voxel count
    dx: f64,              // Voxel size X (mm)
    dy: f64,              // Voxel size Y (mm)
    dz: f64,              // Voxel size Z (mm)
    tr: f64,              // Repetition time (seconds)
    xform: [[f64; 4]; 4], // Affine transformation matrix
    data: [f64; 4096],    // Voxel data (volumetric)
}
```

### Creating and Accessing NIfTI Data

```sio
use fmri::nifti::{nifti_create, get_voxel, voxel_idx, voxel_to_world}

fn example_nifti_operations() {
    // Create a 10x10x10 image
    var img = nifti_create(10, 10, 10)

    // Access voxel at coordinates (5, 5, 5)
    let value = get_voxel(&img, 5, 5, 5)

    // Convert voxel indices to world (MNI) coordinates
    let world = voxel_to_world(&img, 5.0, 5.0, 5.0)
    let mni_x = world.0
    let mni_y = world.1
    let mni_z = world.2

    // Direct index calculation for efficiency
    let idx = voxel_idx(img.nx, img.ny, 5, 5, 5)
}
```

## Preprocessing Pipeline

The preprocessing pipeline follows established fMRIPrep best practices (Esteban et al., 2019).

### Motion Parameters

Motion is characterized by 6 degrees of freedom (3 translations, 3 rotations):

```sio
use fmri::preprocess::{MotionParams, framewise_displacement}
use fmri::pipeline::{MotionParams6, MotionTimeseries, calculate_fd_timeseries}

/// 6 DOF motion parameters
struct MotionParams6 {
    tx: f64,    // Translation X (mm)
    ty: f64,    // Translation Y (mm)
    tz: f64,    // Translation Z (mm)
    rx: f64,    // Rotation pitch (radians)
    ry: f64,    // Rotation roll (radians)
    rz: f64,    // Rotation yaw (radians)
}

// Compute framewise displacement between consecutive volumes
// FD = |dtx| + |dty| + |dtz| + r*(|drx| + |dry| + |drz|)
// where r = 50mm (approximate head radius)
fn compute_motion_quality() {
    let prev = MotionParams6 {
        tx: 0.0, ty: 0.0, tz: 0.0,
        rx: 0.0, ry: 0.0, rz: 0.0
    }
    let curr = MotionParams6 {
        tx: 0.5, ty: 0.3, tz: 0.1,
        rx: 0.002, ry: 0.001, rz: 0.003
    }

    let fd = framewise_displacement(prev, curr, 50.0)
    print("Framewise displacement: ", fd, " mm\n")
}
```

### Temporal Filtering

resting-state fMRI requires bandpass filtering to isolate BOLD signal fluctuations (typically 0.01-0.1 Hz):

```sio
use fmri::preprocess::{BandpassConfig, bandpass_config_rsfmri}
use fmri::pipeline::{highpass_dct}

/// Bandpass filter configuration
struct BandpassConfig {
    low_cutoff: f64,    // High-pass cutoff (Hz), e.g., 0.01
    high_cutoff: f64,   // Low-pass cutoff (Hz), e.g., 0.1
    tr: f64,            // Repetition time (seconds)
}

fn configure_bandpass() {
    // Standard resting-state configuration
    let config = bandpass_config_rsfmri(2.0)  // TR = 2 seconds

    // Custom configuration
    let custom = BandpassConfig {
        low_cutoff: 0.008,   // More aggressive high-pass
        high_cutoff: 0.15,   // Include higher frequencies
        tr: 1.5,
    }
}

// Apply DCT-based high-pass filter
fn apply_highpass(timeseries: [f64; 500], n: i64, cutoff_hz: f64, tr: f64) {
    let filtered = highpass_dct(timeseries, n, cutoff_hz, tr)
    // filtered now contains high-pass filtered data
}
```

### Detrending

Remove scanner drift and polynomial trends:

```sio
use fmri::preprocess::{detrend_linear, demean, zscore}

fn preprocess_timeseries(raw_data: [f64; 200], n: i64) {
    // Step 1: Remove linear trend
    let detrended = detrend_linear(raw_data, n)

    // Step 2: Demean (center at zero)
    let demeaned = demean(detrended, n)

    // Step 3: Z-score normalize (mean=0, std=1)
    let normalized = zscore(demeaned, n)
}
```

### Nuisance Regression

Configure confound regression strategy:

```sio
use fmri::preprocess::{NuisanceConfig, nuisance_config_default, nuisance_config_aggressive}

/// Nuisance regression configuration
struct NuisanceConfig {
    use_motion: bool,           // 6 motion parameters
    use_motion_deriv: bool,     // Temporal derivatives of motion
    use_wm_csf: bool,           // White matter and CSF signals
    use_global_signal: bool,    // Global signal regression (controversial)
}

fn setup_nuisance_regression() {
    // Conservative: motion + WM/CSF, no global signal
    let config = nuisance_config_default()

    // Aggressive: includes global signal regression
    let aggressive = nuisance_config_aggressive()
}
```

### Gaussian Smoothing

Spatial smoothing to improve SNR:

```sio
use fmri::preprocess::{SmoothKernel, gaussian_kern_new}

/// Create Gaussian smoothing kernel
fn create_smoother(fwhm_mm: f64, voxel_size: f64) {
    // FWHM = 2.355 * sigma
    let kernel = gaussian_kern_new(6.0, 2.0)  // 6mm FWHM, 2mm voxels

    // kernel.weights contains 1D Gaussian weights
    // Apply separably in each dimension
}
```

## Quality Control

### Temporal Signal-to-Noise Ratio (tSNR)

tSNR measures signal stability over time:

```sio
use fmri::pipeline::{VoxelQuality, calculate_tsnr}

/// Quality metrics for a voxel timeseries
struct VoxelQuality {
    mean: f64,      // Mean signal intensity
    std: f64,       // Standard deviation
    tsnr: f64,      // Temporal SNR = mean / std
}

fn assess_data_quality(timeseries: [f64; 500], n: i64) {
    let quality = calculate_tsnr(timeseries, n)

    // Typical tSNR values:
    // - Grey matter: 50-100
    // - White matter: 100-200
    // - CSF: 10-30 (high variability)

    if quality.tsnr < 30.0 {
        print("Warning: Low tSNR may indicate data quality issues\n")
    }
}
```

### DVARS (Temporal Derivative of RMS Variance)

DVARS measures volume-to-volume intensity changes:

```sio
use fmri::pipeline::{calculate_dvars}

fn monitor_dvars(vol1: [f64; 500], vol2: [f64; 500], n_voxels: i64) {
    let dvars = calculate_dvars(vol1, vol2, n_voxels)

    // High DVARS indicates sudden intensity changes (motion, artifacts)
    // Typical threshold: 1.5 (standardized DVARS)
}
```

### Scrubbing / Censoring

Remove high-motion volumes:

```sio
use fmri::pipeline::{ScrubResult, identify_scrub_volumes}

/// Scrubbing result
struct ScrubResult {
    good_volumes: [bool; 500],  // Which volumes to keep
    n_good: i64,                // Number of good volumes
    n_scrubbed: i64,            // Number of removed volumes
}

fn apply_scrubbing(
    fd: [f64; 500],
    dvars: [f64; 500],
    n_volumes: i64
) {
    let result = identify_scrub_volumes(
        fd, dvars, n_volumes,
        0.5,    // FD threshold (mm)
        1.5,    // DVARS threshold
        1,      // Scrub 1 volume before bad volume
        2       // Scrub 2 volumes after bad volume
    )

    print("Good volumes: ", result.n_good, "\n")
    print("Scrubbed: ", result.n_scrubbed, "\n")

    // Check if enough data remains
    if result.n_good < 100 {
        print("Warning: Insufficient data after scrubbing\n")
    }
}
```

## Complete Pipeline Configuration

```sio
use fmri::pipeline::{PipelineConfig, pipeline_config_default, pipeline_config_strict}

/// Complete preprocessing pipeline configuration
struct PipelineConfig {
    tr: f64,                    // Repetition time (seconds)
    high_pass_hz: f64,          // High-pass cutoff (Hz)
    fd_threshold: f64,          // FD threshold for scrubbing (mm)
    dvars_threshold: f64,       // DVARS threshold
    scrub_before: i64,          // Volumes to remove before bad
    scrub_after: i64,           // Volumes to remove after bad
    min_volumes: i64,           // Minimum volumes after scrubbing
    max_fd_mean: f64,           // Max mean FD for inclusion
    head_radius: f64,           // Head radius for FD (mm)
}

fn setup_pipeline() {
    // Standard configuration
    let config = pipeline_config_default()
    // tr: 2.0, high_pass_hz: 0.01, fd_threshold: 0.5
    // min_volumes: 100, max_fd_mean: 0.3

    // Strict motion criteria
    let strict = pipeline_config_strict()
    // fd_threshold: 0.3, max_fd_mean: 0.2
}
```

## Quality Check Results

```sio
use fmri::pipeline::{QualityCheck, run_quality_checks}

/// Overall quality assessment
struct QualityCheck {
    passed: bool,               // Did the run pass QC?
    fd_mean: f64,               // Mean framewise displacement
    fd_max: f64,                // Maximum FD
    n_outliers: i64,            // Volumes exceeding threshold
    n_volumes_used: i64,        // Volumes after scrubbing
    n_volumes_scrubbed: i64,    // Removed volumes
    mean_tsnr: f64,             // Average tSNR
    reason: i32,                // 0=passed, 1=too few volumes, 2=fd too high
}

fn run_qc(motion: MotionTimeseries, scrub: ScrubResult, mean_tsnr: f64) {
    let config = pipeline_config_default()
    let qc = run_quality_checks(motion, scrub, mean_tsnr, config)

    if qc.passed {
        print("QC PASSED\n")
    } else {
        match qc.reason {
            1 => print("QC FAILED: Too few volumes after scrubbing\n"),
            2 => print("QC FAILED: Mean FD exceeds threshold\n"),
            _ => print("QC FAILED: Unknown reason\n"),
        }
    }
}
```

## Statistical Maps with Uncertainty

Every statistical output carries uncertainty information:

```sio
use fmri::connectivity_epistemic::{ConnectivityEdge, inflate_uncertainty_for_motion}

// Adjust uncertainty based on data quality
fn motion_aware_connectivity(
    edge: ConnectivityEdge,
    mean_fd: f64,
    scrub_fraction: f64
) -> f64 {
    // Inflate uncertainty for motion-affected data
    // 2x per mm mean FD + 50% max from scrubbing
    let adjusted_uncertainty = inflate_uncertainty_for_motion(
        edge.uncertainty,
        mean_fd,
        scrub_fraction
    )

    adjusted_uncertainty
}
```

## Example: Complete fMRI Preprocessing

```sio
use fmri::nifti::{NiftiImage, nifti_create}
use fmri::preprocess::{detrend_linear, zscore, bandpass_config_rsfmri}
use fmri::pipeline::{
    PipelineConfig, pipeline_config_default,
    MotionParams6, calculate_fd_timeseries,
    calculate_tsnr, identify_scrub_volumes,
    run_quality_checks
}

fn preprocess_fmri_run(
    timeseries: [[f64; 200]; 100],  // 100 ROIs x 200 timepoints
    motion: [MotionParams6; 200],
    n_volumes: i64
) {
    // 1. Configure pipeline
    let config = pipeline_config_default()

    // 2. Compute motion timeseries
    let motion_ts = calculate_fd_timeseries(
        motion, n_volumes, config.head_radius, config.fd_threshold
    )

    print("Mean FD: ", motion_ts.fd_mean, " mm\n")
    print("Max FD: ", motion_ts.fd_max, " mm\n")
    print("Outlier volumes: ", motion_ts.n_outliers, "\n")

    // 3. Compute DVARS (simplified - would need full volume data)
    var dvars: [f64; 500] = [0.0; 500]

    // 4. Identify volumes to scrub
    let scrub = identify_scrub_volumes(
        motion_ts.fd, dvars, n_volumes,
        config.fd_threshold, config.dvars_threshold,
        config.scrub_before, config.scrub_after
    )

    // 5. Preprocess each ROI timeseries
    var i: i64 = 0
    while i < 100 {
        // Detrend
        var ts: [f64; 200] = timeseries[i as usize]
        let detrended = detrend_linear(ts, n_volumes)

        // Z-score normalize
        let normalized = zscore(detrended, n_volumes)

        // Store preprocessed data
        // timeseries[i as usize] = normalized

        i = i + 1
    }

    // 6. Calculate mean tSNR across ROIs
    var mean_tsnr = 0.0
    i = 0
    while i < 100 {
        var ts_500: [f64; 500] = [0.0; 500]
        var j: i64 = 0
        while j < n_volumes {
            ts_500[j as usize] = timeseries[i as usize][j as usize]
            j = j + 1
        }
        let qv = calculate_tsnr(ts_500, n_volumes)
        mean_tsnr = mean_tsnr + qv.tsnr
        i = i + 1
    }
    mean_tsnr = mean_tsnr / 100.0

    // 7. Run quality checks
    let qc = run_quality_checks(motion_ts, scrub, mean_tsnr, config)

    if qc.passed {
        print("Preprocessing complete. Data passes QC.\n")
        print("Volumes used: ", qc.n_volumes_used, "\n")
        print("Mean tSNR: ", qc.mean_tsnr, "\n")
    } else {
        print("WARNING: Data fails QC criteria.\n")
    }
}
```

## Best Practices

### Motion Thresholds

| Population | FD Threshold | Max Mean FD |
|------------|--------------|-------------|
| Adults (healthy) | 0.5 mm | 0.3 mm |
| Adults (strict) | 0.3 mm | 0.2 mm |
| Children | 0.5 mm | 0.4 mm |
| Clinical | 0.5 mm | 0.4 mm |

### Minimum Data Requirements

- **Resting-state**: Minimum 5 minutes of usable data (150+ volumes at TR=2s)
- **Task fMRI**: Minimum 4 repetitions per condition after scrubbing
- **Connectivity**: Minimum 100 timepoints for stable correlation estimates

### Preprocessing Order

1. Slice timing correction (if needed)
2. Motion correction (realignment)
3. Distortion correction (field maps)
4. Normalization to MNI space
5. Spatial smoothing (4-6mm FWHM)
6. Temporal filtering (0.01-0.1 Hz for rs-fMRI)
7. Nuisance regression
8. Scrubbing/censoring

## References

1. Power JD, et al. (2012). "Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion." *NeuroImage* 59(3):2142-54.

2. Power JD, et al. (2014). "Methods to detect, characterize, and remove motion artifact in resting state fMRI." *NeuroImage* 84:320-41.

3. Esteban O, et al. (2019). "fMRIPrep: a robust preprocessing pipeline for functional MRI." *Nat Methods* 16(1):111-116.

4. Ciric R, et al. (2017). "Benchmarking of participant-level confound regression strategies for the control of motion artifact in studies of functional connectivity." *NeuroImage* 154:174-187.

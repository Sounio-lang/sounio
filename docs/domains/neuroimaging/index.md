# Neuroimaging with Sounio

Sounio provides comprehensive support for neuroimaging analysis with first-class epistemic uncertainty tracking. This documentation covers fMRI analysis, brain connectivity, and atlas-based parcellation.

## Why Sounio for Neuroimaging?

Neuroimaging research inherently deals with uncertainty at every stage:

- **Measurement uncertainty**: Scanner noise, physiological artifacts, motion
- **Preprocessing uncertainty**: Spatial normalization accuracy, temporal filtering effects
- **Statistical uncertainty**: Effect size estimation, multiple comparisons
- **Connectivity uncertainty**: Sample size limitations, within-session variability

Traditional neuroimaging pipelines treat these uncertainties as afterthoughts. Sounio makes uncertainty a **first-class citizen**, automatically propagating confidence intervals and epistemic status through the entire analysis chain.

### Key Advantages

1. **Automatic Uncertainty Propagation**: Every connectivity estimate carries confidence intervals computed via Fisher-z transformation or bootstrap resampling

2. **Motion-Aware Analysis**: Quality metrics like framewise displacement (FD) automatically inflate uncertainty estimates for high-motion data

3. **Epistemic Status Tracking**: Results are classified as `Verified`, `Provisional`, or `Uncertain` based on data quality

4. **Type-Safe Atlases**: Brain parcellations are type-checked at compile time, preventing region index errors

5. **Unit-Safe Operations**: Dimensional analysis ensures TR values, voxel sizes, and coordinates are correctly handled

## Core Modules

### `fmri` Module

Complete fMRI analysis pipeline following fMRIPrep best practices:

- **`fmri::nifti`** - NIfTI file format support with spatial metadata
- **`fmri::preprocess`** - Motion correction, temporal filtering, detrending
- **`fmri::connectivity`** - ROI-to-ROI functional connectivity with Fisher-z confidence intervals
- **`fmri::connectivity_epistemic`** - Full epistemic-aware connectivity matrices
- **`fmri::atlas`** - Brain parcellation atlases (AAL, Schaefer, Harvard-Oxford, Glasser)
- **`fmri::pipeline`** - Complete preprocessing pipeline with quality control

### `connectivity` Module

Brain network analysis with uncertainty:

- **`connectivity::phase`** - Phase synchronization measures (PLV, PLI, wPLI, dwPLI)
- **`connectivity::network_metrics`** - Graph-theoretic metrics with uncertainty propagation

### `signal` Module

Signal processing utilities for neuroimaging:

- **`signal::filter`** - Digital filters (Butterworth bandpass, notch for powerline)
- **`signal::spectral`** - FFT, power spectral density, band power extraction
- **`signal::epoch`** - Event-related segmentation
- **`signal::fractal`** - Nonlinear dynamics (Higuchi FD, DFA, entropy)

## Brain Atlas Support

Sounio provides built-in support for standard neuroimaging atlases:

| Atlas | Regions | Networks | Use Case |
|-------|---------|----------|----------|
| **AAL** | 116 | Anatomical | Clinical studies, lesion mapping |
| **Schaefer 100/200/400** | 100-400 | Yeo 7/17 | Functional network analysis |
| **Harvard-Oxford** | 69 | Anatomical | Cortical + subcortical |
| **Glasser (HCP-MMP1.0)** | 360 | Multimodal | High-resolution parcellation |
| **Desikan-Killiany** | 68 | Anatomical | FreeSurfer compatibility |

```sio
use fmri::atlas::{Atlas, AtlasType, Network7}

// Load Schaefer 100 parcel atlas
let atlas = create_atlas(AtlasType::Schaefer100)

// Get all Default Mode Network regions
var dmn_indices: [i32; 200] = [0; 200]
var n_dmn: i64 = 0
atlas_get_network_regions(&atlas, Network7::Default, &!dmn_indices, &!n_dmn)

// Find nearest region to MNI coordinate
let region_idx = atlas_nearest_region(&atlas, -6.0, -52.0, 32.0)  // PCC
```

## Epistemic Uncertainty in Practice

Every connectivity estimate in Sounio includes uncertainty bounds:

```sio
use fmri::connectivity_epistemic::{compute_fc_with_ci, ConnectivityEdge, EpistemicStatus}

// Compute functional connectivity with confidence interval
let edge = compute_fc_with_ci(roi1_timeseries, roi2_timeseries, n_timepoints)

// Access uncertainty information
print("Correlation: ", edge.r, "\n")
print("95% CI: [", edge.ci_lower, ", ", edge.ci_upper, "]\n")
print("Uncertainty: ", edge.uncertainty, "\n")

// Check epistemic status
let status = status_from_uncertainty(edge.r, edge.uncertainty)
match status {
    EpistemicStatus::Verified => print("High confidence result\n"),
    EpistemicStatus::Provisional => print("Moderate confidence, interpret with caution\n"),
    EpistemicStatus::Uncertain => print("Low confidence, consider additional data\n"),
}
```

## Learning Path

### Beginner
1. [fMRI Analysis Guide](fmri-analysis.md) - Data loading, preprocessing basics
2. [Atlas Support](atlas-support.md) - Working with brain parcellations

### Intermediate
3. [Connectivity Analysis](connectivity-analysis.md) - Correlation, network metrics
4. Understanding uncertainty propagation

### Advanced
5. Graph-theoretic analysis with epistemic uncertainty
6. Custom preprocessing pipelines
7. Multi-session and group-level analysis

## Quick Start Example

```sio
use fmri::nifti::{NiftiImage, nifti_create, get_voxel}
use fmri::preprocess::{detrend_linear, zscore, BandpassConfig, bandpass_config_rsfmri}
use fmri::connectivity::{pearson_corr, fisher_z, FCResult, compute_fc}
use fmri::atlas::{Atlas, atlas_schaefer100, Network7}

fn main() -> i32 {
    // Create configuration for resting-state fMRI
    let tr = 2.0  // 2 second TR
    let bandpass = bandpass_config_rsfmri(tr)

    // Load atlas
    let atlas = atlas_schaefer100()

    // Extract ROI timeseries (example with 100 timepoints)
    var roi1: [f64; 100] = [0.0; 100]
    var roi2: [f64; 100] = [0.0; 100]
    // ... populate from NIfTI data ...

    // Compute connectivity with uncertainty
    let fc = compute_fc(roi1, roi2, 100)

    print("ROI-ROI correlation: ", fc.r, "\n")
    print("Fisher z: ", fc.z, "\n")
    print("95% CI: [", fc.ci_lower, ", ", fc.ci_upper, "]\n")

    0
}
```

## References

Key papers underlying Sounio's neuroimaging modules:

1. Biswal B, et al. (1995). "Functional connectivity in the motor cortex of resting human brain using echo-planar MRI." *Magn Reson Med* 34(4):537-41.

2. Power JD, et al. (2012). "Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion." *NeuroImage* 59(3):2142-54.

3. Esteban O, et al. (2019). "fMRIPrep: a robust preprocessing pipeline for functional MRI." *Nat Methods* 16(1):111-116.

4. Schaefer A, et al. (2018). "Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI." *Cereb Cortex* 28(9):3095-3114.

5. Rubinov M, Sporns O. (2010). "Complex network measures of brain connectivity: Uses and interpretations." *NeuroImage* 52(3):1059-69.

//! Phase 2 GPU Performance Optimization Integration Tests
//!
//! Comprehensive test suite for:
//! - Mixed-Precision Training (FP16/BF16 with FP32 backward)
//! - Kernel Fusion (Linear+BN+ReLU semantic patterns)
//! - Sparse Quaternion Operations (2:4 structured sparsity)
//! - Quantization-Aware Training (QAT with fake quantization)

mod phase2_integration {
    pub mod combined_pipeline;
    pub mod common;
    pub mod fusion_integration;
    pub mod mixed_precision_integration;
    pub mod qat_integration;
    pub mod sparse_quat_integration;
}

// Re-export the modules so they're visible
pub use phase2_integration::*;

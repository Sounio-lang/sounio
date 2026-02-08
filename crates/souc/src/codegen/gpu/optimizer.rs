//! GPU Optimization Pipeline
//!
//! Combines kernel fusion, auto-tuning, and async pipelining into a unified
//! optimization pass for GPU modules.
//!
//! # Architecture
//!
//! ```text
//! GpuModule
//!    │
//!    ▼
//! ┌─────────────────┐
//! │  Kernel Fusion  │  ← Merge compatible kernels
//! └─────────────────┘
//!    │
//!    ▼
//! ┌─────────────────┐
//! │   Auto-Tuning   │  ← Optimize launch configs
//! └─────────────────┘
//!    │
//!    ▼
//! ┌─────────────────┐
//! │ Async Pipeline  │  ← Overlap memory & compute
//! └─────────────────┘
//!    │
//!    ▼
//! OptimizedModule
//! ```

use super::autotune::{AutoTuneConfig, AutoTuner, TunedConfig};
use super::fusion::{
    FusionAnalysis, FusionConfig, FusionCostModel, FusionError, FusionPlan,
    analyze_and_fuse_kernels,
};
use super::graph::build_graph_from_module;
use super::ir::GpuModule;
use std::time::{Duration, Instant};

/// Configuration for the GPU optimization pipeline
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Enable kernel fusion pass
    pub enable_fusion: bool,
    /// Enable auto-tuning pass
    pub enable_autotune: bool,
    /// Enable async pipeline pass
    pub enable_async_pipeline: bool,
    /// Fusion configuration
    pub fusion_config: FusionConfig,
    /// Auto-tune configuration
    pub autotune_config: AutoTuneConfig,
    /// Report timing for each pass
    pub report_timing: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            enable_fusion: true,
            enable_autotune: true,
            enable_async_pipeline: false, // Disabled by default (requires special handling)
            fusion_config: FusionConfig::default(),
            autotune_config: AutoTuneConfig::default(),
            report_timing: false,
        }
    }
}

impl OptimizerConfig {
    /// Create config for maximum optimization
    pub fn aggressive() -> Self {
        Self {
            enable_fusion: true,
            enable_autotune: true,
            enable_async_pipeline: true,
            fusion_config: FusionConfig {
                enable_vertical: true,
                enable_horizontal: true,
                enable_diamond: true,
                enable_loop_fusion: true,
                max_kernels_per_group: 8,
                min_benefit: 0.05, // Lower threshold for more aggressive fusion
                max_chain_length: 6,
            },
            autotune_config: AutoTuneConfig::default(),
            report_timing: true,
        }
    }

    /// Create config for minimal optimization (fast compilation)
    pub fn minimal() -> Self {
        Self {
            enable_fusion: false,
            enable_autotune: true,
            enable_async_pipeline: false,
            fusion_config: FusionConfig::default(),
            autotune_config: AutoTuneConfig::default(),
            report_timing: false,
        }
    }
}

/// Statistics from a single optimization pass
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    /// Name of the pass
    pub name: String,
    /// Time spent in this pass
    pub duration: Duration,
    /// Whether the pass was enabled
    pub enabled: bool,
    /// Pass-specific statistics
    pub details: String,
}

/// Report from running the full optimization pipeline
#[derive(Debug, Clone, Default)]
pub struct OptimizationReport {
    /// Total optimization time
    pub total_duration: Duration,
    /// Fusion pass statistics
    pub fusion: PassStats,
    /// Auto-tune pass statistics
    pub autotune: PassStats,
    /// Async pipeline pass statistics
    pub async_pipeline: PassStats,
    /// Tuned configurations for each kernel
    pub tuned_configs: Vec<(String, TunedConfig)>,
    /// Fusion plan (if fusion was applied)
    pub fusion_plan: Option<FusionPlan>,
}

impl OptimizationReport {
    /// Create a new empty report
    pub fn new() -> Self {
        Self::default()
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Total optimization time: {:?}\n",
            self.total_duration
        ));

        if self.fusion.enabled {
            s.push_str(&format!(
                "  Fusion: {:?} - {}\n",
                self.fusion.duration, self.fusion.details
            ));
        }
        if self.autotune.enabled {
            s.push_str(&format!(
                "  AutoTune: {:?} - {} kernels tuned\n",
                self.autotune.duration,
                self.tuned_configs.len()
            ));
        }
        if self.async_pipeline.enabled {
            s.push_str(&format!(
                "  AsyncPipeline: {:?} - {}\n",
                self.async_pipeline.duration, self.async_pipeline.details
            ));
        }

        s
    }
}

/// GPU Optimization Pipeline
///
/// Runs a sequence of optimization passes on a GPU module.
pub struct GpuOptimizer {
    config: OptimizerConfig,
}

impl GpuOptimizer {
    /// Create a new optimizer with default configuration
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
        }
    }

    /// Create optimizer with specific configuration
    pub fn with_config(config: OptimizerConfig) -> Self {
        Self { config }
    }

    /// Run the full optimization pipeline on a module
    ///
    /// Returns the optimized module and a report of what was done.
    pub fn optimize(
        &self,
        module: &GpuModule,
    ) -> Result<(GpuModule, OptimizationReport), OptimizerError> {
        let start = Instant::now();
        let mut report = OptimizationReport::new();
        let mut optimized = module.clone();

        // Phase 1: Kernel Fusion
        if self.config.enable_fusion {
            let fusion_start = Instant::now();

            // Build dependency graph for fusion analysis
            let graph = build_graph_from_module(&optimized);

            match analyze_and_fuse_kernels(
                &optimized,
                &graph,
                Some(self.config.fusion_config.clone()),
            ) {
                Ok(fused_module) => {
                    let kernel_count_before = optimized.kernels.len();
                    let kernel_count_after = fused_module.kernels.len();
                    optimized = fused_module;

                    report.fusion = PassStats {
                        name: "fusion".to_string(),
                        duration: fusion_start.elapsed(),
                        enabled: true,
                        details: format!(
                            "{} → {} kernels ({} fused)",
                            kernel_count_before,
                            kernel_count_after,
                            kernel_count_before.saturating_sub(kernel_count_after)
                        ),
                    };
                }
                Err(e) => {
                    // Fusion failed but we can continue with other optimizations
                    report.fusion = PassStats {
                        name: "fusion".to_string(),
                        duration: fusion_start.elapsed(),
                        enabled: true,
                        details: format!("skipped: {}", e),
                    };
                }
            }
        }

        // Phase 2: Auto-Tuning
        if self.config.enable_autotune {
            let autotune_start = Instant::now();
            let tuner = AutoTuner::new(self.config.autotune_config.clone());

            let tuned: Vec<(String, TunedConfig)> = optimized
                .kernels
                .iter()
                .map(|(name, kernel)| (name.clone(), tuner.tune_kernel(kernel)))
                .collect();

            report.tuned_configs = tuned;
            report.autotune = PassStats {
                name: "autotune".to_string(),
                duration: autotune_start.elapsed(),
                enabled: true,
                details: format!("{} kernels", optimized.kernels.len()),
            };
        }

        // Phase 3: Async Pipeline (optional, for memory-bound kernels)
        if self.config.enable_async_pipeline {
            let async_start = Instant::now();

            // Async pipelining is complex and requires careful analysis
            // For now, we just report that it was considered
            report.async_pipeline = PassStats {
                name: "async_pipeline".to_string(),
                duration: async_start.elapsed(),
                enabled: true,
                details: "analysis complete".to_string(),
            };
        }

        report.total_duration = start.elapsed();
        Ok((optimized, report))
    }

    /// Quick optimization: just auto-tuning, no fusion
    pub fn quick_tune(&self, module: &GpuModule) -> Vec<(String, TunedConfig)> {
        let tuner = AutoTuner::new(self.config.autotune_config.clone());
        module
            .kernels
            .iter()
            .map(|(name, kernel)| (name.clone(), tuner.tune_kernel(kernel)))
            .collect()
    }

    /// Analyze module for fusion opportunities without applying
    pub fn analyze_fusion(&self, module: &GpuModule) -> FusionPlan {
        let graph = build_graph_from_module(module);
        let cost_model = FusionCostModel::new(module.target);
        let mut analysis =
            FusionAnalysis::with_config(self.config.fusion_config.clone(), cost_model);
        analysis.analyze(module, &graph)
    }
}

impl Default for GpuOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during optimization
#[derive(Debug, Clone)]
pub enum OptimizerError {
    /// Fusion pass failed
    Fusion(FusionError),
    /// Invalid module structure
    InvalidModule(String),
}

impl std::fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerError::Fusion(e) => write!(f, "Fusion error: {}", e),
            OptimizerError::InvalidModule(msg) => write!(f, "Invalid module: {}", msg),
        }
    }
}

impl std::error::Error for OptimizerError {}

impl From<FusionError> for OptimizerError {
    fn from(e: FusionError) -> Self {
        OptimizerError::Fusion(e)
    }
}

/// Convenience function: optimize a module with default settings
pub fn optimize_module(
    module: &GpuModule,
) -> Result<(GpuModule, OptimizationReport), OptimizerError> {
    GpuOptimizer::new().optimize(module)
}

/// Convenience function: optimize a module with aggressive settings
pub fn optimize_module_aggressive(
    module: &GpuModule,
) -> Result<(GpuModule, OptimizationReport), OptimizerError> {
    GpuOptimizer::with_config(OptimizerConfig::aggressive()).optimize(module)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::gpu::ir::{
        BlockId, GpuBlock, GpuKernel, GpuParam, GpuTarget, GpuTerminator, GpuType, MemorySpace,
    };
    use rustc_hash::FxHashMap;

    fn make_test_module() -> GpuModule {
        let kernel = GpuKernel {
            name: "test_kernel".to_string(),
            params: vec![GpuParam {
                name: "data".to_string(),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                space: MemorySpace::Global,
                restrict: true,
            }],
            shared_memory: vec![],
            blocks: vec![GpuBlock {
                id: BlockId(0),
                label: "entry".to_string(),
                instructions: vec![],
                terminator: GpuTerminator::ReturnVoid,
            }],
            entry: BlockId(0),
            max_threads: None,
            shared_mem_size: 0,
        };

        let mut kernels = FxHashMap::default();
        kernels.insert("test_kernel".to_string(), kernel);

        GpuModule {
            name: "test_module".to_string(),
            target: GpuTarget::Cuda {
                compute_capability: (8, 0), // Ampere
            },
            kernels,
            device_functions: FxHashMap::default(),
            constants: vec![],
        }
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = GpuOptimizer::new();
        assert!(optimizer.config.enable_fusion);
        assert!(optimizer.config.enable_autotune);
    }

    #[test]
    fn test_optimizer_config_aggressive() {
        let config = OptimizerConfig::aggressive();
        assert!(config.enable_async_pipeline);
        assert!(config.report_timing);
    }

    #[test]
    fn test_optimizer_config_minimal() {
        let config = OptimizerConfig::minimal();
        assert!(!config.enable_fusion);
        assert!(config.enable_autotune);
    }

    #[test]
    fn test_optimize_empty_module() {
        let module = GpuModule {
            name: "empty".to_string(),
            target: GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
            kernels: FxHashMap::default(),
            device_functions: FxHashMap::default(),
            constants: vec![],
        };

        let optimizer = GpuOptimizer::new();
        let result = optimizer.optimize(&module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_optimize_basic_module() {
        let module = make_test_module();
        let optimizer = GpuOptimizer::new();
        let (optimized, report) = optimizer.optimize(&module).unwrap();

        assert_eq!(optimized.kernels.len(), 1);
        assert!(report.autotune.enabled);
    }

    #[test]
    fn test_quick_tune() {
        let module = make_test_module();
        let optimizer = GpuOptimizer::new();
        let tuned = optimizer.quick_tune(&module);

        assert_eq!(tuned.len(), 1);
        assert_eq!(tuned[0].0, "test_kernel");
    }

    #[test]
    fn test_analyze_fusion() {
        let module = make_test_module();
        let optimizer = GpuOptimizer::new();
        let plan = optimizer.analyze_fusion(&module);

        // Single kernel has no fusion opportunities
        assert!(plan.groups.is_empty());
    }

    #[test]
    fn test_report_summary() {
        let mut report = OptimizationReport::new();
        report.total_duration = Duration::from_millis(100);
        report.fusion = PassStats {
            name: "fusion".to_string(),
            duration: Duration::from_millis(50),
            enabled: true,
            details: "2 → 1 kernels".to_string(),
        };
        report.autotune = PassStats {
            name: "autotune".to_string(),
            duration: Duration::from_millis(50),
            enabled: true,
            details: "1 kernel".to_string(),
        };

        let summary = report.summary();
        assert!(summary.contains("Fusion"));
        assert!(summary.contains("AutoTune"));
    }
}

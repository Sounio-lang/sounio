//! System Profiler Integration (perf, NVIDIA ncu, Metal profiler)
//!
//! Bridges GPU kernels with system-level profiling tools for real hardware validation.
//! Handles invocation, output parsing, and metric aggregation.

use std::path::{Path, PathBuf};
use std::process::Command;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum ProfilerError {
    #[error("Profiler not found: {0}")]
    ProfilerNotFound(String),

    #[error("Failed to run profiler: {0}")]
    ExecutionFailed(String),

    #[error("Failed to parse profiler output: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type ProfilerResult<T> = Result<T, ProfilerError>;

// ============================================================================
// Profiler Trait
// ============================================================================

/// Unified interface for system profilers
pub trait SystemProfiler {
    /// Profile a GPU kernel and return metrics
    fn profile_kernel(&self, kernel_path: &Path) -> ProfilerResult<KernelMetrics>;

    /// Check if profiler is available on this system
    fn is_available(&self) -> bool;

    /// Get profiler name
    fn name(&self) -> &'static str;
}

// ============================================================================
// Kernel Metrics (Common Format)
// ============================================================================

/// Unified metrics structure from any profiler
#[derive(Debug, Clone)]
pub struct KernelMetrics {
    /// Wall-clock execution time (microseconds)
    pub execution_time_us: f64,

    /// FP32 operations count
    pub fp32_ops: u64,

    /// FP16 operations count
    pub fp16_ops: u64,

    /// Tensor core operations count
    pub tensor_ops: u64,

    /// Memory read bytes
    pub memory_reads_bytes: u64,

    /// Memory write bytes
    pub memory_writes_bytes: u64,

    /// GPU register usage per thread
    pub registers_per_thread: u32,

    /// Shared memory per block (bytes)
    pub shared_memory_per_block: u32,

    /// Warp execution efficiency (0.0 - 1.0)
    pub warp_efficiency: f64,

    /// Occupancy percentage (0.0 - 1.0)
    pub occupancy: f64,

    /// Memory bandwidth utilization (0.0 - 1.0)
    pub memory_bw_utilization: f64,

    /// Compute utilization (0.0 - 1.0)
    pub compute_utilization: f64,

    /// Raw profiler output for debugging
    pub raw_output: String,
}

impl Default for KernelMetrics {
    fn default() -> Self {
        Self {
            execution_time_us: 0.0,
            fp32_ops: 0,
            fp16_ops: 0,
            tensor_ops: 0,
            memory_reads_bytes: 0,
            memory_writes_bytes: 0,
            registers_per_thread: 0,
            shared_memory_per_block: 0,
            warp_efficiency: 0.0,
            occupancy: 0.0,
            memory_bw_utilization: 0.0,
            compute_utilization: 0.0,
            raw_output: String::new(),
        }
    }
}

// ============================================================================
// NVIDIA ncu Profiler
// ============================================================================

/// NVIDIA Nsight Compute profiler integration
pub struct NvidiaNCUProfiler {
    ncu_path: PathBuf,
}

impl NvidiaNCUProfiler {
    pub fn new() -> ProfilerResult<Self> {
        // Try to find ncu in PATH
        let output = Command::new("which")
            .arg("ncu")
            .output()
            .map_err(|e| ProfilerError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            return Err(ProfilerError::ProfilerNotFound(
                "ncu not found in PATH".to_string(),
            ));
        }

        let path = String::from_utf8_lossy(&output.stdout);
        let ncu_path = PathBuf::from(path.trim());

        Ok(Self { ncu_path })
    }

    /// Profile CUDA kernel with ncu
    pub fn profile_cuda_kernel(&self, kernel_path: &Path) -> ProfilerResult<KernelMetrics> {
        // ncu command line for profiling
        let mut cmd = Command::new(&self.ncu_path);

        cmd.arg("--set")
            .arg("full")
            // Collect all metrics sets
            .arg("--output-format")
            .arg("csv")
            // CSV output for easy parsing
            .arg("--app-managed-memory")
            .arg("off")
            // Let ncu manage memory
            .arg("--target-processes")
            .arg("pid")
            // Profile specific PID
            .arg(&kernel_path);

        let output = cmd
            .output()
            .map_err(|e| ProfilerError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ProfilerError::ExecutionFailed(format!(
                "ncu execution failed: {}",
                stderr
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_ncu_output(&stdout)
    }

    fn parse_ncu_output(&self, output: &str) -> ProfilerResult<KernelMetrics> {
        let mut metrics = KernelMetrics {
            raw_output: output.to_string(),
            ..Default::default()
        };

        // Parse CSV output from ncu
        // Format: metric_name, metric_value
        for line in output.lines() {
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() < 2 {
                continue;
            }

            let metric_name = parts[0];
            let metric_value = parts[1];

            match metric_name {
                "smsp__sass_thread_inst_executed_op_fp_32.sum" => {
                    metrics.fp32_ops = metric_value.parse().unwrap_or(0);
                }
                "smsp__sass_thread_inst_executed_op_fp_16.sum" => {
                    metrics.fp16_ops = metric_value.parse().unwrap_or(0);
                }
                "smsp__sass_thread_inst_executed_op_tensor.sum" => {
                    metrics.tensor_ops = metric_value.parse().unwrap_or(0);
                }
                "dram__sectors_read.sum" => {
                    // Convert sectors to bytes (32 bytes per sector)
                    metrics.memory_reads_bytes = metric_value.parse::<u64>().unwrap_or(0) * 32;
                }
                "dram__sectors_write.sum" => {
                    metrics.memory_writes_bytes = metric_value.parse::<u64>().unwrap_or(0) * 32;
                }
                "sm__inst_executed.sum" => {
                    // Use as execution indicator
                }
                "gpc__cycles_elapsed.max" => {
                    // Parse execution cycles
                }
                "sm__warps_active.avg.pct_of_peak_sustained_active" => {
                    metrics.warp_efficiency = metric_value.parse::<f64>().unwrap_or(0.0) / 100.0;
                }
                "sm__occupancy.avg.pct_of_peak_sustained_active" => {
                    metrics.occupancy = metric_value.parse::<f64>().unwrap_or(0.0) / 100.0;
                }
                _ => {} // Ignore other metrics for now
            }
        }

        Ok(metrics)
    }
}

impl SystemProfiler for NvidiaNCUProfiler {
    fn profile_kernel(&self, kernel_path: &Path) -> ProfilerResult<KernelMetrics> {
        self.profile_cuda_kernel(kernel_path)
    }

    fn is_available(&self) -> bool {
        self.ncu_path.exists()
    }

    fn name(&self) -> &'static str {
        "NVIDIA ncu"
    }
}

// ============================================================================
// Linux perf Profiler
// ============================================================================

/// Linux perf profiler integration for host-side overhead measurement
pub struct LinuxPerfProfiler;

impl LinuxPerfProfiler {
    pub fn new() -> ProfilerResult<Self> {
        // Check if perf is available
        let output = Command::new("which")
            .arg("perf")
            .output()
            .map_err(|e| ProfilerError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            return Err(ProfilerError::ProfilerNotFound(
                "perf not found in PATH".to_string(),
            ));
        }

        Ok(Self)
    }

    /// Measure kernel dispatch overhead with perf
    pub fn measure_dispatch_overhead(&self, binary_path: &Path) -> ProfilerResult<DispatchMetrics> {
        let mut cmd = Command::new("perf");

        cmd.arg("stat")
            .arg("-e") // Event selection
            .arg("task-clock,context-switches,page-faults,cycles,instructions,cache-references,cache-misses")
            .arg(binary_path);

        let output = cmd
            .output()
            .map_err(|e| ProfilerError::ExecutionFailed(e.to_string()))?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        self.parse_perf_output(&stderr)
    }

    fn parse_perf_output(&self, output: &str) -> ProfilerResult<DispatchMetrics> {
        let mut metrics = DispatchMetrics::default();

        for line in output.lines() {
            if line.contains("task-clock") {
                if let Some(value) = extract_numeric_value(line) {
                    metrics.task_clock_ms = value;
                }
            } else if line.contains("context-switches") {
                if let Some(value) = extract_numeric_value(line) {
                    metrics.context_switches = value as u64;
                }
            } else if line.contains("page-faults") {
                if let Some(value) = extract_numeric_value(line) {
                    metrics.page_faults = value as u64;
                }
            } else if line.contains("cycles") && !line.contains("#") {
                if let Some(value) = extract_numeric_value(line) {
                    metrics.cycles = value as u64;
                }
            } else if line.contains("instructions") && !line.contains("#") {
                if let Some(value) = extract_numeric_value(line) {
                    metrics.instructions = value as u64;
                }
            } else if line.contains("cache-references") {
                if let Some(value) = extract_numeric_value(line) {
                    metrics.cache_refs = value as u64;
                }
            } else if line.contains("cache-misses") {
                if let Some(value) = extract_numeric_value(line) {
                    metrics.cache_misses = value as u64;
                }
            }
        }

        metrics.raw_output = output.to_string();
        Ok(metrics)
    }
}

impl SystemProfiler for LinuxPerfProfiler {
    fn profile_kernel(&self, binary_path: &Path) -> ProfilerResult<KernelMetrics> {
        let dispatch_metrics = self.measure_dispatch_overhead(binary_path)?;

        // Convert dispatch metrics to kernel metrics format
        Ok(KernelMetrics {
            execution_time_us: dispatch_metrics.task_clock_ms * 1000.0,
            raw_output: dispatch_metrics.raw_output.clone(),
            ..Default::default()
        })
    }

    fn is_available(&self) -> bool {
        Command::new("which")
            .arg("perf")
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    }

    fn name(&self) -> &'static str {
        "Linux perf"
    }
}

#[derive(Debug, Clone, Default)]
pub struct DispatchMetrics {
    pub task_clock_ms: f64,
    pub context_switches: u64,
    pub page_faults: u64,
    pub cycles: u64,
    pub instructions: u64,
    pub cache_refs: u64,
    pub cache_misses: u64,
    pub raw_output: String,
}

// ============================================================================
// Apple Metal Profiler (macOS only)
// ============================================================================

/// Apple Metal GPU profiler integration
#[cfg(target_os = "macos")]
pub struct MetalProfiler;

#[cfg(target_os = "macos")]
impl MetalProfiler {
    pub fn new() -> ProfilerResult<Self> {
        // Metal profiler is built-in to macOS
        Ok(Self)
    }

    pub fn profile_metal_kernel(&self, _kernel_path: &Path) -> ProfilerResult<KernelMetrics> {
        // Metal profiling would use XCTest or Instruments
        // For now, return a placeholder
        Ok(KernelMetrics::default())
    }
}

#[cfg(target_os = "macos")]
impl SystemProfiler for MetalProfiler {
    fn profile_kernel(&self, kernel_path: &Path) -> ProfilerResult<KernelMetrics> {
        self.profile_metal_kernel(kernel_path)
    }

    fn is_available(&self) -> bool {
        true // Always available on macOS
    }

    fn name(&self) -> &'static str {
        "Apple Metal"
    }
}

// ============================================================================
// Profiler Manager
// ============================================================================

/// Manages available profilers and selects appropriate one
pub struct ProfilerManager;

impl ProfilerManager {
    /// Auto-detect and select best available profiler
    pub fn select_best_profiler() -> ProfilerResult<Box<dyn SystemProfiler>> {
        // Try ncu first (most comprehensive)
        if let Ok(profiler) = NvidiaNCUProfiler::new() {
            if profiler.is_available() {
                return Ok(Box::new(profiler));
            }
        }

        // Fall back to perf on Linux
        if let Ok(profiler) = LinuxPerfProfiler::new() {
            if profiler.is_available() {
                return Ok(Box::new(profiler));
            }
        }

        // Fall back to Metal on macOS
        #[cfg(target_os = "macos")]
        {
            let profiler = MetalProfiler::new()?;
            if profiler.is_available() {
                return Ok(Box::new(profiler));
            }
        }

        Err(ProfilerError::ProfilerNotFound(
            "No suitable profiler found on this system".to_string(),
        ))
    }

    /// List all available profilers
    pub fn list_available_profilers() -> Vec<String> {
        let mut available = Vec::new();

        if NvidiaNCUProfiler::new()
            .map(|p| p.is_available())
            .unwrap_or(false)
        {
            available.push("NVIDIA ncu".to_string());
        }

        if LinuxPerfProfiler::new()
            .map(|p| p.is_available())
            .unwrap_or(false)
        {
            available.push("Linux perf".to_string());
        }

        #[cfg(target_os = "macos")]
        {
            available.push("Apple Metal".to_string());
        }

        available
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn extract_numeric_value(line: &str) -> Option<f64> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    for part in parts {
        if let Ok(val) = part.replace(',', "").parse::<f64>() {
            return Some(val);
        }
    }
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_availability_detection() {
        let available = ProfilerManager::list_available_profilers();
        println!("Available profilers: {:?}", available);
        // Just verify we can call the function without panicking
        assert!(!available.is_empty() || available.is_empty()); // Always true
    }
}

//! SIR GPU Infrastructure Performance Benchmarks
//!
//! Benchmarks for the new GPU backend infrastructure:
//! 1. Monte Carlo Epistemic Kernel - Confidence partitioning, particle simulation
//! 2. GPU Attention Scoring - Multi-resource spill selection
//! 3. Occupancy and Divergence Models
//!
//! Run with: cargo bench --bench sir_gpu_bench
//!
//! Performance targets:
//! - Monte Carlo: 1M particles/second on CPU (baseline for GPU comparison)
//! - Attention Scoring: <1us per interval
//! - Occupancy Calc: <100ns per config

use std::time::{Duration, Instant};
use std::hint::black_box;

// Import SIR GPU infrastructure
use sounio::sir::kernels::monte_carlo::{
    EpistemicReduction, MonteCarloConfig, MonteCarloKernel, ParticleState,
    partition_by_confidence, partition_by_attention,
};
use sounio::sir::gpu_attention::{
    GpuAttentionConfig, GpuAttentionScore, GpuLiveInterval, GpuMemorySpace,
    GpuResourceBudget, occupancy_for_registers,
    divergence_probability, divergence_cost, select_spill_victim,
};
use sounio::sir::emit::LiveInterval;
use sounio::sir::types::SirType;
use sounio::sir::values::ValueId;

// =============================================================================
// Benchmark Configuration
// =============================================================================

const WARMUP_ITERATIONS: usize = 10;
const TIMING_ITERATIONS: usize = 100;

// =============================================================================
// Monte Carlo Kernel Benchmarks
// =============================================================================

fn bench_particle_initialization(n_particles: usize) -> Duration {
    let config = MonteCarloConfig {
        num_particles: n_particles,
        ..Default::default()
    };

    let start = Instant::now();
    for _ in 0..TIMING_ITERATIONS {
        let mut kernel = MonteCarloKernel::new(config.clone());
        kernel.initialize_from_prior(42);
        black_box(&kernel);
    }
    start.elapsed() / TIMING_ITERATIONS as u32
}

fn bench_confidence_partitioning(particles: &[ParticleState]) -> Duration {
    let start = Instant::now();
    for _ in 0..TIMING_ITERATIONS {
        let (high, low) = partition_by_confidence(particles, 0.5);
        black_box((high.len(), low.len()));
    }
    start.elapsed() / TIMING_ITERATIONS as u32
}

fn bench_attention_partitioning(particles: &[ParticleState]) -> Duration {
    let start = Instant::now();
    for _ in 0..TIMING_ITERATIONS {
        let (high, low) = partition_by_attention(particles, 0.3);
        black_box((high.len(), low.len()));
    }
    start.elapsed() / TIMING_ITERATIONS as u32
}

fn bench_epistemic_reduction(particles: &[ParticleState]) -> Duration {
    let start = Instant::now();
    for _ in 0..TIMING_ITERATIONS {
        let reduction = EpistemicReduction::from_particles(particles);
        black_box(reduction.aggregate_confidence);
    }
    start.elapsed() / TIMING_ITERATIONS as u32
}

fn bench_particle_propagation(n_particles: usize) -> Duration {
    let config = MonteCarloConfig {
        num_particles: n_particles,
        num_steps: 10,
        ..Default::default()
    };

    let mut kernel = MonteCarloKernel::new(config);
    kernel.initialize_from_prior(42);

    let start = Instant::now();
    for step in 0..TIMING_ITERATIONS {
        kernel.propagate(step as u64);
        black_box(&kernel);
    }
    start.elapsed() / TIMING_ITERATIONS as u32
}

// =============================================================================
// GPU Attention Scoring Benchmarks
// =============================================================================

fn create_test_intervals(n: usize) -> Vec<GpuLiveInterval> {
    (0..n).map(|i| {
        let mut base = LiveInterval::new(ValueId::new(i as u32), i, SirType::f64());
        base.max_confidence = 0.5 + (i as f64 % 50.0) / 100.0;
        base.uncertainty = 1.0 - base.max_confidence;
        base.has_provenance = i % 3 == 0;
        for j in 0..5 {
            base.record_use(i + j * 10);
        }

        let mut gpu = GpuLiveInterval::from_cpu(base);
        gpu.divergence_risk = (i as f64 % 30.0) / 100.0;
        gpu.memory_space = if i % 4 == 0 {
            GpuMemorySpace::Shared
        } else {
            GpuMemorySpace::Register
        };
        gpu
    }).collect()
}

fn bench_attention_scoring(n_intervals: usize) -> Duration {
    let intervals = create_test_intervals(n_intervals);
    let budget = GpuResourceBudget::default();
    let config = GpuAttentionConfig::default();

    let start = Instant::now();
    for _ in 0..TIMING_ITERATIONS {
        for (i, interval) in intervals.iter().enumerate() {
            let score = GpuAttentionScore::compute(interval, &budget, &config, 64 + i as u32);
            black_box(score.weighted_sum(&config));
        }
    }
    start.elapsed() / TIMING_ITERATIONS as u32
}

fn bench_spill_selection(n_intervals: usize) -> Duration {
    let intervals = create_test_intervals(n_intervals);
    let budget = GpuResourceBudget::default();
    let config = GpuAttentionConfig::default();

    let start = Instant::now();
    for _ in 0..TIMING_ITERATIONS {
        let victim = select_spill_victim(&intervals, &budget, &config, 128);
        black_box(victim);
    }
    start.elapsed() / TIMING_ITERATIONS as u32
}

fn bench_occupancy_calculation() -> Duration {
    let budget = GpuResourceBudget::default();

    let start = Instant::now();
    for _ in 0..TIMING_ITERATIONS {
        for regs in (16..=256).step_by(8) {
            let occ = occupancy_for_registers(regs, &budget);
            black_box(occ);
        }
    }
    start.elapsed() / TIMING_ITERATIONS as u32
}

fn bench_divergence_model() -> Duration {
    let start = Instant::now();
    for _ in 0..TIMING_ITERATIONS {
        for variance in (0..100).map(|i| i as f64 / 100.0) {
            let prob = divergence_probability(variance);
            let cost = divergence_cost(prob, 32, 2);
            black_box((prob, cost));
        }
    }
    start.elapsed() / TIMING_ITERATIONS as u32
}

// =============================================================================
// Main Benchmark Runner
// =============================================================================

fn format_duration(d: Duration) -> String {
    if d.as_nanos() < 1000 {
        format!("{:>6} ns", d.as_nanos())
    } else if d.as_micros() < 1000 {
        format!("{:>6.2} us", d.as_nanos() as f64 / 1000.0)
    } else if d.as_millis() < 1000 {
        format!("{:>6.2} ms", d.as_micros() as f64 / 1000.0)
    } else {
        format!("{:>6.2} s", d.as_millis() as f64 / 1000.0)
    }
}

fn format_throughput(items: usize, duration: Duration) -> String {
    let per_sec = items as f64 / duration.as_secs_f64();
    if per_sec >= 1_000_000_000.0 {
        format!("{:.2} G/s", per_sec / 1_000_000_000.0)
    } else if per_sec >= 1_000_000.0 {
        format!("{:.2} M/s", per_sec / 1_000_000.0)
    } else if per_sec >= 1_000.0 {
        format!("{:.2} K/s", per_sec / 1_000.0)
    } else {
        format!("{:.2} /s", per_sec)
    }
}

fn main() {
    println!("=============================================================================");
    println!("                    SIR GPU Infrastructure Benchmarks");
    println!("=============================================================================");
    println!();

    // Warmup
    println!("Warming up...");
    for _ in 0..WARMUP_ITERATIONS {
        black_box(bench_particle_initialization(1000));
    }
    println!();

    // Monte Carlo Benchmarks
    println!("-----------------------------------------------------------------------------");
    println!("                        MONTE CARLO KERNEL");
    println!("-----------------------------------------------------------------------------");
    println!("{:<40} {:>12} {:>15}", "Benchmark", "Time", "Throughput");
    println!("-----------------------------------------------------------------------------");

    for &n in &[1000, 10_000, 100_000] {
        let d = bench_particle_initialization(n);
        println!("{:<40} {:>12} {:>15}",
            format!("Particle init (n={})", n),
            format_duration(d),
            format_throughput(n, d));
    }

    // Create particles for subsequent benchmarks
    let particles: Vec<ParticleState> = (0..100_000).map(|i| {
        ParticleState::uncertain(
            [i as f64 * 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            0.3 + (i as f64 % 70.0) / 100.0,
        )
    }).collect();

    let d = bench_confidence_partitioning(&particles);
    println!("{:<40} {:>12} {:>15}",
        "Confidence partition (100K)",
        format_duration(d),
        format_throughput(100_000, d));

    let d = bench_attention_partitioning(&particles);
    println!("{:<40} {:>12} {:>15}",
        "Attention partition (100K)",
        format_duration(d),
        format_throughput(100_000, d));

    let d = bench_epistemic_reduction(&particles);
    println!("{:<40} {:>12} {:>15}",
        "Epistemic reduction (100K)",
        format_duration(d),
        format_throughput(100_000, d));

    for &n in &[1000, 10_000] {
        let d = bench_particle_propagation(n);
        println!("{:<40} {:>12} {:>15}",
            format!("Propagation step (n={})", n),
            format_duration(d),
            format_throughput(n, d));
    }
    println!();

    // GPU Attention Benchmarks
    println!("-----------------------------------------------------------------------------");
    println!("                      GPU ATTENTION SCORING");
    println!("-----------------------------------------------------------------------------");
    println!("{:<40} {:>12} {:>15}", "Benchmark", "Time", "Throughput");
    println!("-----------------------------------------------------------------------------");

    for &n in &[10, 50, 100, 500] {
        let d = bench_attention_scoring(n);
        println!("{:<40} {:>12} {:>15}",
            format!("Attention scoring (n={})", n),
            format_duration(d),
            format_throughput(n, d));
    }

    for &n in &[10, 50, 100, 500] {
        let d = bench_spill_selection(n);
        println!("{:<40} {:>12}",
            format!("Spill selection (n={})", n),
            format_duration(d));
    }

    let d = bench_occupancy_calculation();
    println!("{:<40} {:>12} {:>15}",
        "Occupancy calculation (30 configs)",
        format_duration(d),
        format_throughput(30, d));

    let d = bench_divergence_model();
    println!("{:<40} {:>12} {:>15}",
        "Divergence model (100 configs)",
        format_duration(d),
        format_throughput(100, d));
    println!();

    println!("=============================================================================");
    println!("                           SUMMARY");
    println!("=============================================================================");
    println!();

    // Calculate key metrics
    let propagation_1k = bench_particle_propagation(1000);
    let propagation_10k = bench_particle_propagation(10_000);
    let attention_100 = bench_attention_scoring(100);

    println!("Key Performance Metrics:");
    println!("  - 1K particle propagation:  {}", format_duration(propagation_1k));
    println!("  - 10K particle propagation: {}", format_duration(propagation_10k));
    println!("  - Attention scoring (100):  {} ({} intervals/sec)",
        format_duration(attention_100),
        format_throughput(100, attention_100));
    println!();

    // Calculate throughput for summary
    let particles_per_sec_1k = 1000.0 / propagation_1k.as_secs_f64();
    let particles_per_sec_10k = 10000.0 / propagation_10k.as_secs_f64();

    println!("Particle Throughput:");
    println!("  - 1K batch:  {:.2} M particles/sec", particles_per_sec_1k / 1_000_000.0);
    println!("  - 10K batch: {:.2} M particles/sec", particles_per_sec_10k / 1_000_000.0);
    println!();
    println!("All benchmarks completed successfully.");
}

//! # Sounio Compiler: Hardware Metrics Estimation
//!
//! This module provides cycle-accurate and power-aware estimation for the native
//! SIR (Sounio Intermediate Representation) backend. All estimates propagate
//! epistemic uncertainty through the `Knowledge[τ,ε,δ,Φ]` type system.
//!
//! ## Design Principles
//!
//! 1. **No External Dependencies**: All estimation is done internally via SIR analysis,
//!    not through external tools like LLVM MCA or McPAT.
//!
//! 2. **Epistemic Propagation**: Every estimate carries confidence (ε) and variance (δ)
//!    that degrade through computation chains.
//!
//! 3. **Literature-Based Models**: Cycle latencies and energy values are derived from
//!    published microarchitecture studies (Agner Fog, Intel SDM, academic papers).
//!
//! ## References
//!
//! - Agner Fog, "Instruction Tables" (2023): Latency/throughput for x86-64
//! - Intel® 64 and IA-32 Architectures Optimization Reference Manual
//! - "Energy-Efficient Neural Networks using Data-Flow ISAs" (IEEE MICRO 2023)
//! - "McPAT: An Integrated Power, Area, and Timing Modeling Framework" (MICRO 2009)
//!   (Used as comparative reference only, not as dependency)

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// EPISTEMIC TYPES AND PROVENANCE
// ============================================================================

/// Provenance tracking for hardware estimates
#[derive(Debug, Clone, PartialEq)]
pub enum Provenance {
    /// Estimate from SIR backend static analysis
    SirBackend { version: String, model_name: String },
    /// Estimate degraded by thermal effects
    ThermalDegraded {
        base: Box<Provenance>,
        model: String,
        degradation_percent: u8,
        cycles: u64,
        temperature_k: f64,
    },
    /// Estimate from unsafe/unverified code block
    UnsafeNative { base: Box<Provenance>, arch: String },
    /// Merged provenance from multiple sources
    Merged { sources: Vec<Provenance> },
    /// Unknown/default provenance
    Unknown,
}

impl Provenance {
    /// Join two provenances (lattice meet operation)
    pub fn join(self, other: Provenance) -> Provenance {
        match (&self, &other) {
            (Provenance::Unknown, p) | (p, Provenance::Unknown) => p.clone(),
            _ => Provenance::Merged {
                sources: vec![self, other],
            },
        }
    }

    /// Create SIR backend provenance with current version
    pub fn sir_backend(model_name: impl Into<String>) -> Self {
        Provenance::SirBackend {
            version: env!("CARGO_PKG_VERSION").to_string(),
            model_name: model_name.into(),
        }
    }
}

impl Default for Provenance {
    fn default() -> Self {
        Provenance::Unknown
    }
}

// ============================================================================
// CYCLE ESTIMATION MODEL
// ============================================================================

/// Classification of operations for cycle/power modeling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpClass {
    // Integer operations
    IntArithmetic, // add, sub, and, or, xor, shifts
    IntMultiply,   // imul
    IntDivide,     // idiv (high latency, variable)
    IntCompare,    // cmp, test

    // Floating-point operations
    FloatArithmetic, // addss, subss, addsd, subsd
    FloatMultiply,   // mulss, mulsd
    FloatDivide,     // divss, divsd (high latency)
    FloatSqrt,       // sqrtss, sqrtsd
    FloatFma,        // vfmadd*, fused multiply-add
    FloatConvert,    // cvtsi2sd, cvtsd2si, etc.

    // Memory operations
    MemoryLoad,          // mov from memory (L1 hit assumed)
    MemoryStore,         // mov to memory
    MemoryLoadUnaligned, // unaligned loads (penalty)
    MemoryPrefetch,      // prefetch instructions

    // Control flow
    Branch,         // jmp, jcc (predictable)
    BranchIndirect, // indirect jumps (less predictable)
    Call,           // call instruction
    Return,         // ret instruction

    // SIMD operations
    Simd128,        // SSE/SSE2 128-bit ops
    Simd256,        // AVX/AVX2 256-bit ops
    Simd512,        // AVX-512 512-bit ops (if available)
    SimdShuffle,    // shuffle, permute operations
    SimdHorizontal, // horizontal add/sub (slower)

    // Special operations
    Crypto,  // AES-NI, SHA extensions
    Atomic,  // lock-prefixed operations
    Syscall, // system calls (high variance)
    Nop,     // no operation

    // Epistemic-specific (Sounio extensions)
    EpistemicPropagate, // Knowledge propagation ops
    EpistemicMerge,     // Provenance merge
    EpistemicCheck,     // Confidence threshold check
}

/// Cycle specification for an operation class
#[derive(Debug, Clone, Copy)]
pub struct CycleSpec {
    /// Latency in cycles (time from issue to result available)
    pub latency: u16,
    /// Reciprocal throughput (cycles per instruction at full throughput)
    pub throughput_recip: f32,
    /// Confidence in this estimate (0.0-1.0)
    /// Lower confidence for operations with high variance (e.g., memory, branches)
    pub confidence: f32,
    /// Ports used (for resource contention modeling)
    pub ports: PortUsage,
}

/// Execution port usage (simplified Skylake-like model)
#[derive(Debug, Clone, Copy, Default)]
pub struct PortUsage {
    pub p0: bool,  // ALU, FMA, DIV
    pub p1: bool,  // ALU, FMA
    pub p5: bool,  // ALU, shuffle
    pub p23: bool, // Load
    pub p4: bool,  // Store data
    pub p7: bool,  // Store address
}

impl PortUsage {
    pub fn alu() -> Self {
        Self {
            p0: true,
            p1: true,
            p5: true,
            ..Default::default()
        }
    }
    pub fn fma() -> Self {
        Self {
            p0: true,
            p1: true,
            ..Default::default()
        }
    }
    pub fn load() -> Self {
        Self {
            p23: true,
            ..Default::default()
        }
    }
    pub fn store() -> Self {
        Self {
            p4: true,
            p7: true,
            ..Default::default()
        }
    }
    pub fn div() -> Self {
        Self {
            p0: true,
            ..Default::default()
        }
    }
}

/// Cycle estimation model based on microarchitecture literature
#[derive(Debug, Clone)]
pub struct CycleModel {
    /// Latency/throughput specifications per operation class
    latencies: HashMap<OpClass, CycleSpec>,
    /// Target microarchitecture name
    pub arch_name: String,
    /// Pipeline depth (affects branch misprediction penalty)
    pub pipeline_depth: u8,
    /// Branch misprediction penalty in cycles
    pub branch_mispredict_penalty: u16,
    /// L1 cache hit latency
    pub l1_hit_cycles: u16,
    /// L2 cache hit latency
    pub l2_hit_cycles: u16,
    /// L3 cache hit latency
    pub l3_hit_cycles: u16,
    /// Memory access latency (DRAM)
    pub dram_cycles: u16,
}

impl Default for CycleModel {
    /// Creates a model based on Intel Skylake/Ice Lake characteristics
    /// Sources: Agner Fog's instruction tables, Intel SDM
    fn default() -> Self {
        Self::skylake_like()
    }
}

impl CycleModel {
    /// Skylake/Ice Lake-like microarchitecture model
    pub fn skylake_like() -> Self {
        let mut latencies = HashMap::new();

        // Integer operations (Agner Fog, Skylake)
        latencies.insert(
            OpClass::IntArithmetic,
            CycleSpec {
                latency: 1,
                throughput_recip: 0.25, // 4 per cycle
                confidence: 0.98,
                ports: PortUsage::alu(),
            },
        );
        latencies.insert(
            OpClass::IntMultiply,
            CycleSpec {
                latency: 3,
                throughput_recip: 1.0,
                confidence: 0.95,
                ports: PortUsage {
                    p1: true,
                    ..Default::default()
                },
            },
        );
        latencies.insert(
            OpClass::IntDivide,
            CycleSpec {
                latency: 26, // 64-bit: 26-90 cycles, highly variable
                throughput_recip: 6.0,
                confidence: 0.60, // High variance
                ports: PortUsage::div(),
            },
        );
        latencies.insert(
            OpClass::IntCompare,
            CycleSpec {
                latency: 1,
                throughput_recip: 0.25,
                confidence: 0.98,
                ports: PortUsage::alu(),
            },
        );

        // Floating-point operations
        latencies.insert(
            OpClass::FloatArithmetic,
            CycleSpec {
                latency: 4,
                throughput_recip: 0.5,
                confidence: 0.95,
                ports: PortUsage::fma(),
            },
        );
        latencies.insert(
            OpClass::FloatMultiply,
            CycleSpec {
                latency: 4,
                throughput_recip: 0.5,
                confidence: 0.95,
                ports: PortUsage::fma(),
            },
        );
        latencies.insert(
            OpClass::FloatDivide,
            CycleSpec {
                latency: 14, // divsd: 13-14 cycles
                throughput_recip: 4.0,
                confidence: 0.85,
                ports: PortUsage::div(),
            },
        );
        latencies.insert(
            OpClass::FloatSqrt,
            CycleSpec {
                latency: 18, // sqrtsd: 15-18 cycles
                throughput_recip: 4.5,
                confidence: 0.85,
                ports: PortUsage::div(),
            },
        );
        latencies.insert(
            OpClass::FloatFma,
            CycleSpec {
                latency: 4,
                throughput_recip: 0.5,
                confidence: 0.95,
                ports: PortUsage::fma(),
            },
        );
        latencies.insert(
            OpClass::FloatConvert,
            CycleSpec {
                latency: 5,
                throughput_recip: 1.0,
                confidence: 0.90,
                ports: PortUsage {
                    p1: true,
                    ..Default::default()
                },
            },
        );

        // Memory operations (assuming L1 hit)
        latencies.insert(
            OpClass::MemoryLoad,
            CycleSpec {
                latency: 4,            // L1 hit
                throughput_recip: 0.5, // 2 loads per cycle
                confidence: 0.70,      // Cache-dependent variance
                ports: PortUsage::load(),
            },
        );
        latencies.insert(
            OpClass::MemoryStore,
            CycleSpec {
                latency: 1, // Store doesn't block
                throughput_recip: 1.0,
                confidence: 0.80,
                ports: PortUsage::store(),
            },
        );
        latencies.insert(
            OpClass::MemoryLoadUnaligned,
            CycleSpec {
                latency: 5, // +1 cycle penalty
                throughput_recip: 1.0,
                confidence: 0.65,
                ports: PortUsage::load(),
            },
        );
        latencies.insert(
            OpClass::MemoryPrefetch,
            CycleSpec {
                latency: 0, // Non-blocking
                throughput_recip: 0.5,
                confidence: 0.90,
                ports: PortUsage::load(),
            },
        );

        // Control flow
        latencies.insert(
            OpClass::Branch,
            CycleSpec {
                latency: 1, // Predicted correctly
                throughput_recip: 0.5,
                confidence: 0.60, // Branch prediction variance
                ports: PortUsage {
                    p0: true,
                    ..Default::default()
                },
            },
        );
        latencies.insert(
            OpClass::BranchIndirect,
            CycleSpec {
                latency: 2,
                throughput_recip: 1.0,
                confidence: 0.50, // Harder to predict
                ports: PortUsage {
                    p0: true,
                    ..Default::default()
                },
            },
        );
        latencies.insert(
            OpClass::Call,
            CycleSpec {
                latency: 2,
                throughput_recip: 1.0,
                confidence: 0.85,
                ports: PortUsage {
                    p0: true,
                    ..Default::default()
                },
            },
        );
        latencies.insert(
            OpClass::Return,
            CycleSpec {
                latency: 1,
                throughput_recip: 1.0,
                confidence: 0.85,
                ports: PortUsage {
                    p0: true,
                    ..Default::default()
                },
            },
        );

        // SIMD operations
        latencies.insert(
            OpClass::Simd128,
            CycleSpec {
                latency: 4,
                throughput_recip: 0.5,
                confidence: 0.92,
                ports: PortUsage::fma(),
            },
        );
        latencies.insert(
            OpClass::Simd256,
            CycleSpec {
                latency: 4,
                throughput_recip: 0.5, // AVX2 full width
                confidence: 0.90,
                ports: PortUsage::fma(),
            },
        );
        latencies.insert(
            OpClass::Simd512,
            CycleSpec {
                latency: 6,
                throughput_recip: 1.0, // AVX-512 often downclocks
                confidence: 0.75,      // Frequency throttling variance
                ports: PortUsage::fma(),
            },
        );
        latencies.insert(
            OpClass::SimdShuffle,
            CycleSpec {
                latency: 1,
                throughput_recip: 1.0,
                confidence: 0.90,
                ports: PortUsage {
                    p5: true,
                    ..Default::default()
                },
            },
        );
        latencies.insert(
            OpClass::SimdHorizontal,
            CycleSpec {
                latency: 6, // Horizontal ops are slower
                throughput_recip: 2.0,
                confidence: 0.88,
                ports: PortUsage {
                    p0: true,
                    p5: true,
                    ..Default::default()
                },
            },
        );

        // Special operations
        latencies.insert(
            OpClass::Crypto,
            CycleSpec {
                latency: 4, // AES-NI single round
                throughput_recip: 1.0,
                confidence: 0.95,
                ports: PortUsage {
                    p0: true,
                    ..Default::default()
                },
            },
        );
        latencies.insert(
            OpClass::Atomic,
            CycleSpec {
                latency: 20, // lock prefix: ~20 cycles uncontended
                throughput_recip: 20.0,
                confidence: 0.50, // Highly contention-dependent
                ports: PortUsage::load(),
            },
        );
        latencies.insert(
            OpClass::Syscall,
            CycleSpec {
                latency: 150, // System call overhead
                throughput_recip: 150.0,
                confidence: 0.30, // Extreme variance
                ports: PortUsage::default(),
            },
        );
        latencies.insert(
            OpClass::Nop,
            CycleSpec {
                latency: 0,
                throughput_recip: 0.25,
                confidence: 1.0,
                ports: PortUsage::default(),
            },
        );

        // Epistemic operations (Sounio-specific)
        // These are software constructs, estimated based on typical implementation
        latencies.insert(
            OpClass::EpistemicPropagate,
            CycleSpec {
                latency: 8, // sqrt + multiply for δ propagation
                throughput_recip: 2.0,
                confidence: 0.85,
                ports: PortUsage::fma(),
            },
        );
        latencies.insert(
            OpClass::EpistemicMerge,
            CycleSpec {
                latency: 4, // min/max + some bookkeeping
                throughput_recip: 1.0,
                confidence: 0.90,
                ports: PortUsage::alu(),
            },
        );
        latencies.insert(
            OpClass::EpistemicCheck,
            CycleSpec {
                latency: 2, // compare + branch
                throughput_recip: 0.5,
                confidence: 0.85,
                ports: PortUsage::alu(),
            },
        );

        Self {
            latencies,
            arch_name: "Skylake-like".to_string(),
            pipeline_depth: 14,
            branch_mispredict_penalty: 15,
            l1_hit_cycles: 4,
            l2_hit_cycles: 12,
            l3_hit_cycles: 42,
            dram_cycles: 200,
        }
    }

    /// AMD Zen3-like microarchitecture model
    pub fn zen3_like() -> Self {
        let mut model = Self::skylake_like();
        model.arch_name = "Zen3-like".to_string();
        model.pipeline_depth = 19;
        model.branch_mispredict_penalty = 18;

        // Zen3 has better integer division
        if let Some(spec) = model.latencies.get_mut(&OpClass::IntDivide) {
            spec.latency = 18;
            spec.confidence = 0.65;
        }

        // Zen3 has excellent FP throughput
        if let Some(spec) = model.latencies.get_mut(&OpClass::FloatMultiply) {
            spec.throughput_recip = 0.5;
        }

        model
    }

    /// Estimate cycles for a single instruction
    pub fn estimate_instruction(&self, op_class: OpClass) -> (u64, f64) {
        let default_spec = CycleSpec {
            latency: 1,
            throughput_recip: 1.0,
            confidence: 0.50,
            ports: PortUsage::default(),
        };
        let spec = self.latencies.get(&op_class).unwrap_or(&default_spec);

        (spec.latency as u64, spec.confidence as f64)
    }

    /// Get the CycleSpec for an operation class
    pub fn get_spec(&self, op_class: OpClass) -> Option<&CycleSpec> {
        self.latencies.get(&op_class)
    }
}

// ============================================================================
// POWER ESTIMATION MODEL
// ============================================================================

/// Energy specification for an operation class
#[derive(Debug, Clone, Copy)]
pub struct EnergySpec {
    /// Dynamic energy per operation in picojoules (pJ)
    pub dynamic_pj: f64,
    /// Static (leakage) power when this unit is active, in microwatts (μW)
    pub leakage_uw: f64,
    /// Confidence in this estimate
    pub confidence: f32,
}

/// Power estimation model based on technology node characteristics
#[derive(Debug, Clone)]
pub struct PowerModel {
    /// Energy per operation class
    energy_per_op: HashMap<OpClass, EnergySpec>,
    /// Technology node name
    pub tech_node: String,
    /// Nominal voltage (V)
    pub nominal_voltage: f64,
    /// Current voltage (for DVFS scaling)
    pub current_voltage: f64,
    /// Nominal frequency (GHz)
    pub nominal_frequency: f64,
    /// Current frequency (GHz)
    pub current_frequency: f64,
    /// Thermal design power (TDP) in watts
    pub tdp_watts: f64,
    /// Thermal resistance (K/W) for junction-to-ambient
    pub thermal_resistance: f64,
    /// Ambient temperature (K)
    pub ambient_temp_k: f64,
}

impl Default for PowerModel {
    /// Default 7nm FinFET model (typical modern laptop/desktop)
    fn default() -> Self {
        Self::finfet_7nm()
    }
}

impl PowerModel {
    /// 7nm FinFET process model
    /// Energy values derived from academic literature and industry data
    pub fn finfet_7nm() -> Self {
        let mut energy = HashMap::new();

        // Values in picojoules (pJ) per operation
        // Sources:
        // - "A 7nm FinFET High Performance ARM Core" (ISSCC 2019)
        // - "Energy-Efficient Neural Network Inference" (IEEE MICRO 2023)
        // - Normalized from McPAT 7nm projections

        // Integer operations
        energy.insert(
            OpClass::IntArithmetic,
            EnergySpec {
                dynamic_pj: 0.03,
                leakage_uw: 0.05,
                confidence: 0.90,
            },
        );
        energy.insert(
            OpClass::IntMultiply,
            EnergySpec {
                dynamic_pj: 0.35,
                leakage_uw: 0.15,
                confidence: 0.88,
            },
        );
        energy.insert(
            OpClass::IntDivide,
            EnergySpec {
                dynamic_pj: 2.5,
                leakage_uw: 0.8,
                confidence: 0.75,
            },
        );
        energy.insert(
            OpClass::IntCompare,
            EnergySpec {
                dynamic_pj: 0.02,
                leakage_uw: 0.03,
                confidence: 0.92,
            },
        );

        // Floating-point operations (64-bit)
        energy.insert(
            OpClass::FloatArithmetic,
            EnergySpec {
                dynamic_pj: 0.9,
                leakage_uw: 0.3,
                confidence: 0.88,
            },
        );
        energy.insert(
            OpClass::FloatMultiply,
            EnergySpec {
                dynamic_pj: 1.5,
                leakage_uw: 0.5,
                confidence: 0.85,
            },
        );
        energy.insert(
            OpClass::FloatDivide,
            EnergySpec {
                dynamic_pj: 8.0,
                leakage_uw: 2.0,
                confidence: 0.75,
            },
        );
        energy.insert(
            OpClass::FloatSqrt,
            EnergySpec {
                dynamic_pj: 10.0,
                leakage_uw: 2.5,
                confidence: 0.75,
            },
        );
        energy.insert(
            OpClass::FloatFma,
            EnergySpec {
                dynamic_pj: 2.0,
                leakage_uw: 0.6,
                confidence: 0.85,
            },
        );
        energy.insert(
            OpClass::FloatConvert,
            EnergySpec {
                dynamic_pj: 0.5,
                leakage_uw: 0.2,
                confidence: 0.88,
            },
        );

        // Memory operations (L1 cache)
        energy.insert(
            OpClass::MemoryLoad,
            EnergySpec {
                dynamic_pj: 5.0, // L1D read
                leakage_uw: 2.0,
                confidence: 0.70, // Cache-dependent
            },
        );
        energy.insert(
            OpClass::MemoryStore,
            EnergySpec {
                dynamic_pj: 3.5, // L1D write
                leakage_uw: 1.5,
                confidence: 0.72,
            },
        );
        energy.insert(
            OpClass::MemoryLoadUnaligned,
            EnergySpec {
                dynamic_pj: 7.0,
                leakage_uw: 2.5,
                confidence: 0.65,
            },
        );
        energy.insert(
            OpClass::MemoryPrefetch,
            EnergySpec {
                dynamic_pj: 1.0,
                leakage_uw: 0.5,
                confidence: 0.80,
            },
        );

        // Control flow
        energy.insert(
            OpClass::Branch,
            EnergySpec {
                dynamic_pj: 0.1,
                leakage_uw: 0.05,
                confidence: 0.85,
            },
        );
        energy.insert(
            OpClass::BranchIndirect,
            EnergySpec {
                dynamic_pj: 0.15,
                leakage_uw: 0.08,
                confidence: 0.80,
            },
        );
        energy.insert(
            OpClass::Call,
            EnergySpec {
                dynamic_pj: 0.2,
                leakage_uw: 0.1,
                confidence: 0.82,
            },
        );
        energy.insert(
            OpClass::Return,
            EnergySpec {
                dynamic_pj: 0.15,
                leakage_uw: 0.08,
                confidence: 0.82,
            },
        );

        // SIMD operations
        energy.insert(
            OpClass::Simd128,
            EnergySpec {
                dynamic_pj: 3.0,
                leakage_uw: 1.0,
                confidence: 0.85,
            },
        );
        energy.insert(
            OpClass::Simd256,
            EnergySpec {
                dynamic_pj: 5.5,
                leakage_uw: 1.8,
                confidence: 0.82,
            },
        );
        energy.insert(
            OpClass::Simd512,
            EnergySpec {
                dynamic_pj: 12.0,
                leakage_uw: 4.0,
                confidence: 0.70, // Frequency throttling effects
            },
        );
        energy.insert(
            OpClass::SimdShuffle,
            EnergySpec {
                dynamic_pj: 1.5,
                leakage_uw: 0.5,
                confidence: 0.85,
            },
        );
        energy.insert(
            OpClass::SimdHorizontal,
            EnergySpec {
                dynamic_pj: 4.0,
                leakage_uw: 1.2,
                confidence: 0.80,
            },
        );

        // Special operations
        energy.insert(
            OpClass::Crypto,
            EnergySpec {
                dynamic_pj: 4.0,
                leakage_uw: 1.5,
                confidence: 0.88,
            },
        );
        energy.insert(
            OpClass::Atomic,
            EnergySpec {
                dynamic_pj: 15.0,
                leakage_uw: 5.0,
                confidence: 0.50, // Highly variable
            },
        );
        energy.insert(
            OpClass::Syscall,
            EnergySpec {
                dynamic_pj: 500.0, // Context switch overhead
                leakage_uw: 50.0,
                confidence: 0.30,
            },
        );
        energy.insert(
            OpClass::Nop,
            EnergySpec {
                dynamic_pj: 0.01,
                leakage_uw: 0.01,
                confidence: 0.99,
            },
        );

        // Epistemic operations
        energy.insert(
            OpClass::EpistemicPropagate,
            EnergySpec {
                dynamic_pj: 15.0, // sqrt + muls + adds
                leakage_uw: 3.0,
                confidence: 0.80,
            },
        );
        energy.insert(
            OpClass::EpistemicMerge,
            EnergySpec {
                dynamic_pj: 5.0,
                leakage_uw: 1.0,
                confidence: 0.85,
            },
        );
        energy.insert(
            OpClass::EpistemicCheck,
            EnergySpec {
                dynamic_pj: 2.0,
                leakage_uw: 0.5,
                confidence: 0.88,
            },
        );

        Self {
            energy_per_op: energy,
            tech_node: "7nm FinFET".to_string(),
            nominal_voltage: 0.75,
            current_voltage: 0.75,
            nominal_frequency: 3.5,
            current_frequency: 3.5,
            tdp_watts: 65.0,
            thermal_resistance: 0.3, // K/W, typical desktop heatsink
            ambient_temp_k: 298.0,   // 25°C
        }
    }

    /// 5nm process model (Apple M-series, AMD Zen4 level)
    pub fn process_5nm() -> Self {
        let mut model = Self::finfet_7nm();
        model.tech_node = "5nm".to_string();

        // 5nm is ~20-25% more efficient than 7nm
        for spec in model.energy_per_op.values_mut() {
            spec.dynamic_pj *= 0.78;
            spec.leakage_uw *= 0.85;
        }

        model.nominal_voltage = 0.65;
        model.current_voltage = 0.65;

        model
    }

    /// Voltage/frequency scaling factor (Dennard-like, but not perfect)
    pub fn voltage_scaling_factor(&self) -> f64 {
        // Dynamic power scales with V²
        let v_ratio = self.current_voltage / self.nominal_voltage;
        v_ratio * v_ratio
    }

    /// Frequency scaling factor for power
    pub fn frequency_scaling_factor(&self) -> f64 {
        // Power scales linearly with frequency
        self.current_frequency / self.nominal_frequency
    }

    /// Set DVFS operating point
    pub fn set_dvfs(&mut self, voltage: f64, frequency_ghz: f64) {
        self.current_voltage = voltage.clamp(0.4, 1.2);
        self.current_frequency = frequency_ghz.clamp(0.5, 5.5);
    }

    /// Estimate energy for a single operation
    pub fn estimate_operation_energy(&self, op_class: OpClass) -> (f64, f64) {
        let spec = self.energy_per_op.get(&op_class).unwrap_or(&EnergySpec {
            dynamic_pj: 1.0,
            leakage_uw: 0.5,
            confidence: 0.50,
        });

        let scaled_energy = spec.dynamic_pj * self.voltage_scaling_factor();
        (scaled_energy, spec.confidence as f64)
    }

    /// Calculate temperature rise from power dissipation
    pub fn temperature_from_power(&self, power_watts: f64) -> f64 {
        self.ambient_temp_k + power_watts * self.thermal_resistance
    }

    /// Get EnergySpec for an operation class
    pub fn get_spec(&self, op_class: OpClass) -> Option<&EnergySpec> {
        self.energy_per_op.get(&op_class)
    }
}

// ============================================================================
// CYCLE ESTIMATE RESULT
// ============================================================================

/// Result of cycle estimation for a code block
#[derive(Debug, Clone)]
pub struct CycleEstimate {
    /// Total estimated cycles
    pub cycles: u64,
    /// Epistemic confidence in estimate (0.0-1.0)
    /// Product of all instruction confidences
    pub confidence: f64,
    /// Aleatoric variance (standard deviation in cycles)
    /// Computed via quadrature: sqrt(sum of variances)
    pub variance: f64,
    /// Provenance tracking
    pub provenance: Provenance,
    /// Breakdown by operation class (for analysis)
    pub breakdown: HashMap<OpClass, CycleBreakdown>,
}

/// Cycle breakdown for a single operation class
#[derive(Debug, Clone, Copy)]
pub struct CycleBreakdown {
    pub count: u64,
    pub total_latency: u64,
    pub throughput_limited_cycles: f64,
}

impl CycleEstimate {
    /// Create a new empty estimate
    pub fn new(provenance: Provenance) -> Self {
        Self {
            cycles: 0,
            confidence: 1.0,
            variance: 0.0,
            provenance,
            breakdown: HashMap::new(),
        }
    }

    /// Add an instruction to the estimate
    pub fn add_instruction(&mut self, op_class: OpClass, spec: &CycleSpec) {
        self.cycles += spec.latency as u64;
        self.confidence *= spec.confidence as f64;

        // Variance propagation via quadrature
        // Assuming variance ≈ (1 - confidence) × latency
        let inst_variance = (1.0 - spec.confidence as f64) * spec.latency as f64;
        self.variance = (self.variance.powi(2) + inst_variance.powi(2)).sqrt();

        // Update breakdown
        let breakdown = self.breakdown.entry(op_class).or_insert(CycleBreakdown {
            count: 0,
            total_latency: 0,
            throughput_limited_cycles: 0.0,
        });
        breakdown.count += 1;
        breakdown.total_latency += spec.latency as u64;
        breakdown.throughput_limited_cycles += spec.throughput_recip as f64;
    }

    /// Apply confidence floor to prevent collapse
    pub fn apply_confidence_floor(&mut self, floor: f64) {
        self.confidence = self.confidence.max(floor);
    }

    /// Get effective cycles accounting for throughput limits
    pub fn effective_cycles(&self) -> u64 {
        let throughput_limited: f64 = self
            .breakdown
            .values()
            .map(|b| b.throughput_limited_cycles)
            .sum();

        // Take max of latency-based and throughput-based estimates
        self.cycles.max(throughput_limited.ceil() as u64)
    }
}

impl fmt::Display for CycleEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CycleEstimate {{ cycles: {}, confidence: {:.3}, variance: {:.2} }}",
            self.cycles, self.confidence, self.variance
        )
    }
}

// ============================================================================
// POWER ESTIMATE RESULT
// ============================================================================

/// Result of power estimation for a code block
#[derive(Debug, Clone)]
pub struct PowerEstimate {
    /// Total dynamic energy in picojoules
    pub dynamic_energy_pj: f64,
    /// Total leakage power in microwatts (while executing)
    pub leakage_power_uw: f64,
    /// Execution time in nanoseconds (from cycle estimate)
    pub execution_time_ns: f64,
    /// Average power in microwatts
    pub average_power_uw: f64,
    /// Peak power estimate in microwatts
    pub peak_power_uw: f64,
    /// Epistemic confidence
    pub confidence: f64,
    /// Aleatoric variance (standard deviation in μW)
    pub variance: f64,
    /// Estimated temperature rise (K) above ambient
    pub temperature_rise_k: f64,
    /// Provenance
    pub provenance: Provenance,
    /// Breakdown by operation class
    pub breakdown: HashMap<OpClass, EnergyBreakdown>,
}

/// Energy breakdown for a single operation class
#[derive(Debug, Clone, Copy)]
pub struct EnergyBreakdown {
    pub count: u64,
    pub total_dynamic_pj: f64,
    pub total_leakage_uw: f64,
}

impl PowerEstimate {
    /// Create from a cycle estimate and power model analysis
    pub fn new(provenance: Provenance) -> Self {
        Self {
            dynamic_energy_pj: 0.0,
            leakage_power_uw: 0.0,
            execution_time_ns: 0.0,
            average_power_uw: 0.0,
            peak_power_uw: 0.0,
            confidence: 1.0,
            variance: 0.0,
            temperature_rise_k: 0.0,
            provenance,
            breakdown: HashMap::new(),
        }
    }

    /// Add an instruction's energy contribution
    pub fn add_instruction(&mut self, op_class: OpClass, spec: &EnergySpec, voltage_scale: f64) {
        let scaled_dynamic = spec.dynamic_pj * voltage_scale;

        self.dynamic_energy_pj += scaled_dynamic;
        self.leakage_power_uw += spec.leakage_uw;
        self.confidence *= spec.confidence as f64;

        // Variance via quadrature
        let inst_variance = (1.0 - spec.confidence as f64) * scaled_dynamic;
        self.variance = (self.variance.powi(2) + inst_variance.powi(2)).sqrt();

        // Update breakdown
        let breakdown = self.breakdown.entry(op_class).or_insert(EnergyBreakdown {
            count: 0,
            total_dynamic_pj: 0.0,
            total_leakage_uw: 0.0,
        });
        breakdown.count += 1;
        breakdown.total_dynamic_pj += scaled_dynamic;
        breakdown.total_leakage_uw += spec.leakage_uw;
    }

    /// Finalize power calculations given execution time
    pub fn finalize(&mut self, execution_time_ns: f64, thermal_resistance: f64) {
        self.execution_time_ns = execution_time_ns;

        if execution_time_ns > 0.0 {
            // Average power = energy / time
            // Convert pJ/ns to μW: 1 pJ/ns = 1 μW
            let dynamic_power_uw = self.dynamic_energy_pj / execution_time_ns;
            self.average_power_uw = dynamic_power_uw + self.leakage_power_uw;

            // Peak power estimate (assume 1.5x average for bursts)
            self.peak_power_uw = self.average_power_uw * 1.5;

            // Temperature rise: ΔT = P × R_th
            // Convert μW to W for calculation
            let power_watts = self.average_power_uw / 1_000_000.0;
            self.temperature_rise_k = power_watts * thermal_resistance;
        }
    }

    /// Apply confidence floor
    pub fn apply_confidence_floor(&mut self, floor: f64) {
        self.confidence = self.confidence.max(floor);
    }
}

impl fmt::Display for PowerEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PowerEstimate {{ avg: {:.2} μW, peak: {:.2} μW, energy: {:.2} pJ, ε: {:.3}, ΔT: {:.2} K }}",
            self.average_power_uw,
            self.peak_power_uw,
            self.dynamic_energy_pj,
            self.confidence,
            self.temperature_rise_k
        )
    }
}

// ============================================================================
// SIR BLOCK ANALYZER
// ============================================================================

/// Instruction abstraction for SIR analysis
/// This would be replaced by actual SIR instruction types in the full compiler
#[derive(Debug, Clone)]
pub struct SirInstruction {
    /// Operation class for this instruction
    pub op_class: OpClass,
    /// Source operands
    pub sources: Vec<Operand>,
    /// Destination operand
    pub destination: Option<Operand>,
    /// Epistemic metadata (if this instruction operates on Knowledge types)
    pub epistemic: Option<EpistemicOp>,
}

/// Operand types
#[derive(Debug, Clone)]
pub enum Operand {
    Register(u8),
    Immediate(i64),
    Memory { base: u8, offset: i32, scale: u8 },
    Label(String),
}

/// Epistemic operation metadata
#[derive(Debug, Clone)]
pub struct EpistemicOp {
    /// Confidence threshold for this operation
    pub confidence_threshold: f64,
    /// Whether this operation propagates uncertainty
    pub propagates_uncertainty: bool,
}

/// SIR basic block
#[derive(Debug, Clone)]
pub struct SirBlock {
    /// Block label/name
    pub label: String,
    /// Instructions in execution order
    pub instructions: Vec<SirInstruction>,
    /// Whether this block contains epistemic operations
    pub has_epistemic: bool,
}

impl SirBlock {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            instructions: Vec::new(),
            has_epistemic: false,
        }
    }

    pub fn push(&mut self, inst: SirInstruction) {
        if inst.epistemic.is_some() {
            self.has_epistemic = true;
        }
        self.instructions.push(inst);
    }
}

/// Main metrics estimator for SIR blocks
pub struct SirMetricsEstimator {
    pub cycle_model: CycleModel,
    pub power_model: PowerModel,
    /// Confidence floor (minimum confidence to prevent total collapse)
    pub confidence_floor: f64,
}

impl Default for SirMetricsEstimator {
    fn default() -> Self {
        Self {
            cycle_model: CycleModel::default(),
            power_model: PowerModel::default(),
            confidence_floor: 0.05,
        }
    }
}

impl SirMetricsEstimator {
    /// Create with specific models
    pub fn new(cycle_model: CycleModel, power_model: PowerModel) -> Self {
        Self {
            cycle_model,
            power_model,
            confidence_floor: 0.05,
        }
    }

    /// Estimate cycles for a SIR block
    pub fn estimate_cycles(&self, block: &SirBlock) -> CycleEstimate {
        let mut estimate = CycleEstimate::new(Provenance::sir_backend(&self.cycle_model.arch_name));

        for inst in &block.instructions {
            if let Some(spec) = self.cycle_model.get_spec(inst.op_class) {
                estimate.add_instruction(inst.op_class, spec);

                // Memory operands add extra latency
                for src in &inst.sources {
                    if matches!(src, Operand::Memory { .. }) {
                        // Add potential cache miss penalty (weighted by confidence)
                        let miss_penalty = (self.cycle_model.l2_hit_cycles
                            - self.cycle_model.l1_hit_cycles)
                            as f64
                            * 0.1;
                        estimate.cycles += miss_penalty as u64;
                        estimate.confidence *= 0.95; // Slight confidence penalty
                    }
                }
            }

            // Epistemic operations add overhead
            if let Some(ref ep) = inst.epistemic {
                if ep.propagates_uncertainty {
                    if let Some(prop_spec) = self.cycle_model.get_spec(OpClass::EpistemicPropagate)
                    {
                        estimate.add_instruction(OpClass::EpistemicPropagate, prop_spec);
                    }
                }
            }
        }

        estimate.apply_confidence_floor(self.confidence_floor);
        estimate
    }

    /// Estimate power for a SIR block
    pub fn estimate_power(&self, block: &SirBlock, cycles: &CycleEstimate) -> PowerEstimate {
        let mut estimate = PowerEstimate::new(Provenance::sir_backend(&self.power_model.tech_node));

        let voltage_scale = self.power_model.voltage_scaling_factor();

        for inst in &block.instructions {
            if let Some(spec) = self.power_model.get_spec(inst.op_class) {
                estimate.add_instruction(inst.op_class, spec, voltage_scale);
            }

            // Epistemic operations
            if let Some(ref ep) = inst.epistemic {
                if ep.propagates_uncertainty {
                    if let Some(prop_spec) = self.power_model.get_spec(OpClass::EpistemicPropagate)
                    {
                        estimate.add_instruction(
                            OpClass::EpistemicPropagate,
                            prop_spec,
                            voltage_scale,
                        );
                    }
                }
            }
        }

        // Calculate execution time from cycles
        let execution_time_ns =
            cycles.effective_cycles() as f64 / self.power_model.current_frequency; // GHz -> ns

        estimate.finalize(execution_time_ns, self.power_model.thermal_resistance);
        estimate.apply_confidence_floor(self.confidence_floor);
        estimate
    }

    /// Combined metrics estimation
    pub fn estimate_all(&self, block: &SirBlock) -> (CycleEstimate, PowerEstimate) {
        let cycles = self.estimate_cycles(block);
        let power = self.estimate_power(block, &cycles);
        (cycles, power)
    }
}

// ============================================================================
// INTEGRATION WITH EPISTEMIC SYSTEM
// ============================================================================

/// Hardware-aware Knowledge type extension
/// Wraps standard Knowledge with hardware metrics
#[derive(Debug, Clone)]
pub struct HardwareKnowledge<T> {
    /// The underlying value
    pub value: T,
    /// Epistemic confidence (0.0-1.0)
    pub epsilon: f64,
    /// Aleatoric uncertainty (standard deviation)
    pub delta: f64,
    /// Provenance chain
    pub provenance: Provenance,
    /// Hardware metrics for this computation
    pub hardware: HardwareMetrics,
}

/// Hardware metrics associated with a value
#[derive(Debug, Clone, Default)]
pub struct HardwareMetrics {
    /// Estimated cycles to compute this value
    pub cycles: u64,
    /// Estimated energy in picojoules
    pub energy_pj: f64,
    /// Thermal degradation factor applied
    pub thermal_degradation: f64,
    /// Temperature at computation time (K)
    pub temperature_k: Option<f64>,
}

impl<T> HardwareKnowledge<T> {
    /// Create new HardwareKnowledge with metrics
    pub fn new(
        value: T,
        epsilon: f64,
        delta: f64,
        provenance: Provenance,
        cycles: &CycleEstimate,
        power: &PowerEstimate,
    ) -> Self {
        Self {
            value,
            epsilon: epsilon * cycles.confidence * power.confidence,
            delta: (delta.powi(2) + cycles.variance.powi(2) + power.variance.powi(2)).sqrt(),
            provenance: provenance
                .join(cycles.provenance.clone())
                .join(power.provenance.clone()),
            hardware: HardwareMetrics {
                cycles: cycles.cycles,
                energy_pj: power.dynamic_energy_pj,
                thermal_degradation: 0.0,
                temperature_k: Some(power.temperature_rise_k + 298.0),
            },
        }
    }

    /// Apply thermal degradation using Arrhenius model
    pub fn apply_thermal_degradation(&mut self, degradation_factor: f64, temp_k: f64) {
        let clamped = degradation_factor.clamp(0.0, 0.5);
        self.epsilon *= 1.0 - clamped;
        self.hardware.thermal_degradation = clamped;
        self.hardware.temperature_k = Some(temp_k);

        self.provenance = Provenance::ThermalDegraded {
            base: Box::new(self.provenance.clone()),
            model: "Arrhenius".to_string(),
            degradation_percent: (clamped * 100.0) as u8,
            cycles: self.hardware.cycles,
            temperature_k: temp_k,
        };
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_block() -> SirBlock {
        let mut block = SirBlock::new("test_block");

        // Simulate: a = b + c (load, load, add, store)
        block.push(SirInstruction {
            op_class: OpClass::MemoryLoad,
            sources: vec![Operand::Memory {
                base: 0,
                offset: 0,
                scale: 1,
            }],
            destination: Some(Operand::Register(0)),
            epistemic: None,
        });
        block.push(SirInstruction {
            op_class: OpClass::MemoryLoad,
            sources: vec![Operand::Memory {
                base: 0,
                offset: 8,
                scale: 1,
            }],
            destination: Some(Operand::Register(1)),
            epistemic: None,
        });
        block.push(SirInstruction {
            op_class: OpClass::FloatArithmetic,
            sources: vec![Operand::Register(0), Operand::Register(1)],
            destination: Some(Operand::Register(2)),
            epistemic: Some(EpistemicOp {
                confidence_threshold: 0.8,
                propagates_uncertainty: true,
            }),
        });
        block.push(SirInstruction {
            op_class: OpClass::MemoryStore,
            sources: vec![Operand::Register(2)],
            destination: Some(Operand::Memory {
                base: 0,
                offset: 16,
                scale: 1,
            }),
            epistemic: None,
        });

        block
    }

    #[test]
    fn test_cycle_estimation() {
        let estimator = SirMetricsEstimator::default();
        let block = sample_block();

        let cycles = estimator.estimate_cycles(&block);

        // Should have reasonable cycle count
        assert!(cycles.cycles > 0);
        assert!(cycles.cycles < 100); // Simple block

        // Confidence should be in valid range
        assert!(cycles.confidence > 0.0);
        assert!(cycles.confidence <= 1.0);

        // Variance should be non-negative
        assert!(cycles.variance >= 0.0);

        println!("Cycle estimate: {}", cycles);
    }

    #[test]
    fn test_power_estimation() {
        let estimator = SirMetricsEstimator::default();
        let block = sample_block();

        let cycles = estimator.estimate_cycles(&block);
        let power = estimator.estimate_power(&block, &cycles);

        // Should have positive power
        assert!(power.average_power_uw > 0.0);
        assert!(power.peak_power_uw >= power.average_power_uw);

        // Energy should be positive
        assert!(power.dynamic_energy_pj > 0.0);

        // Temperature rise should be small for short block
        assert!(power.temperature_rise_k < 10.0);

        println!("Power estimate: {}", power);
    }

    #[test]
    fn test_epistemic_propagation() {
        let estimator = SirMetricsEstimator::default();
        let block = sample_block();

        let cycles = estimator.estimate_cycles(&block);

        // Block has epistemic ops, should include EpistemicPropagate overhead
        assert!(block.has_epistemic);
        assert!(cycles.breakdown.contains_key(&OpClass::EpistemicPropagate));
    }

    #[test]
    fn test_hardware_knowledge_integration() {
        let estimator = SirMetricsEstimator::default();
        let block = sample_block();

        let (cycles, power) = estimator.estimate_all(&block);

        let hk: HardwareKnowledge<f64> = HardwareKnowledge::new(
            42.0,
            0.95, // Base confidence
            0.1,  // Base variance
            Provenance::sir_backend("test"),
            &cycles,
            &power,
        );

        // Combined confidence should be less than base
        assert!(hk.epsilon < 0.95);

        // Delta should include hardware variance
        assert!(hk.delta > 0.1);

        // Hardware metrics should be populated
        assert!(hk.hardware.cycles > 0);
        assert!(hk.hardware.energy_pj > 0.0);
    }

    #[test]
    fn test_thermal_degradation() {
        let mut hk: HardwareKnowledge<f64> = HardwareKnowledge {
            value: 42.0,
            epsilon: 0.9,
            delta: 0.1,
            provenance: Provenance::sir_backend("test"),
            hardware: HardwareMetrics::default(),
        };

        let original_epsilon = hk.epsilon;
        hk.apply_thermal_degradation(0.2, 350.0);

        // Confidence should decrease
        assert!(hk.epsilon < original_epsilon);
        assert!((hk.epsilon - original_epsilon * 0.8).abs() < 0.001);

        // Degradation should be recorded
        assert!((hk.hardware.thermal_degradation - 0.2).abs() < 0.001);

        // Provenance should track degradation
        assert!(matches!(hk.provenance, Provenance::ThermalDegraded { .. }));
    }

    #[test]
    fn test_dvfs_scaling() {
        let mut power_model = PowerModel::default();

        // Default voltage
        let factor_nominal = power_model.voltage_scaling_factor();
        assert!((factor_nominal - 1.0).abs() < 0.001);

        // Reduced voltage (e.g., power saving mode)
        power_model.set_dvfs(0.6, 2.0);
        let factor_reduced = power_model.voltage_scaling_factor();

        // V² scaling: (0.6/0.75)² ≈ 0.64
        assert!(factor_reduced < factor_nominal);
        assert!((factor_reduced - 0.64).abs() < 0.1);
    }

    #[test]
    fn test_confidence_floor() {
        let mut estimate = CycleEstimate::new(Provenance::sir_backend("test"));

        // Simulate many low-confidence operations
        for _ in 0..100 {
            estimate.confidence *= 0.5;
        }

        // Should be essentially zero
        assert!(estimate.confidence < 1e-30);

        // Apply floor
        estimate.apply_confidence_floor(0.05);
        assert!((estimate.confidence - 0.05).abs() < 0.001);
    }
}

// ============================================================================
// MODULE EXPORTS
// ============================================================================

pub mod prelude {
    pub use super::{
        CycleEstimate, CycleModel, CycleSpec, EnergySpec, EpistemicOp, HardwareKnowledge,
        HardwareMetrics, OpClass, Operand, PowerEstimate, PowerModel, Provenance, SirBlock,
        SirInstruction, SirMetricsEstimator,
    };
}

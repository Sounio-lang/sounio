//! Tests for the Sounio Native Backend infrastructure

use sounio::backend::native::{
    metrics::{CycleModel, PowerModel, SirMetricsEstimator, SirBlock, SirInstruction, OpClass, Operand},
    thermal::{ArrheniusModel, ThermalState, apply_degradation_full},
    alloc::{EpistemicAllocator, LiveInterval, VirtReg, RegClass},
    NativeBackend,
};

#[test]
fn test_cycle_model_skylake() {
    let model = CycleModel::skylake_like();
    assert_eq!(model.arch_name, "Skylake-like");
    
    let (cycles, conf) = model.estimate_instruction(OpClass::IntArithmetic);
    assert_eq!(cycles, 1);
    assert!(conf > 0.9);
}

#[test]
fn test_power_model_7nm() {
    let model = PowerModel::finfet_7nm();
    assert_eq!(model.tech_node, "7nm FinFET");
    
    let (energy, conf) = model.estimate_operation_energy(OpClass::IntArithmetic);
    assert!(energy > 0.0);
    assert!(conf > 0.8);
}

#[test]
fn test_thermal_arrhenius() {
    let model = ArrheniusModel::combined_7nm();
    
    // Zero cycles = no degradation
    assert_eq!(model.degradation_factor(0, 350.0), 0.0);
    
    // More cycles = more degradation
    let deg_1m = model.degradation_factor(1_000_000, 350.0);
    let deg_10m = model.degradation_factor(10_000_000, 350.0);
    assert!(deg_10m > deg_1m);
}

#[test]
fn test_epistemic_degradation() {
    let model = ArrheniusModel::default();
    
    let result = apply_degradation_full(0.95, 0.1, &model, 1_000_000, 350.0);
    
    // Confidence should decrease
    assert!(result.epsilon < 0.95);
    assert!(result.epsilon > 0.0);
    
    // Variance should increase
    assert!(result.delta > 0.1);
}

#[test]
fn test_register_allocator() {
    let mut allocator = EpistemicAllocator::new();
    
    let mut intervals = Vec::new();
    for i in 0..5 {
        let mut interval = LiveInterval::new(
            VirtReg(i),
            RegClass::GeneralPurpose,
            (i * 10) as usize,
        );
        interval.end = ((i + 1) * 10) as usize;
        interval.epistemic.confidence = if i < 2 { 0.2 } else { 0.9 };
        intervals.push(interval);
    }
    
    let result = allocator.allocate(intervals);
    
    assert_eq!(result.stats.total_intervals, 5);
    assert!(result.stats.allocated_to_reg > 0);
    
    // Low confidence values should be spilled preferentially
    if result.stats.spilled > 0 {
        assert!(result.stats.avg_spilled_confidence < result.stats.avg_allocated_confidence);
    }
}

#[test]
fn test_metrics_estimator() {
    let estimator = SirMetricsEstimator::default();
    
    let mut block = SirBlock::new("test");
    block.push(SirInstruction {
        op_class: OpClass::MemoryLoad,
        sources: vec![Operand::Memory { base: 0, offset: 0, scale: 1 }],
        destination: Some(Operand::Register(0)),
        epistemic: None,
    });
    block.push(SirInstruction {
        op_class: OpClass::FloatArithmetic,
        sources: vec![Operand::Register(0), Operand::Register(1)],
        destination: Some(Operand::Register(2)),
        epistemic: None,
    });
    
    let (cycles, power) = estimator.estimate_all(&block);
    
    assert!(cycles.cycles > 0);
    assert!(power.average_power_uw > 0.0);
    assert!(cycles.confidence > 0.0);
    assert!(power.confidence > 0.0);
}

#[test]
fn test_native_backend_creation() {
    let backend = NativeBackend::new();
    assert!(backend.config().enable_thermal_tracking);
}

#[test]
fn test_thermal_state_accumulation() {
    let model = ArrheniusModel::default();
    let mut state = ThermalState::with_history();
    
    state.update(&model, 350.0, 1_000_000);
    assert_eq!(state.accumulated_cycles, 1_000_000);
    assert!(state.accumulated_degradation > 0.0);
    
    state.update(&model, 380.0, 500_000);
    assert_eq!(state.peak_temp_k, 380.0);
    assert!(state.accumulated_degradation > 0.0);
}

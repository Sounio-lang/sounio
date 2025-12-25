//! Prefetch Codegen: Generate prefetch instructions from semantic analysis.
//!
//! This module bridges high-level prefetch hints to low-level instructions,
//! supporting multiple backends (LLVM, hardware-specific intrinsics).

use super::access::{AccessPattern, StridePattern};
use super::prefetch::{PrefetchHint, PrefetchPriority};
use super::types::Locality;
use std::collections::HashMap;

/// A prefetch instruction to be emitted.
#[derive(Debug, Clone)]
pub struct PrefetchInstruction {
    /// The address expression to prefetch
    pub address: AddressExpr,
    /// Read or write intent
    pub intent: PrefetchIntent,
    /// Cache level hint
    pub locality: CacheHint,
    /// Whether this is temporal (reused) or non-temporal (streaming)
    pub temporal: bool,
    /// Source location for diagnostics
    pub source_loc: Option<String>,
    /// Reason for this prefetch
    pub reason: String,
}

impl PrefetchInstruction {
    /// Create a new prefetch instruction.
    pub fn new(address: AddressExpr) -> Self {
        Self {
            address,
            intent: PrefetchIntent::Read,
            locality: CacheHint::L2,
            temporal: true,
            source_loc: None,
            reason: String::new(),
        }
    }

    /// Set the intent.
    pub fn with_intent(mut self, intent: PrefetchIntent) -> Self {
        self.intent = intent;
        self
    }

    /// Set the cache hint.
    pub fn with_locality(mut self, locality: CacheHint) -> Self {
        self.locality = locality;
        self
    }

    /// Set as non-temporal (streaming).
    pub fn non_temporal(mut self) -> Self {
        self.temporal = false;
        self
    }

    /// Set the reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// Generate LLVM IR for this instruction.
    pub fn to_llvm_ir(&self) -> String {
        let (rw, locality) = match (&self.intent, &self.locality) {
            (PrefetchIntent::Read, CacheHint::L1) => (0, 3),
            (PrefetchIntent::Read, CacheHint::L2) => (0, 2),
            (PrefetchIntent::Read, CacheHint::L3) => (0, 1),
            (PrefetchIntent::Read, CacheHint::NonTemporal) => (0, 0),
            (PrefetchIntent::Write, CacheHint::L1) => (1, 3),
            (PrefetchIntent::Write, CacheHint::L2) => (1, 2),
            (PrefetchIntent::Write, CacheHint::L3) => (1, 1),
            (PrefetchIntent::Write, CacheHint::NonTemporal) => (1, 0),
        };

        format!(
            "call void @llvm.prefetch(ptr {}, i32 {}, i32 {}, i32 1)",
            self.address.to_llvm(),
            rw,
            locality
        )
    }

    /// Generate x86 assembly for this instruction.
    pub fn to_x86_asm(&self) -> String {
        let addr = self.address.to_asm();

        match (&self.intent, &self.locality) {
            (PrefetchIntent::Read, CacheHint::NonTemporal) => {
                format!("prefetchnta [{}]", addr)
            }
            (PrefetchIntent::Read, CacheHint::L1) => {
                format!("prefetcht0 [{}]", addr)
            }
            (PrefetchIntent::Read, CacheHint::L2) => {
                format!("prefetcht1 [{}]", addr)
            }
            (PrefetchIntent::Read, CacheHint::L3) => {
                format!("prefetcht2 [{}]", addr)
            }
            (PrefetchIntent::Write, _) => {
                format!("prefetchw [{}]", addr)
            }
        }
    }

    /// Generate ARM assembly for this instruction.
    pub fn to_arm_asm(&self) -> String {
        let addr = self.address.to_asm();

        match &self.locality {
            CacheHint::L1 => format!("prfm pldl1keep, [{}]", addr),
            CacheHint::L2 => format!("prfm pldl2keep, [{}]", addr),
            CacheHint::L3 => format!("prfm pldl3keep, [{}]", addr),
            CacheHint::NonTemporal => format!("prfm pldl1strm, [{}]", addr),
        }
    }
}

/// Address expression for prefetching.
#[derive(Debug, Clone)]
pub enum AddressExpr {
    /// Direct register
    Register(String),
    /// Base + offset
    BaseOffset { base: String, offset: i64 },
    /// Base + index * scale + offset
    Indexed {
        base: String,
        index: String,
        scale: u8,
        offset: i64,
    },
    /// Symbol reference
    Symbol(String),
}

impl AddressExpr {
    /// Create from base and offset.
    pub fn base_offset(base: impl Into<String>, offset: i64) -> Self {
        Self::BaseOffset {
            base: base.into(),
            offset,
        }
    }

    /// Create indexed address.
    pub fn indexed(
        base: impl Into<String>,
        index: impl Into<String>,
        scale: u8,
        offset: i64,
    ) -> Self {
        Self::Indexed {
            base: base.into(),
            index: index.into(),
            scale,
            offset,
        }
    }

    /// Convert to LLVM IR representation.
    pub fn to_llvm(&self) -> String {
        match self {
            AddressExpr::Register(r) => format!("%{}", r),
            AddressExpr::BaseOffset { base, offset } => {
                if *offset == 0 {
                    format!("%{}", base)
                } else {
                    format!("getelementptr i8, ptr %{}, i64 {}", base, offset)
                }
            }
            AddressExpr::Indexed {
                base,
                index,
                scale,
                offset,
            } => {
                format!(
                    "getelementptr i8, ptr %{}, i64 (add (mul %{}, {}), {})",
                    base, index, scale, offset
                )
            }
            AddressExpr::Symbol(s) => format!("@{}", s),
        }
    }

    /// Convert to assembly representation.
    pub fn to_asm(&self) -> String {
        match self {
            AddressExpr::Register(r) => r.clone(),
            AddressExpr::BaseOffset { base, offset } => {
                if *offset == 0 {
                    base.clone()
                } else if *offset > 0 {
                    format!("{} + {}", base, offset)
                } else {
                    format!("{} - {}", base, -offset)
                }
            }
            AddressExpr::Indexed {
                base,
                index,
                scale,
                offset,
            } => {
                let mut s = format!("{} + {} * {}", base, index, scale);
                if *offset > 0 {
                    s.push_str(&format!(" + {}", offset));
                } else if *offset < 0 {
                    s.push_str(&format!(" - {}", -offset));
                }
                s
            }
            AddressExpr::Symbol(s) => s.clone(),
        }
    }

    /// Add an offset to this address.
    pub fn add_offset(&self, additional: i64) -> Self {
        match self {
            AddressExpr::Register(r) => AddressExpr::BaseOffset {
                base: r.clone(),
                offset: additional,
            },
            AddressExpr::BaseOffset { base, offset } => AddressExpr::BaseOffset {
                base: base.clone(),
                offset: offset + additional,
            },
            AddressExpr::Indexed {
                base,
                index,
                scale,
                offset,
            } => AddressExpr::Indexed {
                base: base.clone(),
                index: index.clone(),
                scale: *scale,
                offset: offset + additional,
            },
            AddressExpr::Symbol(s) => AddressExpr::BaseOffset {
                base: s.clone(),
                offset: additional,
            },
        }
    }
}

/// Prefetch intent (read or write).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchIntent {
    /// Data will be read
    Read,
    /// Data will be written (exclusive access)
    Write,
}

/// Cache level hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheHint {
    /// L1 cache (closest to CPU)
    L1,
    /// L2 cache
    L2,
    /// L3 cache (LLC)
    L3,
    /// Non-temporal (streaming, bypass cache)
    NonTemporal,
}

impl From<Locality> for CacheHint {
    fn from(loc: Locality) -> Self {
        match loc {
            Locality::Register | Locality::L1 => CacheHint::L1,
            Locality::L2 => CacheHint::L2,
            Locality::L3 => CacheHint::L3,
            _ => CacheHint::NonTemporal,
        }
    }
}

impl From<PrefetchPriority> for CacheHint {
    fn from(priority: PrefetchPriority) -> Self {
        match priority {
            PrefetchPriority::Critical | PrefetchPriority::High => CacheHint::L1,
            PrefetchPriority::Medium => CacheHint::L2,
            PrefetchPriority::Low => CacheHint::L3,
            PrefetchPriority::None => CacheHint::NonTemporal,
        }
    }
}

/// The prefetch codegen engine.
pub struct PrefetchCodegen {
    /// Target architecture
    target: Target,
    /// Generated instructions
    instructions: Vec<PrefetchInstruction>,
    /// Variable to register mapping
    registers: HashMap<String, String>,
    /// Current instruction counter
    counter: usize,
    /// Configuration
    config: CodegenConfig,
}

/// Target architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// x86-64 with SSE/AVX prefetch
    X86_64,
    /// ARM64 with PRFM
    Arm64,
    /// Generic LLVM IR
    LLVM,
    /// RISC-V
    RiscV,
}

/// Codegen configuration.
#[derive(Debug, Clone)]
pub struct CodegenConfig {
    /// Maximum prefetch distance (in cache lines)
    pub max_distance: usize,
    /// Minimum priority to emit prefetch
    pub min_priority: PrefetchPriority,
    /// Whether to emit non-temporal hints
    pub use_non_temporal: bool,
    /// Whether to emit write prefetches
    pub prefetch_writes: bool,
    /// Cache line size
    pub cache_line_size: usize,
}

impl Default for CodegenConfig {
    fn default() -> Self {
        Self {
            max_distance: 16,
            min_priority: PrefetchPriority::Low,
            use_non_temporal: true,
            prefetch_writes: false,
            cache_line_size: 64,
        }
    }
}

impl PrefetchCodegen {
    /// Create a new codegen instance.
    pub fn new(target: Target) -> Self {
        Self {
            target,
            instructions: Vec::new(),
            registers: HashMap::new(),
            counter: 0,
            config: CodegenConfig::default(),
        }
    }

    /// Create with configuration.
    pub fn with_config(target: Target, config: CodegenConfig) -> Self {
        Self {
            target,
            instructions: Vec::new(),
            registers: HashMap::new(),
            counter: 0,
            config,
        }
    }

    /// Bind a variable to a register.
    pub fn bind_register(&mut self, var: impl Into<String>, reg: impl Into<String>) {
        self.registers.insert(var.into(), reg.into());
    }

    /// Generate prefetch from a hint.
    pub fn from_hint(
        &mut self,
        hint: &PrefetchHint,
        base_addr: AddressExpr,
    ) -> Option<PrefetchInstruction> {
        if hint.priority < self.config.min_priority {
            return None;
        }

        let cache_hint = CacheHint::from(hint.target_locality);

        let inst = PrefetchInstruction::new(base_addr)
            .with_locality(cache_hint)
            .with_reason(hint.reason.clone());

        self.instructions.push(inst.clone());
        Some(inst)
    }

    /// Generate stride prefetch for a loop.
    pub fn for_stride(
        &mut self,
        pattern: &StridePattern,
        base: &str,
        element_size: usize,
    ) -> Vec<PrefetchInstruction> {
        let mut results = Vec::new();

        let distance = pattern.prefetch_distance();
        let reg = self
            .registers
            .get(base)
            .cloned()
            .unwrap_or_else(|| base.to_string());

        for i in 1..=distance.min(self.config.max_distance) {
            let offset = (i * pattern.stride) as i64;
            let addr = AddressExpr::base_offset(&reg, offset);

            let temporal = i <= 4; // First few prefetches are temporal
            let locality = if temporal {
                CacheHint::L2
            } else if self.config.use_non_temporal {
                CacheHint::NonTemporal
            } else {
                CacheHint::L3
            };

            let inst = PrefetchInstruction::new(addr)
                .with_locality(locality)
                .with_reason(format!("stride prefetch +{} elements", i));

            if !temporal {
                let inst = inst.non_temporal();
                results.push(inst);
            } else {
                results.push(inst);
            }
        }

        self.instructions.extend(results.clone());
        results
    }

    /// Generate prefetches for an access pattern.
    pub fn for_pattern(&mut self, pattern: &AccessPattern) -> Vec<PrefetchInstruction> {
        let mut results = Vec::new();

        // Generate prefetches for stride patterns
        for (field, stride) in &pattern.strides {
            if stride.is_constant && stride.count >= 3 {
                let parts: Vec<_> = field.split('.').collect();
                let base = parts.first().copied().unwrap_or("ptr");

                results.extend(self.for_stride(stride, base, stride.stride));
            }
        }

        results
    }

    /// Emit all generated instructions.
    pub fn emit(&self) -> String {
        let mut output = String::new();

        for inst in &self.instructions {
            let code = match self.target {
                Target::X86_64 => inst.to_x86_asm(),
                Target::Arm64 => inst.to_arm_asm(),
                Target::LLVM => inst.to_llvm_ir(),
                Target::RiscV => {
                    // RISC-V doesn't have prefetch instructions in base ISA
                    // Use cache hint extension if available
                    format!(
                        "; prefetch {} (not available on base RISC-V)",
                        inst.address.to_asm()
                    )
                }
            };

            if !inst.reason.is_empty() {
                output.push_str(&format!("; {}\n", inst.reason));
            }
            output.push_str(&code);
            output.push('\n');
        }

        output
    }

    /// Get all generated instructions.
    pub fn instructions(&self) -> &[PrefetchInstruction] {
        &self.instructions
    }

    /// Clear generated instructions.
    pub fn clear(&mut self) {
        self.instructions.clear();
    }

    /// Generate a unique label.
    pub fn fresh_label(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{}_{}", prefix, self.counter)
    }
}

/// Prefetch sequence for a data structure.
#[derive(Debug, Clone)]
pub struct PrefetchSequence {
    /// The type being prefetched
    pub type_name: String,
    /// Instructions in order
    pub instructions: Vec<PrefetchInstruction>,
    /// Total estimated cycles saved
    pub estimated_benefit: u32,
}

impl PrefetchSequence {
    /// Create a new sequence.
    pub fn new(type_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            instructions: Vec::new(),
            estimated_benefit: 0,
        }
    }

    /// Add an instruction.
    pub fn add(&mut self, inst: PrefetchInstruction, cycles_saved: u32) {
        self.instructions.push(inst);
        self.estimated_benefit += cycles_saved;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_expr() {
        let reg = AddressExpr::Register("rax".to_string());
        assert_eq!(reg.to_asm(), "rax");

        let base_off = AddressExpr::base_offset("rax", 64);
        assert_eq!(base_off.to_asm(), "rax + 64");

        let indexed = AddressExpr::indexed("rax", "rcx", 8, 0);
        assert_eq!(indexed.to_asm(), "rax + rcx * 8");
    }

    #[test]
    fn test_address_add_offset() {
        let addr = AddressExpr::base_offset("rax", 32);
        let new_addr = addr.add_offset(64);

        match new_addr {
            AddressExpr::BaseOffset { base, offset } => {
                assert_eq!(base, "rax");
                assert_eq!(offset, 96);
            }
            _ => panic!("Expected BaseOffset"),
        }
    }

    #[test]
    fn test_prefetch_instruction_llvm() {
        let inst = PrefetchInstruction::new(AddressExpr::Register("ptr".to_string()))
            .with_locality(CacheHint::L1)
            .with_intent(PrefetchIntent::Read);

        let ir = inst.to_llvm_ir();
        assert!(ir.contains("@llvm.prefetch"));
        assert!(ir.contains("i32 0")); // read
        assert!(ir.contains("i32 3")); // L1
    }

    #[test]
    fn test_prefetch_instruction_x86() {
        let inst = PrefetchInstruction::new(AddressExpr::Register("rax".to_string()))
            .with_locality(CacheHint::L1);

        let asm = inst.to_x86_asm();
        assert_eq!(asm, "prefetcht0 [rax]");

        let nontemporal = PrefetchInstruction::new(AddressExpr::Register("rax".to_string()))
            .with_locality(CacheHint::NonTemporal);

        assert_eq!(nontemporal.to_x86_asm(), "prefetchnta [rax]");
    }

    #[test]
    fn test_prefetch_instruction_arm() {
        let inst = PrefetchInstruction::new(AddressExpr::Register("x0".to_string()))
            .with_locality(CacheHint::L2);

        let asm = inst.to_arm_asm();
        assert!(asm.contains("prfm"));
        assert!(asm.contains("pldl2keep"));
    }

    #[test]
    fn test_codegen_stride() {
        let mut codegen = PrefetchCodegen::new(Target::X86_64);
        codegen.bind_register("arr", "rax");

        let stride = StridePattern::new(64);
        let insts = codegen.for_stride(&stride, "arr", 64);

        assert!(!insts.is_empty());
    }

    #[test]
    fn test_codegen_emit() {
        let mut codegen = PrefetchCodegen::new(Target::X86_64);

        let inst = PrefetchInstruction::new(AddressExpr::base_offset("rax", 128))
            .with_reason("prefetch next cache line");

        codegen.instructions.push(inst);

        let output = codegen.emit();
        assert!(output.contains("prefetch next cache line"));
        assert!(output.contains("prefetcht1"));
    }

    #[test]
    fn test_cache_hint_from_locality() {
        assert_eq!(CacheHint::from(Locality::L1), CacheHint::L1);
        assert_eq!(CacheHint::from(Locality::L2), CacheHint::L2);
        assert_eq!(CacheHint::from(Locality::L3), CacheHint::L3);
        assert_eq!(CacheHint::from(Locality::Local), CacheHint::NonTemporal);
    }

    #[test]
    fn test_cache_hint_from_priority() {
        assert_eq!(CacheHint::from(PrefetchPriority::Critical), CacheHint::L1);
        assert_eq!(CacheHint::from(PrefetchPriority::High), CacheHint::L1);
        assert_eq!(CacheHint::from(PrefetchPriority::Medium), CacheHint::L2);
        assert_eq!(CacheHint::from(PrefetchPriority::Low), CacheHint::L3);
    }

    #[test]
    fn test_prefetch_sequence() {
        let mut seq = PrefetchSequence::new("Patient");

        seq.add(
            PrefetchInstruction::new(AddressExpr::Register("r0".to_string())),
            50,
        );
        seq.add(
            PrefetchInstruction::new(AddressExpr::base_offset("r0", 64)),
            30,
        );

        assert_eq!(seq.instructions.len(), 2);
        assert_eq!(seq.estimated_benefit, 80);
    }
}

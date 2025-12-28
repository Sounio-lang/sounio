//! SIR Basic Blocks and Control Flow
//!
//! SIR uses a standard CFG (Control Flow Graph) structure with basic blocks.
//! Each block ends with a terminator instruction.

use super::ops::SirInst;
use super::types::SirType;
use super::values::{BlockId, FuncId, Value, ValueId};
use std::fmt;

/// A single instruction with its result value
#[derive(Debug, Clone)]
pub struct Instruction {
    /// The result value (None for void-returning instructions)
    pub result: Option<ValueId>,
    /// The instruction itself
    pub inst: SirInst,
}

impl Instruction {
    pub fn new(result: Option<ValueId>, inst: SirInst) -> Self {
        Self { result, inst }
    }

    pub fn with_result(result: ValueId, inst: SirInst) -> Self {
        Self {
            result: Some(result),
            inst,
        }
    }

    pub fn void(inst: SirInst) -> Self {
        Self { result: None, inst }
    }
}

/// Block terminator instructions
#[derive(Debug, Clone, PartialEq)]
pub enum Terminator {
    /// Return from function
    Return(Option<ValueId>),

    /// Unconditional branch
    Br(BlockId),

    /// Conditional branch
    CondBr {
        cond: ValueId,
        then_block: BlockId,
        else_block: BlockId,
    },

    /// Multi-way branch (switch)
    Switch {
        val: ValueId,
        default: BlockId,
        cases: Vec<(i64, BlockId)>,
    },

    /// Unreachable (for optimization hints)
    Unreachable,

    /// Tail call (transfers control, never returns)
    TailCall {
        callee: super::ops::Callee,
        args: Vec<ValueId>,
    },
}

impl Terminator {
    /// Get successor blocks
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Return(_) => vec![],
            Terminator::Br(blk) => vec![*blk],
            Terminator::CondBr {
                then_block,
                else_block,
                ..
            } => vec![*then_block, *else_block],
            Terminator::Switch { default, cases, .. } => {
                let mut succs = vec![*default];
                succs.extend(cases.iter().map(|(_, blk)| *blk));
                succs
            }
            Terminator::Unreachable => vec![],
            Terminator::TailCall { .. } => vec![],
        }
    }

    /// Get operand values
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Terminator::Return(Some(v)) => vec![*v],
            Terminator::Return(None) => vec![],
            Terminator::Br(_) => vec![],
            Terminator::CondBr { cond, .. } => vec![*cond],
            Terminator::Switch { val, .. } => vec![*val],
            Terminator::Unreachable => vec![],
            Terminator::TailCall { args, .. } => args.clone(),
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Terminator::Return(None) => write!(f, "ret void"),
            Terminator::Return(Some(v)) => write!(f, "ret {}", v),
            Terminator::Br(blk) => write!(f, "br {}", blk),
            Terminator::CondBr {
                cond,
                then_block,
                else_block,
            } => {
                write!(f, "br {}, {}, {}", cond, then_block, else_block)
            }
            Terminator::Switch {
                val,
                default,
                cases,
            } => {
                write!(f, "switch {} [default: {}", val, default)?;
                for (v, blk) in cases {
                    write!(f, ", {}: {}", v, blk)?;
                }
                write!(f, "]")
            }
            Terminator::Unreachable => write!(f, "unreachable"),
            Terminator::TailCall { callee, args } => {
                write!(f, "tailcall {:?}({:?})", callee, args)
            }
        }
    }
}

/// A basic block in the CFG
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Block identifier
    pub id: BlockId,
    /// Optional name (for debugging)
    pub name: Option<String>,
    /// Block parameters (for SSA phi elimination)
    pub params: Vec<Value>,
    /// Instructions in the block
    pub instructions: Vec<Instruction>,
    /// Terminator (required)
    pub terminator: Option<Terminator>,

    // === Analysis Metadata ===
    /// Predecessor blocks (computed)
    pub predecessors: Vec<BlockId>,
    /// Is this a loop header?
    pub is_loop_header: bool,
    /// Loop nesting depth
    pub loop_depth: u32,
    /// Dominator (immediate dominator in dominator tree)
    pub idom: Option<BlockId>,
}

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            name: None,
            params: vec![],
            instructions: vec![],
            terminator: None,
            predecessors: vec![],
            is_loop_header: false,
            loop_depth: 0,
            idom: None,
        }
    }

    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_params(mut self, params: Vec<Value>) -> Self {
        self.params = params;
        self
    }

    /// Add an instruction to the block
    pub fn push(&mut self, inst: Instruction) {
        self.instructions.push(inst);
    }

    /// Set the terminator
    pub fn terminate(&mut self, term: Terminator) {
        self.terminator = Some(term);
    }

    /// Is this block complete (has terminator)?
    pub fn is_complete(&self) -> bool {
        self.terminator.is_some()
    }

    /// Get all values defined in this block
    pub fn defined_values(&self) -> Vec<ValueId> {
        let mut vals: Vec<ValueId> = self.params.iter().map(|v| v.id).collect();
        vals.extend(self.instructions.iter().filter_map(|i| i.result));
        vals
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Block header
        if let Some(name) = &self.name {
            writeln!(f, "{}:", name)?;
        } else {
            writeln!(f, "{}:", self.id)?;
        }

        // Block parameters
        if !self.params.is_empty() {
            write!(f, "  ; params: ")?;
            for (i, p) in self.params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}: {}", p.id, p.ty)?;
            }
            writeln!(f)?;
        }

        // Instructions
        for inst in &self.instructions {
            if let Some(result) = &inst.result {
                writeln!(f, "  {} = {}", result, inst.inst)?;
            } else {
                writeln!(f, "  {}", inst.inst)?;
            }
        }

        // Terminator
        if let Some(term) = &self.terminator {
            writeln!(f, "  {}", term)?;
        } else {
            writeln!(f, "  ; <no terminator>")?;
        }

        Ok(())
    }
}

/// A SIR function
#[derive(Debug, Clone)]
pub struct SirFunction {
    /// Function identifier
    pub id: FuncId,
    /// Function name
    pub name: String,
    /// Parameter types and names
    pub params: Vec<(String, SirType)>,
    /// Return type
    pub ret_ty: SirType,
    /// Basic blocks (entry block is first)
    pub blocks: Vec<BasicBlock>,
    /// Is this function externally visible?
    pub linkage: Linkage,
    /// Calling convention
    pub cc: super::types::CallingConv,

    // === Function Attributes ===
    /// Function never returns
    pub noreturn: bool,
    /// Function has no side effects
    pub pure: bool,
    /// Function can be inlined
    pub inline: InlineHint,
    /// Function does not throw
    pub nounwind: bool,
    /// Function is a hot path
    pub hot: bool,
    /// Function is cold (unlikely to be called)
    pub cold: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Linkage {
    /// Visible to other modules
    #[default]
    External,
    /// Internal to this module
    Internal,
    /// Weak linkage (can be overridden)
    Weak,
    /// Available externally (inline)
    AvailableExternally,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InlineHint {
    /// No preference
    #[default]
    None,
    /// Should be inlined
    Always,
    /// Should not be inlined
    Never,
    /// Inline if profitable
    Hint,
}

impl SirFunction {
    pub fn new(
        id: FuncId,
        name: impl Into<String>,
        params: Vec<(String, SirType)>,
        ret_ty: SirType,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            params,
            ret_ty,
            blocks: vec![],
            linkage: Linkage::External,
            cc: super::types::CallingConv::Default,
            noreturn: false,
            pure: false,
            inline: InlineHint::None,
            nounwind: false,
            hot: false,
            cold: false,
        }
    }

    /// Get the entry block
    pub fn entry_block(&self) -> Option<&BasicBlock> {
        self.blocks.first()
    }

    /// Get the entry block mutably
    pub fn entry_block_mut(&mut self) -> Option<&mut BasicBlock> {
        self.blocks.first_mut()
    }

    /// Get block by ID
    pub fn block(&self, id: BlockId) -> Option<&BasicBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }

    /// Get block by ID mutably
    pub fn block_mut(&mut self, id: BlockId) -> Option<&mut BasicBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }

    /// Add a new block
    pub fn add_block(&mut self, block: BasicBlock) -> BlockId {
        let id = block.id;
        self.blocks.push(block);
        id
    }
}

impl fmt::Display for SirFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Function signature
        write!(f, "define ")?;
        if self.linkage != Linkage::External {
            write!(f, "{:?} ", self.linkage)?;
        }
        write!(f, "{} @{}(", self.ret_ty, self.name)?;

        for (i, (name, ty)) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{} %{}", ty, name)?;
        }
        writeln!(f, ") {{")?;

        // Blocks
        for block in &self.blocks {
            write!(f, "{}", block)?;
        }

        writeln!(f, "}}")
    }
}

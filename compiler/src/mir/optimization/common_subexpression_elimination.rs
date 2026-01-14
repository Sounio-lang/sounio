//! Common Subexpression Elimination (CSE) optimization pass
//! 
//! This module implements CSE, which identifies identical expressions
//! that compute the same value and reuses the result instead of recomputing.

use crate::mir::analysis::dominators::DominatorTree;
use crate::mir::types::{Type, Value};
use crate::mir::{BasicBlock, Function, Instruction, Module};
use crate::mir::analysis::AvailableExpressions;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

/// Represents an expression that can be tracked for CSE
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CseExpression {
    /// Binary operations
    Binary {
        op: String,
        left: Value,
        right: Value,
        ty: Type,
    },
    /// Unary operations
    Unary {
        op: String,
        operand: Value,
        ty: Type,
    },
    /// Load from memory
    Load {
        address: Value,
        ty: Type,
    },
    /// Constant values
    Constant {
        value: String,
        ty: Type,
    },
    /// Function calls
    Call {
        func: String,
        args: Vec<Value>,
        ty: Type,
    },
    /// Phi functions (handled specially)
    Phi {
        args: Vec<Value>,
        ty: Type,
    },
}

impl CseExpression {
    /// Create a CSE expression from an instruction
    pub fn from_instruction(instr: &Instruction, values: &HashMap<Value, String>) -> Option<Self> {
        match instr {
            Instruction::Binary { op, left, right, ty, .. } => Some(CseExpression::Binary {
                op: op.clone(),
                left: *left,
                right: *right,
                ty: ty.clone(),
            }),
            Instruction::Unary { op, operand, ty, .. } => Some(CseExpression::Unary {
                op: op.clone(),
                operand: *operand,
                ty: ty.clone(),
            }),
            Instruction::Load { address, ty, .. } => Some(CseExpression::Load {
                address: *address,
                ty: ty.clone(),
            }),
            Instruction::Const { value, ty, .. } => Some(CseExpression::Constant {
                value: value.clone(),
                ty: ty.clone(),
            }),
            Instruction::Call { func, args, ty, .. } => Some(CseExpression::Call {
                func: func.clone(),
                args: args.clone(),
                ty: ty.clone(),
            }),
            Instruction::Phi { args, ty, .. } => Some(CseExpression::Phi {
                args: args.clone(),
                ty: ty.clone(),
            }),
            _ => None,
        }
    }

    /// Get the type of this expression
    pub fn get_type(&self) -> &Type {
        match self {
            CseExpression::Binary { ty, .. } => ty,
            CseExpression::Unary { ty, .. } => ty,
            CseExpression::Load { ty, .. } => ty,
            CseExpression::Constant { ty, .. } => ty,
            CseExpression::Call { ty, .. } => ty,
            CseExpression::Phi { ty, .. } => ty,
        }
    }
}

/// Available expressions analysis result
#[derive(Debug, Clone)]
pub struct CseAnalysis {
    /// Map from expression to the value that computes it
    pub expressions: HashMap<CseExpression, Value>,
    /// Set of available expressions at each point
    pub available_at: HashMap<String, HashSet<CseExpression>>,
}

impl CseAnalysis {
    /// Perform available expressions analysis
    pub fn analyze_function(func: &Function) -> Self {
        let mut expressions = HashMap::new();
        let mut available_at = HashMap::new();

        // Initialize with empty sets for all basic blocks
        for (block_id, _) in &func.blocks {
            available_at.insert(block_id.clone(), HashSet::new());
        }

        // Perform data flow analysis
        for (block_id, block) in &func.blocks {
            let mut block_expressions = HashSet::new();

            for instr in &block.instructions {
                if let Some(expr) = CseExpression::from_instruction(instr, &HashMap::new()) {
                    // If this expression is already available, reuse it
                    if block_expressions.contains(&expr) {
                        // Expression is already computed in this block
                        continue;
                    }

                    // Add expression to available set
                    block_expressions.insert(expr.clone());
                    
                    // Record that this expression computes a value
                    if let Some(result) = instr.get_result() {
                        expressions.insert(expr, result);
                    }
                }
            }

            available_at.insert(block_id.clone(), block_expressions);
        }

        CseAnalysis {
            expressions,
            available_at,
        }
    }
}

/// Common Subexpression Elimination pass
pub struct CommonSubexpressionElimination {
    /// Analysis results
    analysis: CseAnalysis,
    /// Mapping from original values to their replacements
    replacements: HashMap<Value, Value>,
    /// Counter for generating new temporary values
    temp_counter: u32,
}

impl CommonSubexpressionElimination {
    /// Create a new CSE pass
    pub fn new() -> Self {
        CommonSubexpressionElimination {
            analysis: CseAnalysis {
                expressions: HashMap::new(),
                available_at: HashMap::new(),
            },
            replacements: HashMap::new(),
            temp_counter: 0,
        }
    }

    /// Generate a new temporary value
    fn new_temp(&mut self, ty: Type) -> Value {
        self.temp_counter += 1;
        Value::Temp(format!("cse_temp_{}", self.temp_counter), ty)
    }

    /// Perform CSE on a function
    pub fn optimize_function(&mut self, func: &mut Function) -> bool {
        // Perform available expressions analysis
        self.analysis = CseAnalysis::analyze_function(func);
        self.replacements.clear();

        let mut changed = false;

        // Apply CSE transformations
        for (block_id, block) in &mut func.blocks {
            let available = self.analysis.available_at.get(block_id).unwrap_or(&HashSet::new()).clone();
            
            let mut new_instructions = Vec::new();

            for instr in &block.instructions.clone() {
                if let Some(expr) = CseExpression::from_instruction(instr, &HashMap::new()) {
                    // Check if this expression is already available
                    if available.contains(&expr) {
                        if let Some(cached_value) = self.analysis.expressions.get(&expr) {
                            // Replace with cached value
                            self.replacements.insert(instr.get_result().unwrap_or(Value::Unit), *cached_value);
                            changed = true;
                            continue;
                        }
                    }
                }

                new_instructions.push(instr.clone());
            }

            block.instructions = new_instructions;
        }

        changed
    }

    /// Perform CSE on a module
    pub fn optimize_module(&mut self, module: &mut Module) -> bool {
        let mut changed = false;

        for func in module.functions.values_mut() {
            if self.optimize_function(func) {
                changed = true;
            }
        }

        changed
    }
}

impl Default for CommonSubexpressionElimination {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cse_expression_from_instruction() {
        // Test creating CSE expression from binary instruction
        let instr = Instruction::Binary {
            op: "+".to_string(),
            left: Value::Const("1".to_string(), Type::I32),
            right: Value::Const("2".to_string(), Type::I32),
            result: Some(Value::Temp("x".to_string(), Type::I32)),
            ty: Type::I32,
        };

        let expr = CseExpression::from_instruction(&instr, &HashMap::new()).unwrap();
        assert!(matches!(expr, CseExpression::Binary { .. }));
    }

    #[test]
    fn test_cse_analysis() {
        // This would test the available expressions analysis
        // For now, just test basic structure
        let analysis = CseAnalysis {
            expressions: HashMap::new(),
            available_at: HashMap::new(),
        };

        assert!(analysis.expressions.is_empty());
    }

    #[test]
    fn test_cse_pass() {
        let mut cse = CommonSubexpressionElimination::new();
        assert_eq!(cse.temp_counter, 0);
        
        let temp = cse.new_temp(Type::I32);
        assert_eq!(cse.temp_counter, 1);
        
        if let Value::Temp(name, _) = temp {
            assert!(name.starts_with("cse_temp_"));
        }
    }
}
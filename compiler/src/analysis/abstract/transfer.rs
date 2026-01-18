//! Transfer Functions for Abstract Interpretation
//!
//! Maps HLIR instructions to abstract transformers that update the
//! abstract state. This is the core of the abstract interpreter.
//!
//! # Theory
//!
//! For each instruction I, we define a transfer function [[I]] that maps
//! abstract states to abstract states:
//!
//! [[I]] : AbstractState -> AbstractState
//!
//! The transfer function must be sound: if concrete state s in gamma(a),
//! then executing I on s produces s' in gamma([[I]](a)).

use super::domain::AbstractDomain;
use super::intervals::Interval;
use super::product::NumericDomain;
use crate::hlir::{BinaryOp, HlirConstant, HlirInstr, HlirType, Op, UnaryOp, ValueId};
use std::collections::HashMap;

/// Abstract state for a program point
///
/// Maps value IDs to their abstract values.
#[derive(Debug, Clone)]
pub struct AbstractState<D: AbstractDomain + Clone> {
    /// Map from value IDs to abstract values
    values: HashMap<ValueId, D>,
    /// Whether this state is reachable (false = bottom state)
    reachable: bool,
}

impl<D: AbstractDomain + Clone> AbstractState<D> {
    /// Create a new empty state
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            reachable: true,
        }
    }

    /// Create an unreachable (bottom) state
    pub fn bottom() -> Self {
        Self {
            values: HashMap::new(),
            reachable: false,
        }
    }

    /// Check if this state is reachable
    pub fn is_reachable(&self) -> bool {
        self.reachable
    }

    /// Mark this state as unreachable
    pub fn set_unreachable(&mut self) {
        self.reachable = false;
    }

    /// Get the abstract value for a value ID
    pub fn get(&self, id: ValueId) -> D {
        if !self.reachable {
            return D::bottom();
        }
        self.values.get(&id).cloned().unwrap_or_else(D::top)
    }

    /// Set the abstract value for a value ID
    pub fn set(&mut self, id: ValueId, value: D) {
        if value.is_bottom() {
            self.reachable = false;
        }
        self.values.insert(id, value);
    }

    /// Remove a value (e.g., when it goes out of scope)
    pub fn remove(&mut self, id: ValueId) {
        self.values.remove(&id);
    }

    /// Join two states
    pub fn join(&self, other: &Self) -> Self {
        if !self.reachable {
            return other.clone();
        }
        if !other.reachable {
            return self.clone();
        }

        let mut result = Self::new();

        // Join all values that appear in either state
        let all_keys: std::collections::HashSet<_> = self
            .values
            .keys()
            .chain(other.values.keys())
            .cloned()
            .collect();

        for id in all_keys {
            let v1 = self.get(id);
            let v2 = other.get(id);
            result.set(id, v1.join(&v2));
        }

        result
    }

    /// Meet two states
    pub fn meet(&self, other: &Self) -> Self {
        if !self.reachable || !other.reachable {
            return Self::bottom();
        }

        let mut result = Self::new();

        let all_keys: std::collections::HashSet<_> = self
            .values
            .keys()
            .chain(other.values.keys())
            .cloned()
            .collect();

        for id in all_keys {
            let v1 = self.get(id);
            let v2 = other.get(id);
            let met = v1.meet(&v2);
            if met.is_bottom() {
                return Self::bottom();
            }
            result.set(id, met);
        }

        result
    }

    /// Widen two states
    pub fn widen(&self, other: &Self) -> Self {
        if !self.reachable {
            return other.clone();
        }
        if !other.reachable {
            return self.clone();
        }

        let mut result = Self::new();

        let all_keys: std::collections::HashSet<_> = self
            .values
            .keys()
            .chain(other.values.keys())
            .cloned()
            .collect();

        for id in all_keys {
            let v1 = self.get(id);
            let v2 = other.get(id);
            result.set(id, v1.widen(&v2));
        }

        result
    }

    /// Check if this state is less than or equal to another
    pub fn leq(&self, other: &Self) -> bool {
        if !self.reachable {
            return true;
        }
        if !other.reachable {
            return false;
        }

        for (id, v1) in &self.values {
            let v2 = other.get(*id);
            if !v1.leq(&v2) {
                return false;
            }
        }
        true
    }

    /// Get all value IDs in this state
    pub fn value_ids(&self) -> impl Iterator<Item = &ValueId> {
        self.values.keys()
    }

    /// Iterate over all (value_id, abstract_value) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&ValueId, &D)> {
        self.values.iter()
    }
}

impl<D: AbstractDomain + Clone> Default for AbstractState<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: AbstractDomain + Clone + PartialEq> PartialEq for AbstractState<D> {
    fn eq(&self, other: &Self) -> bool {
        if self.reachable != other.reachable {
            return false;
        }
        if !self.reachable {
            return true; // Both unreachable
        }
        self.values == other.values
    }
}

impl<D: AbstractDomain + Clone + Eq> Eq for AbstractState<D> {}

/// Transfer function for interval domain
pub struct IntervalTransfer;

impl IntervalTransfer {
    /// Transfer function for an instruction
    pub fn transfer(state: &mut AbstractState<Interval>, instr: &HlirInstr) {
        if !state.is_reachable() {
            return;
        }

        match &instr.op {
            Op::Const(c) => {
                if let Some(result) = instr.result {
                    let interval = Self::const_to_interval(c);
                    state.set(result, interval);
                }
            }

            Op::Copy(src) => {
                if let Some(result) = instr.result {
                    let value = state.get(*src);
                    state.set(result, value);
                }
            }

            Op::Binary { op, left, right } => {
                if let Some(result) = instr.result {
                    let lhs = state.get(*left);
                    let rhs = state.get(*right);
                    let res = Self::binary_transfer(*op, &lhs, &rhs);
                    state.set(result, res);
                }
            }

            Op::Unary { op, operand } => {
                if let Some(result) = instr.result {
                    let val = state.get(*operand);
                    let res = Self::unary_transfer(*op, &val);
                    state.set(result, res);
                }
            }

            Op::Phi { incoming } => {
                if let Some(result) = instr.result {
                    // Join all incoming values
                    let mut joined = Interval::Bottom;
                    for (_block, val) in incoming {
                        let v = state.get(*val);
                        joined = joined.join(&v);
                    }
                    state.set(result, joined);
                }
            }

            Op::Call { args, .. } | Op::CallDirect { args, .. } => {
                // Conservative: calls can return any value
                if let Some(result) = instr.result {
                    state.set(result, Interval::top());
                }
                // Mark arguments as potentially modified (for mutable refs)
                // This is conservative - could be more precise with summaries
            }

            Op::Load { ptr } => {
                // Conservative: loads can return any value
                if let Some(result) = instr.result {
                    state.set(result, Interval::top());
                }
            }

            Op::Store { .. } => {
                // Store doesn't produce a value
            }

            Op::GetFieldPtr { .. } | Op::GetElementPtr { .. } => {
                // Pointer arithmetic - result is a pointer
                if let Some(result) = instr.result {
                    state.set(result, Interval::top());
                }
            }

            Op::Alloca { .. } => {
                // Returns a pointer
                if let Some(result) = instr.result {
                    state.set(result, Interval::top());
                }
            }

            Op::Cast { value, .. } => {
                // Conservative for now - could be more precise for specific casts
                if let Some(result) = instr.result {
                    let val = state.get(*value);
                    state.set(result, val);
                }
            }

            Op::ExtractValue { base, .. } => {
                if let Some(result) = instr.result {
                    state.set(result, Interval::top());
                }
            }

            Op::InsertValue { base, .. } => {
                if let Some(result) = instr.result {
                    let val = state.get(*base);
                    state.set(result, val);
                }
            }

            Op::Tuple(_) | Op::Array(_) | Op::Struct { .. } => {
                // Aggregate construction
                if let Some(result) = instr.result {
                    state.set(result, Interval::top());
                }
            }

            Op::PerformEffect { .. } | Op::DispatchEffect { .. } => {
                // Effect operations - conservative
                if let Some(result) = instr.result {
                    state.set(result, Interval::top());
                }
            }

            Op::PushHandler { .. } | Op::PopHandler => {
                // Handler operations don't produce values
            }
        }
    }

    /// Convert constant to interval
    fn const_to_interval(c: &HlirConstant) -> Interval {
        match c {
            HlirConstant::Int(n, _) => Interval::constant(*n),
            HlirConstant::Bool(true) => Interval::constant(1),
            HlirConstant::Bool(false) => Interval::constant(0),
            _ => Interval::top(),
        }
    }

    /// Binary operation transfer
    fn binary_transfer(op: BinaryOp, lhs: &Interval, rhs: &Interval) -> Interval {
        match op {
            BinaryOp::Add | BinaryOp::FAdd => lhs.add(rhs),
            BinaryOp::Sub | BinaryOp::FSub => lhs.sub(rhs),
            BinaryOp::Mul | BinaryOp::FMul => lhs.mul(rhs),
            BinaryOp::SDiv | BinaryOp::UDiv | BinaryOp::FDiv => lhs.div(rhs),
            BinaryOp::SRem | BinaryOp::URem | BinaryOp::FRem => lhs.rem(rhs),

            // Bitwise operations - conservative
            BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => Interval::top(),
            BinaryOp::Shl | BinaryOp::AShr | BinaryOp::LShr => Interval::top(),

            // Comparisons return 0 or 1
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::SLt
            | BinaryOp::SLe
            | BinaryOp::SGt
            | BinaryOp::SGe
            | BinaryOp::ULt
            | BinaryOp::ULe
            | BinaryOp::UGt
            | BinaryOp::UGe
            | BinaryOp::FOEq
            | BinaryOp::FONe
            | BinaryOp::FOLt
            | BinaryOp::FOLe
            | BinaryOp::FOGt
            | BinaryOp::FOGe => Interval::from_range(0, 1),

            BinaryOp::Concat => Interval::top(),
        }
    }

    /// Unary operation transfer
    fn unary_transfer(op: UnaryOp, val: &Interval) -> Interval {
        match op {
            UnaryOp::Neg | UnaryOp::FNeg => val.neg(),
            UnaryOp::Not => {
                // Boolean not: 0 -> 1, non-zero -> 0
                if val.is_constant() == Some(0) {
                    Interval::constant(1)
                } else if !val.may_contain_zero() {
                    Interval::constant(0)
                } else {
                    Interval::from_range(0, 1)
                }
            }
        }
    }

    /// Transfer for conditional branch - returns (then_state, else_state)
    pub fn transfer_cond_branch(
        state: &AbstractState<Interval>,
        condition: ValueId,
    ) -> (AbstractState<Interval>, AbstractState<Interval>) {
        if !state.is_reachable() {
            return (AbstractState::bottom(), AbstractState::bottom());
        }

        let cond = state.get(condition);

        // Check if condition is definitely true or false
        if cond.is_constant() == Some(0) {
            // Definitely false - only else branch reachable
            return (AbstractState::bottom(), state.clone());
        }
        if !cond.may_contain_zero() {
            // Definitely true - only then branch reachable
            return (state.clone(), AbstractState::bottom());
        }

        // Both branches may be taken
        (state.clone(), state.clone())
    }
}

/// Transfer function for NumericDomain (combined interval+sign+congruence)
pub struct NumericTransfer;

impl NumericTransfer {
    /// Transfer function for an instruction
    pub fn transfer(state: &mut AbstractState<NumericDomain>, instr: &HlirInstr) {
        if !state.is_reachable() {
            return;
        }

        match &instr.op {
            Op::Const(c) => {
                if let Some(result) = instr.result {
                    let value = Self::const_to_numeric(c);
                    state.set(result, value);
                }
            }

            Op::Copy(src) => {
                if let Some(result) = instr.result {
                    let value = state.get(*src);
                    state.set(result, value);
                }
            }

            Op::Binary { op, left, right } => {
                if let Some(result) = instr.result {
                    let lhs = state.get(*left);
                    let rhs = state.get(*right);
                    let res = Self::binary_transfer(*op, &lhs, &rhs);
                    state.set(result, res);
                }
            }

            Op::Unary { op, operand } => {
                if let Some(result) = instr.result {
                    let val = state.get(*operand);
                    let res = Self::unary_transfer(*op, &val);
                    state.set(result, res);
                }
            }

            Op::Phi { incoming } => {
                if let Some(result) = instr.result {
                    let mut joined = NumericDomain::bottom();
                    for (_block, val) in incoming {
                        let v = state.get(*val);
                        joined = joined.join(&v);
                    }
                    state.set(result, joined);
                }
            }

            _ => {
                // Conservative for other operations
                if let Some(result) = instr.result {
                    state.set(result, NumericDomain::top());
                }
            }
        }
    }

    /// Convert constant to NumericDomain
    fn const_to_numeric(c: &HlirConstant) -> NumericDomain {
        match c {
            HlirConstant::Int(n, _) => NumericDomain::constant(*n),
            HlirConstant::Bool(true) => NumericDomain::constant(1),
            HlirConstant::Bool(false) => NumericDomain::constant(0),
            _ => NumericDomain::top(),
        }
    }

    /// Binary operation transfer
    fn binary_transfer(op: BinaryOp, lhs: &NumericDomain, rhs: &NumericDomain) -> NumericDomain {
        match op {
            BinaryOp::Add | BinaryOp::FAdd => lhs.add(rhs),
            BinaryOp::Sub | BinaryOp::FSub => lhs.sub(rhs),
            BinaryOp::Mul | BinaryOp::FMul => lhs.mul(rhs),
            _ => NumericDomain::top(),
        }
    }

    /// Unary operation transfer
    fn unary_transfer(op: UnaryOp, val: &NumericDomain) -> NumericDomain {
        match op {
            UnaryOp::Neg | UnaryOp::FNeg => val.neg(),
            UnaryOp::Not => NumericDomain::top(),
        }
    }
}

/// Condition refinement - refine abstract state based on branch condition
pub struct ConditionRefinement;

impl ConditionRefinement {
    /// Refine interval state assuming condition is true
    pub fn refine_for_true_interval(
        state: &mut AbstractState<Interval>,
        instr: &HlirInstr,
        cond_result: ValueId,
    ) {
        // Look at the instruction that produced the condition
        // and refine the operands accordingly
        match &instr.op {
            Op::Binary { op, left, right } if instr.result == Some(cond_result) => {
                let lhs = state.get(*left);
                let rhs = state.get(*right);

                match op {
                    BinaryOp::SLt => {
                        // left < right is true
                        // Refine: left.hi < right.lo if possible
                        if let Some(r_lo) = rhs.lo().and_then(|b| b.as_finite()) {
                            state.set(*left, lhs.assume_lt(r_lo));
                        }
                        if let Some(l_hi) = lhs.hi().and_then(|b| b.as_finite()) {
                            state.set(*right, rhs.assume_gt(l_hi));
                        }
                    }
                    BinaryOp::SLe => {
                        if let Some(r_hi) = rhs.hi().and_then(|b| b.as_finite()) {
                            state.set(*left, lhs.assume_le(r_hi));
                        }
                        if let Some(l_lo) = lhs.lo().and_then(|b| b.as_finite()) {
                            state.set(*right, rhs.assume_ge(l_lo));
                        }
                    }
                    BinaryOp::SGt => {
                        if let Some(r_hi) = rhs.hi().and_then(|b| b.as_finite()) {
                            state.set(*left, lhs.assume_gt(r_hi));
                        }
                        if let Some(l_lo) = lhs.lo().and_then(|b| b.as_finite()) {
                            state.set(*right, rhs.assume_lt(l_lo));
                        }
                    }
                    BinaryOp::SGe => {
                        if let Some(r_lo) = rhs.lo().and_then(|b| b.as_finite()) {
                            state.set(*left, lhs.assume_ge(r_lo));
                        }
                        if let Some(l_hi) = lhs.hi().and_then(|b| b.as_finite()) {
                            state.set(*right, rhs.assume_le(l_hi));
                        }
                    }
                    BinaryOp::Eq => {
                        // Both should have same value
                        let refined = lhs.meet(&rhs);
                        state.set(*left, refined);
                        state.set(*right, refined);
                    }
                    BinaryOp::Ne => {
                        // If one is constant, refine the other to exclude it
                        if let Some(c) = lhs.is_constant() {
                            // right != c - can't easily represent this
                        }
                        if let Some(c) = rhs.is_constant() {
                            // left != c
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Refine interval state assuming condition is false
    pub fn refine_for_false_interval(
        state: &mut AbstractState<Interval>,
        instr: &HlirInstr,
        cond_result: ValueId,
    ) {
        // Opposite of refine_for_true
        match &instr.op {
            Op::Binary { op, left, right } if instr.result == Some(cond_result) => {
                let lhs = state.get(*left);
                let rhs = state.get(*right);

                match op {
                    BinaryOp::SLt => {
                        // left < right is false means left >= right
                        if let Some(r_lo) = rhs.lo().and_then(|b| b.as_finite()) {
                            state.set(*left, lhs.assume_ge(r_lo));
                        }
                    }
                    BinaryOp::SLe => {
                        // left <= right is false means left > right
                        if let Some(r_hi) = rhs.hi().and_then(|b| b.as_finite()) {
                            state.set(*left, lhs.assume_gt(r_hi));
                        }
                    }
                    BinaryOp::SGt => {
                        // left > right is false means left <= right
                        if let Some(r_hi) = rhs.hi().and_then(|b| b.as_finite()) {
                            state.set(*left, lhs.assume_le(r_hi));
                        }
                    }
                    BinaryOp::SGe => {
                        // left >= right is false means left < right
                        if let Some(r_lo) = rhs.lo().and_then(|b| b.as_finite()) {
                            state.set(*left, lhs.assume_lt(r_lo));
                        }
                    }
                    BinaryOp::Eq => {
                        // They are not equal
                    }
                    BinaryOp::Ne => {
                        // They are equal
                        let refined = lhs.meet(&rhs);
                        state.set(*left, refined);
                        state.set(*right, refined);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abstract_state_basic() {
        let mut state = AbstractState::<Interval>::new();

        state.set(ValueId(0), Interval::from_range(0, 10));
        state.set(ValueId(1), Interval::from_range(5, 15));

        assert!(state.get(ValueId(0)).contains(5));
        assert!(!state.get(ValueId(0)).contains(11));
    }

    #[test]
    fn test_abstract_state_join() {
        let mut s1 = AbstractState::<Interval>::new();
        s1.set(ValueId(0), Interval::from_range(0, 5));

        let mut s2 = AbstractState::<Interval>::new();
        s2.set(ValueId(0), Interval::from_range(10, 15));

        let joined = s1.join(&s2);
        let interval = joined.get(ValueId(0));

        assert!(interval.contains(0));
        assert!(interval.contains(15));
    }

    #[test]
    fn test_abstract_state_meet() {
        let mut s1 = AbstractState::<Interval>::new();
        s1.set(ValueId(0), Interval::from_range(0, 10));

        let mut s2 = AbstractState::<Interval>::new();
        s2.set(ValueId(0), Interval::from_range(5, 15));

        let met = s1.meet(&s2);
        let interval = met.get(ValueId(0));

        assert!(!interval.contains(4));
        assert!(interval.contains(5));
        assert!(interval.contains(10));
        assert!(!interval.contains(11));
    }

    #[test]
    fn test_interval_transfer_const() {
        let mut state = AbstractState::<Interval>::new();

        let instr = HlirInstr {
            result: Some(ValueId(0)),
            op: Op::Const(HlirConstant::Int(42, HlirType::I64)),
            ty: HlirType::I64,
        };

        IntervalTransfer::transfer(&mut state, &instr);

        assert_eq!(state.get(ValueId(0)).is_constant(), Some(42));
    }

    #[test]
    fn test_interval_transfer_add() {
        let mut state = AbstractState::<Interval>::new();
        state.set(ValueId(0), Interval::from_range(0, 10));
        state.set(ValueId(1), Interval::from_range(5, 15));

        let instr = HlirInstr {
            result: Some(ValueId(2)),
            op: Op::Binary {
                op: BinaryOp::Add,
                left: ValueId(0),
                right: ValueId(1),
            },
            ty: HlirType::I64,
        };

        IntervalTransfer::transfer(&mut state, &instr);

        let result = state.get(ValueId(2));
        assert!(result.contains(5)); // 0 + 5
        assert!(result.contains(25)); // 10 + 15
    }

    #[test]
    fn test_interval_transfer_comparison() {
        let mut state = AbstractState::<Interval>::new();
        state.set(ValueId(0), Interval::from_range(0, 10));
        state.set(ValueId(1), Interval::from_range(5, 15));

        let instr = HlirInstr {
            result: Some(ValueId(2)),
            op: Op::Binary {
                op: BinaryOp::SLt,
                left: ValueId(0),
                right: ValueId(1),
            },
            ty: HlirType::Bool,
        };

        IntervalTransfer::transfer(&mut state, &instr);

        let result = state.get(ValueId(2));
        // Comparison returns 0 or 1
        assert!(result.contains(0));
        assert!(result.contains(1));
        assert!(!result.contains(2));
    }

    #[test]
    fn test_cond_branch_transfer() {
        let mut state = AbstractState::<Interval>::new();
        state.set(ValueId(0), Interval::constant(1)); // Definitely true

        let (then_state, else_state) = IntervalTransfer::transfer_cond_branch(&state, ValueId(0));

        assert!(then_state.is_reachable());
        assert!(!else_state.is_reachable());
    }

    #[test]
    fn test_numeric_transfer() {
        let mut state = AbstractState::<NumericDomain>::new();

        let instr = HlirInstr {
            result: Some(ValueId(0)),
            op: Op::Const(HlirConstant::Int(42, HlirType::I64)),
            ty: HlirType::I64,
        };

        NumericTransfer::transfer(&mut state, &instr);

        let result = state.get(ValueId(0));
        assert_eq!(result.interval.is_constant(), Some(42));
        assert!(result.sign.is_definitely_positive());
        assert!(result.congruence.is_even());
    }

    #[test]
    fn test_unreachable_state() {
        let state = AbstractState::<Interval>::bottom();

        assert!(!state.is_reachable());
        assert!(state.get(ValueId(0)).is_bottom());
    }
}

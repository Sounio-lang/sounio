//! HIR to HLIR lowering
//!
//! This module transforms the typed HIR into SSA-form HLIR with explicit
//! control flow and basic blocks.

use super::builder::{FunctionBuilder, ModuleBuilder};
use super::ir::*;
use crate::hir::*;
use std::collections::HashMap;

/// Lower HIR to HLIR
pub fn lower(hir: &Hir) -> HlirModule {
    let lowering = HirToHlir::new();
    lowering.lower_module(hir)
}

/// HIR to HLIR lowering context
struct HirToHlir {
    module_builder: ModuleBuilder,
    /// Map from function names to their signatures (for call resolution)
    functions: HashMap<String, HlirType>,
    /// Map from enum names to their variant info
    enums: HashMap<String, Vec<(String, Vec<HlirType>)>>,
    /// Map from struct names to their field info
    structs: HashMap<String, Vec<(String, HlirType)>>,
    /// Map from effect names to their operations
    effects: HashMap<String, Vec<(String, Vec<HlirType>, HlirType)>>,
    /// Map from handler names to their effect
    handlers: HashMap<String, String>,
}

impl HirToHlir {
    fn new() -> Self {
        Self {
            module_builder: ModuleBuilder::new("main"),
            functions: HashMap::new(),
            enums: HashMap::new(),
            structs: HashMap::new(),
            effects: HashMap::new(),
            handlers: HashMap::new(),
        }
    }

    fn lower_module(mut self, hir: &Hir) -> HlirModule {
        // First pass: collect function signatures, type definitions, and effects
        for item in &hir.items {
            match item {
                HirItem::Function(f) => {
                    let ret_ty = HlirType::from_hir(&f.ty.return_type);
                    self.functions.insert(f.name.clone(), ret_ty);
                }
                HirItem::Struct(s) => {
                    let fields: Vec<_> = s
                        .fields
                        .iter()
                        .map(|f| (f.name.clone(), HlirType::from_hir(&f.ty)))
                        .collect();
                    self.structs.insert(s.name.clone(), fields.clone());
                    self.module_builder.add_type_def(HlirTypeDef {
                        name: s.name.clone(),
                        kind: HlirTypeDefKind::Struct(fields),
                    });
                }
                HirItem::Enum(e) => {
                    let variants: Vec<_> = e
                        .variants
                        .iter()
                        .map(|v| {
                            (
                                v.name.clone(),
                                v.fields.iter().map(HlirType::from_hir).collect(),
                            )
                        })
                        .collect();
                    self.enums.insert(e.name.clone(), variants.clone());
                    self.module_builder.add_type_def(HlirTypeDef {
                        name: e.name.clone(),
                        kind: HlirTypeDefKind::Enum(variants),
                    });
                }
                HirItem::Effect(eff) => {
                    let ops: Vec<_> = eff
                        .operations
                        .iter()
                        .map(|op| {
                            (
                                op.name.clone(),
                                op.params.iter().map(HlirType::from_hir).collect(),
                                HlirType::from_hir(&op.return_type),
                            )
                        })
                        .collect();
                    self.effects.insert(eff.name.clone(), ops);
                }
                HirItem::Handler(h) => {
                    self.handlers.insert(h.name.clone(), h.effect.clone());
                }
                HirItem::Global(g) => {
                    let global = HlirGlobal {
                        id: ValueId(0),
                        name: g.name.clone(),
                        ty: HlirType::from_hir(&g.ty),
                        init: None,
                        is_const: g.is_const,
                    };
                    self.module_builder.add_global(global);
                }
                _ => {}
            }
        }

        // Also collect extern function signatures for call resolution
        for extern_block in &hir.externs {
            for ext_fn in &extern_block.functions {
                self.functions
                    .insert(ext_fn.name.clone(), HlirType::from_hir(&ext_fn.return_type));
            }
        }

        // Declare extern functions (imports) as declaration-only HLIR functions
        for extern_block in &hir.externs {
            for ext_fn in &extern_block.functions {
                let hlir_func = self.lower_extern_fn(extern_block.abi.clone(), ext_fn);
                self.module_builder.add_function(hlir_func);
            }
        }

        // Second pass: lower functions
        for item in &hir.items {
            if let HirItem::Function(f) = item {
                let hlir_func = self.lower_function(f);
                self.module_builder.add_function(hlir_func);
            }
        }

        self.module_builder.build()
    }

    fn lower_extern_fn(&mut self, abi: crate::ast::Abi, f: &HirExternFn) -> HlirFunction {
        let func_id = self.module_builder.fresh_func_id();
        let return_type = HlirType::from_hir(&f.return_type);

        let mut func_builder = FunctionBuilder::new(func_id, &f.name, return_type);
        func_builder.set_abi(abi);
        func_builder.set_exported(false);
        func_builder.set_link_name(f.link_name.clone());
        func_builder.set_variadic(f.is_variadic);

        for param in &f.params {
            func_builder.add_param(&param.name, HlirType::from_hir(&param.ty));
        }

        // No blocks/body: declaration-only import.
        func_builder.build()
    }

    fn lower_function(&mut self, f: &HirFn) -> HlirFunction {
        let func_id = self.module_builder.fresh_func_id();
        let return_type = HlirType::from_hir(&f.ty.return_type);

        let mut func_builder = FunctionBuilder::new(func_id, &f.name, return_type.clone());

        // Set ABI and export status
        func_builder.set_abi(f.abi.clone());
        func_builder.set_exported(f.is_exported);

        // Add parameters
        for param in &f.ty.params {
            let ty = HlirType::from_hir(&param.ty);
            func_builder.add_param(&param.name, ty);
        }

        // Create entry block
        let entry = func_builder.create_block("entry");
        func_builder.switch_to_block(entry);

        // Lower function body
        let mut ctx = LoweringContext::new(
            &mut func_builder,
            &self.functions,
            &self.enums,
            &self.structs,
            &self.effects,
            &self.handlers,
        );
        let result = ctx.lower_block(&f.body);

        // Add return if not already terminated
        if !ctx.is_terminated() {
            if return_type == HlirType::Void {
                ctx.builder.build_return(None);
            } else {
                ctx.builder.build_return(result);
            }
        }

        func_builder.build()
    }
}

/// Context for lowering expressions within a function
struct LoweringContext<'a> {
    builder: &'a mut FunctionBuilder,
    functions: &'a HashMap<String, HlirType>,
    enums: &'a HashMap<String, Vec<(String, Vec<HlirType>)>>,
    structs: &'a HashMap<String, Vec<(String, HlirType)>>,
    effects: &'a HashMap<String, Vec<(String, Vec<HlirType>, HlirType)>>,
    handlers: &'a HashMap<String, String>,
    /// Track if current block is terminated
    terminated: bool,
    /// Loop context for break/continue
    loop_stack: Vec<LoopContext>,
    /// Closure environment (captured variables)
    closure_env: Option<ClosureEnv>,
}

struct LoopContext {
    continue_block: BlockId,
    break_block: BlockId,
    /// Values from break expressions (for loop expressions)
    break_values: Vec<(BlockId, ValueId)>,
}

/// Closure environment for captured variables
struct ClosureEnv {
    /// Map from captured variable names to their indices in the environment
    captures: HashMap<String, usize>,
    /// The environment pointer value
    env_ptr: ValueId,
}

impl<'a> LoweringContext<'a> {
    fn new(
        builder: &'a mut FunctionBuilder,
        functions: &'a HashMap<String, HlirType>,
        enums: &'a HashMap<String, Vec<(String, Vec<HlirType>)>>,
        structs: &'a HashMap<String, Vec<(String, HlirType)>>,
        effects: &'a HashMap<String, Vec<(String, Vec<HlirType>, HlirType)>>,
        handlers: &'a HashMap<String, String>,
    ) -> Self {
        Self {
            builder,
            functions,
            enums,
            structs,
            effects,
            handlers,
            terminated: false,
            loop_stack: Vec::new(),
            closure_env: None,
        }
    }

    fn is_terminated(&self) -> bool {
        self.terminated
    }

    fn lower_block(&mut self, block: &HirBlock) -> Option<ValueId> {
        let mut last_value = None;

        for stmt in &block.stmts {
            if self.terminated {
                break;
            }
            last_value = self.lower_stmt(stmt);
        }

        last_value
    }

    fn lower_stmt(&mut self, stmt: &HirStmt) -> Option<ValueId> {
        match stmt {
            HirStmt::Let {
                name,
                ty,
                value,
                is_mut,
                layout_hint: _, // Layout hints are used by codegen, not HLIR lowering
            } => {
                // Use the type from the initializer if the declared type is a type variable
                let hlir_ty = if let Some(init) = value {
                    // Prefer the initializer's type (it's concrete after type checking)
                    HlirType::from_hir(&init.ty)
                } else {
                    HlirType::from_hir(ty)
                };

                if *is_mut {
                    // Mutable variable: allocate stack slot
                    let slot = self.builder.alloc_var(name, hlir_ty.clone());
                    if let Some(init) = value {
                        let init_val = self.lower_expr(init)?;
                        self.builder.build_store(slot, init_val);
                    }
                } else {
                    // Immutable variable: SSA binding
                    if let Some(init) = value {
                        let init_val = self.lower_expr(init)?;
                        self.builder.bind_var(name, init_val);
                    }
                }
                None
            }
            HirStmt::Expr(expr) => self.lower_expr(expr),
            HirStmt::Assign { target, value } => {
                let val = self.lower_expr(value)?;
                self.lower_assign(target, val);
                None
            }
        }
    }

    fn lower_assign(&mut self, target: &HirExpr, value: ValueId) {
        match &target.kind {
            HirExprKind::Local(name) => {
                // Check if it's a mutable variable with a slot
                if let Some(slot) = self.builder.get_var_slot(name) {
                    self.builder.build_store(slot, value);
                }
            }
            HirExprKind::Deref(inner) => {
                if let Some(ptr) = self.lower_expr(inner) {
                    self.builder.build_store(ptr, value);
                }
            }
            HirExprKind::Unary {
                op: HirUnaryOp::Deref,
                expr: inner,
            } => {
                if let Some(ptr) = self.lower_expr(inner) {
                    self.builder.build_store(ptr, value);
                }
            }
            HirExprKind::Field { base, field } => {
                // Aggregate values (structs/refs) are represented as pointers in HLIR/codegen,
                // so for field stores we need the base *value* (a pointer), not the address of a slot.
                let base_ptr = match HlirType::from_hir(&base.ty) {
                    HlirType::Ptr(_) | HlirType::Struct(_) => self.lower_expr(base),
                    _ => self.lower_lvalue(base),
                };
                if let Some(base_ptr) = base_ptr {
                    let field_idx = self.get_field_index(&base.ty, field);
                    let field_ty = HlirType::from_hir(&target.ty);
                    let field_ptr = self.builder.build_field_ptr(base_ptr, field_idx, field_ty);
                    self.builder.build_store(field_ptr, value);
                }
            }
            HirExprKind::Index { base, index } => {
                // For pointer bases, use lower_expr (get pointer value)
                // Arrays are represented as pointers in codegen, so also use lower_expr.
                let base_ptr = match HlirType::from_hir(&base.ty) {
                    HlirType::Ptr(_) | HlirType::Array(_, _) => self.lower_expr(base),
                    _ => self.lower_lvalue(base),
                };
                if let Some(base_ptr) = base_ptr
                    && let Some(idx) = self.lower_expr(index)
                {
                    let elem_ty = HlirType::from_hir(&target.ty);
                    let elem_ptr = self.builder.build_elem_ptr(base_ptr, idx, elem_ty);
                    self.builder.build_store(elem_ptr, value);
                }
            }
            _ => {}
        }
    }

    fn lower_lvalue(&mut self, expr: &HirExpr) -> Option<ValueId> {
        match &expr.kind {
            HirExprKind::Local(name) => self.builder.get_var_slot(name),
            HirExprKind::Deref(inner) => self.lower_expr(inner),
            HirExprKind::Unary {
                op: HirUnaryOp::Deref,
                expr: inner,
            } => self.lower_expr(inner),
            HirExprKind::Field { base, field } => {
                let base_ptr = match HlirType::from_hir(&base.ty) {
                    HlirType::Ptr(_) | HlirType::Struct(_) => self.lower_expr(base)?,
                    _ => self.lower_lvalue(base)?,
                };
                let field_idx = self.get_field_index(&base.ty, field);
                let field_ty = HlirType::from_hir(&expr.ty);
                Some(self.builder.build_field_ptr(base_ptr, field_idx, field_ty))
            }
            HirExprKind::Index { base, index } => {
                let base_ptr = match HlirType::from_hir(&base.ty) {
                    HlirType::Ptr(_) | HlirType::Array(_, _) => self.lower_expr(base)?,
                    _ => self.lower_lvalue(base)?,
                };
                let idx = self.lower_expr(index)?;
                let elem_ty = HlirType::from_hir(&expr.ty);
                Some(self.builder.build_elem_ptr(base_ptr, idx, elem_ty))
            }
            _ => None,
        }
    }

    fn get_field_index(&self, ty: &HirType, field: &str) -> usize {
        if let HirType::Named { name, .. } = ty
            && let Some(fields) = self.structs.get(name)
        {
            for (i, (f_name, _)) in fields.iter().enumerate() {
                if f_name == field {
                    return i;
                }
            }
        }
        0
    }

    /// Get the variant tag value for an enum variant
    fn get_variant_tag(&self, enum_name: &str, variant: &str) -> i64 {
        if let Some(variants) = self.enums.get(enum_name) {
            for (i, (v_name, _)) in variants.iter().enumerate() {
                if v_name == variant {
                    return i as i64;
                }
            }
        }
        0
    }

    /// Get the variant fields for an enum variant
    fn get_variant_fields(&self, enum_name: &str, variant: &str) -> Vec<HlirType> {
        if let Some(variants) = self.enums.get(enum_name) {
            for (v_name, fields) in variants {
                if v_name == variant {
                    return fields.clone();
                }
            }
        }
        Vec::new()
    }

    fn lower_expr(&mut self, expr: &HirExpr) -> Option<ValueId> {
        if self.terminated {
            return None;
        }

        let ty = HlirType::from_hir(&expr.ty);

        match &expr.kind {
            HirExprKind::Literal(lit) => Some(self.lower_literal(lit, &ty)),

            HirExprKind::Local(name) => {
                // Try immutable binding first
                if let Some(val) = self.builder.get_var(name) {
                    return Some(val);
                }
                // Try mutable variable (load from slot)
                if let Some(val) = self.builder.load_var(name, ty.clone()) {
                    return Some(val);
                }
                // Try closure environment
                if let Some(ref env) = self.closure_env
                    && let Some(&idx) = env.captures.get(name)
                {
                    let field_ptr = self.builder.build_field_ptr(env.env_ptr, idx, ty.clone());
                    return Some(self.builder.build_load(field_ptr, ty));
                }
                // Try function reference
                if self.functions.contains_key(name) {
                    return Some(self.builder.build_const(
                        HlirConstant::FunctionRef(name.clone()),
                        HlirType::Ptr(Box::new(HlirType::Void)),
                    ));
                }
                None
            }

            HirExprKind::Global(name) => Some(self.builder.build_const(
                HlirConstant::GlobalRef(name.clone()),
                HlirType::Ptr(Box::new(ty)),
            )),

            HirExprKind::Binary { op, left, right } => {
                let left_val = self.lower_expr(left)?;
                let right_val = self.lower_expr(right)?;
                let left_ty = HlirType::from_hir(&left.ty);
                // For arithmetic ops, use operand type if result type is Void (unresolved type var)
                let result_ty = if ty == HlirType::Void && !op.is_comparison() {
                    left_ty.clone()
                } else if ty == HlirType::Void && op.is_comparison() {
                    HlirType::Bool
                } else {
                    ty.clone()
                };
                Some(self.lower_binary_op(*op, left_val, right_val, &left_ty, &result_ty))
            }

            HirExprKind::Unary { op, expr: inner } => {
                // Special handling for Ref/RefMut: get address, not value
                if matches!(op, HirUnaryOp::Ref | HirUnaryOp::RefMut) {
                    return self.lower_lvalue(inner);
                }
                if matches!(op, HirUnaryOp::Deref) {
                    let ptr = self.lower_expr(inner)?;
                    return Some(self.builder.build_load(ptr, ty));
                }
                let operand = self.lower_expr(inner)?;
                let inner_ty = HlirType::from_hir(&inner.ty);
                Some(self.lower_unary_op(*op, operand, &inner_ty))
            }

            HirExprKind::Call { func, args } => {
                let arg_vals: Vec<_> = args.iter().filter_map(|a| self.lower_expr(a)).collect();

                // Check if it's a direct function call
                // Try Local first, then Global (builtins can appear as either)
                let func_name = match &func.kind {
                    HirExprKind::Local(name) => Some(name.as_str()),
                    HirExprKind::Global(name) => Some(name.as_str()),
                    _ => None,
                };

                if let Some(name) = func_name {
                    // Check user-defined functions first
                    if self.functions.contains_key(name) {
                        return Some(self.builder.build_call(name, arg_vals, ty));
                    }

                    // Check for builtins that should be lowered to direct calls
                    // These are handled specially by code generators
                    if matches!(
                        name,
                        "print" | "println" | "dbg" | "panic" | "assert" | "assert_eq"
                    ) {
                        return Some(self.builder.build_call(name, arg_vals, ty));
                    }
                }

                // Indirect call
                let func_val = self.lower_expr(func)?;
                Some(self.builder.build_call_indirect(func_val, arg_vals, ty))
            }

            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => self.lower_if(condition, then_branch, else_branch.as_deref(), &ty),

            HirExprKind::Block(block) => self.lower_block(block),

            HirExprKind::Return(value) => {
                let ret_val = value.as_ref().and_then(|v| self.lower_expr(v));
                self.builder.build_return(ret_val);
                self.terminated = true;
                None
            }

            HirExprKind::Loop(body) => self.lower_loop(body, &ty),

            HirExprKind::While { condition, body } => self.lower_while(condition, body, &ty),

            HirExprKind::Break(value) => {
                let break_val = value.as_ref().and_then(|v| self.lower_expr(v));

                if let Some(loop_ctx) = self.loop_stack.last_mut() {
                    if let Some(val) = break_val {
                        let current_block = self.builder.current_block().unwrap();
                        loop_ctx.break_values.push((current_block, val));
                    }
                    let break_block = loop_ctx.break_block;
                    self.builder.build_branch(break_block);
                    self.terminated = true;
                }
                None
            }

            HirExprKind::Continue => {
                if let Some(loop_ctx) = self.loop_stack.last() {
                    let continue_block = loop_ctx.continue_block;
                    self.builder.build_branch(continue_block);
                    self.terminated = true;
                }
                None
            }

            HirExprKind::Tuple(elems) => {
                let vals: Vec<_> = elems.iter().filter_map(|e| self.lower_expr(e)).collect();
                Some(self.builder.build_tuple(vals, ty))
            }

            HirExprKind::Array(elems) => {
                let vals: Vec<_> = elems.iter().filter_map(|e| self.lower_expr(e)).collect();
                Some(self.builder.build_array(vals, ty))
            }

            HirExprKind::Range {
                start,
                end,
                inclusive: _,
            } => {
                // Lower range to a struct with start/end fields
                let start_val = start.as_ref().and_then(|s| self.lower_expr(s));
                let end_val = end.as_ref().and_then(|e| self.lower_expr(e));
                // For now, just return a tuple of (start, end)
                let range_fields = vec![
                    (
                        "start".to_string(),
                        start_val.unwrap_or_else(|| self.builder.build_i64(0)),
                    ),
                    (
                        "end".to_string(),
                        end_val.unwrap_or_else(|| self.builder.build_i64(i64::MAX)),
                    ),
                ];
                Some(self.builder.build_struct("Range", range_fields, ty))
            }

            HirExprKind::Struct { name, fields } => {
                let field_vals: Vec<_> = fields
                    .iter()
                    .filter_map(|(n, e)| self.lower_expr(e).map(|v| (n.clone(), v)))
                    .collect();
                Some(self.builder.build_struct(name, field_vals, ty))
            }

            HirExprKind::Field { base, field } => {
                let base_val = self.lower_expr(base)?;
                let field_idx = self.get_field_index(&base.ty, field);
                Some(self.builder.build_extract(base_val, field_idx, ty))
            }

            HirExprKind::TupleField { base, index } => {
                let base_val = self.lower_expr(base)?;
                Some(self.builder.build_extract(base_val, *index, ty))
            }

            HirExprKind::Index { base, index } => {
                // For arrays/pointers, we need pointer arithmetic
                let base_val = self.lower_expr(base)?;
                let idx_val = self.lower_expr(index)?;
                let elem_ptr = self.builder.build_elem_ptr(base_val, idx_val, ty.clone());
                Some(self.builder.build_load(elem_ptr, ty))
            }

            HirExprKind::Ref {
                mutable: _,
                expr: inner,
            } => {
                // Get address of inner expression
                self.lower_lvalue(inner)
            }

            HirExprKind::Deref(inner) => {
                let ptr = self.lower_expr(inner)?;
                Some(self.builder.build_load(ptr, ty))
            }

            HirExprKind::Cast {
                expr: inner,
                target,
            } => {
                let val = self.lower_expr(inner)?;
                let source_ty = HlirType::from_hir(&inner.ty);
                let target_ty = HlirType::from_hir(target);
                Some(self.builder.build_cast(val, source_ty, target_ty))
            }

            HirExprKind::Match { scrutinee, arms } => self.lower_match(scrutinee, arms, &ty),

            HirExprKind::Closure { params, body } => self.lower_closure(params, body, &ty),

            HirExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                // Desugar to regular function call with receiver as first argument
                let mut all_args = vec![self.lower_expr(receiver)?];
                all_args.extend(args.iter().filter_map(|a| self.lower_expr(a)));
                Some(self.builder.build_call(method, all_args, ty))
            }

            HirExprKind::Variant {
                enum_name,
                variant,
                fields,
            } => self.lower_variant(enum_name, variant, fields, &ty),

            HirExprKind::Perform { effect, op, args } => {
                self.lower_effect_perform(effect, op, args, &ty)
            }

            HirExprKind::Handle { expr, handler } => self.lower_effect_handle(expr, handler, &ty),

            HirExprKind::Sample(dist) => self.lower_sample(dist, &ty),

            // ==================== EPISTEMIC EXPRESSIONS ====================
            HirExprKind::Knowledge {
                value,
                epsilon,
                validity,
                provenance,
            } => {
                // Lower Knowledge to a struct containing value and epsilon
                // For now, just return the value - full implementation would create Knowledge struct
                self.lower_expr(value)
            }

            HirExprKind::Do { variable, value } => {
                // Do intervention - for now, lower as an assignment side effect
                // In a full causal inference system, this would modify the causal graph
                let val = self.lower_expr(value)?;
                Some(
                    self.builder
                        .build_call(format!("__epistemic_do_{}", variable), vec![val], ty),
                )
            }

            HirExprKind::Counterfactual {
                factual,
                intervention,
                outcome,
            } => {
                // Counterfactual requires: abduction, action, prediction
                // For now, lower as a call to runtime counterfactual function
                let factual_val = self.lower_expr(factual)?;
                let intervention_val = self.lower_expr(intervention)?;
                let outcome_val = self.lower_expr(outcome)?;
                Some(self.builder.build_call(
                    "__epistemic_counterfactual",
                    vec![factual_val, intervention_val, outcome_val],
                    ty,
                ))
            }

            HirExprKind::Query {
                target,
                given,
                interventions,
            } => {
                // Probabilistic query - P(target | given, do(interventions))
                let target_val = self.lower_expr(target)?;
                let given_vals: Vec<_> = given.iter().filter_map(|g| self.lower_expr(g)).collect();
                let intervention_vals: Vec<_> = interventions
                    .iter()
                    .filter_map(|i| self.lower_expr(i))
                    .collect();

                // Build argument array for runtime query
                let mut args = vec![target_val];
                args.extend(given_vals);
                args.extend(intervention_vals);

                Some(self.builder.build_call("__epistemic_query", args, ty))
            }

            HirExprKind::Observe { variable, value } => {
                // Observe for probabilistic programming - condition on observed value
                let val = self.lower_expr(value)?;
                Some(self.builder.build_call(
                    format!("__epistemic_observe_{}", variable),
                    vec![val],
                    ty,
                ))
            }

            HirExprKind::EpsilonOf(expr) => {
                // Extract epsilon (confidence) from Knowledge value
                let val = self.lower_expr(expr)?;
                Some(
                    self.builder
                        .build_call("__epistemic_get_epsilon", vec![val], ty),
                )
            }

            HirExprKind::ProvenanceOf(expr) => {
                // Extract provenance from Knowledge value
                let val = self.lower_expr(expr)?;
                Some(
                    self.builder
                        .build_call("__epistemic_get_provenance", vec![val], ty),
                )
            }

            HirExprKind::ValidityOf(expr) => {
                // Extract validity from Knowledge value
                let val = self.lower_expr(expr)?;
                Some(
                    self.builder
                        .build_call("__epistemic_get_validity", vec![val], ty),
                )
            }

            HirExprKind::Unwrap(expr) => {
                // Unwrap Knowledge to get inner value
                self.lower_expr(expr)
            }

            HirExprKind::OntologyTerm { namespace, term } => {
                // Ontology terms are represented as string constants at runtime
                // Format: "namespace:term"
                let term_str = format!("{}:{}", namespace, term);
                let string_ty = HlirType::Ptr(Box::new(HlirType::U8));
                Some(
                    self.builder
                        .build_const(HlirConstant::String(term_str), string_ty),
                )
            }

            // ==================== ASYNC EXPRESSIONS ====================
            HirExprKind::Await { future } => {
                // For HLIR lowering, await becomes a runtime call
                // In a full implementation, this would call into the async runtime
                // For now, we lower it as a placeholder call
                let future_val = self.lower_expr(future)?;
                // Call runtime await function
                Some(self.builder.build_call(
                    "sounio_await".to_string(),
                    vec![future_val],
                    HlirType::from_hir(&expr.ty),
                ))
            }

            HirExprKind::Spawn { expr: spawn_expr } => {
                // Spawn creates a new task - call runtime spawn
                let inner_val = self.lower_expr(spawn_expr)?;
                Some(self.builder.build_call(
                    "sounio_spawn".to_string(),
                    vec![inner_val],
                    HlirType::Ptr(Box::new(HlirType::Struct("Task".to_string()))),
                ))
            }

            HirExprKind::AsyncBlock { body } => {
                // Async block creates a future - lower the body and wrap
                let block_val = self.lower_block(body);
                // Get the value or unit (avoiding borrow issues with closure)
                let arg_val = match block_val {
                    Some(v) => v,
                    None => self.builder.build_unit(),
                };
                // Wrap in a future structure
                Some(self.builder.build_call(
                    "sounio_make_future".to_string(),
                    vec![arg_val],
                    HlirType::Ptr(Box::new(HlirType::Struct("Future".to_string()))),
                ))
            }

            HirExprKind::Join { futures } => {
                // Join waits for multiple futures concurrently
                let mut future_vals = Vec::new();
                for f in futures {
                    if let Some(v) = self.lower_expr(f) {
                        future_vals.push(v);
                    }
                }
                // For now, return a tuple of results
                Some(self.builder.build_call(
                    "sounio_join".to_string(),
                    future_vals,
                    HlirType::from_hir(&expr.ty),
                ))
            }

            HirExprKind::Select { arms } => {
                // Select waits for the first ready future
                // For now, just evaluate the first arm
                if let Some(arm) = arms.first() {
                    let future_val = self.lower_expr(&arm.future)?;
                    Some(self.builder.build_call(
                        "sounio_select".to_string(),
                        vec![future_val],
                        HlirType::from_hir(&expr.ty),
                    ))
                } else {
                    Some(self.builder.build_unit())
                }
            }
        }
    }

    fn lower_literal(&mut self, lit: &HirLiteral, ty: &HlirType) -> ValueId {
        match lit {
            HirLiteral::Unit => self.builder.build_unit(),
            HirLiteral::Bool(b) => self.builder.build_bool(*b),
            HirLiteral::Int(i) => self
                .builder
                .build_const(HlirConstant::Int(*i, ty.clone()), ty.clone()),
            HirLiteral::Float(f) => self
                .builder
                .build_const(HlirConstant::Float(*f, ty.clone()), ty.clone()),
            HirLiteral::Char(c) => self
                .builder
                .build_const(HlirConstant::Int(*c as i64, HlirType::U32), HlirType::U32),
            HirLiteral::String(s) => self
                .builder
                .build_const(HlirConstant::String(s.clone()), ty.clone()),
        }
    }

    fn lower_binary_op(
        &mut self,
        op: HirBinaryOp,
        left: ValueId,
        right: ValueId,
        operand_ty: &HlirType,
        result_ty: &HlirType,
    ) -> ValueId {
        let is_float = operand_ty.is_float();
        let is_signed = operand_ty.is_signed();

        let hlir_op = match op {
            HirBinaryOp::Add => {
                if is_float {
                    BinaryOp::FAdd
                } else {
                    BinaryOp::Add
                }
            }
            HirBinaryOp::Sub => {
                if is_float {
                    BinaryOp::FSub
                } else {
                    BinaryOp::Sub
                }
            }
            HirBinaryOp::Mul => {
                if is_float {
                    BinaryOp::FMul
                } else {
                    BinaryOp::Mul
                }
            }
            HirBinaryOp::Div => {
                if is_float {
                    BinaryOp::FDiv
                } else if is_signed {
                    BinaryOp::SDiv
                } else {
                    BinaryOp::UDiv
                }
            }
            HirBinaryOp::Rem => {
                if is_float {
                    BinaryOp::FRem
                } else if is_signed {
                    BinaryOp::SRem
                } else {
                    BinaryOp::URem
                }
            }
            HirBinaryOp::Eq => {
                if is_float {
                    BinaryOp::FOEq
                } else {
                    BinaryOp::Eq
                }
            }
            HirBinaryOp::Ne => {
                if is_float {
                    BinaryOp::FONe
                } else {
                    BinaryOp::Ne
                }
            }
            HirBinaryOp::Lt => {
                if is_float {
                    BinaryOp::FOLt
                } else if is_signed {
                    BinaryOp::SLt
                } else {
                    BinaryOp::ULt
                }
            }
            HirBinaryOp::Le => {
                if is_float {
                    BinaryOp::FOLe
                } else if is_signed {
                    BinaryOp::SLe
                } else {
                    BinaryOp::ULe
                }
            }
            HirBinaryOp::Gt => {
                if is_float {
                    BinaryOp::FOGt
                } else if is_signed {
                    BinaryOp::SGt
                } else {
                    BinaryOp::UGt
                }
            }
            HirBinaryOp::Ge => {
                if is_float {
                    BinaryOp::FOGe
                } else if is_signed {
                    BinaryOp::SGe
                } else {
                    BinaryOp::UGe
                }
            }
            HirBinaryOp::And => BinaryOp::And,
            HirBinaryOp::Or => BinaryOp::Or,
            HirBinaryOp::BitAnd => BinaryOp::And,
            HirBinaryOp::BitOr => BinaryOp::Or,
            HirBinaryOp::BitXor => BinaryOp::Xor,
            HirBinaryOp::Shl => BinaryOp::Shl,
            HirBinaryOp::Shr => {
                if is_signed {
                    BinaryOp::AShr
                } else {
                    BinaryOp::LShr
                }
            }
            HirBinaryOp::PlusMinus => {
                // Treat PlusMinus as Add (uncertainty is handled at type-check time)
                if is_float {
                    BinaryOp::FAdd
                } else {
                    BinaryOp::Add
                }
            }
            HirBinaryOp::Concat => {
                // Array/slice concatenation
                BinaryOp::Concat
            }
        };

        self.builder
            .build_binary(hlir_op, left, right, result_ty.clone())
    }

    fn lower_unary_op(
        &mut self,
        op: HirUnaryOp,
        operand: ValueId,
        operand_ty: &HlirType,
    ) -> ValueId {
        match op {
            HirUnaryOp::Neg => {
                if operand_ty.is_float() {
                    self.builder.build_fneg(operand, operand_ty.clone())
                } else {
                    self.builder.build_neg(operand, operand_ty.clone())
                }
            }
            HirUnaryOp::Not => self.builder.build_not(operand),
            HirUnaryOp::Ref | HirUnaryOp::RefMut => {
                // Address-of is handled in lower_lvalue
                operand
            }
            HirUnaryOp::Deref => {
                // Dereference is handled in lower_expr
                operand
            }
        }
    }

    fn lower_if(
        &mut self,
        condition: &HirExpr,
        then_branch: &HirBlock,
        else_branch: Option<&HirExpr>,
        ty: &HlirType,
    ) -> Option<ValueId> {
        let cond_val = self.lower_expr(condition)?;

        let then_block = self.builder.create_block("if.then");
        let else_block = self.builder.create_block("if.else");
        let merge_block = self.builder.create_block("if.merge");

        self.builder
            .build_cond_branch(cond_val, then_block, else_block);

        // Then branch
        self.builder.switch_to_block(then_block);
        self.terminated = false;
        let then_val = self.lower_block(then_branch);
        let then_terminated = self.terminated;
        let then_exit_block = self.builder.current_block().unwrap();
        if !then_terminated {
            self.builder.build_branch(merge_block);
        }

        // Else branch
        self.builder.switch_to_block(else_block);
        self.terminated = false;
        let else_val = if let Some(else_expr) = else_branch {
            self.lower_expr(else_expr)
        } else {
            Some(self.builder.build_unit())
        };
        let else_terminated = self.terminated;
        let else_exit_block = self.builder.current_block().unwrap();
        if !else_terminated {
            self.builder.build_branch(merge_block);
        }

        // Merge block
        self.builder.switch_to_block(merge_block);
        self.terminated = then_terminated && else_terminated;

        if self.terminated {
            return None;
        }

        // Build phi if both branches produce values
        if *ty != HlirType::Void
            && let (Some(tv), Some(ev)) = (then_val, else_val)
        {
            let mut incoming = Vec::new();
            if !then_terminated {
                incoming.push((then_exit_block, tv));
            }
            if !else_terminated {
                incoming.push((else_exit_block, ev));
            }
            if !incoming.is_empty() {
                return Some(self.builder.build_phi(incoming, ty.clone()));
            }
        }

        None
    }

    fn lower_loop(&mut self, body: &HirBlock, ty: &HlirType) -> Option<ValueId> {
        let loop_block = self.builder.create_block("loop.body");
        let exit_block = self.builder.create_block("loop.exit");

        // Jump to loop
        self.builder.build_branch(loop_block);

        // Push loop context
        self.loop_stack.push(LoopContext {
            continue_block: loop_block,
            break_block: exit_block,
            break_values: Vec::new(),
        });

        // Loop body
        self.builder.switch_to_block(loop_block);
        self.terminated = false;
        self.lower_block(body);

        // If body didn't terminate, loop back
        if !self.terminated {
            self.builder.build_branch(loop_block);
        }

        // Pop loop context and collect break values
        let loop_ctx = self.loop_stack.pop().unwrap();

        // Exit block
        self.builder.switch_to_block(exit_block);
        self.terminated = false;

        // Build phi for break values
        if *ty != HlirType::Void && !loop_ctx.break_values.is_empty() {
            Some(self.builder.build_phi(loop_ctx.break_values, ty.clone()))
        } else {
            None
        }
    }

    fn lower_while(
        &mut self,
        condition: &HirExpr,
        body: &HirBlock,
        ty: &HlirType,
    ) -> Option<ValueId> {
        let cond_block = self.builder.create_block("while.cond");
        let body_block = self.builder.create_block("while.body");
        let exit_block = self.builder.create_block("while.exit");

        // Jump to condition check
        self.builder.build_branch(cond_block);

        // Push loop context (continue goes to condition, break goes to exit)
        self.loop_stack.push(LoopContext {
            continue_block: cond_block,
            break_block: exit_block,
            break_values: Vec::new(),
        });

        // Condition block - re-evaluate condition each iteration
        self.builder.switch_to_block(cond_block);
        self.terminated = false;
        let cond_val = self.lower_expr(condition);

        if let Some(cond) = cond_val {
            // Branch based on condition
            self.builder.build_cond_branch(cond, body_block, exit_block);
        } else {
            // If condition lowering failed, assume true and enter body
            self.builder.build_branch(body_block);
        }

        // Body block
        self.builder.switch_to_block(body_block);
        self.terminated = false;
        self.lower_block(body);

        // If body didn't terminate, jump back to condition check
        if !self.terminated {
            self.builder.build_branch(cond_block);
        }

        // Pop loop context and collect break values
        let loop_ctx = self.loop_stack.pop().unwrap();

        // Exit block
        self.builder.switch_to_block(exit_block);
        self.terminated = false;

        // Build phi for break values
        if *ty != HlirType::Void && !loop_ctx.break_values.is_empty() {
            Some(self.builder.build_phi(loop_ctx.break_values, ty.clone()))
        } else {
            None
        }
    }

    fn lower_match(
        &mut self,
        scrutinee: &HirExpr,
        arms: &[HirMatchArm],
        ty: &HlirType,
    ) -> Option<ValueId> {
        let scrut_val = self.lower_expr(scrutinee)?;
        let scrut_ty = HlirType::from_hir(&scrutinee.ty);

        // For simple integer matches with literals (and optional wildcard/binding), use switch
        if scrut_ty.is_integer()
            && arms.iter().all(|a| {
                matches!(
                    a.pattern,
                    HirPattern::Literal(_) | HirPattern::Wildcard | HirPattern::Binding { .. }
                ) && a.guard.is_none()
            })
            && arms
                .iter()
                .any(|a| matches!(a.pattern, HirPattern::Literal(_)))
        {
            return self.lower_match_switch(scrut_val, arms, ty);
        }

        // For enum variant matching, use tag-based switch
        if let HirType::Named { name, .. } = &scrutinee.ty
            && self.enums.contains_key(name)
        {
            return self.lower_match_enum(scrut_val, name, arms, ty);
        }

        // General case: chain of if-else
        self.lower_match_chain(scrut_val, &scrut_ty, arms, ty)
    }

    fn lower_match_switch(
        &mut self,
        scrut: ValueId,
        arms: &[HirMatchArm],
        ty: &HlirType,
    ) -> Option<ValueId> {
        let merge_block = self.builder.create_block("match.merge");
        let default_block = self.builder.create_block("match.default");

        let mut cases = Vec::new();
        let mut arm_results = Vec::new();
        let mut has_wildcard = false;
        let mut wildcard_arm: Option<&HirMatchArm> = None;

        // First pass: collect cases and check for wildcard
        for arm in arms {
            match &arm.pattern {
                HirPattern::Literal(HirLiteral::Int(n)) => {
                    let arm_block = self.builder.create_block("match.case");
                    cases.push((*n, arm_block));
                }
                HirPattern::Wildcard | HirPattern::Binding { .. } => {
                    has_wildcard = true;
                    wildcard_arm = Some(arm);
                }
                _ => {}
            }
        }

        // Build the switch from current block
        let current = self.builder.current_block().unwrap();
        self.builder.switch_to_block(current);
        self.builder
            .build_switch(scrut, default_block, cases.clone());

        // Generate code for each case
        for (arm, (_, arm_block)) in arms
            .iter()
            .filter(|a| matches!(a.pattern, HirPattern::Literal(HirLiteral::Int(_))))
            .zip(cases.iter())
        {
            self.builder.switch_to_block(*arm_block);
            self.terminated = false;
            let result = self.lower_expr(&arm.body);
            let arm_exit = self.builder.current_block().unwrap();
            if !self.terminated {
                self.builder.build_branch(merge_block);
                if let Some(v) = result {
                    arm_results.push((arm_exit, v));
                }
            }
        }

        // Default block (wildcard or unreachable)
        self.builder.switch_to_block(default_block);
        self.terminated = false;
        if has_wildcard {
            if let Some(arm) = wildcard_arm {
                // Bind the value if it's a binding pattern
                if let HirPattern::Binding { name, .. } = &arm.pattern {
                    self.builder.bind_var(name, scrut);
                }
                let result = self.lower_expr(&arm.body);
                let arm_exit = self.builder.current_block().unwrap();
                if !self.terminated {
                    self.builder.build_branch(merge_block);
                    if let Some(v) = result {
                        arm_results.push((arm_exit, v));
                    }
                }
            }
        } else {
            self.builder.build_unreachable();
        }

        self.builder.switch_to_block(merge_block);
        self.terminated = false;

        if *ty != HlirType::Void && !arm_results.is_empty() {
            Some(self.builder.build_phi(arm_results, ty.clone()))
        } else {
            None
        }
    }

    fn lower_match_enum(
        &mut self,
        scrut: ValueId,
        enum_name: &str,
        arms: &[HirMatchArm],
        ty: &HlirType,
    ) -> Option<ValueId> {
        let merge_block = self.builder.create_block("match.merge");
        let default_block = self.builder.create_block("match.default");

        // Extract the tag from the enum value (first field)
        let tag = self.builder.build_extract(scrut, 0, HlirType::I64);

        let mut cases = Vec::new();
        let mut arm_results = Vec::new();
        let mut has_wildcard = false;
        let mut wildcard_arm: Option<&HirMatchArm> = None;

        // First pass: collect variant cases
        for arm in arms {
            match &arm.pattern {
                HirPattern::Variant {
                    enum_name: _,
                    variant,
                    patterns,
                } => {
                    let tag_val = self.get_variant_tag(enum_name, variant);
                    let arm_block = self.builder.create_block(format!("match.{}", variant));
                    cases.push((tag_val, arm_block, variant.clone(), patterns.clone()));
                }
                HirPattern::Wildcard | HirPattern::Binding { .. } => {
                    has_wildcard = true;
                    wildcard_arm = Some(arm);
                }
                _ => {}
            }
        }

        // Build the switch
        let switch_cases: Vec<_> = cases.iter().map(|(t, b, _, _)| (*t, *b)).collect();
        self.builder.build_switch(tag, default_block, switch_cases);

        // Generate code for each variant case
        for (arm, (_, arm_block, variant, patterns)) in arms
            .iter()
            .filter(|a| matches!(a.pattern, HirPattern::Variant { .. }))
            .zip(cases.iter())
        {
            self.builder.switch_to_block(*arm_block);
            self.terminated = false;

            // Bind variant fields to pattern variables
            let field_types = self.get_variant_fields(enum_name, variant);
            for (i, pattern) in patterns.iter().enumerate() {
                if let HirPattern::Binding { name, .. } = pattern {
                    let field_ty = field_types.get(i).cloned().unwrap_or(HlirType::Void);
                    // Fields start at index 1 (index 0 is the tag)
                    let field_val = self.builder.build_extract(scrut, i + 1, field_ty);
                    self.builder.bind_var(name, field_val);
                }
            }

            // Handle guard if present
            if let Some(guard) = &arm.guard {
                let guard_block = self.builder.create_block("match.guard");
                let next_block = self.builder.create_block("match.next");

                let guard_val = self.lower_expr(guard);
                if let Some(gv) = guard_val {
                    self.builder.build_cond_branch(gv, guard_block, next_block);

                    self.builder.switch_to_block(guard_block);
                    self.terminated = false;
                    let result = self.lower_expr(&arm.body);
                    let arm_exit = self.builder.current_block().unwrap();
                    if !self.terminated {
                        self.builder.build_branch(merge_block);
                        if let Some(v) = result {
                            arm_results.push((arm_exit, v));
                        }
                    }

                    // next_block falls through to default
                    self.builder.switch_to_block(next_block);
                    self.builder.build_branch(default_block);
                }
            } else {
                let result = self.lower_expr(&arm.body);
                let arm_exit = self.builder.current_block().unwrap();
                if !self.terminated {
                    self.builder.build_branch(merge_block);
                    if let Some(v) = result {
                        arm_results.push((arm_exit, v));
                    }
                }
            }
        }

        // Default block
        self.builder.switch_to_block(default_block);
        self.terminated = false;
        if has_wildcard {
            if let Some(arm) = wildcard_arm {
                if let HirPattern::Binding { name, .. } = &arm.pattern {
                    self.builder.bind_var(name, scrut);
                }
                let result = self.lower_expr(&arm.body);
                let arm_exit = self.builder.current_block().unwrap();
                if !self.terminated {
                    self.builder.build_branch(merge_block);
                    if let Some(v) = result {
                        arm_results.push((arm_exit, v));
                    }
                }
            }
        } else {
            self.builder.build_unreachable();
        }

        self.builder.switch_to_block(merge_block);
        self.terminated = false;

        if *ty != HlirType::Void && !arm_results.is_empty() {
            Some(self.builder.build_phi(arm_results, ty.clone()))
        } else {
            None
        }
    }

    fn lower_match_chain(
        &mut self,
        scrut: ValueId,
        scrut_ty: &HlirType,
        arms: &[HirMatchArm],
        ty: &HlirType,
    ) -> Option<ValueId> {
        if arms.is_empty() {
            return None;
        }

        let merge_block = self.builder.create_block("match.merge");
        let mut arm_results = Vec::new();

        for (i, arm) in arms.iter().enumerate() {
            let is_last = i == arms.len() - 1;
            let next_block = if is_last {
                merge_block
            } else {
                self.builder.create_block(format!("match.test.{}", i + 1))
            };

            let arm_block = self.builder.create_block(format!("match.arm.{}", i));

            // Check pattern
            let matches = self.lower_pattern_check(&arm.pattern, scrut, scrut_ty);

            if let Some(cond) = matches {
                // Check guard if present
                if let Some(guard) = &arm.guard {
                    let guard_check_block = self.builder.create_block("match.guard.check");
                    self.builder
                        .build_cond_branch(cond, guard_check_block, next_block);

                    self.builder.switch_to_block(guard_check_block);
                    self.terminated = false;
                    // First bind pattern variables so guard can use them
                    self.bind_pattern(&arm.pattern, scrut);
                    let guard_val = self.lower_expr(guard);
                    if let Some(gv) = guard_val {
                        self.builder.build_cond_branch(gv, arm_block, next_block);
                    } else {
                        self.builder.build_branch(arm_block);
                    }
                } else {
                    self.builder.build_cond_branch(cond, arm_block, next_block);
                }
            } else {
                // Wildcard or binding - always matches, but check guard
                if let Some(guard) = &arm.guard {
                    self.bind_pattern(&arm.pattern, scrut);
                    let guard_val = self.lower_expr(guard);
                    if let Some(gv) = guard_val {
                        self.builder.build_cond_branch(gv, arm_block, next_block);
                    } else {
                        self.builder.build_branch(arm_block);
                    }
                } else {
                    self.builder.build_branch(arm_block);
                }
            }

            // Arm body
            self.builder.switch_to_block(arm_block);
            self.terminated = false;

            // Bind pattern variables (if not already bound for guard)
            if arm.guard.is_none() {
                self.bind_pattern(&arm.pattern, scrut);
            }

            let result = self.lower_expr(&arm.body);
            let arm_exit = self.builder.current_block().unwrap();

            if !self.terminated {
                self.builder.build_branch(merge_block);
                if let Some(v) = result {
                    arm_results.push((arm_exit, v));
                }
            }

            if !is_last {
                self.builder.switch_to_block(next_block);
            }
        }

        self.builder.switch_to_block(merge_block);
        self.terminated = false;

        if *ty != HlirType::Void && !arm_results.is_empty() {
            Some(self.builder.build_phi(arm_results, ty.clone()))
        } else {
            None
        }
    }

    fn lower_pattern_check(
        &mut self,
        pattern: &HirPattern,
        scrut: ValueId,
        scrut_ty: &HlirType,
    ) -> Option<ValueId> {
        match pattern {
            HirPattern::Wildcard => None,
            HirPattern::Binding { .. } => None,
            HirPattern::Literal(lit) => {
                let lit_val = self.lower_literal(lit, scrut_ty);
                Some(self.builder.build_eq(scrut, lit_val))
            }
            HirPattern::Tuple(patterns) => {
                // Check all tuple elements
                let mut combined: Option<ValueId> = None;
                for (i, p) in patterns.iter().enumerate() {
                    let elem_ty = if let HlirType::Tuple(elems) = scrut_ty {
                        elems.get(i).cloned().unwrap_or(HlirType::Void)
                    } else {
                        HlirType::Void
                    };
                    let elem = self.builder.build_extract(scrut, i, elem_ty.clone());
                    if let Some(check) = self.lower_pattern_check(p, elem, &elem_ty) {
                        combined = Some(match combined {
                            Some(prev) => self.builder.build_binary(
                                BinaryOp::And,
                                prev,
                                check,
                                HlirType::Bool,
                            ),
                            None => check,
                        });
                    }
                }
                combined
            }
            HirPattern::Or(patterns) => {
                // Check if any pattern matches
                let mut combined: Option<ValueId> = None;
                for p in patterns {
                    if let Some(check) = self.lower_pattern_check(p, scrut, scrut_ty) {
                        combined = Some(match combined {
                            Some(prev) => {
                                self.builder
                                    .build_binary(BinaryOp::Or, prev, check, HlirType::Bool)
                            }
                            None => check,
                        });
                    }
                }
                combined
            }
            HirPattern::Struct { name, fields } => {
                // Check struct fields
                let mut combined: Option<ValueId> = None;
                if let Some(struct_fields) = self.structs.get(name) {
                    for (field_name, field_pattern) in fields {
                        // Find field index
                        if let Some(idx) = struct_fields.iter().position(|(n, _)| n == field_name) {
                            let field_ty = struct_fields[idx].1.clone();
                            let field_val =
                                self.builder.build_extract(scrut, idx, field_ty.clone());
                            if let Some(check) =
                                self.lower_pattern_check(field_pattern, field_val, &field_ty)
                            {
                                combined = Some(match combined {
                                    Some(prev) => self.builder.build_binary(
                                        BinaryOp::And,
                                        prev,
                                        check,
                                        HlirType::Bool,
                                    ),
                                    None => check,
                                });
                            }
                        }
                    }
                }
                combined
            }
            HirPattern::Variant {
                enum_name,
                variant,
                patterns,
            } => {
                // Check tag and then fields
                let tag_val = self.get_variant_tag(enum_name, variant);
                let tag = self.builder.build_extract(scrut, 0, HlirType::I64);
                let tag_const = self
                    .builder
                    .build_const(HlirConstant::Int(tag_val, HlirType::I64), HlirType::I64);
                let tag_check = self.builder.build_eq(tag, tag_const);

                // Check field patterns
                let field_types = self.get_variant_fields(enum_name, variant);
                let mut combined = Some(tag_check);

                for (i, pattern) in patterns.iter().enumerate() {
                    let field_ty = field_types.get(i).cloned().unwrap_or(HlirType::Void);
                    let field_val = self.builder.build_extract(scrut, i + 1, field_ty.clone());
                    if let Some(check) = self.lower_pattern_check(pattern, field_val, &field_ty) {
                        combined = Some(match combined {
                            Some(prev) => self.builder.build_binary(
                                BinaryOp::And,
                                prev,
                                check,
                                HlirType::Bool,
                            ),
                            None => check,
                        });
                    }
                }

                combined
            }
        }
    }

    fn bind_pattern(&mut self, pattern: &HirPattern, value: ValueId) {
        match pattern {
            HirPattern::Binding { name, .. } => {
                self.builder.bind_var(name, value);
            }
            HirPattern::Tuple(patterns) => {
                for (i, p) in patterns.iter().enumerate() {
                    let elem = self.builder.build_extract(value, i, HlirType::Void);
                    self.bind_pattern(p, elem);
                }
            }
            HirPattern::Struct { name, fields } => {
                if let Some(struct_fields) = self.structs.get(name).cloned() {
                    for (field_name, field_pattern) in fields {
                        if let Some(idx) = struct_fields.iter().position(|(n, _)| n == field_name) {
                            let field_ty = struct_fields[idx].1.clone();
                            let field_val = self.builder.build_extract(value, idx, field_ty);
                            self.bind_pattern(field_pattern, field_val);
                        }
                    }
                }
            }
            HirPattern::Variant {
                enum_name,
                variant,
                patterns,
            } => {
                let field_types = self.get_variant_fields(enum_name, variant);
                for (i, pattern) in patterns.iter().enumerate() {
                    let field_ty = field_types.get(i).cloned().unwrap_or(HlirType::Void);
                    let field_val = self.builder.build_extract(value, i + 1, field_ty);
                    self.bind_pattern(pattern, field_val);
                }
            }
            HirPattern::Or(patterns) => {
                // For or patterns, bind the first pattern (they should all bind the same vars)
                if let Some(p) = patterns.first() {
                    self.bind_pattern(p, value);
                }
            }
            _ => {}
        }
    }

    /// Lower closure expression
    fn lower_closure(
        &mut self,
        params: &[HirParam],
        body: &HirExpr,
        ty: &HlirType,
    ) -> Option<ValueId> {
        // For now, we implement closures as function pointers with an environment
        // The environment is a tuple of captured variables

        // Collect free variables in the body (simplified - in practice we'd do a proper analysis)
        // For now, just create a closure struct with the function pointer

        // Build the closure type: { fn_ptr, env_ptr }
        let closure_ty = ty.clone();

        // For a simple implementation, we'll just return a function pointer
        // A full implementation would:
        // 1. Analyze captured variables
        // 2. Allocate environment struct
        // 3. Store captured values
        // 4. Create a wrapper function that unpacks the environment

        // Generate a unique name for the closure function
        let closure_name = format!("__closure_{}", self.builder.fresh_value().0);

        // For now, create a simple function reference
        // This is a simplified implementation - full closures would require
        // environment capture analysis
        let fn_ptr = self.builder.build_const(
            HlirConstant::FunctionRef(closure_name),
            HlirType::Ptr(Box::new(HlirType::Void)),
        );

        // Create a null environment pointer (no captures for now)
        let env_ptr = self.builder.build_const(
            HlirConstant::Null(HlirType::Ptr(Box::new(HlirType::Void))),
            HlirType::Ptr(Box::new(HlirType::Void)),
        );

        // Build the closure tuple
        Some(self.builder.build_tuple(vec![fn_ptr, env_ptr], closure_ty))
    }

    /// Lower enum variant construction
    fn lower_variant(
        &mut self,
        enum_name: &str,
        variant: &str,
        fields: &[HirExpr],
        ty: &HlirType,
    ) -> Option<ValueId> {
        // Enum variants are represented as: { tag: i64, field0, field1, ... }
        let tag = self.get_variant_tag(enum_name, variant);
        let tag_val = self
            .builder
            .build_const(HlirConstant::Int(tag, HlirType::I64), HlirType::I64);

        let mut values = vec![tag_val];
        for field in fields {
            if let Some(v) = self.lower_expr(field) {
                values.push(v);
            }
        }

        // Build as a tuple (tag + fields)
        Some(self.builder.build_tuple(values, ty.clone()))
    }

    /// Lower effect perform operation
    fn lower_effect_perform(
        &mut self,
        effect: &str,
        op: &str,
        args: &[HirExpr],
        ty: &HlirType,
    ) -> Option<ValueId> {
        let arg_vals: Vec<_> = args.iter().filter_map(|a| self.lower_expr(a)).collect();

        // Look up effect operation return type
        let ret_ty = if let Some(ops) = self.effects.get(effect) {
            ops.iter()
                .find(|(name, _, _)| name == op)
                .map(|(_, _, ret)| ret.clone())
                .unwrap_or_else(|| ty.clone())
        } else {
            ty.clone()
        };

        // Emit the perform effect operation
        let result = self.builder.fresh_value();
        let instr = HlirInstr {
            result: Some(result),
            op: Op::PerformEffect {
                effect: effect.to_string(),
                op: op.to_string(),
                args: arg_vals,
            },
            ty: ret_ty,
        };
        self.builder
            .func
            .get_block_mut(self.builder.current_block().unwrap())
            .unwrap()
            .instructions
            .push(instr);

        Some(result)
    }

    /// Lower effect handle expression
    fn lower_effect_handle(
        &mut self,
        expr: &HirExpr,
        handler: &str,
        ty: &HlirType,
    ) -> Option<ValueId> {
        // Effect handlers require continuation support
        // For now, we implement a simplified version:
        // 1. Create a handler context
        // 2. Execute the expression
        // 3. The handler intercepts effect operations

        // Look up the handler's effect
        let _effect = self.handlers.get(handler).cloned();

        // For a simplified implementation, we just evaluate the expression
        // A full implementation would:
        // 1. Push the handler onto a handler stack
        // 2. Execute the expression
        // 3. When a perform is encountered, look up the handler
        // 4. Execute the handler case with the continuation

        // Create blocks for the handler structure
        let handle_block = self.builder.create_block("handle.body");
        let resume_block = self.builder.create_block("handle.resume");

        self.builder.build_branch(handle_block);

        // Execute the expression in the handle block
        self.builder.switch_to_block(handle_block);
        self.terminated = false;
        let result = self.lower_expr(expr);

        if !self.terminated {
            self.builder.build_branch(resume_block);
        }

        // Resume block collects the result
        self.builder.switch_to_block(resume_block);
        self.terminated = false;

        // Return the result
        if let Some(r) = result {
            Some(r)
        } else {
            Some(self.builder.build_unit())
        }
    }

    /// Lower probabilistic sample expression
    fn lower_sample(&mut self, dist: &HirExpr, ty: &HlirType) -> Option<ValueId> {
        // Sampling from distributions is handled as a special operation
        // In a full implementation, this would:
        // 1. Evaluate the distribution expression
        // 2. Call a runtime sampling function
        // 3. Return the sampled value

        let dist_val = self.lower_expr(dist)?;

        // Call a runtime sampling function
        // The actual implementation depends on the distribution type
        Some(
            self.builder
                .build_call("__sample", vec![dist_val], ty.clone()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_hir() -> Hir {
        use crate::common::NodeId;

        Hir {
            items: vec![HirItem::Function(HirFn {
                id: NodeId(0),
                name: "add".to_string(),
                ty: HirFnType {
                    params: vec![
                        HirParam {
                            id: NodeId(1),
                            name: "a".to_string(),
                            ty: HirType::I64,
                            is_mut: false,
                        },
                        HirParam {
                            id: NodeId(2),
                            name: "b".to_string(),
                            ty: HirType::I64,
                            is_mut: false,
                        },
                    ],
                    return_type: Box::new(HirType::I64),
                    effects: Vec::new(),
                },
                body: HirBlock {
                    stmts: vec![HirStmt::Expr(HirExpr {
                        id: NodeId(3),
                        kind: HirExprKind::Binary {
                            op: HirBinaryOp::Add,
                            left: Box::new(HirExpr {
                                id: NodeId(4),
                                kind: HirExprKind::Local("a".to_string()),
                                ty: HirType::I64,
                            }),
                            right: Box::new(HirExpr {
                                id: NodeId(5),
                                kind: HirExprKind::Local("b".to_string()),
                                ty: HirType::I64,
                            }),
                        },
                        ty: HirType::I64,
                    })],
                    ty: HirType::I64,
                },
                abi: crate::ast::Abi::Rust,
                is_exported: false,
            })],
            externs: Vec::new(),
        }
    }

    #[test]
    fn test_lower_simple_function() {
        let hir = make_simple_hir();
        let hlir = lower(&hir);

        assert_eq!(hlir.functions.len(), 1);
        let func = &hlir.functions[0];
        assert_eq!(func.name, "add");
        assert_eq!(func.params.len(), 2);
        assert!(!func.blocks.is_empty());
    }

    fn make_match_hir() -> Hir {
        use crate::common::NodeId;

        Hir {
            items: vec![HirItem::Function(HirFn {
                id: NodeId(0),
                name: "classify".to_string(),
                ty: HirFnType {
                    params: vec![HirParam {
                        id: NodeId(1),
                        name: "x".to_string(),
                        ty: HirType::I64,
                        is_mut: false,
                    }],
                    return_type: Box::new(HirType::I64),
                    effects: Vec::new(),
                },
                body: HirBlock {
                    stmts: vec![HirStmt::Expr(HirExpr {
                        id: NodeId(2),
                        kind: HirExprKind::Match {
                            scrutinee: Box::new(HirExpr {
                                id: NodeId(3),
                                kind: HirExprKind::Local("x".to_string()),
                                ty: HirType::I64,
                            }),
                            arms: vec![
                                HirMatchArm {
                                    pattern: HirPattern::Literal(HirLiteral::Int(0)),
                                    guard: None,
                                    body: HirExpr {
                                        id: NodeId(4),
                                        kind: HirExprKind::Literal(HirLiteral::Int(0)),
                                        ty: HirType::I64,
                                    },
                                },
                                HirMatchArm {
                                    pattern: HirPattern::Literal(HirLiteral::Int(1)),
                                    guard: None,
                                    body: HirExpr {
                                        id: NodeId(5),
                                        kind: HirExprKind::Literal(HirLiteral::Int(10)),
                                        ty: HirType::I64,
                                    },
                                },
                                HirMatchArm {
                                    pattern: HirPattern::Wildcard,
                                    guard: None,
                                    body: HirExpr {
                                        id: NodeId(6),
                                        kind: HirExprKind::Literal(HirLiteral::Int(100)),
                                        ty: HirType::I64,
                                    },
                                },
                            ],
                        },
                        ty: HirType::I64,
                    })],
                    ty: HirType::I64,
                },
                abi: crate::ast::Abi::Rust,
                is_exported: false,
            })],
            externs: Vec::new(),
        }
    }

    #[test]
    fn test_lower_match_with_switch() {
        let hir = make_match_hir();
        let hlir = lower(&hir);

        assert_eq!(hlir.functions.len(), 1);
        let func = &hlir.functions[0];
        assert_eq!(func.name, "classify");

        // Should have multiple blocks for the match
        assert!(func.blocks.len() > 1);

        // Check that we have a switch terminator
        let has_switch = func
            .blocks
            .iter()
            .any(|b| matches!(b.terminator, HlirTerminator::Switch { .. }));
        assert!(has_switch, "Expected switch terminator for integer match");
    }

    #[test]
    fn test_lower_match_with_guard() {
        use crate::common::NodeId;

        let hir = Hir {
            items: vec![HirItem::Function(HirFn {
                id: NodeId(0),
                name: "guarded".to_string(),
                ty: HirFnType {
                    params: vec![HirParam {
                        id: NodeId(1),
                        name: "x".to_string(),
                        ty: HirType::I64,
                        is_mut: false,
                    }],
                    return_type: Box::new(HirType::I64),
                    effects: Vec::new(),
                },
                body: HirBlock {
                    stmts: vec![HirStmt::Expr(HirExpr {
                        id: NodeId(2),
                        kind: HirExprKind::Match {
                            scrutinee: Box::new(HirExpr {
                                id: NodeId(3),
                                kind: HirExprKind::Local("x".to_string()),
                                ty: HirType::I64,
                            }),
                            arms: vec![
                                HirMatchArm {
                                    pattern: HirPattern::Binding {
                                        name: "n".to_string(),
                                        mutable: false,
                                    },
                                    guard: Some(Box::new(HirExpr {
                                        id: NodeId(4),
                                        kind: HirExprKind::Binary {
                                            op: HirBinaryOp::Gt,
                                            left: Box::new(HirExpr {
                                                id: NodeId(5),
                                                kind: HirExprKind::Local("n".to_string()),
                                                ty: HirType::I64,
                                            }),
                                            right: Box::new(HirExpr {
                                                id: NodeId(6),
                                                kind: HirExprKind::Literal(HirLiteral::Int(10)),
                                                ty: HirType::I64,
                                            }),
                                        },
                                        ty: HirType::Bool,
                                    })),
                                    body: HirExpr {
                                        id: NodeId(7),
                                        kind: HirExprKind::Literal(HirLiteral::Int(1)),
                                        ty: HirType::I64,
                                    },
                                },
                                HirMatchArm {
                                    pattern: HirPattern::Wildcard,
                                    guard: None,
                                    body: HirExpr {
                                        id: NodeId(8),
                                        kind: HirExprKind::Literal(HirLiteral::Int(0)),
                                        ty: HirType::I64,
                                    },
                                },
                            ],
                        },
                        ty: HirType::I64,
                    })],
                    ty: HirType::I64,
                },
                abi: crate::ast::Abi::Rust,
                is_exported: false,
            })],
            externs: Vec::new(),
        };

        let hlir = lower(&hir);
        assert_eq!(hlir.functions.len(), 1);
        let func = &hlir.functions[0];

        // Should have conditional branches for guards
        let has_cond_branch = func
            .blocks
            .iter()
            .any(|b| matches!(b.terminator, HlirTerminator::CondBranch { .. }));
        assert!(has_cond_branch, "Expected conditional branch for guard");
    }

    #[test]
    fn test_lower_pointer_index_assign() {
        use crate::common::NodeId;

        let ptr_ty = HirType::RawPointer {
            mutable: true,
            inner: Box::new(HirType::U8),
        };
        let idx_ty = HirType::Usize;
        let val_ty = HirType::U8;

        let hir = Hir {
            items: vec![HirItem::Function(HirFn {
                id: NodeId(0),
                name: "write_byte".to_string(),
                ty: HirFnType {
                    params: vec![
                        HirParam {
                            id: NodeId(1),
                            name: "out".to_string(),
                            ty: ptr_ty.clone(),
                            is_mut: false,
                        },
                        HirParam {
                            id: NodeId(2),
                            name: "idx".to_string(),
                            ty: idx_ty.clone(),
                            is_mut: false,
                        },
                        HirParam {
                            id: NodeId(3),
                            name: "val".to_string(),
                            ty: val_ty.clone(),
                            is_mut: false,
                        },
                    ],
                    return_type: Box::new(HirType::Unit),
                    effects: Vec::new(),
                },
                body: HirBlock {
                    stmts: vec![HirStmt::Assign {
                        target: HirExpr {
                            id: NodeId(4),
                            kind: HirExprKind::Index {
                                base: Box::new(HirExpr {
                                    id: NodeId(5),
                                    kind: HirExprKind::Local("out".to_string()),
                                    ty: ptr_ty,
                                }),
                                index: Box::new(HirExpr {
                                    id: NodeId(6),
                                    kind: HirExprKind::Local("idx".to_string()),
                                    ty: idx_ty,
                                }),
                            },
                            ty: HirType::U8,
                        },
                        value: HirExpr {
                            id: NodeId(7),
                            kind: HirExprKind::Local("val".to_string()),
                            ty: val_ty,
                        },
                    }],
                    ty: HirType::Unit,
                },
                abi: crate::ast::Abi::Rust,
                is_exported: false,
            })],
            externs: Vec::new(),
        };

        let hlir = lower(&hir);
        let func = &hlir.functions[0];

        let mut has_gep = false;
        let mut has_store = false;

        for instr in func.blocks.iter().flat_map(|b| b.instructions.iter()) {
            if matches!(&instr.op, Op::GetElementPtr { .. }) {
                has_gep = true;
            }
            if matches!(&instr.op, Op::Store { .. }) {
                has_store = true;
            }
        }

        assert!(has_gep, "Expected GEP for pointer index assignment");
        assert!(has_store, "Expected store for pointer index assignment");
    }
}

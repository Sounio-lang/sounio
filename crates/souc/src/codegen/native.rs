//! Minimal HIR -> x86-64 machine code backend.
//!
//! Current scope:
//! - Linux x86-64 process binaries via `runtime::elf::NativeEmitter`
//! - Integer/bool literals, arithmetic, comparisons
//! - Locals (`let`, assignment), calls, `if`, `while`, `return`
//! - Builtin `print("...")` and `exit(code)` through `RuntimeCodegen` stubs

use crate::hir::{
    Hir, HirBinaryOp, HirBlock, HirExpr, HirExprKind, HirFn, HirItem, HirLiteral, HirStmt,
    HirUnaryOp,
};
use crate::runtime::elf::NativeEmitter;
use crate::runtime::runtime::RuntimeCodegen;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
struct PendingRelocation {
    imm_offset: usize,
    target: RelocationTarget,
}

#[derive(Debug, Clone)]
enum RelocationTarget {
    FunctionOrBuiltin(String),
    Builtin(String),
}

pub struct NativeCodegen {
    emitter: NativeEmitter,
    runtime: RuntimeCodegen,
    /// User function name -> code offset in `emitter.code`.
    fn_offsets: HashMap<String, usize>,
    /// `call rel32` patches to resolve after all functions are laid out.
    relocations: Vec<PendingRelocation>,
    /// User-defined function names declared in the module.
    user_functions: HashSet<String>,
    /// Local variable name -> rbp-relative slot offset (negative).
    locals: HashMap<String, i32>,
    /// Current function stack frame size in bytes (16-byte aligned).
    frame_size: i32,
    /// `return` jumps that must branch to the function epilogue.
    return_jumps: Vec<usize>,
}

impl NativeCodegen {
    pub fn new() -> Self {
        Self {
            emitter: NativeEmitter::new(),
            runtime: RuntimeCodegen::new(),
            fn_offsets: HashMap::new(),
            relocations: Vec::new(),
            user_functions: HashSet::new(),
            locals: HashMap::new(),
            frame_size: 0,
            return_jumps: Vec::new(),
        }
    }

    /// Compile a complete HIR module to a native executable image.
    pub fn compile(&mut self, module: &Hir) -> Result<Vec<u8>, String> {
        if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            return Err("native backend currently supports Linux x86-64 only".to_string());
        }

        self.reset_module_state();

        self.runtime.emit_runtime();
        self.emitter = self.runtime.emitter().clone();

        let functions: Vec<&HirFn> = module
            .items
            .iter()
            .filter_map(|item| match item {
                HirItem::Function(f) => Some(f),
                _ => None,
            })
            .collect();

        if functions.is_empty() {
            return Err("No functions found in HIR module".to_string());
        }

        self.user_functions = functions.iter().map(|f| f.name.clone()).collect();

        for func in functions {
            self.compile_fn(func)?;
        }

        if !self.fn_offsets.contains_key("main") {
            return Err("No main function found".to_string());
        }

        // The native entry trampoline calls `main()` with no arguments.
        // Reject other signatures early to avoid running with garbage registers.
        if let Some(main_fn) = module.items.iter().find_map(|item| match item {
            HirItem::Function(f) if f.name == "main" => Some(f),
            _ => None,
        }) {
            if !main_fn.ty.params.is_empty() {
                return Err(format!(
                    "native backend requires `fn main()` to take 0 parameters (got {})",
                    main_fn.ty.params.len()
                ));
            }
        }

        let entry_offset = self.emit_entry_trampoline();
        self.apply_relocations()?;
        Ok(self.emitter.finalize(entry_offset))
    }

    fn reset_module_state(&mut self) {
        self.emitter = NativeEmitter::new();
        self.runtime = RuntimeCodegen::new();
        self.fn_offsets.clear();
        self.relocations.clear();
        self.user_functions.clear();
        self.locals.clear();
        self.frame_size = 0;
        self.return_jumps.clear();
    }

    fn compile_fn(&mut self, func: &HirFn) -> Result<(), String> {
        if self.fn_offsets.contains_key(&func.name) {
            return Err(format!("Duplicate function '{}'", func.name));
        }

        let offset = self.emitter.code.len();
        self.fn_offsets.insert(func.name.clone(), offset);

        self.prepare_frame(func);
        self.emit_prologue();
        self.spill_params(func)?;
        self.compile_block(&func.body)?;

        let epilogue_offset = self.emitter.code.len();
        self.emit_epilogue();
        self.patch_return_jumps(epilogue_offset)?;
        Ok(())
    }

    fn prepare_frame(&mut self, func: &HirFn) {
        self.locals.clear();
        self.frame_size = 0;
        self.return_jumps.clear();

        for param in &func.ty.params {
            self.allocate_local_slot(&param.name);
        }

        let mut let_bindings = Vec::new();
        collect_let_bindings_block(&func.body, &mut let_bindings);
        for name in let_bindings {
            self.allocate_local_slot(&name);
        }

        let raw_size = (self.locals.len() as i32) * 8;
        self.frame_size = align16(raw_size);
    }

    fn allocate_local_slot(&mut self, name: &str) {
        if self.locals.contains_key(name) {
            return;
        }
        let slot_index = self.locals.len() as i32 + 1;
        self.locals.insert(name.to_string(), -(slot_index * 8));
    }

    fn emit_prologue(&mut self) {
        self.emitter.emit_code(&[
            0x55, // push rbp
            0x48, 0x89, 0xe5, // mov rbp, rsp
        ]);

        if self.frame_size > 0 {
            self.emitter.emit_code(&[0x48, 0x81, 0xec]); // sub rsp, imm32
            self.emitter
                .emit_code(&(self.frame_size as u32).to_le_bytes());
        }
    }

    fn spill_params(&mut self, func: &HirFn) -> Result<(), String> {
        if func.ty.params.len() > 6 {
            return Err(format!(
                "Function '{}' has {} params; max supported is 6",
                func.name,
                func.ty.params.len()
            ));
        }

        for (i, param) in func.ty.params.iter().enumerate() {
            let offset = self
                .locals
                .get(&param.name)
                .copied()
                .ok_or_else(|| format!("Missing local slot for param '{}'", param.name))?;
            self.emit_store_param_reg(offset, i)?;
        }

        Ok(())
    }

    fn emit_store_param_reg(&mut self, offset: i32, param_index: usize) -> Result<(), String> {
        match param_index {
            0 => self.emitter.emit_code(&[0x48, 0x89, 0xbd]), // mov [rbp+disp], rdi
            1 => self.emitter.emit_code(&[0x48, 0x89, 0xb5]), // mov [rbp+disp], rsi
            2 => self.emitter.emit_code(&[0x48, 0x89, 0x95]), // mov [rbp+disp], rdx
            3 => self.emitter.emit_code(&[0x48, 0x89, 0x8d]), // mov [rbp+disp], rcx
            4 => self.emitter.emit_code(&[0x4c, 0x89, 0x85]), // mov [rbp+disp], r8
            5 => self.emitter.emit_code(&[0x4c, 0x89, 0x8d]), // mov [rbp+disp], r9
            _ => {
                return Err(format!(
                    "Unsupported parameter register index {}",
                    param_index
                ));
            }
        }
        self.emitter.emit_code(&offset.to_le_bytes());
        Ok(())
    }

    fn emit_epilogue(&mut self) {
        self.emitter.emit_code(&[
            0x48, 0x89, 0xec, // mov rsp, rbp
            0x5d, // pop rbp
            0xc3, // ret
        ]);
    }

    fn patch_return_jumps(&mut self, epilogue_offset: usize) -> Result<(), String> {
        let jumps = std::mem::take(&mut self.return_jumps);
        for imm_offset in jumps {
            self.patch_rel32(imm_offset, epilogue_offset)?;
        }
        Ok(())
    }

    fn compile_block(&mut self, block: &HirBlock) -> Result<(), String> {
        let mut has_tail_expr = false;

        for stmt in &block.stmts {
            match stmt {
                HirStmt::Expr(expr) => {
                    self.compile_expr(expr)?;
                    has_tail_expr = true;
                }
                HirStmt::Let { .. } | HirStmt::Assign { .. } => {
                    self.compile_stmt(stmt)?;
                    has_tail_expr = false;
                }
            }
        }

        if !has_tail_expr {
            self.emit_zero_rax();
        }

        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &HirStmt) -> Result<(), String> {
        match stmt {
            HirStmt::Let { name, value, .. } => {
                let offset = self.local_offset(name)?;
                if let Some(expr) = value {
                    self.compile_expr(expr)?;
                } else {
                    self.emit_zero_rax();
                }
                self.emit_store_rax_local(offset);
                Ok(())
            }
            HirStmt::Assign { target, value } => {
                self.compile_expr(value)?;
                match &target.kind {
                    HirExprKind::Local(name) => {
                        let offset = self.local_offset(name)?;
                        self.emit_store_rax_local(offset);
                        Ok(())
                    }
                    _ => Err(format!(
                        "Unsupported assignment target in native backend: {:?}",
                        target.kind
                    )),
                }
            }
            HirStmt::Expr(expr) => self.compile_expr(expr),
        }
    }

    fn local_offset(&self, name: &str) -> Result<i32, String> {
        self.locals
            .get(name)
            .copied()
            .ok_or_else(|| format!("Undefined local '{}'", name))
    }

    fn compile_expr(&mut self, expr: &HirExpr) -> Result<(), String> {
        match &expr.kind {
            HirExprKind::Literal(lit) => self.compile_literal(lit),
            HirExprKind::Local(name) => {
                let offset = self.local_offset(name)?;
                self.emit_load_local_rax(offset);
                Ok(())
            }
            HirExprKind::Binary { op, left, right } => self.compile_binary(*op, left, right),
            HirExprKind::Unary { op, expr } => self.compile_unary(*op, expr),
            HirExprKind::Call { func, args } => self.compile_call(func, args),
            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => self.compile_if(condition, then_branch, else_branch.as_deref()),
            HirExprKind::Block(block) => self.compile_block(block),
            HirExprKind::Return(value) => self.compile_return(value.as_deref()),
            HirExprKind::While { condition, body } => self.compile_while(condition, body),
            _ => Err(format!(
                "Unsupported expression in native backend: {:?}",
                expr.kind
            )),
        }
    }

    fn compile_literal(&mut self, lit: &HirLiteral) -> Result<(), String> {
        match lit {
            HirLiteral::Unit => {
                self.emit_zero_rax();
                Ok(())
            }
            HirLiteral::Bool(true) => {
                self.emit_mov_rax_imm64(1);
                Ok(())
            }
            HirLiteral::Bool(false) => {
                self.emit_zero_rax();
                Ok(())
            }
            HirLiteral::Int(n) => {
                self.emit_mov_rax_imm64(*n);
                Ok(())
            }
            HirLiteral::String(_) => Err(
                "String literals are currently supported only in print(\"...\") calls".to_string(),
            ),
            _ => Err(format!("Unsupported literal in native backend: {:?}", lit)),
        }
    }

    fn compile_unary(&mut self, op: HirUnaryOp, expr: &HirExpr) -> Result<(), String> {
        self.compile_expr(expr)?;
        match op {
            HirUnaryOp::Neg => {
                self.emitter.emit_code(&[0x48, 0xf7, 0xd8]); // neg rax
                Ok(())
            }
            HirUnaryOp::Not => {
                self.emitter.emit_code(&[0x48, 0x85, 0xc0]); // test rax, rax
                self.emitter.emit_code(&[0x0f, 0x94, 0xc0]); // sete al
                self.emitter.emit_code(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
                Ok(())
            }
            _ => Err(format!("Unsupported unary op in native backend: {:?}", op)),
        }
    }

    fn compile_binary(
        &mut self,
        op: HirBinaryOp,
        left: &HirExpr,
        right: &HirExpr,
    ) -> Result<(), String> {
        match op {
            HirBinaryOp::And => return self.compile_logical_and(left, right),
            HirBinaryOp::Or => return self.compile_logical_or(left, right),
            _ => {}
        }

        self.compile_expr(left)?;
        self.emitter.emit_code(&[0x50]); // push rax
        // Keep the stack 16-byte aligned while evaluating the RHS. The System V ABI requires
        // `%rsp` to be 16-byte aligned *before* a `call`. A single push misaligns by 8, so we
        // add a dummy 8-byte adjustment that we later undo.
        self.emitter.emit_code(&[0x48, 0x83, 0xec, 0x08]); // sub rsp, 8
        self.compile_expr(right)?;
        self.emitter.emit_code(&[0x48, 0x83, 0xc4, 0x08]); // add rsp, 8
        self.emitter.emit_code(&[0x41, 0x5a]); // pop r10

        match op {
            HirBinaryOp::Add => {
                self.emitter.emit_code(&[0x4c, 0x01, 0xd0]); // add rax, r10
                Ok(())
            }
            HirBinaryOp::Sub => {
                self.emitter.emit_code(&[0x49, 0x29, 0xc2]); // sub r10, rax
                self.emitter.emit_code(&[0x4c, 0x89, 0xd0]); // mov rax, r10
                Ok(())
            }
            HirBinaryOp::Mul => {
                self.emitter.emit_code(&[0x49, 0x0f, 0xaf, 0xc2]); // imul rax, r10
                Ok(())
            }
            HirBinaryOp::Div => {
                self.emitter.emit_code(&[0x48, 0x89, 0xc1]); // mov rcx, rax (rhs)
                self.emitter.emit_code(&[0x4c, 0x89, 0xd0]); // mov rax, r10 (lhs)
                self.emitter.emit_code(&[0x48, 0x99]); // cqo
                self.emitter.emit_code(&[0x48, 0xf7, 0xf9]); // idiv rcx
                Ok(())
            }
            HirBinaryOp::Rem => {
                self.emitter.emit_code(&[0x48, 0x89, 0xc1]); // mov rcx, rax (rhs)
                self.emitter.emit_code(&[0x4c, 0x89, 0xd0]); // mov rax, r10 (lhs)
                self.emitter.emit_code(&[0x48, 0x99]); // cqo
                self.emitter.emit_code(&[0x48, 0xf7, 0xf9]); // idiv rcx
                self.emitter.emit_code(&[0x48, 0x89, 0xd0]); // mov rax, rdx (remainder)
                Ok(())
            }
            HirBinaryOp::Eq => {
                self.emitter.emit_code(&[0x49, 0x39, 0xc2]); // cmp r10, rax
                self.emitter.emit_code(&[0x0f, 0x94, 0xc0]); // sete al
                self.emitter.emit_code(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
                Ok(())
            }
            HirBinaryOp::Ne => {
                self.emitter.emit_code(&[0x49, 0x39, 0xc2]); // cmp r10, rax
                self.emitter.emit_code(&[0x0f, 0x95, 0xc0]); // setne al
                self.emitter.emit_code(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
                Ok(())
            }
            HirBinaryOp::Lt => {
                self.emitter.emit_code(&[0x49, 0x39, 0xc2]); // cmp r10, rax
                self.emitter.emit_code(&[0x0f, 0x9c, 0xc0]); // setl al
                self.emitter.emit_code(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
                Ok(())
            }
            HirBinaryOp::Le => {
                self.emitter.emit_code(&[0x49, 0x39, 0xc2]); // cmp r10, rax
                self.emitter.emit_code(&[0x0f, 0x9e, 0xc0]); // setle al
                self.emitter.emit_code(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
                Ok(())
            }
            HirBinaryOp::Gt => {
                self.emitter.emit_code(&[0x49, 0x39, 0xc2]); // cmp r10, rax
                self.emitter.emit_code(&[0x0f, 0x9f, 0xc0]); // setg al
                self.emitter.emit_code(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
                Ok(())
            }
            HirBinaryOp::Ge => {
                self.emitter.emit_code(&[0x49, 0x39, 0xc2]); // cmp r10, rax
                self.emitter.emit_code(&[0x0f, 0x9d, 0xc0]); // setge al
                self.emitter.emit_code(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
                Ok(())
            }
            _ => Err(format!("Unsupported binary op in native backend: {:?}", op)),
        }
    }

    fn compile_logical_and(&mut self, left: &HirExpr, right: &HirExpr) -> Result<(), String> {
        self.compile_expr(left)?;
        self.emitter.emit_code(&[0x48, 0x85, 0xc0]); // test rax, rax

        let jz_false_pos = self.emitter.code.len();
        self.emitter
            .emit_code(&[0x0f, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32

        self.compile_expr(right)?;
        self.emit_bool_from_rax();

        let jmp_end_pos = self.emitter.code.len();
        self.emitter.emit_code(&[0xe9, 0x00, 0x00, 0x00, 0x00]); // jmp rel32

        let false_pos = self.emitter.code.len();
        self.emit_zero_rax();

        let end_pos = self.emitter.code.len();
        self.patch_rel32(jz_false_pos + 2, false_pos)?;
        self.patch_rel32(jmp_end_pos + 1, end_pos)?;
        Ok(())
    }

    fn compile_logical_or(&mut self, left: &HirExpr, right: &HirExpr) -> Result<(), String> {
        self.compile_expr(left)?;
        self.emitter.emit_code(&[0x48, 0x85, 0xc0]); // test rax, rax

        let jnz_true_pos = self.emitter.code.len();
        self.emitter
            .emit_code(&[0x0f, 0x85, 0x00, 0x00, 0x00, 0x00]); // jnz rel32

        self.compile_expr(right)?;
        self.emit_bool_from_rax();

        let jmp_end_pos = self.emitter.code.len();
        self.emitter.emit_code(&[0xe9, 0x00, 0x00, 0x00, 0x00]); // jmp rel32

        let true_pos = self.emitter.code.len();
        self.emit_mov_rax_imm64(1);

        let end_pos = self.emitter.code.len();
        self.patch_rel32(jnz_true_pos + 2, true_pos)?;
        self.patch_rel32(jmp_end_pos + 1, end_pos)?;
        Ok(())
    }

    fn compile_call(&mut self, func: &HirExpr, args: &[HirExpr]) -> Result<(), String> {
        let callee = self.resolve_callee_name(func)?;

        if callee == "print" && !self.user_functions.contains("print") {
            return self.compile_print_call(args);
        }

        if args.len() > 6 {
            return Err(format!(
                "Function call '{}' has {} args; max supported is 6",
                callee,
                args.len()
            ));
        }

        // Evaluate arguments left-to-right while keeping the stack 16-byte aligned.
        //
        // Important: we cannot simply `push` each arg value, because after the first push the
        // stack becomes misaligned and any nested call during later arg evaluation would violate
        // the System V ABI.
        if !args.is_empty() {
            let scratch_size = (args.len() as i32) * 8;
            let scratch_total = align16(scratch_size);

            // sub rsp, scratch_total
            self.emitter.emit_code(&[0x48, 0x81, 0xec]);
            self.emitter
                .emit_code(&(scratch_total as u32).to_le_bytes());

            for (i, arg) in args.iter().enumerate() {
                self.compile_expr(arg)?;
                self.emit_store_rax_rsp_disp32((i as i32) * 8);
            }

            for i in 0..args.len() {
                self.emit_load_arg_reg_from_rsp_disp32(i, (i as i32) * 8)?;
            }

            // add rsp, scratch_total
            self.emitter.emit_code(&[0x48, 0x81, 0xc4]);
            self.emitter
                .emit_code(&(scratch_total as u32).to_le_bytes());
        }

        self.emit_call_relocation_any(callee);
        Ok(())
    }

    fn resolve_callee_name(&self, expr: &HirExpr) -> Result<String, String> {
        match &expr.kind {
            HirExprKind::Local(name) | HirExprKind::Global(name) => Ok(name.clone()),
            _ => Err(format!(
                "Unsupported call target in native backend: {:?}",
                expr.kind
            )),
        }
    }

    fn emit_load_arg_reg_from_rsp_disp32(
        &mut self,
        index: usize,
        disp: i32,
    ) -> Result<(), String> {
        match index {
            // mov rdi, [rsp + disp32]
            0 => self.emitter.emit_code(&[0x48, 0x8b, 0xbc, 0x24]),
            // mov rsi, [rsp + disp32]
            1 => self.emitter.emit_code(&[0x48, 0x8b, 0xb4, 0x24]),
            // mov rdx, [rsp + disp32]
            2 => self.emitter.emit_code(&[0x48, 0x8b, 0x94, 0x24]),
            // mov rcx, [rsp + disp32]
            3 => self.emitter.emit_code(&[0x48, 0x8b, 0x8c, 0x24]),
            // mov r8, [rsp + disp32]
            4 => self.emitter.emit_code(&[0x4c, 0x8b, 0x84, 0x24]),
            // mov r9, [rsp + disp32]
            5 => self.emitter.emit_code(&[0x4c, 0x8b, 0x8c, 0x24]),
            _ => return Err(format!("Unsupported argument register index {}", index)),
        }
        self.emitter.emit_code(&disp.to_le_bytes());
        Ok(())
    }

    fn emit_store_rax_rsp_disp32(&mut self, disp: i32) {
        // mov [rsp + disp32], rax
        self.emitter.emit_code(&[0x48, 0x89, 0x84, 0x24]);
        self.emitter.emit_code(&disp.to_le_bytes());
    }

    fn compile_print_call(&mut self, args: &[HirExpr]) -> Result<(), String> {
        if args.len() != 1 {
            return Err(format!(
                "print currently supports exactly 1 argument, got {}",
                args.len()
            ));
        }

        let HirExprKind::Literal(HirLiteral::String(text)) = &args[0].kind else {
            return Err("print currently supports only string literals".to_string());
        };

        self.emit_inline_print_string(text)?;
        self.emit_zero_rax(); // print returns unit
        Ok(())
    }

    fn emit_inline_print_string(&mut self, text: &str) -> Result<(), String> {
        let text_len = u32::try_from(text.len())
            .map_err(|_| "String literal too large for native print".to_string())?;

        let lea_pos = self.emitter.code.len();
        self.emitter.emit_code(&[
            0x48, 0x8d, 0x3d, 0x00, 0x00, 0x00, 0x00, // lea rdi, [rip + disp32]
        ]);

        self.emitter.emit_code(&[0x48, 0xc7, 0xc6]); // mov rsi, imm32
        self.emitter.emit_code(&text_len.to_le_bytes());

        self.emit_call_relocation_builtin("print".to_string());

        let jmp_pos = self.emitter.code.len();
        self.emitter.emit_code(&[0xe9, 0x00, 0x00, 0x00, 0x00]); // jmp over inline bytes

        let data_pos = self.emitter.code.len();
        self.emitter.emit_code(text.as_bytes());

        let continue_pos = self.emitter.code.len();

        let lea_next = lea_pos + 7;
        let lea_disp = rel32(lea_next, data_pos)?;
        self.emitter.code[lea_pos + 3..lea_pos + 7].copy_from_slice(&lea_disp.to_le_bytes());

        let jmp_rel = rel32(jmp_pos + 5, continue_pos)?;
        self.emitter.code[jmp_pos + 1..jmp_pos + 5].copy_from_slice(&jmp_rel.to_le_bytes());

        Ok(())
    }

    fn compile_if(
        &mut self,
        condition: &HirExpr,
        then_branch: &HirBlock,
        else_expr: Option<&HirExpr>,
    ) -> Result<(), String> {
        self.compile_expr(condition)?;
        self.emitter.emit_code(&[0x48, 0x85, 0xc0]); // test rax, rax

        let jz_pos = self.emitter.code.len();
        self.emitter
            .emit_code(&[0x0f, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32

        self.compile_block(then_branch)?;

        let jmp_end_pos = self.emitter.code.len();
        self.emitter.emit_code(&[0xe9, 0x00, 0x00, 0x00, 0x00]); // jmp rel32

        let else_pos = self.emitter.code.len();
        if let Some(expr) = else_expr {
            self.compile_expr(expr)?;
        } else {
            self.emit_zero_rax();
        }

        let end_pos = self.emitter.code.len();

        self.patch_rel32(jz_pos + 2, else_pos)?;
        self.patch_rel32(jmp_end_pos + 1, end_pos)?;
        Ok(())
    }

    fn compile_while(&mut self, condition: &HirExpr, body: &HirBlock) -> Result<(), String> {
        let cond_pos = self.emitter.code.len();
        self.compile_expr(condition)?;
        self.emitter.emit_code(&[0x48, 0x85, 0xc0]); // test rax, rax

        let jz_pos = self.emitter.code.len();
        self.emitter
            .emit_code(&[0x0f, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32

        self.compile_block(body)?;

        let back_jump_pos = self.emitter.code.len();
        self.emitter.emit_code(&[0xe9, 0x00, 0x00, 0x00, 0x00]); // jmp rel32
        self.patch_rel32(back_jump_pos + 1, cond_pos)?;

        let end_pos = self.emitter.code.len();
        self.patch_rel32(jz_pos + 2, end_pos)?;

        self.emit_zero_rax();
        Ok(())
    }

    fn compile_return(&mut self, value: Option<&HirExpr>) -> Result<(), String> {
        if let Some(expr) = value {
            self.compile_expr(expr)?;
        } else {
            self.emit_zero_rax();
        }

        let jmp_pos = self.emitter.code.len();
        self.emitter.emit_code(&[0xe9, 0x00, 0x00, 0x00, 0x00]); // jmp rel32
        self.return_jumps.push(jmp_pos + 1);
        Ok(())
    }

    fn emit_entry_trampoline(&mut self) -> usize {
        let entry = self.emitter.code.len();

        self.emit_call_relocation_any("main".to_string());
        self.emitter.emit_code(&[0x48, 0x89, 0xc7]); // mov rdi, rax
        self.emit_call_relocation_builtin("exit".to_string());
        self.emitter.emit_code(&[0x0f, 0x0b]); // ud2 (should not return)

        entry
    }

    fn emit_call_relocation_any(&mut self, target_name: String) {
        let call_pos = self.emitter.code.len();
        self.emitter.emit_code(&[0xe8, 0x00, 0x00, 0x00, 0x00]); // call rel32
        self.relocations.push(PendingRelocation {
            imm_offset: call_pos + 1,
            target: RelocationTarget::FunctionOrBuiltin(target_name),
        });
    }

    fn emit_call_relocation_builtin(&mut self, target_name: String) {
        let call_pos = self.emitter.code.len();
        self.emitter.emit_code(&[0xe8, 0x00, 0x00, 0x00, 0x00]); // call rel32
        self.relocations.push(PendingRelocation {
            imm_offset: call_pos + 1,
            target: RelocationTarget::Builtin(target_name),
        });
    }

    fn apply_relocations(&mut self) -> Result<(), String> {
        let relocs = std::mem::take(&mut self.relocations);
        for reloc in relocs {
            let target = match &reloc.target {
                RelocationTarget::FunctionOrBuiltin(name) => self
                    .fn_offsets
                    .get(name)
                    .copied()
                    .or_else(|| self.runtime.builtin_offset(name))
                    .ok_or_else(|| format!("Undefined function '{}'", name))?,
                RelocationTarget::Builtin(name) => self
                    .runtime
                    .builtin_offset(name)
                    .ok_or_else(|| format!("Undefined builtin function '{}'", name))?,
            };
            self.patch_rel32(reloc.imm_offset, target)?;
        }
        Ok(())
    }

    fn patch_rel32(&mut self, imm_offset: usize, target: usize) -> Result<(), String> {
        let next_ip = imm_offset + 4;
        let rel = rel32(next_ip, target)?;
        self.emitter.code[imm_offset..imm_offset + 4].copy_from_slice(&rel.to_le_bytes());
        Ok(())
    }

    fn emit_mov_rax_imm64(&mut self, value: i64) {
        self.emitter.emit_code(&[0x48, 0xb8]); // mov rax, imm64
        self.emitter.emit_code(&value.to_le_bytes());
    }

    fn emit_zero_rax(&mut self) {
        self.emitter.emit_code(&[0x48, 0x31, 0xc0]); // xor rax, rax
    }

    fn emit_bool_from_rax(&mut self) {
        self.emitter.emit_code(&[0x48, 0x85, 0xc0]); // test rax, rax
        self.emitter.emit_code(&[0x0f, 0x95, 0xc0]); // setne al
        self.emitter.emit_code(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
    }

    fn emit_store_rax_local(&mut self, offset: i32) {
        self.emitter.emit_code(&[0x48, 0x89, 0x85]); // mov [rbp+disp32], rax
        self.emitter.emit_code(&offset.to_le_bytes());
    }

    fn emit_load_local_rax(&mut self, offset: i32) {
        self.emitter.emit_code(&[0x48, 0x8b, 0x85]); // mov rax, [rbp+disp32]
        self.emitter.emit_code(&offset.to_le_bytes());
    }
}

impl Default for NativeCodegen {
    fn default() -> Self {
        Self::new()
    }
}

fn align16(n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }
    (n + 15) & !15
}

fn rel32(from_next_ip: usize, to_target: usize) -> Result<i32, String> {
    let rel = (to_target as i64) - (from_next_ip as i64);
    i32::try_from(rel)
        .map_err(|_| format!("rel32 out of range: from {} to {}", from_next_ip, to_target))
}

fn collect_let_bindings_block(block: &HirBlock, out: &mut Vec<String>) {
    for stmt in &block.stmts {
        match stmt {
            HirStmt::Let { name, value, .. } => {
                out.push(name.clone());
                if let Some(v) = value {
                    collect_let_bindings_expr(v, out);
                }
            }
            HirStmt::Expr(expr) => collect_let_bindings_expr(expr, out),
            HirStmt::Assign { target, value } => {
                collect_let_bindings_expr(target, out);
                collect_let_bindings_expr(value, out);
            }
        }
    }
}

fn collect_let_bindings_expr(expr: &HirExpr, out: &mut Vec<String>) {
    match &expr.kind {
        HirExprKind::Binary { left, right, .. } => {
            collect_let_bindings_expr(left, out);
            collect_let_bindings_expr(right, out);
        }
        HirExprKind::Unary { expr, .. } => collect_let_bindings_expr(expr, out),
        HirExprKind::Call { func, args } => {
            collect_let_bindings_expr(func, out);
            for arg in args {
                collect_let_bindings_expr(arg, out);
            }
        }
        HirExprKind::MethodCall { receiver, args, .. } => {
            collect_let_bindings_expr(receiver, out);
            for arg in args {
                collect_let_bindings_expr(arg, out);
            }
        }
        HirExprKind::Field { base, .. } | HirExprKind::TupleField { base, .. } => {
            collect_let_bindings_expr(base, out);
        }
        HirExprKind::Index { base, index } => {
            collect_let_bindings_expr(base, out);
            collect_let_bindings_expr(index, out);
        }
        HirExprKind::Cast { expr, .. } => collect_let_bindings_expr(expr, out),
        HirExprKind::Block(block) | HirExprKind::Loop(block) => {
            collect_let_bindings_block(block, out)
        }
        HirExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_let_bindings_expr(condition, out);
            collect_let_bindings_block(then_branch, out);
            if let Some(e) = else_branch {
                collect_let_bindings_expr(e, out);
            }
        }
        HirExprKind::Match { scrutinee, arms } => {
            collect_let_bindings_expr(scrutinee, out);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    collect_let_bindings_expr(g, out);
                }
                collect_let_bindings_expr(&arm.body, out);
            }
        }
        HirExprKind::While { condition, body } => {
            collect_let_bindings_expr(condition, out);
            collect_let_bindings_block(body, out);
        }
        HirExprKind::Return(v) | HirExprKind::Break(v) => {
            if let Some(v) = v {
                collect_let_bindings_expr(v, out);
            }
        }
        HirExprKind::Closure { body, .. } => collect_let_bindings_expr(body, out),
        HirExprKind::Tuple(items) | HirExprKind::Array(items) => {
            for item in items {
                collect_let_bindings_expr(item, out);
            }
        }
        HirExprKind::ArrayRepeat { value, .. } => collect_let_bindings_expr(value, out),
        HirExprKind::Range { start, end, .. } => {
            if let Some(s) = start {
                collect_let_bindings_expr(s, out);
            }
            if let Some(e) = end {
                collect_let_bindings_expr(e, out);
            }
        }
        HirExprKind::Struct { fields, .. } => {
            for (_, value) in fields {
                collect_let_bindings_expr(value, out);
            }
        }
        HirExprKind::Variant { fields, .. } => {
            for field in fields {
                collect_let_bindings_expr(field, out);
            }
        }
        HirExprKind::Ref { expr, .. } => collect_let_bindings_expr(expr, out),
        HirExprKind::Deref(expr)
        | HirExprKind::Sample(expr)
        | HirExprKind::EpsilonOf(expr)
        | HirExprKind::ProvenanceOf(expr)
        | HirExprKind::ValidityOf(expr)
        | HirExprKind::Unwrap(expr) => collect_let_bindings_expr(expr, out),
        HirExprKind::Perform { args, .. } => {
            for arg in args {
                collect_let_bindings_expr(arg, out);
            }
        }
        HirExprKind::Handle { expr, .. } => collect_let_bindings_expr(expr, out),
        HirExprKind::Resume { value } => collect_let_bindings_expr(value, out),
        HirExprKind::Knowledge {
            value,
            epsilon,
            validity,
            ..
        } => {
            collect_let_bindings_expr(value, out);
            collect_let_bindings_expr(epsilon, out);
            if let Some(v) = validity {
                collect_let_bindings_expr(v, out);
            }
        }
        HirExprKind::Do { value, .. } => collect_let_bindings_expr(value, out),
        HirExprKind::Counterfactual {
            factual,
            intervention,
            outcome,
        } => {
            collect_let_bindings_expr(factual, out);
            collect_let_bindings_expr(intervention, out);
            collect_let_bindings_expr(outcome, out);
        }
        HirExprKind::Query {
            target,
            given,
            interventions,
        } => {
            collect_let_bindings_expr(target, out);
            for g in given {
                collect_let_bindings_expr(g, out);
            }
            for i in interventions {
                collect_let_bindings_expr(i, out);
            }
        }
        HirExprKind::Observe { value, .. } => collect_let_bindings_expr(value, out),
        HirExprKind::Await { future } => collect_let_bindings_expr(future, out),
        HirExprKind::Spawn { expr } => collect_let_bindings_expr(expr, out),
        HirExprKind::AsyncBlock { body } => collect_let_bindings_block(body, out),
        HirExprKind::Join { futures } => {
            for future in futures {
                collect_let_bindings_expr(future, out);
            }
        }
        HirExprKind::Select { arms } => {
            for arm in arms {
                collect_let_bindings_expr(&arm.future, out);
                if let Some(g) = &arm.guard {
                    collect_let_bindings_expr(g, out);
                }
                collect_let_bindings_expr(&arm.body, out);
            }
        }
        HirExprKind::Literal(_)
        | HirExprKind::Local(_)
        | HirExprKind::Global(_)
        | HirExprKind::Continue
        | HirExprKind::OntologyTerm { .. } => {}
    }
}

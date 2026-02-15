//! Bytecode Virtual Machine for Sounio
//!
//! A stack-based interpreter for executing Sounio bytecode.
//! Supports basic arithmetic, control flow, memory ops, and FFI calls.

pub mod bytecode;
pub mod memory;
pub mod serialize;
pub mod stack;

pub use bytecode::{Bytecode, Value};
pub use memory::Heap;
use rustc_hash::FxHashMap as HashMap;
pub use serialize::{BYTECODE_MAGIC, BYTECODE_VERSION, SerializeError, deserialize, serialize};
use std::io::Read;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq)]
pub enum VmError {
    StackUnderflow,
    StackOverflow,
    TypeMismatch(String),
    DivisionByZero,
    InvalidJump(usize),
    InvalidFunctionCall(usize),
    FfiError(String),
    MemoryError(String),
}

pub type VmResult<T> = Result<T, VmError>;

/// Call frame for function calls
#[derive(Debug, Clone)]
pub struct CallFrame {
    pub return_ip: usize,
    pub locals: Vec<Value>,
}

/// The bytecode virtual machine
pub struct BytecodeVM {
    pub stack: Vec<Value>,
    pub heap: Heap,
    pub call_stack: Vec<CallFrame>,
    /// Slot-based locals used by the bytecode compiler backend.
    pub locals: Vec<Value>,
    pub globals: HashMap<String, Value>,
    pub functions: HashMap<usize, Vec<Bytecode>>,
    pub next_fn_id: usize,
    stdin_bytes: Option<Vec<u8>>,
    stdin_pos: usize,
    user_args: Vec<String>,
    progress_mode: bool,
    ffi_call_counter: u64,
}

impl BytecodeVM {
    /// Creates a new VM instance
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(4096),
            heap: Heap::new(),
            call_stack: Vec::new(),
            locals: Vec::new(),
            globals: HashMap::default(),
            functions: HashMap::default(),
            next_fn_id: 0,
            stdin_bytes: None,
            stdin_pos: 0,
            user_args: Vec::new(),
            progress_mode: false,
            ffi_call_counter: 0,
        }
    }

    /// Sets user CLI args exposed to `arg_count`/`get_arg` builtins.
    pub fn set_user_args(&mut self, args: Vec<String>) {
        self.user_args = args;
    }

    /// Registers a function in the function table
    pub fn register_function(&mut self, bytecode: Vec<Bytecode>) -> usize {
        let id = self.next_fn_id;
        self.next_fn_id += 1;
        self.functions.insert(id, bytecode);
        id
    }

    /// Executes a bytecode program
    pub fn execute(&mut self, bytecode: &[Bytecode]) -> VmResult<Value> {
        // Local slots are per-execution state.
        self.locals.clear();
        self.stdin_bytes = None;
        self.stdin_pos = 0;
        let mut ip = 0;
        let progress_enabled = std::env::var_os("SOUNIO_VM_PROGRESS").is_some();
        self.progress_mode = progress_enabled;
        self.ffi_call_counter = 0;
        let progress_every: u64 = std::env::var("SOUNIO_VM_PROGRESS_EVERY")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(5_000_000);
        let instruction_limit: Option<u64> = std::env::var("SOUNIO_VM_INSTRUCTION_LIMIT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|v| *v > 0);
        let mut instruction_count: u64 = 0;
        let mut call_counts = if progress_enabled {
            Some(HashMap::<usize, u64>::default())
        } else {
            None
        };
        let bytecode_len = bytecode.len();
        let track_backedges = progress_enabled || instruction_limit.is_some();
        let mut backedge_jumps: u64 = 0;
        let mut next_progress_tick = progress_every;

        if progress_enabled {
            eprintln!(
                "VM_PROGRESS start bytecode_len={} progress_every={} instruction_limit={}",
                bytecode_len,
                progress_every,
                instruction_limit
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string())
            );
        }

        while ip < bytecode_len {
            instruction_count = instruction_count.saturating_add(1);
            if let Some(limit) = instruction_limit {
                if instruction_count > limit {
                    return Err(VmError::FfiError(format!(
                        "Instruction limit exceeded: count={} limit={} ip={} stack_len={} call_depth={} backedges={}",
                        instruction_count,
                        limit,
                        ip,
                        self.stack.len(),
                        self.call_stack.len(),
                        backedge_jumps
                    )));
                }
            }

            let instr = &bytecode[ip];
            if progress_enabled && instruction_count >= next_progress_tick {
                let mut hot_calls: Vec<(usize, u64)> = call_counts
                    .as_ref()
                    .map(|counts| counts.iter().map(|(k, v)| (*k, *v)).collect())
                    .unwrap_or_default();
                hot_calls.sort_by(|a, b| b.1.cmp(&a.1));
                hot_calls.truncate(5);
                let hot_summary = if hot_calls.is_empty() {
                    "none".to_string()
                } else {
                    hot_calls
                        .into_iter()
                        .map(|(addr, count)| format!("{}:{}", addr, count))
                        .collect::<Vec<_>>()
                        .join(",")
                };

                eprintln!(
                    "VM_PROGRESS tick count={} ip={} instr={:?} stack_len={} call_depth={} backedges={} hot_calls=[{}]",
                    instruction_count,
                    ip,
                    instr,
                    self.stack.len(),
                    self.call_stack.len(),
                    backedge_jumps,
                    hot_summary
                );
                next_progress_tick = next_progress_tick.saturating_add(progress_every);
            }

            match instr {
                // Stack operations
                Bytecode::Push(val) => {
                    self.stack.push(val.clone());
                }
                Bytecode::Pop => {
                    self.stack.pop().ok_or(VmError::StackUnderflow)?;
                }
                Bytecode::Dup => {
                    let val = self.stack.last().ok_or(VmError::StackUnderflow)?.clone();
                    self.stack.push(val);
                }
                Bytecode::Swap => {
                    let len = self.stack.len();
                    if len >= 2 {
                        self.stack.swap(len - 1, len - 2);
                    } else {
                        return Err(VmError::StackUnderflow);
                    }
                }

                // Arithmetic
                Bytecode::Add => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
                        (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
                        (Value::Int(a), Value::Float(b)) => Value::Float(a as f64 + b),
                        (Value::Float(a), Value::Int(b)) => Value::Float(a + b as f64),
                        _ => return Err(VmError::TypeMismatch("Add".to_string())),
                    };
                    self.stack.push(result);
                }
                Bytecode::Sub => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
                        (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
                        (Value::Int(a), Value::Float(b)) => Value::Float(a as f64 - b),
                        (Value::Float(a), Value::Int(b)) => Value::Float(a - b as f64),
                        _ => return Err(VmError::TypeMismatch("Sub".to_string())),
                    };
                    self.stack.push(result);
                }
                Bytecode::Mul => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
                        (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
                        (Value::Int(a), Value::Float(b)) => Value::Float(a as f64 * b),
                        (Value::Float(a), Value::Int(b)) => Value::Float(a * b as f64),
                        _ => return Err(VmError::TypeMismatch("Mul".to_string())),
                    };
                    self.stack.push(result);
                }
                Bytecode::Div => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            if b == 0 {
                                return Err(VmError::DivisionByZero);
                            }
                            Value::Int(a / b)
                        }
                        (Value::Float(a), Value::Float(b)) => {
                            if b == 0.0 {
                                return Err(VmError::DivisionByZero);
                            }
                            Value::Float(a / b)
                        }
                        (Value::Int(a), Value::Float(b)) => {
                            if b == 0.0 {
                                return Err(VmError::DivisionByZero);
                            }
                            Value::Float(a as f64 / b)
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            if b == 0 {
                                return Err(VmError::DivisionByZero);
                            }
                            Value::Float(a / b as f64)
                        }
                        _ => return Err(VmError::TypeMismatch("Div".to_string())),
                    };
                    self.stack.push(result);
                }
                Bytecode::Mod => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            if b == 0 {
                                return Err(VmError::DivisionByZero);
                            }
                            Value::Int(a % b)
                        }
                        _ => return Err(VmError::TypeMismatch("Mod".to_string())),
                    };
                    self.stack.push(result);
                }
                Bytecode::Neg => {
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match a {
                        Value::Int(n) => Value::Int(-n),
                        Value::Float(f) => Value::Float(-f),
                        _ => return Err(VmError::TypeMismatch("Neg".to_string())),
                    };
                    self.stack.push(result);
                }
                Bytecode::ToInt => {
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match a {
                        Value::Int(n) => Value::Int(n),
                        Value::Bool(b) => Value::Int(if b { 1 } else { 0 }),
                        Value::Float(f) => Value::Int(f as i64),
                        other => {
                            return Err(VmError::TypeMismatch(format!(
                                "ToInt expects int/bool/float, got {:?} at ip={}",
                                other, ip
                            )));
                        }
                    };
                    self.stack.push(result);
                }
                Bytecode::ToFloat => {
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match a {
                        Value::Float(f) => Value::Float(f),
                        Value::Int(n) => Value::Float(n as f64),
                        Value::Bool(b) => Value::Float(if b { 1.0 } else { 0.0 }),
                        other => {
                            return Err(VmError::TypeMismatch(format!(
                                "ToFloat expects int/bool/float, got {:?} at ip={}",
                                other, ip
                            )));
                        }
                    };
                    self.stack.push(result);
                }
                Bytecode::ToBool => {
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match a {
                        Value::Bool(b) => Value::Bool(b),
                        Value::Int(n) => Value::Bool(n != 0),
                        Value::Float(f) => Value::Bool(f != 0.0),
                        other => {
                            return Err(VmError::TypeMismatch(format!(
                                "ToBool expects int/bool/float, got {:?} at ip={}",
                                other, ip
                            )));
                        }
                    };
                    self.stack.push(result);
                }
                Bytecode::Shl => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a << b),
                        _ => return Err(VmError::TypeMismatch("Shl".to_string())),
                    };
                    self.stack.push(result);
                }
                Bytecode::Shr => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a >> b),
                        _ => return Err(VmError::TypeMismatch("Shr".to_string())),
                    };
                    self.stack.push(result);
                }
                Bytecode::Xor => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a ^ b),
                        _ => return Err(VmError::TypeMismatch("Xor".to_string())),
                    };
                    self.stack.push(result);
                }
                Bytecode::Eq => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    self.stack.push(Value::Bool(a == b));
                }
                Bytecode::Ne => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    self.stack.push(Value::Bool(a != b));
                }
                Bytecode::Lt => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => a < b,
                        (Value::Float(a), Value::Float(b)) => a < b,
                        (Value::Int(a), Value::Float(b)) => (a as f64) < b,
                        (Value::Float(a), Value::Int(b)) => a < (b as f64),
                        _ => false,
                    };
                    self.stack.push(Value::Bool(result));
                }
                Bytecode::Le => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => a <= b,
                        (Value::Float(a), Value::Float(b)) => a <= b,
                        (Value::Int(a), Value::Float(b)) => (a as f64) <= b,
                        (Value::Float(a), Value::Int(b)) => a <= (b as f64),
                        _ => false,
                    };
                    self.stack.push(Value::Bool(result));
                }
                Bytecode::Gt => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => a > b,
                        (Value::Float(a), Value::Float(b)) => a > b,
                        (Value::Int(a), Value::Float(b)) => (a as f64) > b,
                        (Value::Float(a), Value::Int(b)) => a > (b as f64),
                        _ => false,
                    };
                    self.stack.push(Value::Bool(result));
                }
                Bytecode::Ge => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => a >= b,
                        (Value::Float(a), Value::Float(b)) => a >= b,
                        (Value::Int(a), Value::Float(b)) => (a as f64) >= b,
                        (Value::Float(a), Value::Int(b)) => a >= (b as f64),
                        _ => false,
                    };
                    self.stack.push(Value::Bool(result));
                }
                Bytecode::And => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match (a, b) {
                        (Value::Bool(a), Value::Bool(b)) => self.stack.push(Value::Bool(a && b)),
                        (lhs, rhs) => {
                            return Err(VmError::TypeMismatch(format!(
                                "And expects bool,bool got lhs={:?} rhs={:?} at ip={}",
                                lhs, rhs, ip
                            )));
                        }
                    }
                }
                Bytecode::Or => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match (a, b) {
                        (Value::Bool(a), Value::Bool(b)) => self.stack.push(Value::Bool(a || b)),
                        (lhs, rhs) => {
                            return Err(VmError::TypeMismatch(format!(
                                "Or expects bool,bool got lhs={:?} rhs={:?} at ip={}",
                                lhs, rhs, ip
                            )));
                        }
                    }
                }
                Bytecode::Not => {
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match a {
                        Value::Bool(b) => self.stack.push(Value::Bool(!b)),
                        other => {
                            return Err(VmError::TypeMismatch(format!(
                                "Not expects bool, got {:?} at ip={}",
                                other, ip
                            )));
                        }
                    }
                }
                Bytecode::BitAnd => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a & b),
                        (Value::Bool(a), Value::Bool(b)) => Value::Bool(a & b),
                        (lhs, rhs) => {
                            return Err(VmError::TypeMismatch(format!(
                                "BitAnd expects int,int or bool,bool got lhs={:?} rhs={:?} at ip={}",
                                lhs, rhs, ip
                            )));
                        }
                    };
                    self.stack.push(result);
                }
                Bytecode::BitOr => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let result = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a | b),
                        (Value::Bool(a), Value::Bool(b)) => Value::Bool(a | b),
                        (lhs, rhs) => {
                            return Err(VmError::TypeMismatch(format!(
                                "BitOr expects int,int or bool,bool got lhs={:?} rhs={:?} at ip={}",
                                lhs, rhs, ip
                            )));
                        }
                    };
                    self.stack.push(result);
                }

                // Control flow
                Bytecode::Jump(addr) => {
                    if *addr < bytecode_len {
                        if track_backedges && *addr <= ip {
                            backedge_jumps = backedge_jumps.saturating_add(1);
                        }
                        ip = *addr;
                        continue;
                    } else {
                        return Err(VmError::InvalidJump(*addr));
                    }
                }
                Bytecode::JumpIf(addr) => {
                    let cond = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match cond {
                        Value::Bool(true) => {
                            if *addr < bytecode_len {
                                if track_backedges && *addr <= ip {
                                    backedge_jumps = backedge_jumps.saturating_add(1);
                                }
                                ip = *addr;
                                continue;
                            } else {
                                return Err(VmError::InvalidJump(*addr));
                            }
                        }
                        Value::Bool(false) => {}
                        _ => return Err(VmError::TypeMismatch("JumpIf".to_string())),
                    }
                }
                Bytecode::Call(fn_id) => {
                    if let Some(call_counts) = call_counts.as_mut() {
                        *call_counts.entry(*fn_id).or_insert(0) += 1;
                    }
                    let frame = CallFrame {
                        return_ip: ip + 1,
                        locals: std::mem::take(&mut self.locals),
                    };
                    self.call_stack.push(frame);

                    if *fn_id >= bytecode_len {
                        return Err(VmError::InvalidFunctionCall(*fn_id));
                    }

                    ip = *fn_id;
                    continue;
                }
                Bytecode::Return => {
                    let result = self.stack.pop().unwrap_or(Value::Unit);
                    if let Some(frame) = self.call_stack.pop() {
                        self.locals = frame.locals;
                        self.stack.push(result);
                        ip = frame.return_ip;
                        continue;
                    }
                    if progress_enabled {
                        eprintln!(
                            "VM_PROGRESS done count={} result={:?} backedges={}",
                            instruction_count, result, backedge_jumps
                        );
                    }
                    return Ok(result);
                }

                // Memory operations
                Bytecode::Load(offset) => {
                    // Slot-based local load. The bytecode backend uses explicit
                    // instructions for structured memory operations; implicit
                    // pointer-sensitive behavior here causes nondeterministic
                    // local corruption when pointers are on the stack.
                    let value = self.locals.get(*offset).cloned().unwrap_or(Value::Unit);
                    self.stack.push(value);
                }
                Bytecode::Store(offset) => {
                    // Slot-based local store. See `Load` for rationale.
                    let val = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    if self.locals.len() <= *offset {
                        self.locals.resize(*offset + 1, Value::Unit);
                    }
                    self.locals[*offset] = val;
                }
                Bytecode::Alloc(size) => {
                    let addr = self.heap.alloc(*size);
                    self.stack.push(Value::Pointer(addr));
                }

                // Field and index operations
                Bytecode::StoreField(field_name) => {
                    // Stack: [... target, value] -> [... updated_target]
                    // Pop value and target, store value in target's field
                    let value = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let target = self.stack.pop().ok_or(VmError::StackUnderflow)?;

                    match target {
                        Value::Ref(rc) => {
                            // Write through the shared mutable reference in-place.
                            let mut inner = rc.lock().unwrap();
                            if let Value::Struct(ref mut fields) = *inner {
                                fields.insert(field_name.clone(), value);
                            }
                            drop(inner);
                            self.stack.push(Value::Ref(rc));
                        }
                        Value::Struct(mut fields) => {
                            fields.insert(field_name.clone(), value);
                            self.stack.push(Value::Struct(fields));
                        }
                        _ => {
                            return Err(VmError::TypeMismatch(
                                "StoreField requires struct target".to_string(),
                            ));
                        }
                    }
                }

                Bytecode::StoreIndex => {
                    // Stack: [... target, index, value] -> [... updated_target]
                    // Pop value, index, and target; store value at target[index]
                    let value = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let index = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let target = self.stack.pop().ok_or(VmError::StackUnderflow)?;

                    if let Value::Ref(rc) = &target {
                        // Write through the shared mutable reference in-place.
                        let idx = match &index {
                            Value::Int(i) => *i as usize,
                            _ => {
                                return Err(VmError::TypeMismatch(
                                    "StoreIndex on Ref requires integer index".to_string(),
                                ));
                            }
                        };
                        let mut inner = rc.lock().unwrap();
                        match &mut *inner {
                            Value::List(items) => {
                                if idx < items.len() {
                                    items[idx] = value;
                                } else {
                                    return Err(VmError::MemoryError(format!(
                                        "Index {} out of bounds for ref-list of length {}",
                                        idx,
                                        items.len()
                                    )));
                                }
                            }
                            Value::SparseList {
                                len,
                                default,
                                overrides,
                            } => {
                                if idx < *len {
                                    if value == **default {
                                        overrides.remove(&idx);
                                    } else {
                                        overrides.insert(idx, value);
                                    }
                                } else {
                                    return Err(VmError::MemoryError(format!(
                                        "Index {} out of bounds for ref-list of length {}",
                                        idx, len
                                    )));
                                }
                            }
                            _ => {
                                return Err(VmError::TypeMismatch(
                                    "StoreIndex on Ref requires list inner type".to_string(),
                                ));
                            }
                        }
                        drop(inner);
                        self.stack.push(target);
                    } else {
                        // Non-ref path: modify by value
                        let mut target = target;
                        match (&mut target, index) {
                            (Value::List(items), Value::Int(idx)) => {
                                let idx = idx as usize;
                                if idx < items.len() {
                                    items[idx] = value;
                                    self.stack.push(target);
                                } else {
                                    return Err(VmError::MemoryError(format!(
                                        "Index {} out of bounds for list of length {}",
                                        idx,
                                        items.len()
                                    )));
                                }
                            }
                            (
                                Value::SparseList {
                                    len,
                                    default,
                                    overrides,
                                },
                                Value::Int(idx),
                            ) => {
                                let idx = idx as usize;
                                if idx < *len {
                                    if value == **default {
                                        overrides.remove(&idx);
                                    } else {
                                        overrides.insert(idx, value);
                                    }
                                    self.stack.push(target);
                                } else {
                                    return Err(VmError::MemoryError(format!(
                                        "Index {} out of bounds for list of length {}",
                                        idx, len
                                    )));
                                }
                            }
                            _ => {
                                return Err(VmError::TypeMismatch(
                                    "StoreIndex requires list and integer index".to_string(),
                                ));
                            }
                        }
                    }
                }

                // GetField: get field from struct/map (auto-derefs through Ref)
                Bytecode::GetField(field_name) => {
                    let target = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match &target {
                        Value::Ref(rc) => {
                            let inner = rc.lock().unwrap();
                            match &*inner {
                                Value::Struct(fields) => {
                                    if let Some(value) = fields.get(field_name) {
                                        self.stack.push(value.clone());
                                    } else {
                                        return Err(VmError::TypeMismatch(format!(
                                            "Field '{}' not found in ref-struct at ip={}",
                                            field_name, ip
                                        )));
                                    }
                                }
                                _ => {
                                    tracing::trace!(
                                        "VM: GetField {} on non-struct Ref, returning Unit",
                                        field_name
                                    );
                                    self.stack.push(Value::Unit);
                                }
                            }
                        }
                        Value::Struct(fields) => {
                            if let Some(value) = fields.get(field_name) {
                                self.stack.push(value.clone());
                            } else {
                                let mut available: Vec<String> = fields.keys().cloned().collect();
                                available.sort();
                                let preview = if available.len() > 8 {
                                    available[..8].join(",")
                                } else {
                                    available.join(",")
                                };
                                let call_trace = self
                                    .call_stack
                                    .iter()
                                    .rev()
                                    .take(8)
                                    .map(|frame| frame.return_ip.to_string())
                                    .collect::<Vec<_>>()
                                    .join("->");
                                return Err(VmError::TypeMismatch(format!(
                                    "Field '{}' not found in struct at ip={} available=[{}] call_stack_return_ips=[{}]",
                                    field_name, ip, preview, call_trace
                                )));
                            }
                        }
                        _ => {
                            // For other types, return a placeholder
                            tracing::trace!(
                                "VM: GetField {} on non-struct, returning Unit",
                                field_name
                            );
                            self.stack.push(Value::Unit);
                        }
                    }
                }

                // GetIndex: get element from tuple/array at constant index (auto-derefs Ref)
                Bytecode::GetIndex(index) => {
                    let target = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match &target {
                        Value::Ref(rc) => {
                            let inner = rc.lock().unwrap();
                            match &*inner {
                                Value::List(items) => {
                                    if *index < items.len() {
                                        self.stack.push(items[*index].clone());
                                    } else {
                                        return Err(VmError::MemoryError(format!(
                                            "Index {} out of bounds for ref-list of length {}",
                                            index,
                                            items.len()
                                        )));
                                    }
                                }
                                Value::SparseList {
                                    len,
                                    default,
                                    overrides,
                                } => {
                                    if *index < *len {
                                        if let Some(value) = overrides.get(index) {
                                            self.stack.push(value.clone());
                                        } else {
                                            self.stack.push((**default).clone());
                                        }
                                    } else {
                                        return Err(VmError::MemoryError(format!(
                                            "Index {} out of bounds for ref-list of length {}",
                                            index, len
                                        )));
                                    }
                                }
                                _ => {
                                    tracing::trace!(
                                        "VM: GetIndex {} on non-list Ref, returning Unit",
                                        index
                                    );
                                    self.stack.push(Value::Unit);
                                }
                            }
                        }
                        Value::List(items) => {
                            if *index < items.len() {
                                self.stack.push(items[*index].clone());
                            } else {
                                return Err(VmError::MemoryError(format!(
                                    "Index {} out of bounds for list of length {}",
                                    index,
                                    items.len()
                                )));
                            }
                        }
                        Value::SparseList {
                            len,
                            default,
                            overrides,
                        } => {
                            if *index < *len {
                                if let Some(value) = overrides.get(index) {
                                    self.stack.push(value.clone());
                                } else {
                                    self.stack.push((**default).clone());
                                }
                            } else {
                                return Err(VmError::MemoryError(format!(
                                    "Index {} out of bounds for list of length {}",
                                    index, len
                                )));
                            }
                        }
                        _ => {
                            tracing::trace!("VM: GetIndex {} on non-list, returning Unit", index);
                            self.stack.push(Value::Unit);
                        }
                    }
                }

                // IndexOp: runtime indexing (array and index on stack, auto-derefs Ref)
                Bytecode::IndexOp => {
                    let index = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let target = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    if let Value::Ref(rc) = &target {
                        let inner = rc.lock().unwrap();
                        match (&*inner, &index) {
                            (Value::List(items), Value::Int(idx)) => {
                                let idx = *idx as usize;
                                if idx < items.len() {
                                    self.stack.push(items[idx].clone());
                                } else {
                                    return Err(VmError::MemoryError(format!(
                                        "Index {} out of bounds for ref-list of length {}",
                                        idx,
                                        items.len()
                                    )));
                                }
                            }
                            (
                                Value::SparseList {
                                    len,
                                    default,
                                    overrides,
                                },
                                Value::Int(idx),
                            ) => {
                                let idx = *idx as usize;
                                if idx < *len {
                                    if let Some(value) = overrides.get(&idx) {
                                        self.stack.push(value.clone());
                                    } else {
                                        self.stack.push((**default).clone());
                                    }
                                } else {
                                    return Err(VmError::MemoryError(format!(
                                        "Index {} out of bounds for ref-list of length {}",
                                        idx, len
                                    )));
                                }
                            }
                            (Value::String(s), Value::Int(idx)) => {
                                let idx = *idx as usize;
                                if s.is_ascii() {
                                    if let Some(&b) = s.as_bytes().get(idx) {
                                        self.stack.push(Value::Int((b as i8) as i64));
                                    } else {
                                        return Err(VmError::MemoryError(format!(
                                            "Index {} out of bounds for ref-string of length {}",
                                            idx,
                                            s.len()
                                        )));
                                    }
                                } else if let Some(c) = s.chars().nth(idx) {
                                    self.stack.push(Value::Int(c as i64));
                                } else {
                                    return Err(VmError::MemoryError(format!(
                                        "Index {} out of bounds for ref-string of length {}",
                                        idx,
                                        s.len()
                                    )));
                                }
                            }
                            _ => {
                                return Err(VmError::TypeMismatch(
                                    "IndexOp on Ref requires list/string and integer".to_string(),
                                ));
                            }
                        }
                    } else {
                        match (target, index) {
                            (Value::List(items), Value::Int(idx)) => {
                                let idx = idx as usize;
                                if idx < items.len() {
                                    self.stack.push(items[idx].clone());
                                } else {
                                    return Err(VmError::MemoryError(format!(
                                        "Index {} out of bounds for list of length {}",
                                        idx,
                                        items.len()
                                    )));
                                }
                            }
                            (
                                Value::SparseList {
                                    len,
                                    default,
                                    overrides,
                                },
                                Value::Int(idx),
                            ) => {
                                let idx = idx as usize;
                                if idx < len {
                                    if let Some(value) = overrides.get(&idx) {
                                        self.stack.push(value.clone());
                                    } else {
                                        self.stack.push((*default).clone());
                                    }
                                } else {
                                    return Err(VmError::MemoryError(format!(
                                        "Index {} out of bounds for list of length {}",
                                        idx, len
                                    )));
                                }
                            }
                            (Value::String(s), Value::Int(idx)) => {
                                let idx = idx as usize;
                                if s.is_ascii() {
                                    if let Some(&b) = s.as_bytes().get(idx) {
                                        self.stack.push(Value::Int((b as i8) as i64));
                                    } else {
                                        return Err(VmError::MemoryError(format!(
                                            "Index {} out of bounds for string of length {}",
                                            idx,
                                            s.len()
                                        )));
                                    }
                                } else if let Some(c) = s.chars().nth(idx) {
                                    self.stack.push(Value::Int(c as i64));
                                } else {
                                    return Err(VmError::MemoryError(format!(
                                        "Index {} out of bounds for string of length {}",
                                        idx,
                                        s.len()
                                    )));
                                }
                            }
                            _ => {
                                return Err(VmError::TypeMismatch(
                                    "IndexOp requires list/string and integer".to_string(),
                                ));
                            }
                        }
                    } // close else block
                }

                // MakeVariant: construct an enum variant
                Bytecode::MakeVariant {
                    enum_name,
                    variant,
                    field_count,
                } => {
                    let mut fields = Vec::with_capacity(*field_count);
                    for _ in 0..*field_count {
                        fields.push(self.stack.pop().ok_or(VmError::StackUnderflow)?);
                    }
                    fields.reverse(); // Fields were pushed in order
                    // Represent variant as a struct with special enum fields
                    let mut variant_fields = HashMap::default();
                    variant_fields.insert("__enum".to_string(), Value::String(enum_name.clone()));
                    variant_fields.insert("__variant".to_string(), Value::String(variant.clone()));
                    for (i, field) in fields.into_iter().enumerate() {
                        variant_fields.insert(format!("_{}", i), field);
                    }
                    self.stack.push(Value::Struct(variant_fields));
                }

                // MakeRange: construct a range value
                Bytecode::MakeRange { inclusive } => {
                    let end = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let start = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let mut range_fields = HashMap::default();
                    range_fields.insert("start".to_string(), start);
                    range_fields.insert("end".to_string(), end);
                    range_fields.insert("inclusive".to_string(), Value::Bool(*inclusive));
                    self.stack.push(Value::Struct(range_fields));
                }

                // Reference creation
                Bytecode::MakeRef => {
                    let val = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match val {
                        // Already a Ref — reuse the same Rc (don't double-wrap)
                        Value::Ref(_) => self.stack.push(val),
                        other => self.stack.push(Value::Ref(Arc::new(Mutex::new(other)))),
                    }
                }

                // FFI calls
                Bytecode::CallExtern(name, arg_count) => {
                    let result = self.call_ffi(name, *arg_count)?;
                    self.stack.push(result);
                }
            }

            ip += 1;
        }

        let result = self.stack.pop().unwrap_or(Value::Unit);
        if progress_enabled {
            eprintln!(
                "VM_PROGRESS done_fallthrough count={} result={:?} backedges={}",
                instruction_count, result, backedge_jumps
            );
        }
        Ok(result)
    }

    /// Calls an FFI function
    fn call_ffi(&mut self, name: &str, arg_count: i32) -> VmResult<Value> {
        self.ffi_call_counter = self.ffi_call_counter.saturating_add(1);
        if self.progress_mode && (self.ffi_call_counter <= 40 || self.ffi_call_counter % 10000 == 0)
        {
            eprintln!(
                "VM_PROGRESS ffi_enter idx={} name={} arg_count={} stack_len={} call_depth={}",
                self.ffi_call_counter,
                name,
                arg_count,
                self.stack.len(),
                self.call_stack.len()
            );
        }

        let result = match name {
            "__sounio_print" => {
                if arg_count < 1 {
                    return Err(VmError::FfiError(
                        "print requires at least 1 argument".to_string(),
                    ));
                }
                let count = arg_count as usize;
                let mut args = Vec::with_capacity(count);
                for _ in 0..count {
                    args.push(self.stack.pop().ok_or(VmError::StackUnderflow)?);
                }
                args.reverse();
                for arg in args {
                    match arg {
                        Value::String(text) => print!("{}", text),
                        other => print!("{}", other),
                    }
                }
                Ok(Value::Unit)
            }
            "__sounio_println" => {
                if arg_count < 1 {
                    return Err(VmError::FfiError(
                        "println requires at least 1 argument".to_string(),
                    ));
                }
                let count = arg_count as usize;
                let mut args = Vec::with_capacity(count);
                for _ in 0..count {
                    args.push(self.stack.pop().ok_or(VmError::StackUnderflow)?);
                }
                args.reverse();
                for arg in args {
                    match arg {
                        Value::String(text) => print!("{}", text),
                        other => print!("{}", other),
                    }
                }
                println!();
                Ok(Value::Unit)
            }
            "__sounio_concat" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError(
                        "__sounio_concat requires exactly 2 arguments".to_string(),
                    ));
                }
                let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match (a, b) {
                    (Value::String(a), Value::String(b)) => Ok(Value::String(a + &b)),
                    (Value::List(mut a), Value::List(b)) => {
                        a.extend(b);
                        Ok(Value::List(a))
                    }
                    _ => Err(VmError::TypeMismatch(
                        "__sounio_concat expects (string,string) or (list,list)".to_string(),
                    )),
                }
            }
            "print_int" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError(
                        "print_int requires exactly 1 argument".to_string(),
                    ));
                }
                let v = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match v {
                    Value::Int(i) => {
                        print!("{}", i);
                        Ok(Value::Unit)
                    }
                    _ => Err(VmError::TypeMismatch("print_int expects int".to_string())),
                }
            }
            "print_char" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError(
                        "print_char requires exactly 1 argument".to_string(),
                    ));
                }
                let v = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match v {
                    Value::Int(i) => {
                        let byte = i as i8 as u8;
                        print!("{}", char::from(byte));
                        Ok(Value::Unit)
                    }
                    _ => Err(VmError::TypeMismatch("print_char expects int".to_string())),
                }
            }
            "read_byte" => {
                if arg_count != 0 {
                    return Err(VmError::FfiError(
                        "read_byte expects 0 arguments".to_string(),
                    ));
                }
                if self.stdin_bytes.is_none() {
                    let mut buf = Vec::new();
                    std::io::stdin()
                        .read_to_end(&mut buf)
                        .map_err(|e| VmError::FfiError(format!("read_byte stdin failed: {}", e)))?;
                    self.stdin_bytes = Some(buf);
                    self.stdin_pos = 0;
                }
                let bytes = self
                    .stdin_bytes
                    .as_ref()
                    .expect("stdin buffer should exist");
                if self.stdin_pos < bytes.len() {
                    let b = bytes[self.stdin_pos];
                    self.stdin_pos += 1;
                    Ok(Value::Int((b as i8) as i64))
                } else {
                    Ok(Value::Int(-1))
                }
            }
            "read_file" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError(
                        "read_file requires exactly 1 argument".to_string(),
                    ));
                }
                let path = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let path = match path {
                    Value::String(s) => s,
                    _ => {
                        return Err(VmError::TypeMismatch(
                            "read_file expects string".to_string(),
                        ));
                    }
                };
                match std::fs::read(&path) {
                    Ok(bytes) => {
                        let list = bytes
                            .into_iter()
                            .map(|b| Value::Int((b as i8) as i64))
                            .collect::<Vec<_>>();
                        Ok(Value::List(list))
                    }
                    Err(_) => Ok(Value::List(Vec::new())),
                }
            }
            "read_file_prefix" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError(
                        "read_file_prefix requires exactly 2 arguments".to_string(),
                    ));
                }
                let max_bytes = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let max_bytes = match max_bytes {
                    Value::Int(i) => i,
                    _ => {
                        return Err(VmError::TypeMismatch(
                            "read_file_prefix expects int byte limit".to_string(),
                        ));
                    }
                };
                if max_bytes <= 0 {
                    return Ok(Value::List(Vec::new()));
                }
                let max_bytes = u64::try_from(max_bytes).unwrap_or(u64::MAX);

                let path = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let path = match path {
                    Value::String(s) => s,
                    _ => {
                        return Err(VmError::TypeMismatch(
                            "read_file_prefix expects string path".to_string(),
                        ));
                    }
                };

                match std::fs::File::open(&path) {
                    Ok(file) => {
                        let mut reader = file.take(max_bytes);
                        let mut bytes = Vec::new();
                        if reader.read_to_end(&mut bytes).is_err() {
                            return Ok(Value::List(Vec::new()));
                        }
                        let list = bytes
                            .into_iter()
                            .map(|b| Value::Int((b as i8) as i64))
                            .collect::<Vec<_>>();
                        Ok(Value::List(list))
                    }
                    Err(_) => Ok(Value::List(Vec::new())),
                }
            }
            "file_size" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError(
                        "file_size requires exactly 1 argument".to_string(),
                    ));
                }
                let path = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let path = match path {
                    Value::String(s) => s,
                    _ => {
                        return Err(VmError::TypeMismatch(
                            "file_size expects string".to_string(),
                        ));
                    }
                };
                match std::fs::metadata(&path) {
                    Ok(meta) => {
                        // Keep `file_size` predictable: directories and other non-regular files
                        // report -1 so Sounio code can distinguish "not a file" from "empty file".
                        if !meta.is_file() {
                            return Ok(Value::Int(-1));
                        }
                        Ok(Value::Int(i64::try_from(meta.len()).unwrap_or(i64::MAX)))
                    }
                    Err(_) => Ok(Value::Int(-1)),
                }
            }
            "is_dir" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError(
                        "is_dir requires exactly 1 argument".to_string(),
                    ));
                }
                let path = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let path = match path {
                    Value::String(s) => s,
                    _ => return Err(VmError::TypeMismatch("is_dir expects string".to_string())),
                };
                match std::fs::metadata(&path) {
                    Ok(meta) => Ok(Value::Bool(meta.is_dir())),
                    Err(_) => Ok(Value::Bool(false)),
                }
            }
            "path_join" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError(
                        "path_join expects 2 arguments".to_string(),
                    ));
                }
                let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let (a, b) = match (a, b) {
                    (Value::String(a), Value::String(b)) => (a, b),
                    _ => {
                        return Err(VmError::TypeMismatch(
                            "path_join expects strings".to_string(),
                        ));
                    }
                };
                let joined = std::path::PathBuf::from(a).join(b);
                Ok(Value::String(joined.to_string_lossy().to_string()))
            }
            "list_dir" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("list_dir expects 1 argument".to_string()));
                }
                let path = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let path = match path {
                    Value::String(s) => s,
                    _ => return Err(VmError::TypeMismatch("list_dir expects string".to_string())),
                };
                let mut out = Vec::new();
                if let Ok(rd) = std::fs::read_dir(&path) {
                    let mut entries: Vec<std::path::PathBuf> =
                        rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
                    // Deterministic ordering: lexicographic by full path.
                    entries.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
                    for p in entries {
                        out.push(Value::String(p.to_string_lossy().to_string()));
                    }
                }
                Ok(Value::List(out))
            }
            "arg_count" => {
                if arg_count != 0 {
                    return Err(VmError::FfiError(
                        "arg_count expects 0 arguments".to_string(),
                    ));
                }
                Ok(Value::Int(self.user_args.len() as i64))
            }
            "get_arg" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("get_arg expects 1 argument".to_string()));
                }
                let idx = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let idx = match idx {
                    Value::Int(i) => i,
                    _ => {
                        return Err(VmError::TypeMismatch(
                            "get_arg expects int index".to_string(),
                        ));
                    }
                };
                if idx < 0 {
                    return Ok(Value::String(String::new()));
                }
                let idx = idx as usize;
                if idx < self.user_args.len() {
                    Ok(Value::String(self.user_args[idx].clone()))
                } else {
                    Ok(Value::String(String::new()))
                }
            }
            "str_len" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("str_len expects 1 argument".to_string()));
                }
                let s = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match s {
                    Value::String(s) => Ok(Value::Int(s.len() as i64)),
                    _ => Err(VmError::TypeMismatch("str_len expects string".to_string())),
                }
            }
            "str_eq" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError("str_eq expects 2 arguments".to_string()));
                }
                let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match (a, b) {
                    (Value::String(a), Value::String(b)) => Ok(Value::Bool(a == b)),
                    _ => Err(VmError::TypeMismatch("str_eq expects strings".to_string())),
                }
            }
            "str_concat" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError(
                        "str_concat expects 2 arguments".to_string(),
                    ));
                }
                let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match (a, b) {
                    (Value::String(a), Value::String(b)) => {
                        Ok(Value::String(format!("{}{}", a, b)))
                    }
                    _ => Err(VmError::TypeMismatch(
                        "str_concat expects strings".to_string(),
                    )),
                }
            }
            "str_char_at" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError(
                        "str_char_at expects 2 arguments".to_string(),
                    ));
                }
                let idx = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let s = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let idx = match idx {
                    Value::Int(i) => i,
                    _ => {
                        return Err(VmError::TypeMismatch(
                            "str_char_at expects (string, int)".to_string(),
                        ));
                    }
                };
                let s = match s {
                    Value::String(s) => s,
                    _ => {
                        return Err(VmError::TypeMismatch(
                            "str_char_at expects (string, int)".to_string(),
                        ));
                    }
                };
                if idx < 0 {
                    return Ok(Value::Int(-1));
                }
                let idx = idx as usize;
                let bytes = s.as_bytes();
                if idx >= bytes.len() {
                    return Ok(Value::Int(-1));
                }
                Ok(Value::Int(bytes[idx] as i64))
            }
            "str_from_bytes" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError(
                        "str_from_bytes expects 2 arguments".to_string(),
                    ));
                }
                let len_val = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let bytes_val = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let length = match len_val {
                    Value::Int(n) => n,
                    _ => {
                        return Err(VmError::TypeMismatch(
                            "str_from_bytes expects (list, int)".to_string(),
                        ));
                    }
                };
                if length <= 0 {
                    return Ok(Value::String(String::new()));
                }
                let max_len = usize::try_from(length).unwrap_or(usize::MAX);

                let mut out = Vec::new();
                match bytes_val {
                    Value::List(items) => {
                        let take = std::cmp::min(max_len, items.len());
                        out.reserve(take);
                        for item in items.iter().take(take) {
                            match item {
                                Value::Int(i) => out.push((*i as i8) as u8),
                                _ => {
                                    return Err(VmError::TypeMismatch(
                                        "str_from_bytes expects a list of ints".to_string(),
                                    ));
                                }
                            }
                        }
                    }
                    Value::SparseList {
                        len,
                        default,
                        overrides,
                    } => {
                        let take = std::cmp::min(max_len, len);
                        out.reserve(take);
                        for i in 0..take {
                            let value = overrides.get(&i).unwrap_or(default.as_ref());
                            match value {
                                Value::Int(v) => out.push((*v as i8) as u8),
                                _ => {
                                    return Err(VmError::TypeMismatch(
                                        "str_from_bytes expects a list of ints".to_string(),
                                    ));
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(VmError::TypeMismatch(
                            "str_from_bytes expects a list of ints".to_string(),
                        ));
                    }
                }

                Ok(Value::String(String::from_utf8_lossy(&out).into_owned()))
            }
            "int_to_str" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError(
                        "int_to_str expects 1 argument".to_string(),
                    ));
                }
                let v = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match v {
                    Value::Int(i) => Ok(Value::String(i.to_string())),
                    _ => Err(VmError::TypeMismatch("int_to_str expects int".to_string())),
                }
            }
            "contains" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError(
                        "contains expects 2 arguments".to_string(),
                    ));
                }
                let needle = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let haystack = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match (haystack, needle) {
                    (Value::String(haystack), Value::String(needle)) => {
                        Ok(Value::Bool(haystack.contains(needle.as_str())))
                    }
                    _ => Err(VmError::TypeMismatch(
                        "contains expects strings".to_string(),
                    )),
                }
            }
            "starts_with" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError(
                        "starts_with expects 2 arguments".to_string(),
                    ));
                }
                let prefix = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let s = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match (s, prefix) {
                    (Value::String(s), Value::String(prefix)) => {
                        Ok(Value::Bool(s.starts_with(prefix.as_str())))
                    }
                    _ => Err(VmError::TypeMismatch(
                        "starts_with expects strings".to_string(),
                    )),
                }
            }
            "ends_with" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError(
                        "ends_with expects 2 arguments".to_string(),
                    ));
                }
                let suffix = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let s = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match (s, suffix) {
                    (Value::String(s), Value::String(suffix)) => {
                        Ok(Value::Bool(s.ends_with(suffix.as_str())))
                    }
                    _ => Err(VmError::TypeMismatch(
                        "ends_with expects strings".to_string(),
                    )),
                }
            }
            "str_slice" => {
                if arg_count != 3 {
                    return Err(VmError::FfiError(
                        "str_slice expects 3 arguments".to_string(),
                    ));
                }
                let end = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let start = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let s = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match (s, start, end) {
                    (Value::String(s), Value::Int(start), Value::Int(end)) => {
                        let len = s.len() as i64;
                        let start = start.clamp(0, len) as usize;
                        let end = end.clamp(0, len) as usize;
                        let (from, to) = if start <= end {
                            (start, end)
                        } else {
                            (end, start)
                        };
                        Ok(Value::String(s[from..to].to_string()))
                    }
                    _ => Err(VmError::TypeMismatch(
                        "str_slice expects (string, int, int)".to_string(),
                    )),
                }
            }
            "len" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("len expects 1 argument".to_string()));
                }
                let v = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match v {
                    Value::String(s) => Ok(Value::Int(s.len() as i64)),
                    Value::List(items) => Ok(Value::Int(items.len() as i64)),
                    Value::SparseList { len, .. } => Ok(Value::Int(len as i64)),
                    _ => Err(VmError::TypeMismatch("len expects string/list".to_string())),
                }
            }
            // ==================== Math intrinsics (f64 & f32 aliases) ====================
            "sqrt" | "sqrtf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("sqrt expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.sqrt()))
            }
            "cbrt" | "cbrtf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("cbrt expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.cbrt()))
            }
            "fabs" | "fabsf" | "abs" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("abs expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.abs()))
            }
            "sin" | "sinf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("sin expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.sin()))
            }
            "cos" | "cosf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("cos expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.cos()))
            }
            "tan" | "tanf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("tan expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.tan()))
            }
            "asin" | "asinf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("asin expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.asin()))
            }
            "acos" | "acosf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("acos expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.acos()))
            }
            "atan" | "atanf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("atan expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.atan()))
            }
            "atan2" | "atan2f" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError("atan2 expects 2 arguments".to_string()));
                }
                let x = self.pop_numeric(name)?; // x is last argument
                let y = self.pop_numeric(name)?; // y is first argument / receiver
                Ok(Value::Float(y.atan2(x)))
            }
            "tanh" | "tanhf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("tanh expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.tanh()))
            }
            "exp" | "expf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("exp expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.exp()))
            }
            "log" | "logf" | "ln" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("log expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.ln()))
            }
            "log10" | "log10f" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("log10 expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.log10()))
            }
            "log2" | "log2f" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("log2 expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.log2()))
            }
            "pow" | "powf" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError("pow expects 2 arguments".to_string()));
                }
                let exp = self.pop_numeric(name)?;
                let base = self.pop_numeric(name)?;
                Ok(Value::Float(base.powf(exp)))
            }
            "floor" | "floorf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("floor expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.floor()))
            }
            "ceil" | "ceilf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("ceil expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.ceil()))
            }
            "round" | "roundf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("round expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.round()))
            }
            "trunc" | "truncf" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError("trunc expects 1 argument".to_string()));
                }
                let x = self.pop_numeric(name)?;
                Ok(Value::Float(x.trunc()))
            }
            "min" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError("min expects 2 arguments".to_string()));
                }
                let b = self.pop_numeric(name)?;
                let a = self.pop_numeric(name)?;
                Ok(Value::Float(a.min(b)))
            }
            "max" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError("max expects 2 arguments".to_string()));
                }
                let b = self.pop_numeric(name)?;
                let a = self.pop_numeric(name)?;
                Ok(Value::Float(a.max(b)))
            }
            "__sounio_make_list" => {
                if arg_count < 0 {
                    return Err(VmError::FfiError(
                        "make_list requires non-negative arg count".to_string(),
                    ));
                }
                let count = arg_count as usize;
                let mut items = vec![Value::Unit; count];
                for i in (0..count).rev() {
                    items[i] = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                }
                Ok(Value::List(items))
            }
            "__sounio_repeat_list" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError(
                        "repeat_list requires 2 arguments".to_string(),
                    ));
                }
                let count = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let value = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let count = match count {
                    Value::Int(n) if n >= 0 => n as usize,
                    Value::Int(_) => {
                        return Err(VmError::TypeMismatch(
                            "repeat_list count must be non-negative".to_string(),
                        ));
                    }
                    _ => {
                        return Err(VmError::TypeMismatch(
                            "repeat_list count must be int".to_string(),
                        ));
                    }
                };
                if self.progress_mode && self.ffi_call_counter <= 200 {
                    let value_kind = match &value {
                        Value::Unit => "unit",
                        Value::Bool(_) => "bool",
                        Value::Int(_) => "int",
                        Value::Float(_) => "float",
                        Value::String(_) => "string",
                        Value::Pointer(_) => "pointer",
                        Value::List(_) => "list",
                        Value::SparseList { .. } => "sparse_list",
                        Value::Struct(_) => "struct",
                        Value::Ref(_) => "ref",
                    };
                    eprintln!(
                        "VM_PROGRESS repeat_list idx={} count={} value_kind={}",
                        self.ffi_call_counter, count, value_kind
                    );
                }
                // Medium and large repeat arrays are far cheaper as sparse defaults.
                // Self-hosted lexer/parser initializes many [0; 256] buffers.
                if count >= 256 {
                    Ok(Value::SparseList {
                        len: count,
                        default: Box::new(value),
                        overrides: HashMap::default(),
                    })
                } else {
                    let mut items = Vec::with_capacity(count);
                    for _ in 0..count {
                        items.push(value.clone());
                    }
                    Ok(Value::List(items))
                }
            }
            "__sounio_make_struct" => {
                if arg_count < 0 || arg_count % 2 != 0 {
                    return Err(VmError::FfiError(
                        "make_struct requires even arg count".to_string(),
                    ));
                }
                let mut pairs = Vec::with_capacity((arg_count / 2) as usize);
                for _ in 0..(arg_count / 2) {
                    let value = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let key = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match key {
                        Value::String(name) => pairs.push((name, value)),
                        _ => {
                            return Err(VmError::TypeMismatch(
                                "make_struct expects string field names".to_string(),
                            ));
                        }
                    }
                }
                pairs.reverse();
                let mut fields: HashMap<String, Value> = HashMap::default();
                fields.reserve(pairs.len());
                for (name, value) in pairs {
                    fields.insert(name, value);
                }
                Ok(Value::Struct(fields))
            }
            "__sounio_get_global" => {
                if arg_count != 1 {
                    return Err(VmError::FfiError(
                        "get_global requires exactly 1 argument".to_string(),
                    ));
                }
                let name = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match name {
                    Value::String(name) => {
                        Ok(self.globals.get(&name).cloned().unwrap_or(Value::Unit))
                    }
                    _ => Err(VmError::TypeMismatch(
                        "get_global expects a string name".to_string(),
                    )),
                }
            }
            "__sounio_set_global" => {
                if arg_count != 2 {
                    return Err(VmError::FfiError(
                        "set_global requires exactly 2 arguments".to_string(),
                    ));
                }
                let value = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                let name = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match name {
                    Value::String(name) => {
                        self.globals.insert(name, value);
                        Ok(Value::Unit)
                    }
                    _ => Err(VmError::TypeMismatch(
                        "set_global expects a string name".to_string(),
                    )),
                }
            }
            _ => Err(VmError::FfiError(format!("Unknown FFI function: {}", name))),
        };

        if self.progress_mode && (self.ffi_call_counter <= 40 || self.ffi_call_counter % 10000 == 0)
        {
            let status = if result.is_ok() { "ok" } else { "err" };
            eprintln!(
                "VM_PROGRESS ffi_exit idx={} name={} status={} stack_len={} call_depth={}",
                self.ffi_call_counter,
                name,
                status,
                self.stack.len(),
                self.call_stack.len()
            );
        }

        result
    }

    fn pop_numeric(&mut self, context: &str) -> VmResult<f64> {
        let v = self.stack.pop().ok_or(VmError::StackUnderflow)?;
        match v {
            Value::Float(f) => Ok(f),
            Value::Int(i) => Ok(i as f64),
            Value::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
            other => Err(VmError::TypeMismatch(format!(
                "{} expects numeric, got {:?}",
                context, other
            ))),
        }
    }
}

impl Default for BytecodeVM {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_slot_store_and_load_roundtrip() {
        let mut vm = BytecodeVM::new();
        let program = vec![
            Bytecode::Push(Value::Int(42)),
            Bytecode::Store(0),
            Bytecode::Load(0),
            Bytecode::Return,
        ];

        let result = vm.execute(&program).expect("vm execute");
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn pointer_store_and_load_still_work() {
        let mut vm = BytecodeVM::new();
        let program = vec![
            Bytecode::Alloc(8),
            Bytecode::Dup,
            Bytecode::Push(Value::Int(7)),
            Bytecode::Store(8),
            Bytecode::Load(8),
            Bytecode::Return,
        ];

        let result = vm.execute(&program).expect("vm execute");
        assert_eq!(result, Value::Int(7));
    }
}

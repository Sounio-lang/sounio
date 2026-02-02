//! Bytecode Virtual Machine for Sounio
//!
//! A stack-based interpreter for executing Sounio bytecode.
//! Supports basic arithmetic, control flow, memory ops, and FFI calls.

pub mod bytecode;
pub mod memory;
pub mod stack;

pub use bytecode::{Bytecode, Value};
pub use memory::Heap;

use crate::runtime::ffi;
use std::collections::HashMap;

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
    pub locals: HashMap<String, Value>,
}

/// The bytecode virtual machine
pub struct BytecodeVM {
    pub stack: Vec<Value>,
    pub heap: Heap,
    pub call_stack: Vec<CallFrame>,
    pub globals: HashMap<String, Value>,
    pub functions: HashMap<usize, Vec<Bytecode>>,
    pub next_fn_id: usize,
}

impl BytecodeVM {
    /// Creates a new VM instance
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(4096),
            heap: Heap::new(),
            call_stack: Vec::new(),
            globals: HashMap::new(),
            functions: HashMap::new(),
            next_fn_id: 0,
        }
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
        let mut ip = 0;

        loop {
            if ip >= bytecode.len() {
                break;
            }

            let instr = &bytecode[ip].clone();
            tracing::trace!(
                "VM: ip={} instr={:?} stack_len={}",
                ip,
                instr,
                self.stack.len()
            );

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
                        _ => return Err(VmError::TypeMismatch("Lt".to_string())),
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
                        _ => return Err(VmError::TypeMismatch("Le".to_string())),
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
                        _ => return Err(VmError::TypeMismatch("Gt".to_string())),
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
                        _ => return Err(VmError::TypeMismatch("Ge".to_string())),
                    };
                    self.stack.push(Value::Bool(result));
                }
                Bytecode::And => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match (a, b) {
                        (Value::Bool(a), Value::Bool(b)) => self.stack.push(Value::Bool(a && b)),
                        _ => return Err(VmError::TypeMismatch("And".to_string())),
                    }
                }
                Bytecode::Or => {
                    let b = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match (a, b) {
                        (Value::Bool(a), Value::Bool(b)) => self.stack.push(Value::Bool(a || b)),
                        _ => return Err(VmError::TypeMismatch("Or".to_string())),
                    }
                }
                Bytecode::Not => {
                    let a = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match a {
                        Value::Bool(b) => self.stack.push(Value::Bool(!b)),
                        _ => return Err(VmError::TypeMismatch("Not".to_string())),
                    }
                }

                // Control flow
                Bytecode::Jump(addr) => {
                    if *addr < bytecode.len() {
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
                            if *addr < bytecode.len() {
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
                    let _fn = self
                        .functions
                        .get(fn_id)
                        .ok_or(VmError::InvalidFunctionCall(*fn_id))?;

                    let frame = CallFrame {
                        return_ip: ip + 1,
                        locals: HashMap::new(),
                    };
                    self.call_stack.push(frame);

                    tracing::trace!("VM: Calling function {}", fn_id);
                    // In a more complete impl, we'd switch to executing the callee's bytecode
                }
                Bytecode::Return => {
                    let result = self.stack.pop().unwrap_or(Value::Unit);
                    self.call_stack.pop();
                    return Ok(result);
                }

                // Memory operations
                Bytecode::Load(offset) => {
                    let ptr = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match ptr {
                        Value::Pointer(addr) => {
                            let data = self.heap.read(addr, *offset)?;
                            let value = if data.len() >= 8 {
                                let bytes = [
                                    data[0], data[1], data[2], data[3], data[4], data[5], data[6],
                                    data[7],
                                ];
                                Value::Int(i64::from_le_bytes(bytes))
                            } else {
                                Value::Int(0)
                            };
                            self.stack.push(value);
                        }
                        _ => return Err(VmError::TypeMismatch("Load expects pointer".to_string())),
                    }
                }
                Bytecode::Store(offset) => {
                    let val = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    let ptr = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                    match (val, ptr) {
                        (Value::Int(n), Value::Pointer(addr)) => {
                            let bytes = n.to_le_bytes();
                            self.heap.write(addr, &bytes[..*offset.min(&bytes.len())])?;
                        }
                        _ => {
                            return Err(VmError::TypeMismatch(
                                "Store expects (value, pointer)".to_string(),
                            ))
                        }
                    }
                }
                Bytecode::Alloc(size) => {
                    let addr = self.heap.alloc(*size);
                    self.stack.push(Value::Pointer(addr));
                    tracing::trace!("VM: Allocated {} bytes at {}", size, addr);
                }

                // FFI calls
                Bytecode::CallExtern(name, arg_count) => {
                    let result = self.call_ffi(name, *arg_count)?;
                    self.stack.push(result);
                }
            }

            ip += 1;
        }

        Ok(self.stack.pop().unwrap_or(Value::Unit))
    }

    /// Calls an FFI function
    fn call_ffi(&mut self, name: &str, arg_count: i32) -> VmResult<Value> {
        tracing::trace!("VM: FFI call: {} (args={})", name, arg_count);

        match name {
            "__sounio_print" => {
                if arg_count < 1 {
                    return Err(VmError::FfiError("print requires 1 argument".to_string()));
                }
                let s = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match s {
                    Value::String(ref text) => {
                        ffi::__sounio_print(text.as_ptr(), text.len());
                        Ok(Value::Unit)
                    }
                    _ => Err(VmError::TypeMismatch("print expects string".to_string())),
                }
            }
            "__sounio_println" => {
                if arg_count < 1 {
                    return Err(VmError::FfiError("println requires 1 argument".to_string()));
                }
                let s = self.stack.pop().ok_or(VmError::StackUnderflow)?;
                match s {
                    Value::String(ref text) => {
                        ffi::__sounio_println(text.as_ptr(), text.len());
                        Ok(Value::Unit)
                    }
                    _ => Err(VmError::TypeMismatch("println expects string".to_string())),
                }
            }
            _ => Err(VmError::FfiError(format!("Unknown FFI function: {}", name))),
        }
    }
}

impl Default for BytecodeVM {
    fn default() -> Self {
        Self::new()
    }
}

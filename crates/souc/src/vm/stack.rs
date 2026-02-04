//! Stack management utilities (currently a placeholder)
//! The main stack is managed in BytecodeVM

use crate::vm::Value;

/// Stack utility functions
pub struct Stack;

impl Stack {
    /// Converts a Value to a string representation (for printing)
    pub fn value_to_string(val: &Value) -> String {
        match val {
            Value::Unit => "()".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::String(s) => s.clone(),
            Value::Pointer(p) => format!("0x{:x}", p),
            Value::List(items) => {
                let strs: Vec<String> = items.iter().map(|v| Self::value_to_string(v)).collect();
                format!("[{}]", strs.join(", "))
            }
            Value::Struct(fields) => {
                let mut entries: Vec<String> = fields
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, Self::value_to_string(v)))
                    .collect();
                entries.sort();
                format!("{{ {} }}", entries.join(", "))
            }
        }
    }

    /// Converts a string to an Int value (if possible)
    pub fn string_to_int(s: &str) -> Option<i64> {
        s.trim().parse().ok()
    }

    /// Converts a string to a Float value (if possible)
    pub fn string_to_float(s: &str) -> Option<f64> {
        s.trim().parse().ok()
    }
}

//! Python ctypes binding generator for Sounio FFI
//!
//! This module generates Python ctypes bindings from Sounio source files,
//! enabling Python code to call Sounio functions compiled as shared libraries.
//!
//! # Usage
//!
//! ```ignore
//! use souc::tools::pybindgen::PythonBindingGenerator;
//!
//! let generator = PythonBindingGenerator::new();
//! let python_code = generator.generate_from_hir(&hir, "libexample.so")?;
//! ```

use crate::hir::{Hir, HirFn, HirItem, HirType};
use std::collections::HashSet;
use std::fmt::Write;

/// Python ctypes binding generator
pub struct PythonBindingGenerator {
    /// Generated Python code
    output: String,
    /// Set of generated type definitions to avoid duplicates
    generated_types: HashSet<String>,
    /// Library name for the generated bindings
    library_name: String,
}

impl PythonBindingGenerator {
    /// Create a new Python binding generator
    pub fn new() -> Self {
        Self {
            output: String::new(),
            generated_types: HashSet::new(),
            library_name: String::new(),
        }
    }

    /// Generate Python ctypes bindings from an HIR module
    pub fn generate_from_hir(&mut self, hir: &Hir, library_path: &str) -> Result<String, String> {
        self.output.clear();
        self.generated_types.clear();
        self.library_name = library_path.to_string();

        // Write header
        self.write_header()?;

        // Write library loading code
        self.write_library_loader(library_path)?;

        // Collect exported functions from items
        let exported_funcs: Vec<&HirFn> = hir
            .items
            .iter()
            .filter_map(|item| {
                if let HirItem::Function(func) = item {
                    if func.is_exported { Some(func) } else { None }
                } else {
                    None
                }
            })
            .collect();

        if exported_funcs.is_empty() {
            return Err("No exported functions found in module".to_string());
        }

        // Generate struct definitions first (for complex return types)
        for func in &exported_funcs {
            self.generate_type_definitions(&func.ty.return_type)?;
            for param in &func.ty.params {
                self.generate_type_definitions(&param.ty)?;
            }
        }

        // Generate function bindings
        writeln!(self.output, "\n# Function signatures").map_err(|e| e.to_string())?;
        for func in &exported_funcs {
            self.generate_function_binding(func)?;
        }

        // Generate Python wrapper functions
        writeln!(self.output, "\n# Python wrapper functions").map_err(|e| e.to_string())?;
        for func in &exported_funcs {
            self.generate_python_wrapper(func)?;
        }

        Ok(self.output.clone())
    }

    /// Write the Python file header with imports
    fn write_header(&mut self) -> Result<(), String> {
        writeln!(
            self.output,
            r#"# Generated Sounio ctypes bindings
# Do not edit manually - regenerate from source

import ctypes
from ctypes import (
    c_void_p, c_char_p, c_int, c_uint, c_long, c_ulong,
    c_int8, c_uint8, c_int16, c_uint16, c_int32, c_uint32,
    c_int64, c_uint64, c_float, c_double, c_bool, c_size_t,
    Structure, POINTER, CFUNCTYPE
)
from pathlib import Path
from typing import Optional, Union, List, Callable, Any, Tuple
import os
"#
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Write library loading code
    fn write_library_loader(&mut self, library_path: &str) -> Result<(), String> {
        writeln!(
            self.output,
            r#"
# Load the shared library
def _load_library(name: str) -> ctypes.CDLL:
    """Load the Sounio shared library, searching common paths."""
    search_paths = [
        name,
        f"./{{name}}",
        f"./lib{{name}}",
        f"./target/release/lib{{name}}",
        f"./target/debug/lib{{name}}",
    ]

    # Add platform-specific extensions
    extensions = ['']
    import sys
    if sys.platform == 'darwin':
        extensions.extend(['.dylib', '.so'])
    elif sys.platform == 'win32':
        extensions.extend(['.dll'])
    else:
        extensions.extend(['.so'])

    for path in search_paths:
        for ext in extensions:
            try:
                full_path = path + ext
                if os.path.exists(full_path):
                    return ctypes.CDLL(full_path)
            except OSError:
                continue

    # Try loading directly (let the system find it)
    return ctypes.CDLL(name)

_lib = _load_library("{library_path}")
"#
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Map Sounio/HIR type to ctypes type string
    fn type_to_ctypes(&self, ty: &HirType) -> String {
        match ty {
            HirType::Unit => "None".to_string(),
            HirType::Bool => "c_bool".to_string(),
            HirType::I8 => "c_int8".to_string(),
            HirType::U8 => "c_uint8".to_string(),
            HirType::I16 => "c_int16".to_string(),
            HirType::U16 => "c_uint16".to_string(),
            HirType::I32 => "c_int32".to_string(),
            HirType::U32 => "c_uint32".to_string(),
            HirType::I64 => "c_int64".to_string(),
            HirType::U64 => "c_uint64".to_string(),
            HirType::I128 | HirType::U128 => {
                // Python ctypes doesn't have 128-bit types
                // Generate a two-field structure for proper representation
                tracing::warn!("128-bit type mapped to two c_uint64 fields (lo, hi)");
                "Int128".to_string()
            }
            HirType::Isize => "c_long".to_string(),
            HirType::Usize => "c_size_t".to_string(),
            HirType::F32 => "c_float".to_string(),
            HirType::F64 => "c_double".to_string(),
            HirType::Char => "c_uint32".to_string(), // Unicode code point
            HirType::String => "c_char_p".to_string(),
            HirType::RawPointer { inner, .. } => {
                let inner_type = self.type_to_ctypes(inner);
                if inner_type == "c_int8" {
                    "c_char_p".to_string()
                } else if inner_type == "None" {
                    "c_void_p".to_string()
                } else {
                    format!("POINTER({})", inner_type)
                }
            }
            HirType::Ref { inner, .. } => {
                let inner_type = self.type_to_ctypes(inner);
                format!("POINTER({})", inner_type)
            }
            HirType::Array { element, .. } => {
                let elem_type = self.type_to_ctypes(element);
                format!("POINTER({})", elem_type)
            }
            HirType::Fn {
                params,
                return_type,
            } => {
                let param_types: Vec<_> = params.iter().map(|p| self.type_to_ctypes(p)).collect();
                let ret_type = self.type_to_ctypes(return_type);
                let ret_str = if ret_type == "None" {
                    "None".to_string()
                } else {
                    ret_type
                };
                format!("CFUNCTYPE({}, {})", ret_str, param_types.join(", "))
            }
            HirType::Named { name, .. } => {
                // Handle common type aliases
                match name.as_str() {
                    "c_void" => "c_void_p".to_string(),
                    "c_char" => "c_char".to_string(),
                    "c_int" => "c_int".to_string(),
                    "c_uint" => "c_uint".to_string(),
                    "c_long" => "c_long".to_string(),
                    "c_ulong" => "c_ulong".to_string(),
                    _ => name.clone(),
                }
            }
            HirType::Tuple(elements) => {
                if elements.is_empty() {
                    "None".to_string()
                } else {
                    // Tuples become structs
                    format!("Tuple{}", elements.len())
                }
            }
            // Linear algebra types - map to arrays
            HirType::Vec2 => "c_float * 2".to_string(),
            HirType::Vec3 => "c_float * 3".to_string(),
            HirType::Vec4 => "c_float * 4".to_string(),
            HirType::Mat2 => "c_float * 4".to_string(),
            HirType::Mat3 => "c_float * 9".to_string(),
            HirType::Mat4 => "c_float * 16".to_string(),
            HirType::Quat => "c_float * 4".to_string(),
            HirType::Dual => "c_double * 2".to_string(),
            _ => {
                tracing::warn!("Unknown HIR type {:?} mapped to c_void_p in Python bindings", ty);
                "c_void_p".to_string()
            }
        }
    }

    /// Generate type definitions for complex types
    fn generate_type_definitions(&mut self, ty: &HirType) -> Result<(), String> {
        match ty {
            HirType::Named { name, .. } => {
                if !self.generated_types.contains(name) {
                    self.generated_types.insert(name.clone());
                    // For now, generate a placeholder struct
                    // In a full implementation, we'd look up the struct definition
                    writeln!(
                        self.output,
                        r#"
class {name}(Structure):
    """Sounio struct: {name}"""
    _fields_ = []  # TODO: Generate actual fields from struct definition
"#
                    )
                    .map_err(|e| e.to_string())?;
                }
            }
            HirType::Tuple(elements) if !elements.is_empty() => {
                let name = format!("Tuple{}", elements.len());
                if !self.generated_types.contains(&name) {
                    self.generated_types.insert(name.clone());
                    let fields: Vec<_> = elements
                        .iter()
                        .enumerate()
                        .map(|(i, t)| format!("    ('_{i}', {})", self.type_to_ctypes(t)))
                        .collect();
                    writeln!(
                        self.output,
                        r#"
class {name}(Structure):
    """Sounio tuple with {len} elements"""
    _fields_ = [
{fields}
    ]
"#,
                        len = elements.len(),
                        fields = fields.join(",\n")
                    )
                    .map_err(|e| e.to_string())?;
                }
            }
            HirType::I128 | HirType::U128 => {
                let name = "Int128";
                if !self.generated_types.contains(name) {
                    self.generated_types.insert(name.to_string());
                    writeln!(
                        self.output,
                        r#"
class Int128(Structure):
    """128-bit integer represented as two 64-bit halves (little-endian)"""
    _fields_ = [
        ('lo', ctypes.c_uint64),
        ('hi', ctypes.c_uint64)
    ]
"#
                    )
                    .map_err(|e| e.to_string())?;
                }
            }
            HirType::RawPointer { inner, .. } | HirType::Ref { inner, .. } => {
                self.generate_type_definitions(inner)?;
            }
            HirType::Array { element, .. } => {
                self.generate_type_definitions(element)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Generate ctypes function signature binding
    fn generate_function_binding(&mut self, func: &HirFn) -> Result<(), String> {
        let func_name = &func.name;

        // Generate argtypes
        let argtypes: Vec<_> = func
            .ty
            .params
            .iter()
            .map(|p| self.type_to_ctypes(&p.ty))
            .collect();

        // Generate restype
        let restype = self.type_to_ctypes(&func.ty.return_type);

        writeln!(
            self.output,
            r#"
# {name}: {signature}
_lib.{func_name}.argtypes = [{argtypes}]
_lib.{func_name}.restype = {restype}
"#,
            name = func.name,
            signature = self.generate_signature_comment(func),
            func_name = func_name,
            argtypes = argtypes.join(", "),
            restype = if restype == "None" {
                "None".to_string()
            } else {
                restype
            }
        )
        .map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Generate human-readable signature comment
    fn generate_signature_comment(&self, func: &HirFn) -> String {
        let params: Vec<_> = func
            .ty
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, self.type_to_python_hint(&p.ty)))
            .collect();
        let ret = self.type_to_python_hint(&func.ty.return_type);
        format!("({}) -> {}", params.join(", "), ret)
    }

    /// Map Sounio type to Python type hint
    fn type_to_python_hint(&self, ty: &HirType) -> String {
        match ty {
            HirType::Unit => "None".to_string(),
            HirType::Bool => "bool".to_string(),
            HirType::I8 | HirType::I16 | HirType::I32 | HirType::I64 | HirType::I128 => {
                "int".to_string()
            }
            HirType::U8 | HirType::U16 | HirType::U32 | HirType::U64 | HirType::U128 => {
                "int".to_string()
            }
            HirType::Isize | HirType::Usize => "int".to_string(),
            HirType::F32 | HirType::F64 => "float".to_string(),
            HirType::Char => "str".to_string(),
            HirType::String => "str".to_string(),
            HirType::RawPointer { .. } | HirType::Ref { .. } => "int".to_string(), // Pointer as int
            HirType::Array { element, .. } => {
                format!("List[{}]", self.type_to_python_hint(element))
            }
            HirType::Tuple(elements) => {
                let inner: Vec<_> = elements
                    .iter()
                    .map(|t| self.type_to_python_hint(t))
                    .collect();
                format!("Tuple[{}]", inner.join(", "))
            }
            HirType::Fn { .. } => "Callable".to_string(),
            HirType::Vec2 | HirType::Vec3 | HirType::Vec4 => "List[float]".to_string(),
            HirType::Mat2 | HirType::Mat3 | HirType::Mat4 => "List[float]".to_string(),
            HirType::Quat => "List[float]".to_string(),
            HirType::Dual => "Tuple[float, float]".to_string(),
            _ => "Any".to_string(),
        }
    }

    /// Generate a Python wrapper function for easier use
    fn generate_python_wrapper(&mut self, func: &HirFn) -> Result<(), String> {
        let func_name = &func.name;

        // Generate parameter list
        let params: Vec<_> = func
            .ty
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, self.type_to_python_hint(&p.ty)))
            .collect();

        let param_names: Vec<_> = func.ty.params.iter().map(|p| p.name.clone()).collect();

        let return_hint = self.type_to_python_hint(&func.ty.return_type);

        writeln!(
            self.output,
            r#"
def {name}({params}) -> {return_hint}:
    """{docstring}"""
    return _lib.{func_name}({args})
"#,
            name = func.name,
            params = params.join(", "),
            return_hint = return_hint,
            docstring = format!("Call Sounio function: {}", func.name),
            func_name = func_name,
            args = param_names.join(", ")
        )
        .map_err(|e| e.to_string())?;

        Ok(())
    }
}

impl Default for PythonBindingGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate Python ctypes bindings from HIR module
pub fn generate_python_bindings(hir: &Hir, library_path: &str) -> Result<String, String> {
    let mut generator = PythonBindingGenerator::new();
    generator.generate_from_hir(hir, library_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_to_ctypes() {
        let generator = PythonBindingGenerator::new();

        assert_eq!(generator.type_to_ctypes(&HirType::I32), "c_int32");
        assert_eq!(generator.type_to_ctypes(&HirType::F64), "c_double");
        assert_eq!(generator.type_to_ctypes(&HirType::Bool), "c_bool");
        assert_eq!(generator.type_to_ctypes(&HirType::Unit), "None");
    }

    #[test]
    fn test_type_to_python_hint() {
        let generator = PythonBindingGenerator::new();

        assert_eq!(generator.type_to_python_hint(&HirType::I32), "int");
        assert_eq!(generator.type_to_python_hint(&HirType::F64), "float");
        assert_eq!(generator.type_to_python_hint(&HirType::Bool), "bool");
        assert_eq!(generator.type_to_python_hint(&HirType::Unit), "None");
    }
}

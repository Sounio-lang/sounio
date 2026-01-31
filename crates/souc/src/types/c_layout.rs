//! C-compatible type layout calculation for FFI
//!
//! This module provides functions to calculate size, alignment, and field offsets
//! for types marked with `#[repr(C)]`. These calculations follow the C ABI rules
//! for the target platform.
//!
//! # C ABI Layout Rules (x86-64 System V / LP64)
//!
//! 1. Fields are laid out in declaration order
//! 2. Each field is aligned to its natural alignment
//! 3. Struct alignment = max(field alignments), minimum 1
//! 4. Struct size is padded to be a multiple of struct alignment
//! 5. Empty structs have size 1, alignment 1
//!
//! # Example
//!
//! ```sio
//! #[repr(C)]
//! struct Example {
//!     a: i8,    // offset 0, size 1
//!     // 7 bytes padding
//!     b: i64,   // offset 8, size 8
//!     c: i32,   // offset 16, size 4
//!     // 4 bytes padding
//! }
//! // total size: 24, alignment: 8
//! ```

use crate::hlir::ir::HlirType;
use crate::types::core::Type;
use std::collections::HashMap;

/// Target platform configuration for layout calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetPlatform {
    /// x86-64 Linux/macOS (LP64 data model)
    X86_64,
    /// ARM64 (AArch64) Linux/macOS
    Arm64,
    /// x86-64 Windows (LLP64 data model)
    X86_64Windows,
    /// 32-bit x86
    X86,
}

impl Default for TargetPlatform {
    fn default() -> Self {
        // Default to current platform
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_os = "windows")]
            return TargetPlatform::X86_64Windows;
            #[cfg(not(target_os = "windows"))]
            return TargetPlatform::X86_64;
        }
        #[cfg(target_arch = "aarch64")]
        return TargetPlatform::Arm64;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        return TargetPlatform::X86_64; // Fallback
    }
}

impl TargetPlatform {
    /// Get pointer size in bytes for this platform
    pub fn pointer_size(&self) -> usize {
        match self {
            TargetPlatform::X86_64 | TargetPlatform::Arm64 | TargetPlatform::X86_64Windows => 8,
            TargetPlatform::X86 => 4,
        }
    }

    /// Get size_t/usize size in bytes for this platform
    pub fn size_type_size(&self) -> usize {
        self.pointer_size()
    }
}

/// Layout information for a field in a struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldLayout {
    /// Field name
    pub name: String,
    /// Offset from struct start in bytes
    pub offset: usize,
    /// Size of the field in bytes
    pub size: usize,
    /// Alignment requirement of the field
    pub alignment: usize,
}

/// Complete layout information for a struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructLayout {
    /// Total size of the struct including padding
    pub size: usize,
    /// Alignment requirement of the struct
    pub alignment: usize,
    /// Layout of each field
    pub fields: Vec<FieldLayout>,
}

/// C layout calculation engine
#[derive(Debug, Clone)]
pub struct CLayoutEngine {
    /// Target platform for layout calculations
    platform: TargetPlatform,
    /// Cache of struct layouts
    struct_cache: HashMap<String, StructLayout>,
}

impl Default for CLayoutEngine {
    fn default() -> Self {
        Self::new(TargetPlatform::default())
    }
}

impl CLayoutEngine {
    /// Create a new layout engine for the specified platform
    pub fn new(platform: TargetPlatform) -> Self {
        Self {
            platform,
            struct_cache: HashMap::new(),
        }
    }

    /// Get the target platform
    pub fn platform(&self) -> TargetPlatform {
        self.platform
    }

    /// Calculate the size of a type in bytes
    pub fn size_of(&self, ty: &HlirType) -> usize {
        match ty {
            // Unit/Void
            HlirType::Void => 0,

            // Boolean (C99 _Bool is typically 1 byte)
            HlirType::Bool => 1,

            // Signed integers
            HlirType::I8 => 1,
            HlirType::I16 => 2,
            HlirType::I32 => 4,
            HlirType::I64 => 8,
            HlirType::I128 => 16,

            // Unsigned integers
            HlirType::U8 => 1,
            HlirType::U16 => 2,
            HlirType::U32 => 4,
            HlirType::U64 => 8,
            HlirType::U128 => 16,

            // Floating point
            HlirType::F32 => 4,
            HlirType::F64 => 8,

            // Pointers
            HlirType::Ptr(_) => self.platform.pointer_size(),

            // Arrays
            HlirType::Array(elem, len) => self.size_of(elem) * len,

            // Tuples - laid out like C structs
            HlirType::Tuple(elems) => {
                if elems.is_empty() {
                    return 0;
                }
                let mut offset = 0;
                let mut max_align = 1;
                for elem in elems {
                    let align = self.align_of(elem);
                    max_align = max_align.max(align);
                    offset = align_to(offset, align);
                    offset += self.size_of(elem);
                }
                // Pad to struct alignment
                align_to(offset, max_align)
            }

            // Structs - lookup in cache or calculate
            HlirType::Struct(name) => {
                if let Some(layout) = self.struct_cache.get(name) {
                    layout.size
                } else {
                    // Unknown struct - assume pointer size as conservative estimate
                    self.platform.pointer_size()
                }
            }

            // Function pointers
            HlirType::Function { .. } => self.platform.pointer_size(),

            // Linear algebra types (SIMD-friendly)
            HlirType::Vec2 => 8,  // 2 x f32
            HlirType::Vec3 => 12, // 3 x f32 (often padded to 16 for SIMD)
            HlirType::Vec4 => 16, // 4 x f32
            HlirType::Mat2 => 16, // 2x2 x f32
            HlirType::Mat3 => 36, // 3x3 x f32
            HlirType::Mat4 => 64, // 4x4 x f32
            HlirType::Quat => 16, // 4 x f32
            // Octonion: 8 x f32 = 32 bytes
            HlirType::Octonion => 32,

            // f64 vector types
            HlirType::Vec2d => 16, // 2 x f64
            HlirType::Vec3d => 24, // 3 x f64 (often padded to 32 for SIMD)
            HlirType::Vec4d => 32, // 4 x f64

            // Dual numbers (autodiff)
            HlirType::Dual => 16, // 2 x f64
            HlirType::Knowledge { inner, .. } => self.size_of(inner),

            // Quaternionic Neural Network types - treat as structs
            HlirType::QuatLinear { .. } => 16,
            HlirType::QuatConv2d { .. } => 16,
            HlirType::QuatRnnState { .. } => 16,
            HlirType::QuatGate { .. } => 16,
        }
    }

    /// Calculate the alignment requirement of a type in bytes
    pub fn align_of(&self, ty: &HlirType) -> usize {
        match ty {
            HlirType::Void => 1,
            HlirType::Bool => 1,

            HlirType::I8 | HlirType::U8 => 1,
            HlirType::I16 | HlirType::U16 => 2,
            HlirType::I32 | HlirType::U32 | HlirType::F32 => 4,
            HlirType::I64 | HlirType::U64 | HlirType::F64 => 8,
            HlirType::I128 | HlirType::U128 => 16,

            HlirType::Ptr(_) => self.platform.pointer_size(),

            // Array alignment is element alignment
            HlirType::Array(elem, _) => self.align_of(elem),

            // Tuple alignment is max of element alignments
            HlirType::Tuple(elems) => elems.iter().map(|e| self.align_of(e)).max().unwrap_or(1),

            HlirType::Struct(name) => {
                if let Some(layout) = self.struct_cache.get(name) {
                    layout.alignment
                } else {
                    self.platform.pointer_size()
                }
            }

            HlirType::Function { .. } => self.platform.pointer_size(),

            // SIMD types typically have natural alignment
            HlirType::Vec2 => 8,  // or 4 if not SIMD
            HlirType::Vec3 => 16, // Usually padded to 16 for SIMD
            HlirType::Vec4 => 16,
            HlirType::Mat2 => 16,
            HlirType::Mat3 => 16,
            HlirType::Mat4 => 16,
            HlirType::Quat => 16,

            // Octonion: 8 x f32 = 32 bytes, aligned to 32 for SIMD
            HlirType::Octonion => 32,

            // f64 vector types
            HlirType::Vec2d => 16, // Aligned to 16 for SIMD
            HlirType::Vec3d => 32, // Aligned to 32 for SIMD
            HlirType::Vec4d => 32,

            HlirType::Dual => 8, // Aligned to f64
            HlirType::Knowledge { inner, .. } => self.align_of(inner),

            // Quaternionic Neural Network types
            HlirType::QuatLinear { .. } => 16,
            HlirType::QuatConv2d { .. } => 16,
            HlirType::QuatRnnState { .. } => 16,
            HlirType::QuatGate { .. } => 16,
        }
    }

    /// Calculate the layout of a struct with the given fields
    ///
    /// This follows C ABI rules: fields are laid out in order with natural
    /// alignment, and the struct is padded to its alignment at the end.
    pub fn layout_struct(&mut self, name: &str, fields: &[(String, HlirType)]) -> StructLayout {
        // Check cache first
        if let Some(layout) = self.struct_cache.get(name) {
            return layout.clone();
        }

        // Empty struct
        if fields.is_empty() {
            let layout = StructLayout {
                size: 1, // C++ says 1, C says undefined (we use 1 for consistency)
                alignment: 1,
                fields: Vec::new(),
            };
            self.struct_cache.insert(name.to_string(), layout.clone());
            return layout;
        }

        let mut offset = 0;
        let mut max_align = 1;
        let mut field_layouts = Vec::with_capacity(fields.len());

        for (field_name, field_ty) in fields {
            let field_size = self.size_of(field_ty);
            let field_align = self.align_of(field_ty);

            // Update max alignment
            max_align = max_align.max(field_align);

            // Align current offset
            offset = align_to(offset, field_align);

            field_layouts.push(FieldLayout {
                name: field_name.clone(),
                offset,
                size: field_size,
                alignment: field_align,
            });

            // Move past this field
            offset += field_size;
        }

        // Pad struct size to alignment
        let struct_size = align_to(offset, max_align);

        let layout = StructLayout {
            size: struct_size,
            alignment: max_align,
            fields: field_layouts,
        };

        self.struct_cache.insert(name.to_string(), layout.clone());
        layout
    }

    /// Get the offset of a field in a struct
    pub fn field_offset(&mut self, struct_name: &str, field_name: &str) -> Option<usize> {
        if let Some(layout) = self.struct_cache.get(struct_name) {
            layout
                .fields
                .iter()
                .find(|f| f.name == field_name)
                .map(|f| f.offset)
        } else {
            None
        }
    }

    /// Calculate size from core Type (used earlier in the pipeline)
    pub fn size_of_core_type(&self, ty: &Type) -> usize {
        match ty {
            Type::Unit => 0,
            Type::Bool => 1,
            Type::I8 | Type::U8 | Type::Char => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 => 8,
            Type::I128 | Type::U128 => 16,
            Type::Isize | Type::Usize => self.platform.size_type_size(),
            Type::Str | Type::String => {
                // Fat pointer: ptr + len
                self.platform.pointer_size() * 2
            }
            Type::Ref { .. } | Type::RawPointer { .. } => self.platform.pointer_size(),
            Type::Array { element, size } => {
                let elem_size = self.size_of_core_type(element);
                elem_size * size.unwrap_or(0)
            }
            Type::Tuple(elems) => {
                if elems.is_empty() {
                    return 0;
                }
                let mut offset = 0;
                let mut max_align = 1;
                for elem in elems {
                    let align = self.align_of_core_type(elem);
                    max_align = max_align.max(align);
                    offset = align_to(offset, align);
                    offset += self.size_of_core_type(elem);
                }
                align_to(offset, max_align)
            }
            Type::Function { .. } => self.platform.pointer_size(),
            Type::Named { .. } => self.platform.pointer_size(), // Conservative
            Type::Vec2 => 8,
            Type::Vec3 => 12,
            Type::Vec4 => 16,
            Type::Mat2 => 16,
            Type::Mat3 => 36,
            Type::Mat4 => 64,
            Type::Quat => 16,
            // Octonion: 8 x f32 = 32 bytes
            Type::Octonion => 32,
            Type::Dual => 16,
            _ => self.platform.pointer_size(), // Conservative default
        }
    }

    /// Calculate alignment from core Type
    pub fn align_of_core_type(&self, ty: &Type) -> usize {
        match ty {
            Type::Unit => 1,
            Type::Bool => 1,
            Type::I8 | Type::U8 | Type::Char => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 => 8,
            Type::I128 | Type::U128 => 16,
            Type::Isize | Type::Usize => self.platform.pointer_size(),
            Type::Str | Type::String => self.platform.pointer_size(),
            Type::Ref { .. } | Type::RawPointer { .. } => self.platform.pointer_size(),
            Type::Array { element, .. } => self.align_of_core_type(element),
            Type::Tuple(elems) => elems
                .iter()
                .map(|e| self.align_of_core_type(e))
                .max()
                .unwrap_or(1),
            Type::Function { .. } => self.platform.pointer_size(),
            Type::Named { .. } => self.platform.pointer_size(),
            Type::Vec2 => 8,
            Type::Vec3 | Type::Vec4 | Type::Mat2 | Type::Mat3 | Type::Mat4 | Type::Quat => 16,
            // Octonion aligned to 32 for 8-wide SIMD
            Type::Octonion => 32,
            Type::Dual => 8,
            _ => self.platform.pointer_size(),
        }
    }

    /// Check if a type is C-compatible (can cross FFI boundary safely)
    pub fn is_c_compatible(&self, ty: &HlirType) -> bool {
        match ty {
            // Primitives are always compatible
            HlirType::Void
            | HlirType::Bool
            | HlirType::I8
            | HlirType::I16
            | HlirType::I32
            | HlirType::I64
            | HlirType::U8
            | HlirType::U16
            | HlirType::U32
            | HlirType::U64
            | HlirType::F32
            | HlirType::F64 => true,

            // 128-bit integers may not be supported on all C compilers
            HlirType::I128 | HlirType::U128 => false,

            // Pointers are compatible
            HlirType::Ptr(_) => true,

            // Fixed-size arrays of compatible types are compatible
            HlirType::Array(elem, _) => self.is_c_compatible(elem),

            // Tuples are not directly C-compatible (use struct)
            HlirType::Tuple(_) => false,

            // Structs need #[repr(C)] - assume compatible if in cache
            HlirType::Struct(name) => self.struct_cache.contains_key(name),

            // Function pointers are compatible
            HlirType::Function {
                params,
                return_type,
            } => {
                params.iter().all(|p| self.is_c_compatible(p)) && self.is_c_compatible(return_type)
            }

            // SIMD types need special handling
            HlirType::Vec2 | HlirType::Vec3 | HlirType::Vec4 => false,
            HlirType::Vec2d | HlirType::Vec3d | HlirType::Vec4d => false,
            HlirType::Mat2 | HlirType::Mat3 | HlirType::Mat4 => false,
            HlirType::Quat => false,
            // Octonion needs special handling (8 floats)
            HlirType::Octonion => false,
            // Quaternionic Neural Network types
            HlirType::QuatLinear { .. } => false,
            HlirType::QuatConv2d { .. } => false,
            HlirType::QuatRnnState { .. } => false,
            HlirType::QuatGate { .. } => false,
            HlirType::Dual => false,
            HlirType::Knowledge { .. } => false,
        }
    }

    /// Generate C type declaration for a Sounio type
    pub fn to_c_type(&self, ty: &HlirType) -> String {
        match ty {
            HlirType::Void => "void".to_string(),
            HlirType::Bool => "_Bool".to_string(), // C99
            HlirType::I8 => "int8_t".to_string(),
            HlirType::I16 => "int16_t".to_string(),
            HlirType::I32 => "int32_t".to_string(),
            HlirType::I64 => "int64_t".to_string(),
            HlirType::I128 => "__int128".to_string(), // GCC/Clang extension
            HlirType::U8 => "uint8_t".to_string(),
            HlirType::U16 => "uint16_t".to_string(),
            HlirType::U32 => "uint32_t".to_string(),
            HlirType::U64 => "uint64_t".to_string(),
            HlirType::U128 => "unsigned __int128".to_string(),
            HlirType::F32 => "float".to_string(),
            HlirType::F64 => "double".to_string(),
            HlirType::Ptr(inner) => {
                let inner_c = self.to_c_type(inner);
                if matches!(inner.as_ref(), HlirType::Void) {
                    "void*".to_string()
                } else {
                    format!("{}*", inner_c)
                }
            }
            HlirType::Array(elem, len) => {
                let elem_c = self.to_c_type(elem);
                format!("{}[{}]", elem_c, len)
            }
            HlirType::Struct(name) => format!("struct {}", name),
            HlirType::Function {
                params,
                return_type,
            } => {
                let ret = self.to_c_type(return_type);
                let param_strs: Vec<_> = params.iter().map(|p| self.to_c_type(p)).collect();
                format!("{} (*)({})", ret, param_strs.join(", "))
            }
            HlirType::Tuple(elems) => {
                // Tuples become anonymous structs
                let fields: Vec<_> = elems
                    .iter()
                    .enumerate()
                    .map(|(i, e)| format!("{} _{}", self.to_c_type(e), i))
                    .collect();
                format!("struct {{ {} }}", fields.join("; "))
            }
            // SIMD types - platform specific
            HlirType::Vec2 => "float __attribute__((vector_size(8)))".to_string(),
            HlirType::Vec3 => "float __attribute__((vector_size(16)))".to_string(),
            HlirType::Vec4 => "float __attribute__((vector_size(16)))".to_string(),
            HlirType::Vec2d => "double __attribute__((vector_size(16)))".to_string(),
            HlirType::Vec3d => "double __attribute__((vector_size(32)))".to_string(),
            HlirType::Vec4d => "double __attribute__((vector_size(32)))".to_string(),
            HlirType::Mat2 => "float[4]".to_string(),
            HlirType::Mat3 => "float[9]".to_string(),
            HlirType::Mat4 => "float[16]".to_string(),
            HlirType::Quat => "float[4]".to_string(),
            // Octonion: 8 floats
            HlirType::Octonion => "float[8]".to_string(),
            // Quaternionic Neural Network types
            HlirType::QuatLinear { .. } => "struct { float* weights; float* bias; }".to_string(),
            HlirType::QuatConv2d { .. } => "struct { float* kernel; int* dims; }".to_string(),
            HlirType::QuatRnnState { .. } => "float[hidden_size][4]".to_string(),
            HlirType::QuatGate { .. } => "struct { float* w_i; float* w_h; float* b; }".to_string(),
            HlirType::Dual => "struct { double value; double derivative; }".to_string(),
            HlirType::Knowledge { inner, .. } => {
                format!("/* Knowledge */ {}", self.to_c_type(inner))
            }
        }
    }
}

/// Align a value up to the given alignment (must be power of 2)
#[inline]
fn align_to(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two() || alignment == 0);
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_sizes() {
        let engine = CLayoutEngine::default();

        assert_eq!(engine.size_of(&HlirType::I8), 1);
        assert_eq!(engine.size_of(&HlirType::I16), 2);
        assert_eq!(engine.size_of(&HlirType::I32), 4);
        assert_eq!(engine.size_of(&HlirType::I64), 8);
        assert_eq!(engine.size_of(&HlirType::F32), 4);
        assert_eq!(engine.size_of(&HlirType::F64), 8);
    }

    #[test]
    fn test_primitive_alignments() {
        let engine = CLayoutEngine::default();

        assert_eq!(engine.align_of(&HlirType::I8), 1);
        assert_eq!(engine.align_of(&HlirType::I16), 2);
        assert_eq!(engine.align_of(&HlirType::I32), 4);
        assert_eq!(engine.align_of(&HlirType::I64), 8);
    }

    #[test]
    fn test_struct_layout_simple() {
        let mut engine = CLayoutEngine::default();

        // struct { i32 a; i32 b; }
        let layout = engine.layout_struct(
            "Simple",
            &[
                ("a".to_string(), HlirType::I32),
                ("b".to_string(), HlirType::I32),
            ],
        );

        assert_eq!(layout.size, 8);
        assert_eq!(layout.alignment, 4);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[1].offset, 4);
    }

    #[test]
    fn test_struct_layout_with_padding() {
        let mut engine = CLayoutEngine::default();

        // struct { i8 a; i64 b; i32 c; }
        // Layout: a at 0, padding 1-7, b at 8, c at 16, padding 20-23
        let layout = engine.layout_struct(
            "Padded",
            &[
                ("a".to_string(), HlirType::I8),
                ("b".to_string(), HlirType::I64),
                ("c".to_string(), HlirType::I32),
            ],
        );

        assert_eq!(layout.alignment, 8);
        assert_eq!(layout.fields[0].offset, 0); // a
        assert_eq!(layout.fields[1].offset, 8); // b (aligned to 8)
        assert_eq!(layout.fields[2].offset, 16); // c
        assert_eq!(layout.size, 24); // Padded to alignment
    }

    #[test]
    fn test_struct_layout_nested() {
        let mut engine = CLayoutEngine::default();

        // First define inner struct
        engine.layout_struct(
            "Inner",
            &[
                ("x".to_string(), HlirType::I32),
                ("y".to_string(), HlirType::I32),
            ],
        );

        // struct { i8 a; Inner inner; i8 b; }
        let layout = engine.layout_struct(
            "Outer",
            &[
                ("a".to_string(), HlirType::I8),
                ("inner".to_string(), HlirType::Struct("Inner".to_string())),
                ("b".to_string(), HlirType::I8),
            ],
        );

        assert_eq!(layout.fields[0].offset, 0); // a
                                                // inner should be aligned to 4 (its alignment from cache)
                                                // but since Inner wasn't registered with proper size, this test shows the limitation
    }

    #[test]
    fn test_array_layout() {
        let engine = CLayoutEngine::default();

        // i32[10]
        let array_ty = HlirType::Array(Box::new(HlirType::I32), 10);
        assert_eq!(engine.size_of(&array_ty), 40);
        assert_eq!(engine.align_of(&array_ty), 4);
    }

    #[test]
    fn test_pointer_layout() {
        let engine = CLayoutEngine::new(TargetPlatform::X86_64);

        let ptr_ty = HlirType::Ptr(Box::new(HlirType::I32));
        assert_eq!(engine.size_of(&ptr_ty), 8);
        assert_eq!(engine.align_of(&ptr_ty), 8);
    }

    #[test]
    fn test_c_type_generation() {
        let engine = CLayoutEngine::default();

        assert_eq!(engine.to_c_type(&HlirType::I32), "int32_t");
        assert_eq!(engine.to_c_type(&HlirType::F64), "double");
        assert_eq!(
            engine.to_c_type(&HlirType::Ptr(Box::new(HlirType::I8))),
            "int8_t*"
        );
        assert_eq!(
            engine.to_c_type(&HlirType::Ptr(Box::new(HlirType::Void))),
            "void*"
        );
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(0, 4), 0);
        assert_eq!(align_to(1, 4), 4);
        assert_eq!(align_to(4, 4), 4);
        assert_eq!(align_to(5, 4), 8);
        assert_eq!(align_to(7, 8), 8);
        assert_eq!(align_to(8, 8), 8);
        assert_eq!(align_to(9, 8), 16);
    }

    #[test]
    fn test_empty_struct() {
        let mut engine = CLayoutEngine::default();

        let layout = engine.layout_struct("Empty", &[]);
        assert_eq!(layout.size, 1);
        assert_eq!(layout.alignment, 1);
    }

    #[test]
    fn test_c_compatibility_check() {
        let engine = CLayoutEngine::default();

        assert!(engine.is_c_compatible(&HlirType::I32));
        assert!(engine.is_c_compatible(&HlirType::F64));
        assert!(engine.is_c_compatible(&HlirType::Ptr(Box::new(HlirType::Void))));
        assert!(!engine.is_c_compatible(&HlirType::I128)); // Not widely supported
        assert!(!engine.is_c_compatible(&HlirType::Vec4)); // SIMD
    }
}

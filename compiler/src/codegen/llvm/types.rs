//! Type mapping from D/HLIR types to LLVM types
//!
//! This module provides conversion from Sounio HLIR types to LLVM types
//! using inkwell's type system.

use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::types::{
    AnyType, AnyTypeEnum, BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FloatType, FunctionType,
    IntType, PointerType, StructType, VoidType,
};

use crate::hlir::HlirType;

/// Type converter from HLIR types to LLVM types
pub struct TypeConverter<'ctx> {
    context: &'ctx Context,
    /// Cached struct types by name
    struct_cache: std::collections::HashMap<String, StructType<'ctx>>,
}

impl<'ctx> TypeConverter<'ctx> {
    /// Create a new type converter
    pub fn new(context: &'ctx Context) -> Self {
        Self {
            context,
            struct_cache: std::collections::HashMap::new(),
        }
    }

    /// Convert an HLIR type to an LLVM basic type
    pub fn convert(&mut self, ty: &HlirType) -> BasicTypeEnum<'ctx> {
        match ty {
            // Void becomes an empty struct (LLVM void is not a basic type)
            HlirType::Void => self.context.struct_type(&[], false).into(),

            // Boolean
            HlirType::Bool => self.context.bool_type().into(),

            // Signed integers
            HlirType::I8 => self.context.i8_type().into(),
            HlirType::I16 => self.context.i16_type().into(),
            HlirType::I32 => self.context.i32_type().into(),
            HlirType::I64 => self.context.i64_type().into(),
            HlirType::I128 => self.context.i128_type().into(),

            // Unsigned integers (same LLVM type, signedness is in operations)
            HlirType::U8 => self.context.i8_type().into(),
            HlirType::U16 => self.context.i16_type().into(),
            HlirType::U32 => self.context.i32_type().into(),
            HlirType::U64 => self.context.i64_type().into(),
            HlirType::U128 => self.context.i128_type().into(),

            // Floating point
            HlirType::F32 => self.context.f32_type().into(),
            HlirType::F64 => self.context.f64_type().into(),

            // Pointers - LLVM 15+ uses opaque pointers, so all pointers are the same type
            HlirType::Ptr(_inner) => {
                // In LLVM 15+ with opaque pointers, use context.ptr_type() instead of
                // calling ptr_type() on a specific element type
                self.context.ptr_type(AddressSpace::default()).into()
            }

            // Arrays - need to match on specific type for array_type() in LLVM 15+
            HlirType::Array(elem, size) => {
                let elem_ty = self.convert(elem);
                let len = *size as u32;
                let arr_ty = match elem_ty {
                    BasicTypeEnum::IntType(t) => t.array_type(len),
                    BasicTypeEnum::FloatType(t) => t.array_type(len),
                    BasicTypeEnum::PointerType(t) => t.array_type(len),
                    BasicTypeEnum::ArrayType(t) => t.array_type(len),
                    BasicTypeEnum::StructType(t) => t.array_type(len),
                    BasicTypeEnum::VectorType(t) => t.array_type(len),
                };
                arr_ty.into()
            }

            // Structs
            HlirType::Struct(name) => {
                if let Some(cached) = self.struct_cache.get(name) {
                    (*cached).into()
                } else {
                    // Create opaque struct for forward references
                    let struct_ty = self.context.opaque_struct_type(name);
                    self.struct_cache.insert(name.clone(), struct_ty);
                    struct_ty.into()
                }
            }

            // Tuples
            HlirType::Tuple(elems) => {
                let elem_types: Vec<BasicTypeEnum<'ctx>> =
                    elems.iter().map(|t| self.convert(t)).collect();
                self.context.struct_type(&elem_types, false).into()
            }

            // Function pointers - LLVM 15+ uses opaque pointers
            HlirType::Function {
                params,
                return_type,
            } => {
                // Function type is still created for call site type checking,
                // but the pointer itself is opaque in LLVM 15+
                let _fn_ty = self.function_type(params, return_type);
                self.context.ptr_type(AddressSpace::default()).into()
            }

            // SIMD vector types
            HlirType::Vec2 => {
                // 2x f32 vector
                self.context.f32_type().vec_type(2).into()
            }
            HlirType::Vec3 => {
                // 3x f32 padded to 4 for SIMD alignment
                self.context.f32_type().vec_type(4).into()
            }
            HlirType::Vec4 => {
                // 4x f32 vector
                self.context.f32_type().vec_type(4).into()
            }
            HlirType::Mat2 => {
                // 2x2 matrix as array of 4 f32
                self.context.f32_type().array_type(4).into()
            }
            HlirType::Mat3 => {
                // 3x3 matrix as array of 12 f32 (padded rows)
                self.context.f32_type().array_type(12).into()
            }
            HlirType::Mat4 => {
                // 4x4 matrix as array of 16 f32
                self.context.f32_type().array_type(16).into()
            }
            HlirType::Quat => {
                // Quaternion as 4x f32 vector
                self.context.f32_type().vec_type(4).into()
            }
            HlirType::Dual => {
                // Dual number: struct { value: f64, derivative: f64 }
                let f64_ty = self.context.f64_type();
                self.context
                    .struct_type(&[f64_ty.into(), f64_ty.into()], false)
                    .into()
            }
        }
    }

    /// Convert an HLIR type to a basic metadata type (for function parameters)
    pub fn convert_to_metadata(&mut self, ty: &HlirType) -> BasicMetadataTypeEnum<'ctx> {
        self.convert(ty).into()
    }

    /// Get the LLVM void type
    pub fn void_type(&self) -> VoidType<'ctx> {
        self.context.void_type()
    }

    /// Get a specific integer type
    pub fn int_type(&self, bits: u32) -> IntType<'ctx> {
        self.context.custom_width_int_type(bits)
    }

    /// Get i8 type
    pub fn i8_type(&self) -> IntType<'ctx> {
        self.context.i8_type()
    }

    /// Get i16 type
    pub fn i16_type(&self) -> IntType<'ctx> {
        self.context.i16_type()
    }

    /// Get i32 type
    pub fn i32_type(&self) -> IntType<'ctx> {
        self.context.i32_type()
    }

    /// Get i64 type
    pub fn i64_type(&self) -> IntType<'ctx> {
        self.context.i64_type()
    }

    /// Get i128 type
    pub fn i128_type(&self) -> IntType<'ctx> {
        self.context.i128_type()
    }

    /// Get f32 type
    pub fn f32_type(&self) -> FloatType<'ctx> {
        self.context.f32_type()
    }

    /// Get f64 type
    pub fn f64_type(&self) -> FloatType<'ctx> {
        self.context.f64_type()
    }

    /// Get bool type (i1)
    pub fn bool_type(&self) -> IntType<'ctx> {
        self.context.bool_type()
    }

    /// Get pointer type for a given element type
    /// Note: In LLVM 15+ with opaque pointers, the element type is ignored
    pub fn ptr_type(&self, _element: BasicTypeEnum<'ctx>) -> PointerType<'ctx> {
        // LLVM 15+ uses opaque pointers - all pointers are the same type
        self.context.ptr_type(AddressSpace::default())
    }

    /// Get generic pointer type (opaque pointer in LLVM 15+)
    pub fn generic_ptr_type(&self) -> PointerType<'ctx> {
        // LLVM 15+ uses opaque pointers
        self.context.ptr_type(AddressSpace::default())
    }

    /// Create a string type (fat pointer: ptr + len)
    pub fn string_type(&self) -> StructType<'ctx> {
        // LLVM 15+ uses opaque pointers
        let ptr_ty = self.context.ptr_type(AddressSpace::default());
        let len_ty = self.context.i64_type();
        self.context
            .struct_type(&[ptr_ty.into(), len_ty.into()], false)
    }

    /// Create a slice type (fat pointer: ptr + len)
    pub fn slice_type(&mut self, _elem: &HlirType) -> StructType<'ctx> {
        // LLVM 15+ uses opaque pointers - element type doesn't affect pointer type
        let ptr_ty = self.context.ptr_type(AddressSpace::default());
        let len_ty = self.context.i64_type();
        self.context
            .struct_type(&[ptr_ty.into(), len_ty.into()], false)
    }

    /// Create a struct type with fields
    pub fn create_struct_type(&mut self, name: &str, field_types: &[HlirType]) -> StructType<'ctx> {
        let fields: Vec<BasicTypeEnum<'ctx>> =
            field_types.iter().map(|t| self.convert(t)).collect();

        if let Some(cached) = self.struct_cache.get(name) {
            // Set body if it was an opaque struct
            if cached.is_opaque() {
                cached.set_body(&fields, false);
            }
            *cached
        } else {
            let struct_ty = self.context.opaque_struct_type(name);
            struct_ty.set_body(&fields, false);
            self.struct_cache.insert(name.to_string(), struct_ty);
            struct_ty
        }
    }

    /// Create a function type
    pub fn function_type(
        &mut self,
        params: &[HlirType],
        return_type: &HlirType,
    ) -> FunctionType<'ctx> {
        self.function_type_variadic(params, return_type, false)
    }

    /// Create a function type with explicit variadic flag.
    pub fn function_type_variadic(
        &mut self,
        params: &[HlirType],
        return_type: &HlirType,
        is_variadic: bool,
    ) -> FunctionType<'ctx> {
        let param_types: Vec<BasicMetadataTypeEnum<'ctx>> =
            params.iter().map(|t| self.convert_to_metadata(t)).collect();

        match return_type {
            HlirType::Void => self.void_type().fn_type(&param_types, is_variadic),
            _ => {
                let ret_ty = self.convert(return_type);
                ret_ty.fn_type(&param_types, is_variadic)
            }
        }
    }

    /// Get size of a type in bits
    pub fn size_bits(&self, ty: &HlirType) -> u64 {
        match ty {
            HlirType::Void => 0,
            HlirType::Bool => 1,
            HlirType::I8 | HlirType::U8 => 8,
            HlirType::I16 | HlirType::U16 => 16,
            HlirType::I32 | HlirType::U32 | HlirType::F32 => 32,
            HlirType::I64 | HlirType::U64 | HlirType::F64 => 64,
            HlirType::I128 | HlirType::U128 => 128,
            HlirType::Ptr(_) => 64, // Assuming 64-bit pointers
            HlirType::Array(elem, size) => self.size_bits(elem) * (*size as u64),
            HlirType::Struct(_) => 64, // Conservative estimate
            HlirType::Tuple(elems) => elems.iter().map(|e| self.size_bits(e)).sum(),
            HlirType::Function { .. } => 64, // Function pointer
            // SIMD types
            HlirType::Vec2 => 64,  // 2x f32 = 64 bits
            HlirType::Vec3 => 128, // 4x f32 (padded) = 128 bits
            HlirType::Vec4 => 128, // 4x f32 = 128 bits
            HlirType::Mat2 => 128, // 4x f32 = 128 bits
            HlirType::Mat3 => 384, // 12x f32 = 384 bits
            HlirType::Mat4 => 512, // 16x f32 = 512 bits
            HlirType::Quat => 128, // 4x f32 = 128 bits
            HlirType::Dual => 128, // 2x f64 = 128 bits
        }
    }

    /// Get size of a type in bytes
    pub fn size_bytes(&self, ty: &HlirType) -> u64 {
        (self.size_bits(ty) + 7) / 8
    }

    /// Get alignment of a type in bytes
    pub fn align_bytes(&self, ty: &HlirType) -> u64 {
        match ty {
            HlirType::Void => 1,
            HlirType::Bool | HlirType::I8 | HlirType::U8 => 1,
            HlirType::I16 | HlirType::U16 => 2,
            HlirType::I32 | HlirType::U32 | HlirType::F32 => 4,
            HlirType::I64
            | HlirType::U64
            | HlirType::F64
            | HlirType::I128
            | HlirType::U128
            | HlirType::Ptr(_) => 8,
            HlirType::Array(elem, _) => self.align_bytes(elem),
            HlirType::Struct(_) => 8, // Conservative
            HlirType::Tuple(elems) => elems.iter().map(|e| self.align_bytes(e)).max().unwrap_or(1),
            HlirType::Function { .. } => 8,
            // SIMD types - aligned to 16 bytes for optimal SIMD performance
            HlirType::Vec2 => 8, // 8-byte alignment for 2x f32
            HlirType::Vec3 | HlirType::Vec4 | HlirType::Quat => 16, // 16-byte for 4x f32
            HlirType::Mat2 => 16, // 16-byte alignment
            HlirType::Mat3 => 16, // 16-byte alignment (rows are padded)
            HlirType::Mat4 => 16, // 16-byte alignment
            HlirType::Dual => 16, // 16-byte for 2x f64
        }
    }

    /// Check if type is an integer type
    pub fn is_integer_type(&self, ty: &HlirType) -> bool {
        matches!(
            ty,
            HlirType::I8
                | HlirType::I16
                | HlirType::I32
                | HlirType::I64
                | HlirType::I128
                | HlirType::U8
                | HlirType::U16
                | HlirType::U32
                | HlirType::U64
                | HlirType::U128
                | HlirType::Bool
        )
    }

    /// Check if type is a floating point type
    pub fn is_float_type(&self, ty: &HlirType) -> bool {
        matches!(ty, HlirType::F32 | HlirType::F64)
    }

    /// Check if type is signed
    pub fn is_signed(&self, ty: &HlirType) -> bool {
        matches!(
            ty,
            HlirType::I8 | HlirType::I16 | HlirType::I32 | HlirType::I64 | HlirType::I128
        )
    }

    /// Get the integer bit width
    pub fn int_bit_width(&self, ty: &HlirType) -> Option<u32> {
        match ty {
            HlirType::Bool => Some(1),
            HlirType::I8 | HlirType::U8 => Some(8),
            HlirType::I16 | HlirType::U16 => Some(16),
            HlirType::I32 | HlirType::U32 => Some(32),
            HlirType::I64 | HlirType::U64 => Some(64),
            HlirType::I128 | HlirType::U128 => Some(128),
            _ => None,
        }
    }

    /// Get an integer type for the given bit width
    pub fn int_type_for_bits(&self, bits: u32) -> IntType<'ctx> {
        match bits {
            1 => self.context.bool_type(),
            8 => self.context.i8_type(),
            16 => self.context.i16_type(),
            32 => self.context.i32_type(),
            64 => self.context.i64_type(),
            128 => self.context.i128_type(),
            _ => self.context.custom_width_int_type(bits),
        }
    }

    /// Get a float type for the given bit width
    pub fn float_type_for_bits(&self, bits: u32) -> FloatType<'ctx> {
        match bits {
            32 => self.context.f32_type(),
            64 => self.context.f64_type(),
            _ => self.context.f64_type(), // Default to f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_type_conversion() {
        let context = Context::create();
        let mut converter = TypeConverter::new(&context);

        // Test integer types
        let i32_ty = converter.convert(&HlirType::I32);
        assert!(i32_ty.is_int_type());

        let i64_ty = converter.convert(&HlirType::I64);
        assert!(i64_ty.is_int_type());

        // Test float types
        let f64_ty = converter.convert(&HlirType::F64);
        assert!(f64_ty.is_float_type());

        // Test bool
        let bool_ty = converter.convert(&HlirType::Bool);
        assert!(bool_ty.is_int_type());
    }

    #[test]
    fn test_pointer_type() {
        let context = Context::create();
        let mut converter = TypeConverter::new(&context);

        let ptr_ty = converter.convert(&HlirType::Ptr(Box::new(HlirType::I32)));
        assert!(ptr_ty.is_pointer_type());
    }

    #[test]
    fn test_array_type() {
        let context = Context::create();
        let mut converter = TypeConverter::new(&context);

        let arr_ty = converter.convert(&HlirType::Array(Box::new(HlirType::I32), 10));
        assert!(arr_ty.is_array_type());
    }

    #[test]
    fn test_tuple_type() {
        let context = Context::create();
        let mut converter = TypeConverter::new(&context);

        let tuple_ty = converter.convert(&HlirType::Tuple(vec![HlirType::I32, HlirType::F64]));
        assert!(tuple_ty.is_struct_type());
    }

    #[test]
    fn test_function_type() {
        let context = Context::create();
        let mut converter = TypeConverter::new(&context);

        let fn_ty = converter.function_type(&[HlirType::I32, HlirType::I32], &HlirType::I64);

        assert_eq!(fn_ty.count_param_types(), 2);
    }

    #[test]
    fn test_size_calculations() {
        let context = Context::create();
        let converter = TypeConverter::new(&context);

        assert_eq!(converter.size_bits(&HlirType::I8), 8);
        assert_eq!(converter.size_bits(&HlirType::I32), 32);
        assert_eq!(converter.size_bits(&HlirType::I64), 64);
        assert_eq!(converter.size_bits(&HlirType::F64), 64);

        assert_eq!(converter.size_bytes(&HlirType::I32), 4);
        assert_eq!(converter.size_bytes(&HlirType::I64), 8);
    }
}

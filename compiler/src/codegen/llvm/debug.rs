//! DWARF debug information generation
//!
//! This module provides functionality to generate debug information
//! for debugging D programs with tools like gdb and lldb.

use inkwell::context::Context;
use inkwell::debug_info::{
    AsDIScope, DICompileUnit, DIFile, DIFlags, DIFlagsConstants, DILexicalBlock, DILocalVariable,
    DILocation, DIScope, DISubprogram, DIType, DWARFEmissionKind, DWARFSourceLanguage,
    DebugInfoBuilder,
};
use inkwell::module::Module;
use inkwell::values::FunctionValue;

use std::path::Path;

use crate::hlir::HlirType;

/// Debug info builder for LLVM
pub struct DebugBuilder<'ctx> {
    /// The debug info builder
    builder: DebugInfoBuilder<'ctx>,

    /// The compile unit
    compile_unit: DICompileUnit<'ctx>,

    /// The main source file
    file: DIFile<'ctx>,

    /// Current scope stack
    scope_stack: Vec<DIScope<'ctx>>,

    /// LLVM context reference
    context: &'ctx Context,
}

impl<'ctx> DebugBuilder<'ctx> {
    /// Create a new debug builder
    pub fn new(
        module: &Module<'ctx>,
        context: &'ctx Context,
        filename: &str,
        directory: &str,
    ) -> Self {
        // Add debug info version flag
        let debug_metadata_version = context.i32_type().const_int(3, false);
        module.add_basic_value_flag(
            "Debug Info Version",
            inkwell::module::FlagBehavior::Warning,
            debug_metadata_version,
        );

        // Create debug info builder
        let (builder, compile_unit) = module.create_debug_info_builder(
            true,                     // allow_unresolved
            DWARFSourceLanguage::C99, // Use C99 as closest match for D
            filename,
            directory,
            "sounio", // producer
            false,       // is_optimized
            "",          // flags
            0,           // runtime_version
            "",          // split_name
            DWARFEmissionKind::Full,
            0,     // dwo_id
            false, // split_debug_inlining
            false, // debug_info_for_profiling
            "",    // sysroot
            "",    // sdk
        );

        let file = builder.create_file(filename, directory);

        let scope_stack = vec![compile_unit.as_debug_info_scope()];

        Self {
            builder,
            compile_unit,
            file,
            scope_stack,
            context,
        }
    }

    /// Create debug info builder from file path
    pub fn from_path(module: &Module<'ctx>, context: &'ctx Context, path: &Path) -> Self {
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown.sio");
        let directory = path.parent().and_then(|p| p.to_str()).unwrap_or(".");

        Self::new(module, context, filename, directory)
    }

    /// Get the compile unit
    pub fn compile_unit(&self) -> DICompileUnit<'ctx> {
        self.compile_unit
    }

    /// Get the file
    pub fn file(&self) -> DIFile<'ctx> {
        self.file
    }

    /// Get current scope
    pub fn current_scope(&self) -> DIScope<'ctx> {
        *self.scope_stack.last().unwrap()
    }

    /// Push a new scope
    pub fn push_scope(&mut self, scope: DIScope<'ctx>) {
        self.scope_stack.push(scope);
    }

    /// Pop the current scope
    pub fn pop_scope(&mut self) {
        if self.scope_stack.len() > 1 {
            self.scope_stack.pop();
        }
    }

    /// Create a function debug info
    pub fn create_function(
        &mut self,
        function: FunctionValue<'ctx>,
        name: &str,
        linkage_name: Option<&str>,
        line: u32,
        return_type: Option<DIType<'ctx>>,
        param_types: &[DIType<'ctx>],
    ) -> DISubprogram<'ctx> {
        let subroutine_type = self.builder.create_subroutine_type(
            self.file,
            return_type,
            param_types,
            DIFlags::PUBLIC,
        );

        let subprogram = self.builder.create_function(
            self.current_scope(),
            name,
            linkage_name,
            self.file,
            line,
            subroutine_type,
            true, // is_local_to_unit
            true, // is_definition
            line, // scope_line
            DIFlags::PUBLIC,
            false, // is_optimized
        );

        function.set_subprogram(subprogram);

        self.push_scope(subprogram.as_debug_info_scope());

        subprogram
    }

    /// Create a lexical block
    pub fn create_lexical_block(&mut self, line: u32, column: u32) -> DILexicalBlock<'ctx> {
        let block =
            self.builder
                .create_lexical_block(self.current_scope(), self.file, line, column);

        self.push_scope(block.as_debug_info_scope());

        block
    }

    /// Create a local variable
    pub fn create_local_variable(
        &self,
        name: &str,
        ty: DIType<'ctx>,
        line: u32,
    ) -> DILocalVariable<'ctx> {
        self.builder.create_auto_variable(
            self.current_scope(),
            name,
            self.file,
            line,
            ty,
            false, // always_preserve
            DIFlags::PUBLIC,
            8, // align_in_bits
        )
    }

    /// Create a parameter variable
    pub fn create_parameter_variable(
        &self,
        name: &str,
        ty: DIType<'ctx>,
        arg_no: u32,
        line: u32,
    ) -> DILocalVariable<'ctx> {
        self.builder.create_parameter_variable(
            self.current_scope(),
            name,
            arg_no,
            self.file,
            line,
            ty,
            false, // always_preserve
            DIFlags::PUBLIC,
        )
    }

    /// Create a debug location
    pub fn create_location(&self, line: u32, column: u32) -> DILocation<'ctx> {
        self.builder.create_debug_location(
            self.context,
            line,
            column,
            self.current_scope(),
            None, // inlined_at
        )
    }

    /// Create basic type debug info
    pub fn create_basic_type(&self, name: &str, bits: u64, encoding: u32) -> DIType<'ctx> {
        self.builder
            .create_basic_type(name, bits, encoding, DIFlags::PUBLIC)
            .unwrap()
            .as_type()
    }

    /// Create pointer type
    pub fn create_pointer_type(&self, pointee: Option<DIType<'ctx>>, name: &str) -> DIType<'ctx> {
        self.builder
            .create_pointer_type(
                name,
                pointee.unwrap(),
                64,
                64,
                inkwell::AddressSpace::default(),
            )
            .as_type()
    }

    /// Create array type
    pub fn create_array_type(&self, element: DIType<'ctx>, size: u64, align: u32) -> DIType<'ctx> {
        // create_subrange was removed in newer inkwell versions
        // Create array type with empty subscripts for now (inkwell API changed)
        // TODO: Find correct API for creating array subscripts in newer inkwell
        self.builder
            .create_array_type(element, size * 8, align, &[])
            .as_type()
    }

    /// Create struct type
    pub fn create_struct_type(
        &self,
        name: &str,
        elements: &[DIType<'ctx>],
        size_bits: u64,
        align_bits: u32,
    ) -> DIType<'ctx> {
        self.builder
            .create_struct_type(
                self.current_scope(),
                name,
                self.file,
                0, // line
                size_bits,
                align_bits,
                DIFlags::PUBLIC,
                None, // derived_from
                elements,
                0,    // runtime_lang
                None, // vtable_holder
                name, // unique_id
            )
            .as_type()
    }

    /// Create type info for HLIR type
    pub fn create_type_info(&self, ty: &HlirType) -> DIType<'ctx> {
        match ty {
            HlirType::Void => self.create_basic_type("void", 0, DW_ATE_SIGNED),
            HlirType::Bool => self.create_basic_type("bool", 8, DW_ATE_BOOLEAN),
            HlirType::I8 => self.create_basic_type("i8", 8, DW_ATE_SIGNED),
            HlirType::I16 => self.create_basic_type("i16", 16, DW_ATE_SIGNED),
            HlirType::I32 => self.create_basic_type("i32", 32, DW_ATE_SIGNED),
            HlirType::I64 => self.create_basic_type("i64", 64, DW_ATE_SIGNED),
            HlirType::I128 => self.create_basic_type("i128", 128, DW_ATE_SIGNED),
            HlirType::U8 => self.create_basic_type("u8", 8, DW_ATE_UNSIGNED),
            HlirType::U16 => self.create_basic_type("u16", 16, DW_ATE_UNSIGNED),
            HlirType::U32 => self.create_basic_type("u32", 32, DW_ATE_UNSIGNED),
            HlirType::U64 => self.create_basic_type("u64", 64, DW_ATE_UNSIGNED),
            HlirType::U128 => self.create_basic_type("u128", 128, DW_ATE_UNSIGNED),
            HlirType::F32 => self.create_basic_type("f32", 32, DW_ATE_FLOAT),
            HlirType::F64 => self.create_basic_type("f64", 64, DW_ATE_FLOAT),
            HlirType::Ptr(inner) => {
                let inner_ty = self.create_type_info(inner);
                self.create_pointer_type(Some(inner_ty), "ptr")
            }
            HlirType::Array(elem, size) => {
                let elem_ty = self.create_type_info(elem);
                self.create_array_type(elem_ty, *size as u64, 8)
            }
            HlirType::Struct(name) => {
                // Create opaque struct type
                self.create_struct_type(name, &[], 0, 8)
            }
            HlirType::Tuple(elems) => {
                let elem_types: Vec<_> = elems.iter().map(|e| self.create_type_info(e)).collect();
                let size: u64 = elems.iter().map(|e| type_size_bits(e)).sum();
                self.create_struct_type("tuple", &elem_types, size, 8)
            }
            HlirType::Function { .. } => {
                // Function types are represented as pointers
                self.create_basic_type("fn_ptr", 64, DW_ATE_ADDRESS)
            }
            // SIMD vector types
            HlirType::Vec2 => self.create_basic_type("vec2", 64, DW_ATE_FLOAT),
            HlirType::Vec3 => self.create_basic_type("vec3", 128, DW_ATE_FLOAT),
            HlirType::Vec4 => self.create_basic_type("vec4", 128, DW_ATE_FLOAT),
            HlirType::Mat2 => self.create_basic_type("mat2", 128, DW_ATE_FLOAT),
            HlirType::Mat3 => self.create_basic_type("mat3", 384, DW_ATE_FLOAT),
            HlirType::Mat4 => self.create_basic_type("mat4", 512, DW_ATE_FLOAT),
            HlirType::Quat => self.create_basic_type("quat", 128, DW_ATE_FLOAT),
            HlirType::Dual => self.create_basic_type("dual", 128, DW_ATE_FLOAT),
        }
    }

    /// Finalize debug info
    pub fn finalize(&self) {
        self.builder.finalize();
    }
}

/// DWARF attribute encodings
pub const DW_ATE_ADDRESS: u32 = 0x01;
pub const DW_ATE_BOOLEAN: u32 = 0x02;
pub const DW_ATE_FLOAT: u32 = 0x04;
pub const DW_ATE_SIGNED: u32 = 0x05;
pub const DW_ATE_SIGNED_CHAR: u32 = 0x06;
pub const DW_ATE_UNSIGNED: u32 = 0x07;
pub const DW_ATE_UNSIGNED_CHAR: u32 = 0x08;

/// Get type size in bits
fn type_size_bits(ty: &HlirType) -> u64 {
    match ty {
        HlirType::Void => 0,
        HlirType::Bool => 8,
        HlirType::I8 | HlirType::U8 => 8,
        HlirType::I16 | HlirType::U16 => 16,
        HlirType::I32 | HlirType::U32 | HlirType::F32 => 32,
        HlirType::I64 | HlirType::U64 | HlirType::F64 => 64,
        HlirType::I128 | HlirType::U128 => 128,
        HlirType::Ptr(_) | HlirType::Function { .. } => 64,
        HlirType::Array(elem, size) => type_size_bits(elem) * (*size as u64),
        HlirType::Struct(_) => 64, // Conservative estimate
        HlirType::Tuple(elems) => elems.iter().map(type_size_bits).sum(),
        // SIMD types
        HlirType::Vec2 => 64,
        HlirType::Vec3 | HlirType::Vec4 | HlirType::Quat => 128,
        HlirType::Mat2 => 128,
        HlirType::Mat3 => 384,
        HlirType::Mat4 => 512,
        HlirType::Dual => 128,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_size_bits() {
        assert_eq!(type_size_bits(&HlirType::I32), 32);
        assert_eq!(type_size_bits(&HlirType::I64), 64);
        assert_eq!(type_size_bits(&HlirType::F64), 64);
        assert_eq!(
            type_size_bits(&HlirType::Array(Box::new(HlirType::I32), 10)),
            320
        );
    }

    #[test]
    fn test_dwarf_encodings() {
        assert_eq!(DW_ATE_SIGNED, 0x05);
        assert_eq!(DW_ATE_UNSIGNED, 0x07);
        assert_eq!(DW_ATE_FLOAT, 0x04);
    }
}

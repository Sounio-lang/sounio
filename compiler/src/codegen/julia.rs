//! Julia code generator for Sounio
//!
//! Generates Julia source code from HLIR for integration with
//! DifferentialEquations.jl and Unitful.jl

use crate::hlir::{HlirModule, HlirFunction, HlirType};
use std::fmt::Write;

/// Julia code generator
pub struct JuliaCodegen {
    output: String,
    indent_level: usize,
}

impl JuliaCodegen {
    /// Create a new Julia code generator
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent_level: 0,
        }
    }

    /// Get current indentation string
    fn indent(&self) -> String {
        "    ".repeat(self.indent_level)
    }

    /// Write indented line
    fn writeln(&mut self, text: &str) {
        let _ = writeln!(self.output, "{}{}", self.indent(), text);
    }

    /// Convert Sounio type to Julia type
    fn convert_type(&self, hlir_type: &HlirType) -> String {
        match hlir_type {
            HlirType::I32 => "Int32".to_string(),
            HlirType::I64 => "Int64".to_string(),
            HlirType::F64 => "Float64".to_string(),
            HlirType::Bool => "Bool".to_string(),
            HlirType::Unit => "Nothing".to_string(),
            HlirType::String => "String".to_string(),
            _ => "Any".to_string(),
        }
    }

    /// Generate module header
    pub fn generate_header(&mut self, module_name: &str) {
        self.writeln(&format!("# {} - Generated from Sounio", module_name));
        self.writeln("# Julia code generator for PBPK modeling");
        self.writeln("");
        self.writeln("using DifferentialEquations");
        self.writeln("using Unitful");
        self.writeln("using Plots");
        self.writeln("");
    }

    /// Generate complete code
    pub fn generate(&mut self, module_name: &str) {
        self.generate_header(module_name);
        self.generate_param_struct(module_name);
        self.generate_ode_function(module_name);
        self.generate_solve_function(module_name);
        self.generate_main(module_name);
    }

    fn generate_param_struct(&mut self, name: &str) {
        self.writeln(&format!("@kwdef struct {}Params", name));
        self.indent_level += 1;
        self.writeln("v_blood::Float64 = 5.0");
        self.writeln("v_liver::Float64 = 1.8");
        self.writeln("cl_hepatic::Float64 = 30.0");
        self.writeln("cl_renal::Float64 = 2.0");
        self.writeln("fu::Float64 = 0.03");
        self.indent_level -= 1;
        self.writeln("end");
        self.writeln("");
    }

    fn generate_ode_function(&mut self, name: &str) {
        self.writeln(&format!("function {}!(du, u, p::{}Params, t)", name, name));
        self.indent_level += 1;
        self.writeln("c = u[1]");
        self.writeln("k = p.cl_hepatic / p.v_blood * p.fu");
        self.writeln("du[1] = -k * c");
        self.indent_level -= 1;
        self.writeln("end");
        self.writeln("");
    }

    fn generate_solve_function(&mut self, name: &str) {
        self.writeln(&format!("function simulate_{}(dose::Float64, tspan::Tuple)", name));
        self.indent_level += 1;
        self.writeln(&format!("p = {}Params()", name));
        self.writeln("u0 = [dose / p.v_blood]");
        self.writeln(&format!("prob = ODEProblem({}!, u0, tspan, p)", name));
        self.writeln("sol = solve(prob, Tsit5())");
        self.writeln("return sol");
        self.indent_level -= 1;
        self.writeln("end");
        self.writeln("");
    }

    fn generate_main(&mut self, name: &str) {
        self.writeln("dose = 2.0");
        self.writeln("tspan = (0.0, 2.0)");
        self.writeln(&format!("sol = simulate_{}(dose, tspan)", name));
        self.writeln("println(sol[1, end])");
    }

    /// Get generated code
    pub fn finish(self) -> String {
        self.output
    }
}

/// Generate Julia code from HLIR module
pub fn generate_julia(module_name: &str) -> Result<String, String> {
    let mut codegen = JuliaCodegen::new();
    codegen.generate(module_name);
    Ok(codegen.finish())
}

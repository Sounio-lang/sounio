//! REPL - Read-Eval-Print Loop for Sounio
//!
//! Provides an interactive shell for evaluating D expressions and statements.

use crate::hir;
use crate::interp::{Interpreter, Value};
use rustyline::error::ReadlineError;
use rustyline::{DefaultEditor, Result as RlResult};
use std::collections::HashMap;

/// REPL configuration
#[derive(Debug, Clone)]
pub struct ReplConfig {
    /// Show AST after parsing
    pub show_ast: bool,
    /// Show HIR after type checking
    pub show_hir: bool,
    /// Show types of expressions
    pub show_types: bool,
    /// Use JIT compilation instead of interpreter
    pub use_jit: bool,
    /// History file path
    pub history_file: Option<String>,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            show_ast: false,
            show_hir: false,
            show_types: true,
            use_jit: false,
            history_file: Some(".sounio_history".to_string()),
        }
    }
}

/// The REPL state
pub struct Repl {
    config: ReplConfig,
    /// Accumulated function definitions
    functions: HashMap<String, String>,
    /// Accumulated type definitions
    types: HashMap<String, String>,
    /// Variable bindings from previous expressions
    bindings: HashMap<String, Value>,
    /// Line counter
    line_count: usize,
}

impl Repl {
    pub fn new(config: ReplConfig) -> Self {
        Self {
            config,
            functions: HashMap::new(),
            types: HashMap::new(),
            bindings: HashMap::new(),
            line_count: 0,
        }
    }

    /// Run the REPL
    pub fn run(&mut self) -> RlResult<()> {
        let mut rl = DefaultEditor::new()?;

        // Load history
        if let Some(ref hist_file) = self.config.history_file {
            let _ = rl.load_history(hist_file);
        }

        println!("Sounio REPL v0.7.0");
        println!("Type :help for help, :quit to exit\n");

        loop {
            let prompt = format!("d[{}]> ", self.line_count);

            match rl.readline(&prompt) {
                Ok(line) => {
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    let _ = rl.add_history_entry(line);

                    // Handle commands
                    if line.starts_with(':') {
                        if self.handle_command(line) {
                            break;
                        }
                        continue;
                    }

                    // Handle multi-line input for function/type definitions
                    let input = if line.starts_with("fn ")
                        || line.starts_with("struct ")
                        || line.starts_with("enum ")
                        || line.starts_with("effect ")
                    {
                        self.read_multiline(&mut rl, line)?
                    } else {
                        line.to_string()
                    };

                    // Evaluate the input
                    self.eval_input(&input);
                    self.line_count += 1;
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!("Goodbye!");
                    break;
                }
                Err(err) => {
                    eprintln!("Error: {:?}", err);
                    break;
                }
            }
        }

        // Save history
        if let Some(ref hist_file) = self.config.history_file {
            let _ = rl.save_history(hist_file);
        }

        Ok(())
    }

    fn read_multiline(&self, rl: &mut DefaultEditor, first_line: &str) -> RlResult<String> {
        let mut lines = vec![first_line.to_string()];
        let mut brace_count =
            first_line.matches('{').count() as i32 - first_line.matches('}').count() as i32;

        while brace_count > 0 {
            match rl.readline("... ") {
                Ok(line) => {
                    brace_count += line.matches('{').count() as i32;
                    brace_count -= line.matches('}').count() as i32;
                    lines.push(line);
                }
                Err(_) => break,
            }
        }

        Ok(lines.join("\n"))
    }

    fn handle_command(&mut self, cmd: &str) -> bool {
        let parts: Vec<&str> = cmd.split_whitespace().collect();

        match parts.first().copied() {
            Some(":quit") | Some(":q") | Some(":exit") => {
                println!("Goodbye!");
                return true;
            }
            Some(":help") | Some(":h") | Some(":?") => {
                self.print_help();
            }
            Some(":clear") => {
                self.functions.clear();
                self.types.clear();
                self.bindings.clear();
                println!("Cleared all definitions and bindings.");
            }
            Some(":ast") => {
                self.config.show_ast = !self.config.show_ast;
                println!("Show AST: {}", self.config.show_ast);
            }
            Some(":hir") => {
                self.config.show_hir = !self.config.show_hir;
                println!("Show HIR: {}", self.config.show_hir);
            }
            Some(":types") => {
                self.config.show_types = !self.config.show_types;
                println!("Show types: {}", self.config.show_types);
            }
            Some(":jit") => {
                self.config.use_jit = !self.config.use_jit;
                println!("Use JIT: {}", self.config.use_jit);
            }
            Some(":env") => {
                self.print_environment();
            }
            Some(":funcs") => {
                self.print_functions();
            }
            Some(":load") if parts.len() > 1 => {
                self.load_file(parts[1]);
            }
            Some(":type") if parts.len() > 1 => {
                let expr = parts[1..].join(" ");
                self.show_type(&expr);
            }
            Some(cmd) => {
                println!("Unknown command: {}", cmd);
                println!("Type :help for available commands.");
            }
            None => {}
        }

        false
    }

    fn print_help(&self) {
        println!("Sounio REPL Commands:");
        println!("  :help, :h, :?    Show this help");
        println!("  :quit, :q        Exit the REPL");
        println!("  :clear           Clear all definitions");
        println!("  :env             Show current bindings");
        println!("  :funcs           Show defined functions");
        println!("  :ast             Toggle AST display");
        println!("  :hir             Toggle HIR display");
        println!("  :types           Toggle type display");
        println!("  :jit             Toggle JIT compilation");
        println!("  :type <expr>     Show type of expression");
        println!("  :load <file>     Load a D source file");
        println!();
        println!("Examples:");
        println!("  1 + 2 * 3        Evaluate an expression");
        println!("  let x = 42       Bind a variable");
        println!("  fn add(a: i64, b: i64) -> i64 {{ a + b }}");
        println!("  add(1, 2)        Call a function");
    }

    fn print_environment(&self) {
        if self.bindings.is_empty() {
            println!("No bindings.");
        } else {
            println!("Current bindings:");
            for (name, value) in &self.bindings {
                println!("  {} = {:?}", name, value);
            }
        }
    }

    fn print_functions(&self) {
        if self.functions.is_empty() {
            println!("No functions defined.");
        } else {
            println!("Defined functions:");
            for name in self.functions.keys() {
                println!("  {}", name);
            }
        }
    }

    fn load_file(&mut self, path: &str) {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                println!("Loading {}...", path);
                self.eval_input(&content);
            }
            Err(e) => {
                eprintln!("Failed to load {}: {}", path, e);
            }
        }
    }

    fn show_type(&mut self, expr: &str) {
        // Build full source with definitions
        let source = self.build_source(expr);

        // Parse using the library functions
        match crate::lexer::lex(&source) {
            Ok(tokens) => {
                match crate::parser::parse(&tokens, &source) {
                    Ok(ast) => {
                        // Type check
                        match crate::check::check(&ast) {
                            Ok(hir) => {
                                // Find the type of the last expression
                                if let Some(item) = hir.items.last()
                                    && let hir::HirItem::Function(f) = item
                                {
                                    println!("Type: {:?}", f.body.ty);
                                }
                            }
                            Err(e) => {
                                eprintln!("Type error: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Parse error: {:?}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Lex error: {:?}", e);
            }
        }
    }

    fn eval_input(&mut self, input: &str) {
        // Check if this is a definition
        if input.starts_with("fn ")
            && let Some(name) = self.extract_fn_name(input)
        {
            self.functions.insert(name.clone(), input.to_string());
            println!("Defined function: {}", name);
            return;
        }

        if input.starts_with("struct ")
            && let Some(name) = self.extract_type_name(input)
        {
            self.types.insert(name.clone(), input.to_string());
            println!("Defined struct: {}", name);
            return;
        }

        if input.starts_with("enum ")
            && let Some(name) = self.extract_type_name(input)
        {
            self.types.insert(name.clone(), input.to_string());
            println!("Defined enum: {}", name);
            return;
        }

        // Check if this is a let binding
        let is_let = input.starts_with("let ");

        // Wrap expression in a main function for evaluation
        let wrapped = if is_let {
            // let x = expr -> we need to capture the value
            format!("{}\n__repl_result__", input)
        } else {
            input.to_string()
        };

        let source = self.build_source(&wrapped);

        if self.config.show_ast {
            println!("--- Source ---\n{}\n", source);
        }

        // Parse using library functions
        let tokens = match crate::lexer::lex(&source) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Lex error: {:?}", e);
                return;
            }
        };

        let ast = match crate::parser::parse(&tokens, &source) {
            Ok(ast) => ast,
            Err(e) => {
                eprintln!("Parse error: {:?}", e);
                return;
            }
        };

        if self.config.show_ast {
            println!("--- AST ---\n{:#?}\n", ast);
        }

        // Type check
        let hir = match crate::check::check(&ast) {
            Ok(hir) => hir,
            Err(e) => {
                eprintln!("Type error: {}", e);
                return;
            }
        };

        if self.config.show_hir {
            println!("--- HIR ---\n{:#?}\n", hir);
        }

        // Execute
        if self.config.use_jit {
            self.eval_jit(&hir);
        } else {
            self.eval_interp(&hir, is_let, input);
        }
    }

    fn eval_interp(&mut self, hir: &hir::Hir, is_let: bool, input: &str) {
        let mut interp = Interpreter::new();

        // Pre-populate environment with existing bindings
        for (name, value) in &self.bindings {
            interp.env_mut().define(name.clone(), value.clone());
        }

        match interp.run(hir) {
            Ok(value) => {
                if is_let {
                    // Extract binding name and store value
                    if let Some(name) = self.extract_let_name(input) {
                        println!("{} = {:?}", name, value);
                        self.bindings.insert(name, value);
                    }
                } else if value != Value::Unit {
                    if self.config.show_types {
                        println!("=> {:?}", value);
                    } else {
                        println!("{:?}", value);
                    }
                }
            }
            Err(e) => {
                eprintln!("Runtime error: {}", e);
            }
        }
    }

    fn eval_jit(&self, hir: &hir::Hir) {
        #[cfg(feature = "jit")]
        {
            use crate::codegen::cranelift::CraneliftJit;
            use crate::hlir;

            // Lower to HLIR
            let hlir_module = hlir::lower(hir);

            // Compile and run
            let jit = CraneliftJit::new();
            match jit.compile_and_run(&hlir_module) {
                Ok(result) => {
                    println!("=> {}", result);
                }
                Err(e) => {
                    eprintln!("JIT error: {}", e);
                }
            }
        }

        #[cfg(not(feature = "jit"))]
        {
            let _ = hir;
            eprintln!("JIT not enabled. Compile with --features jit");
        }
    }

    fn build_source(&self, expr: &str) -> String {
        let mut source = String::new();

        // Add type definitions
        for def in self.types.values() {
            source.push_str(def);
            source.push('\n');
        }

        // Add function definitions
        for def in self.functions.values() {
            source.push_str(def);
            source.push('\n');
        }

        // Wrap expression in main function
        source.push_str(&format!("fn main() -> i64 {{\n    {}\n}}\n", expr));

        source
    }

    fn extract_fn_name(&self, input: &str) -> Option<String> {
        // fn name(...) -> ...
        let input = input.strip_prefix("fn ")?.trim_start();
        let end = input.find('(')?;
        Some(input[..end].trim().to_string())
    }

    fn extract_type_name(&self, input: &str) -> Option<String> {
        // struct/enum name { ... }
        let input = if input.starts_with("struct ") {
            input.strip_prefix("struct ")?
        } else if input.starts_with("enum ") {
            input.strip_prefix("enum ")?
        } else {
            return None;
        };
        let input = input.trim_start();
        let end = input.find(|c: char| c == '{' || c == '<' || c.is_whitespace())?;
        Some(input[..end].trim().to_string())
    }

    fn extract_let_name(&self, input: &str) -> Option<String> {
        // let name = ...
        let input = input.strip_prefix("let ")?.trim_start();
        let end = input.find(|c: char| c == '=' || c == ':' || c.is_whitespace())?;
        Some(input[..end].trim().to_string())
    }
}

/// Run the REPL with default configuration
pub fn run() -> RlResult<()> {
    let mut repl = Repl::new(ReplConfig::default());
    repl.run()
}

/// Run the REPL with custom configuration
pub fn run_with_config(config: ReplConfig) -> RlResult<()> {
    let mut repl = Repl::new(config);
    repl.run()
}

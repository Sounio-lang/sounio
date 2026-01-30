//! Error Codes Registry for the Sounio Compiler
//!
//! This module defines all error codes used by the compiler along with
//! their documentation, examples, and explanations.

use std::collections::HashMap;

/// An error code with documentation
#[derive(Debug, Clone)]
pub struct ErrorCode {
    /// The error code (e.g., "E0001")
    pub code: &'static str,
    /// Short description
    pub title: &'static str,
    /// Detailed explanation
    pub explanation: &'static str,
    /// Example code that triggers this error
    pub example: Option<&'static str>,
    /// Category of the error
    pub category: ErrorCategory,
}

/// Error categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Lexer/tokenization errors
    Lexer,
    /// Parser/syntax errors
    Parser,
    /// Name resolution errors
    Resolve,
    /// Type checking errors
    Type,
    /// Effect system errors
    Effect,
    /// Ownership/linearity errors
    Ownership,
    /// Pattern matching errors
    Pattern,
    /// Macro errors
    Macro,
    /// Import/module errors
    Module,
    /// Code generation errors
    Codegen,
    /// Internal compiler errors
    Internal,
}

impl ErrorCategory {
    /// Get the category prefix for error codes
    pub fn prefix(&self) -> &'static str {
        match self {
            ErrorCategory::Lexer => "L",
            ErrorCategory::Parser => "P",
            ErrorCategory::Resolve => "R",
            ErrorCategory::Type => "T",
            ErrorCategory::Effect => "F",
            ErrorCategory::Ownership => "O",
            ErrorCategory::Pattern => "M",
            ErrorCategory::Macro => "X",
            ErrorCategory::Module => "I",
            ErrorCategory::Codegen => "C",
            ErrorCategory::Internal => "E",
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            ErrorCategory::Lexer => "Lexer Error",
            ErrorCategory::Parser => "Syntax Error",
            ErrorCategory::Resolve => "Name Resolution Error",
            ErrorCategory::Type => "Type Error",
            ErrorCategory::Effect => "Effect Error",
            ErrorCategory::Ownership => "Ownership Error",
            ErrorCategory::Pattern => "Pattern Error",
            ErrorCategory::Macro => "Macro Error",
            ErrorCategory::Module => "Import Error",
            ErrorCategory::Codegen => "Code Generation Error",
            ErrorCategory::Internal => "Internal Error",
        }
    }
}

/// Error index containing all registered error codes
pub struct ErrorIndex {
    codes: HashMap<&'static str, ErrorCode>,
}

impl ErrorIndex {
    /// Create a new error index with all registered codes
    pub fn new() -> Self {
        let mut index = ErrorIndex {
            codes: HashMap::new(),
        };
        index.register_all();
        index
    }

    /// Look up an error code
    pub fn get(&self, code: &str) -> Option<&ErrorCode> {
        self.codes.get(code)
    }

    /// Get all error codes
    pub fn all(&self) -> impl Iterator<Item = &ErrorCode> {
        self.codes.values()
    }

    /// Get codes by category
    pub fn by_category(&self, category: ErrorCategory) -> Vec<&ErrorCode> {
        self.codes
            .values()
            .filter(|c| c.category == category)
            .collect()
    }

    /// Generate markdown documentation for all error codes
    pub fn generate_docs(&self) -> String {
        let mut doc = String::new();
        doc.push_str("# Sounio Compiler Error Index\n\n");
        doc.push_str("This document lists all error codes produced by the Sounio compiler.\n\n");

        // Group by category
        let categories = [
            ErrorCategory::Lexer,
            ErrorCategory::Parser,
            ErrorCategory::Resolve,
            ErrorCategory::Type,
            ErrorCategory::Effect,
            ErrorCategory::Ownership,
            ErrorCategory::Pattern,
            ErrorCategory::Macro,
            ErrorCategory::Module,
            ErrorCategory::Codegen,
            ErrorCategory::Internal,
        ];

        for category in categories {
            let codes = self.by_category(category);
            if codes.is_empty() {
                continue;
            }

            doc.push_str(&format!("## {}\n\n", category.name()));

            for code in codes {
                doc.push_str(&format!("### {}: {}\n\n", code.code, code.title));
                doc.push_str(&format!("{}\n\n", code.explanation));

                if let Some(example) = code.example {
                    doc.push_str("**Example:**\n\n```d\n");
                    doc.push_str(example);
                    doc.push_str("\n```\n\n");
                }
            }
        }

        doc
    }

    /// Register all error codes
    fn register_all(&mut self) {
        // Lexer errors (L0xxx)
        self.register(ErrorCode {
            code: "L0001",
            title: "Invalid character",
            explanation: "The source file contains a character that is not valid in Sounio source code.",
            example: Some("let x = @invalid;  // '@' is not a valid character"),
            category: ErrorCategory::Lexer,
        });

        self.register(ErrorCode {
            code: "L0002",
            title: "Unterminated string literal",
            explanation: "A string literal was started but never closed with a matching quote.",
            example: Some("let s = \"hello;  // missing closing quote"),
            category: ErrorCategory::Lexer,
        });

        self.register(ErrorCode {
            code: "L0003",
            title: "Invalid number literal",
            explanation: "A number literal has an invalid format.",
            example: Some("let x = 0x;      // hex literal with no digits\nlet y = 1.2.3;   // multiple decimal points"),
            category: ErrorCategory::Lexer,
        });

        self.register(ErrorCode {
            code: "L0004",
            title: "Unterminated block comment",
            explanation: "A block comment /* was started but never closed with */.",
            example: Some("/* This comment\n   never ends"),
            category: ErrorCategory::Lexer,
        });

        // Parser errors (P0xxx)
        self.register(ErrorCode {
            code: "P0001",
            title: "Unexpected token",
            explanation: "The parser encountered a token that was not expected at this position.",
            example: Some("fn foo( {  // expected parameter or ')', found '{'"),
            category: ErrorCategory::Parser,
        });

        self.register(ErrorCode {
            code: "P0002",
            title: "Expected expression",
            explanation: "An expression was expected but not found.",
            example: Some("let x = ;  // missing expression after '='"),
            category: ErrorCategory::Parser,
        });

        self.register(ErrorCode {
            code: "P0003",
            title: "Expected type annotation",
            explanation: "A type annotation was expected but not found.",
            example: Some("fn foo(x) {}  // parameter 'x' needs a type annotation"),
            category: ErrorCategory::Parser,
        });

        self.register(ErrorCode {
            code: "P0004",
            title: "Missing semicolon",
            explanation: "A semicolon was expected to terminate a statement.",
            example: Some("let x = 1\nlet y = 2  // missing ';' after first statement"),
            category: ErrorCategory::Parser,
        });

        self.register(ErrorCode {
            code: "P0005",
            title: "Mismatched brackets",
            explanation: "Opening and closing brackets do not match.",
            example: Some("let arr = [1, 2, 3);  // '[' closed with ')'"),
            category: ErrorCategory::Parser,
        });

        self.register(ErrorCode {
            code: "P0006",
            title: "Invalid pattern",
            explanation: "The pattern syntax is invalid.",
            example: Some("match x {\n    1 + 2 => {}  // patterns cannot contain operators\n}"),
            category: ErrorCategory::Parser,
        });

        // Parser errors for unimplemented/unsupported features (P001x)
        self.register(ErrorCode {
            code: "P0010",
            title: "Refinement type syntax not implemented",
            explanation: "Refinement types using the syntax `{ x: Type | constraint }` are planned but not yet available in Sounio. Refinement types allow constraining values beyond their base type, such as `{ x: i32 | x > 0 }` for positive integers.",
            example: Some("type Positive = { x: i32 | x > 0 };  // not yet implemented\n// Use a type alias with runtime validation instead:\ntype Positive = i32;"),
            category: ErrorCategory::Parser,
        });

        self.register(ErrorCode {
            code: "P0011",
            title: "Rust-style mutable reference",
            explanation: "Sounio uses `&!T` for mutable (exclusive) references, not Rust's `&mut T` syntax. This is a deliberate design choice to emphasize the exclusive nature of mutable borrows.",
            example: Some("fn modify(x: &mut i32) {}  // WRONG: Rust syntax\nfn modify(x: &!i32) {}     // CORRECT: Sounio syntax"),
            category: ErrorCategory::Parser,
        });

        self.register(ErrorCode {
            code: "P0012",
            title: "Tuple destructuring not implemented",
            explanation: "Destructuring tuples directly in let bindings or patterns is not yet implemented. Access tuple elements using `.0`, `.1`, etc.",
            example: Some("let (a, b) = pair;  // not yet implemented\n// Use instead:\nlet a = pair.0;\nlet b = pair.1;"),
            category: ErrorCategory::Parser,
        });

        self.register(ErrorCode {
            code: "P0013",
            title: "Rust macro syntax not supported",
            explanation: "Sounio does not support Rust-style macro invocations with `!`. Functions like print, assert, etc. are regular functions in Sounio.",
            example: Some("println!(\"hello\");  // WRONG: Rust macro syntax\nprint(\"hello\");      // CORRECT: Sounio function (with IO effect)"),
            category: ErrorCategory::Parser,
        });

        self.register(ErrorCode {
            code: "P0014",
            title: "Rust attribute not supported",
            explanation: "Rust-specific attributes like `#[derive(...)]`, `#[test]`, and `#[cfg(...)]` are not available in Sounio. Sounio has its own attribute system.",
            example: Some("#[derive(Debug)]  // not supported\nstruct Point { x: i32, y: i32 }"),
            category: ErrorCategory::Parser,
        });

        self.register(ErrorCode {
            code: "P0015",
            title: "Closure tuple destructuring not supported",
            explanation: "Tuple destructuring in closure parameters is not supported. Use a single parameter and access tuple elements explicitly.",
            example: Some("|(a, b)| a + b  // not supported\n|pair| pair.0 + pair.1  // use instead"),
            category: ErrorCategory::Parser,
        });

        self.register(ErrorCode {
            code: "P0020",
            title: "Feature not implemented",
            explanation: "This language feature is planned but not yet available in the current version of Sounio.",
            example: None,
            category: ErrorCategory::Parser,
        });

        self.register(ErrorCode {
            code: "P0021",
            title: "Invalid module-level item",
            explanation: "This syntax is not valid at the module level. Only declarations (fn, struct, enum, type, etc.) can appear at the top level.",
            example: Some("// At module level:\nlet x = 5;  // WRONG: statements not allowed\nfn main() { let x = 5; }  // CORRECT: inside a function"),
            category: ErrorCategory::Parser,
        });

        // Name resolution errors (R0xxx)
        self.register(ErrorCode {
            code: "R0001",
            title: "Undefined variable",
            explanation: "The variable has not been declared in this scope or any enclosing scope.",
            example: Some("fn foo() {\n    println(x);  // 'x' is not defined\n}"),
            category: ErrorCategory::Resolve,
        });

        self.register(ErrorCode {
            code: "R0002",
            title: "Undefined type",
            explanation: "The type name does not refer to any known type.",
            example: Some("fn foo(x: Undefined) {}  // type 'Undefined' does not exist"),
            category: ErrorCategory::Resolve,
        });

        self.register(ErrorCode {
            code: "R0003",
            title: "Undefined function",
            explanation: "No function with this name exists in scope.",
            example: Some("fn main() {\n    unknown_function();  // function not found\n}"),
            category: ErrorCategory::Resolve,
        });

        self.register(ErrorCode {
            code: "R0004",
            title: "Duplicate definition",
            explanation: "An item with this name has already been defined in this scope.",
            example: Some("let x = 1;\nlet x = 2;  // 'x' already defined"),
            category: ErrorCategory::Resolve,
        });

        self.register(ErrorCode {
            code: "R0005",
            title: "Import not found",
            explanation: "The specified module or item could not be found.",
            example: Some("use nonexistent::module;  // module does not exist"),
            category: ErrorCategory::Resolve,
        });

        self.register(ErrorCode {
            code: "R0006",
            title: "Private item",
            explanation: "The item exists but is not accessible from this location.",
            example: Some("use other_module::private_fn;  // 'private_fn' is not public"),
            category: ErrorCategory::Resolve,
        });

        // Type errors (T0xxx)
        self.register(ErrorCode {
            code: "T0001",
            title: "Type mismatch",
            explanation: "The expected type does not match the actual type of the expression.",
            example: Some(
                "fn foo() -> int {\n    return true;  // expected 'int', found 'bool'\n}",
            ),
            category: ErrorCategory::Type,
        });

        self.register(ErrorCode {
            code: "T0002",
            title: "Cannot infer type",
            explanation: "The type of this expression cannot be determined. Add a type annotation.",
            example: Some("let x = [];  // cannot infer element type of empty array"),
            category: ErrorCategory::Type,
        });

        self.register(ErrorCode {
            code: "T0003",
            title: "Invalid binary operation",
            explanation: "The binary operator cannot be applied to these types.",
            example: Some("let x = \"hello\" - 1;  // cannot subtract int from string"),
            category: ErrorCategory::Type,
        });

        self.register(ErrorCode {
            code: "T0004",
            title: "Invalid unary operation",
            explanation: "The unary operator cannot be applied to this type.",
            example: Some("let x = -\"hello\";  // cannot negate a string"),
            category: ErrorCategory::Type,
        });

        self.register(ErrorCode {
            code: "T0005",
            title: "Not callable",
            explanation: "This expression cannot be called as a function.",
            example: Some("let x = 5;\nx();  // 'int' is not callable"),
            category: ErrorCategory::Type,
        });

        self.register(ErrorCode {
            code: "T0006",
            title: "Wrong number of arguments",
            explanation: "The function was called with the wrong number of arguments.",
            example: Some("fn foo(a: int, b: int) {}\nfoo(1);  // expected 2 arguments, got 1"),
            category: ErrorCategory::Type,
        });

        self.register(ErrorCode {
            code: "T0007",
            title: "Not indexable",
            explanation: "This type does not support indexing.",
            example: Some("let x = 5;\nlet y = x[0];  // 'int' cannot be indexed"),
            category: ErrorCategory::Type,
        });

        self.register(ErrorCode {
            code: "T0008",
            title: "Field not found",
            explanation: "The struct or type does not have a field with this name.",
            example: Some("struct Point { x: int, y: int }\nlet p = Point { x: 0, y: 0 };\nlet z = p.z;  // 'Point' has no field 'z'"),
            category: ErrorCategory::Type,
        });

        self.register(ErrorCode {
            code: "T0009",
            title: "Method not found",
            explanation: "No method with this name exists for this type.",
            example: Some("let x = 5;\nx.unknown();  // 'int' has no method 'unknown'"),
            category: ErrorCategory::Type,
        });

        self.register(ErrorCode {
            code: "T0010",
            title: "Infinite type",
            explanation: "Type inference resulted in an infinite type, which is not allowed.",
            example: Some("fn foo(x) { foo(foo) }  // type of 'foo' would be infinite"),
            category: ErrorCategory::Type,
        });

        self.register(ErrorCode {
            code: "T0011",
            title: "Unit constraint violated",
            explanation: "A units-of-measure constraint was violated.",
            example: Some("let distance: f64<m> = 5.0<kg>;  // expected meters, got kilograms"),
            category: ErrorCategory::Type,
        });

        self.register(ErrorCode {
            code: "T0012",
            title: "Refinement type violation",
            explanation: "The value does not satisfy the refinement type's predicate.",
            example: Some("type Positive = int where x => x > 0;\nlet x: Positive = -5;  // -5 does not satisfy x > 0"),
            category: ErrorCategory::Type,
        });

        // Effect errors (F0xxx)
        self.register(ErrorCode {
            code: "F0001",
            title: "Unhandled effect",
            explanation: "The function performs an effect that is not declared in its signature.",
            example: Some("fn foo() {  // missing 'with IO'\n    println(\"hello\");\n}"),
            category: ErrorCategory::Effect,
        });

        self.register(ErrorCode {
            code: "F0002",
            title: "Effect not available",
            explanation: "The required effect is not available in the current context.",
            example: Some(
                "fn pure_fn() {\n    io_fn();  // cannot call IO function from pure context\n}",
            ),
            category: ErrorCategory::Effect,
        });

        self.register(ErrorCode {
            code: "F0003",
            title: "Effect handler not found",
            explanation: "No handler for this effect was found in scope.",
            example: Some("fn main() {\n    perform MyEffect;  // no handler for 'MyEffect'\n}"),
            category: ErrorCategory::Effect,
        });

        self.register(ErrorCode {
            code: "F0004",
            title: "Invalid effect handler",
            explanation: "The effect handler is not valid for this effect.",
            example: Some("handle IO with {  // incorrect handler signature\n    read => {}\n}"),
            category: ErrorCategory::Effect,
        });

        // Ownership errors (O0xxx)
        self.register(ErrorCode {
            code: "O0001",
            title: "Use of moved value",
            explanation: "The value has been moved and can no longer be used.",
            example: Some("let x = vec![1, 2, 3];\nlet y = x;  // x moved here\nprintln(x);  // error: x has been moved"),
            category: ErrorCategory::Ownership,
        });

        self.register(ErrorCode {
            code: "O0002",
            title: "Cannot borrow as mutable",
            explanation: "The value cannot be borrowed mutably because it is not declared as mutable.",
            example: Some("let x = 5;\nincrement(&!x);  // cannot borrow 'x' as mutable"),
            category: ErrorCategory::Ownership,
        });

        self.register(ErrorCode {
            code: "O0003",
            title: "Cannot borrow while already borrowed",
            explanation: "The value is already borrowed and cannot be borrowed again in this way.",
            example: Some(
                "let r1 = &x;\nlet r2 = &!x;  // cannot borrow mutably while immutably borrowed",
            ),
            category: ErrorCategory::Ownership,
        });

        self.register(ErrorCode {
            code: "O0004",
            title: "Linear value not used",
            explanation: "A linear value must be used exactly once, but it was not used.",
            example: Some("fn foo() {\n    let handle: linear FileHandle = open(\"file.txt\");\n}  // error: 'handle' must be used"),
            category: ErrorCategory::Ownership,
        });

        self.register(ErrorCode {
            code: "O0005",
            title: "Linear value used multiple times",
            explanation: "A linear value can only be used once, but it was used multiple times.",
            example: Some("fn foo(x: linear Resource) {\n    use(x);\n    use(x);  // error: 'x' already used\n}"),
            category: ErrorCategory::Ownership,
        });

        self.register(ErrorCode {
            code: "O0006",
            title: "Reference outlives value",
            explanation: "The reference would outlive the value it refers to.",
            example: Some("fn foo() -> &int {\n    let x = 5;\n    return &x;  // 'x' does not live long enough\n}"),
            category: ErrorCategory::Ownership,
        });

        self.register(ErrorCode {
            code: "O0007",
            title: "Cannot copy linear type",
            explanation: "Linear types cannot be implicitly copied.",
            example: Some("linear struct Unique { value: int }\nlet a = Unique { value: 1 };\nlet b = a;  // move, not copy\nlet c = a;  // error: 'a' already moved"),
            category: ErrorCategory::Ownership,
        });

        // Pattern errors (M0xxx)
        self.register(ErrorCode {
            code: "M0001",
            title: "Non-exhaustive patterns",
            explanation: "The match expression does not cover all possible cases.",
            example: Some("enum Color { Red, Green, Blue }\nmatch color {\n    Color::Red => {}\n    Color::Green => {}\n    // missing Color::Blue\n}"),
            category: ErrorCategory::Pattern,
        });

        self.register(ErrorCode {
            code: "M0002",
            title: "Unreachable pattern",
            explanation: "This pattern will never be matched because previous patterns cover all cases.",
            example: Some("match x {\n    _ => {}\n    1 => {}  // unreachable: '_' matches everything\n}"),
            category: ErrorCategory::Pattern,
        });

        self.register(ErrorCode {
            code: "M0003",
            title: "Invalid pattern for type",
            explanation: "This pattern cannot be used with this type.",
            example: Some(
                "let x: int = 5;\nmatch x {\n    Some(n) => {}  // 'int' is not an Option\n}",
            ),
            category: ErrorCategory::Pattern,
        });

        // Import errors (I0xxx)
        self.register(ErrorCode {
            code: "I0001",
            title: "Circular import",
            explanation: "There is a circular dependency between modules.",
            example: Some("// a.d\nuse b;\n// b.d\nuse a;  // circular dependency"),
            category: ErrorCategory::Module,
        });

        self.register(ErrorCode {
            code: "I0002",
            title: "Module not found",
            explanation: "The specified module could not be found.",
            example: Some("use nonexistent_module;"),
            category: ErrorCategory::Module,
        });

        self.register(ErrorCode {
            code: "I0003",
            title: "Ambiguous import",
            explanation: "The import is ambiguous because multiple items match.",
            example: Some(
                "use module_a::*;\nuse module_b::*;\nfoo();  // 'foo' exists in both modules",
            ),
            category: ErrorCategory::Module,
        });

        // Codegen errors (C0xxx)
        self.register(ErrorCode {
            code: "C0001",
            title: "FFI type not supported",
            explanation: "This type cannot be used in FFI declarations.",
            example: Some("extern \"C\" {\n    fn foo(x: String);  // 'String' is not FFI-safe\n}"),
            category: ErrorCategory::Codegen,
        });

        self.register(ErrorCode {
            code: "C0002",
            title: "Invalid inline assembly",
            explanation: "The inline assembly syntax or constraints are invalid.",
            example: Some("asm!(\"invalid instruction\");"),
            category: ErrorCategory::Codegen,
        });

        // Internal errors (E0xxx)
        self.register(ErrorCode {
            code: "E0001",
            title: "Internal compiler error",
            explanation: "An unexpected internal error occurred. This is a bug in the compiler.",
            example: None,
            category: ErrorCategory::Internal,
        });
    }

    /// Register an error code
    fn register(&mut self, code: ErrorCode) {
        self.codes.insert(code.code, code);
    }
}

impl Default for ErrorIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Look up error explanation by code
pub fn explain_error(code: &str) -> Option<String> {
    let index = ErrorIndex::new();
    index.get(code).map(|e| {
        let mut explanation = format!("# {} - {}\n\n", e.code, e.title);
        explanation.push_str(&format!("**Category:** {}\n\n", e.category.name()));
        explanation.push_str(&format!("## Explanation\n\n{}\n", e.explanation));

        if let Some(example) = e.example {
            explanation.push_str("\n## Example\n\n```d\n");
            explanation.push_str(example);
            explanation.push_str("\n```\n");
        }

        explanation
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_index() {
        let index = ErrorIndex::new();

        // Check that common errors exist
        assert!(index.get("T0001").is_some());
        assert!(index.get("R0001").is_some());
        assert!(index.get("O0001").is_some());

        // Check error categories
        assert_eq!(index.get("T0001").unwrap().category, ErrorCategory::Type);
        assert_eq!(
            index.get("O0001").unwrap().category,
            ErrorCategory::Ownership
        );
    }

    #[test]
    fn test_explain_error() {
        let explanation = explain_error("T0001");
        assert!(explanation.is_some());
        assert!(explanation.unwrap().contains("Type mismatch"));
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(ErrorCategory::Type.prefix(), "T");
        assert_eq!(ErrorCategory::Ownership.prefix(), "O");
        assert_eq!(ErrorCategory::Effect.prefix(), "F");
    }
}

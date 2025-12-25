//! Code completion provider
//!
//! Provides intelligent code completion suggestions.

use std::collections::HashMap;
use tower_lsp::lsp_types::*;

use super::analysis::AnalysisResult;

/// Provider for code completion
pub struct CompletionProvider;

impl CompletionProvider {
    /// Create a new completion provider
    pub fn new() -> Self {
        Self
    }

    /// Get completions at a position
    pub fn complete(
        &self,
        source: &str,
        offset: usize,
        context: &str,
        cache: &HashMap<Url, AnalysisResult>,
    ) -> Vec<CompletionItem> {
        let mut items = Vec::new();

        // Determine completion context
        let ctx = self.analyze_context(source, offset, context);

        match ctx {
            CompletionContext::TopLevel => {
                items.extend(self.top_level_completions());
            }
            CompletionContext::Type => {
                items.extend(self.type_completions(cache));
            }
            CompletionContext::Expression => {
                items.extend(self.expression_completions(cache));
            }
            CompletionContext::Effect => {
                items.extend(self.effect_completions());
            }
            CompletionContext::Unit => {
                items.extend(self.unit_completions());
            }
            CompletionContext::Member(_base_type) => {
                // TODO: Look up type and provide field/method completions
                items.extend(self.common_methods());
            }
            CompletionContext::Pattern => {
                items.extend(self.pattern_completions(cache));
            }
        }

        items
    }

    /// Analyze the completion context
    fn analyze_context(&self, source: &str, offset: usize, context: &str) -> CompletionContext {
        let before = if offset <= source.len() {
            &source[..offset]
        } else {
            source
        };

        // Check for member access
        if context.ends_with('.') {
            return CompletionContext::Member("unknown".to_string());
        }

        // Check for type position (after colon)
        if context.contains(": ") && !context.contains(" = ") {
            let last_colon = context.rfind(": ");
            if let Some(pos) = last_colon {
                let after_colon = &context[pos + 2..];
                if !after_colon.contains(' ') {
                    return CompletionContext::Type;
                }
            }
        }

        // Check for effect position
        if context.ends_with("with ") || (context.contains("with ") && context.ends_with(", ")) {
            return CompletionContext::Effect;
        }

        // Check for unit position (after underscore following number)
        if context.ends_with('_') {
            let trimmed = context.trim_end_matches('_');
            if trimmed
                .chars()
                .last()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
            {
                return CompletionContext::Unit;
            }
        }

        // Check for pattern position (after match, if let, etc.)
        if context.contains("match ") || context.contains("=> ") {
            return CompletionContext::Pattern;
        }

        // Check for top-level
        let trimmed = before.trim();
        if trimmed.is_empty()
            || trimmed.ends_with('}')
            || trimmed.ends_with(';')
            || before
                .lines()
                .last()
                .map(|l| l.trim().is_empty())
                .unwrap_or(true)
        {
            // Check if we're inside a function or at module level
            let brace_depth =
                before.matches('{').count() as i32 - before.matches('}').count() as i32;
            if brace_depth <= 0 {
                return CompletionContext::TopLevel;
            }
        }

        CompletionContext::Expression
    }

    /// Top-level completions (fn, struct, enum, etc.)
    fn top_level_completions(&self) -> Vec<CompletionItem> {
        vec![
            snippet_item(
                "fn",
                "fn ${1:name}(${2:params}) -> ${3:Type} {\n\t$0\n}",
                "Function declaration",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "pub fn",
                "pub fn ${1:name}(${2:params}) -> ${3:Type} {\n\t$0\n}",
                "Public function declaration",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "struct",
                "struct ${1:Name} {\n\t${2:field}: ${3:Type},\n}",
                "Structure definition",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "enum",
                "enum ${1:Name} {\n\t${2:Variant},\n}",
                "Enumeration definition",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "type",
                "type ${1:Name} = ${2:Type}",
                "Type alias",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "trait",
                "trait ${1:Name} {\n\t$0\n}",
                "Trait definition",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "impl",
                "impl ${1:Type} {\n\t$0\n}",
                "Implementation block",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "impl for",
                "impl ${1:Trait} for ${2:Type} {\n\t$0\n}",
                "Trait implementation",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "linear struct",
                "linear struct ${1:Name} {\n\t${2:field}: ${3:Type},\n}",
                "Linear type (must be consumed exactly once)",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "affine struct",
                "affine struct ${1:Name} {\n\t${2:field}: ${3:Type},\n}",
                "Affine type (can be consumed at most once)",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "kernel fn",
                "kernel fn ${1:name}(${2:params}) {\n\t$0\n}",
                "GPU kernel function",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "effect",
                "effect ${1:Name} {\n\t${2:op}(${3:params}) -> ${4:Type},\n}",
                "Effect definition",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "handler",
                "handler ${1:Name} for ${2:Effect} {\n\t$0\n}",
                "Effect handler",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "module",
                "module ${1:name}",
                "Module declaration",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "import",
                "import ${1:path}",
                "Import declaration",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "const",
                "const ${1:NAME}: ${2:Type} = ${3:value}",
                "Constant declaration",
                CompletionItemKind::KEYWORD,
            ),
        ]
    }

    /// Type completions
    fn type_completions(&self, cache: &HashMap<Url, AnalysisResult>) -> Vec<CompletionItem> {
        let mut items = vec![
            // Primitive types
            simple_type("i8", "8-bit signed integer"),
            simple_type("i16", "16-bit signed integer"),
            simple_type("i32", "32-bit signed integer"),
            simple_type("i64", "64-bit signed integer"),
            simple_type("i128", "128-bit signed integer"),
            simple_type("isize", "Pointer-sized signed integer"),
            simple_type("u8", "8-bit unsigned integer"),
            simple_type("u16", "16-bit unsigned integer"),
            simple_type("u32", "32-bit unsigned integer"),
            simple_type("u64", "64-bit unsigned integer"),
            simple_type("u128", "128-bit unsigned integer"),
            simple_type("usize", "Pointer-sized unsigned integer"),
            simple_type("f32", "32-bit floating point"),
            simple_type("f64", "64-bit floating point"),
            simple_type("bool", "Boolean (true/false)"),
            simple_type("char", "Unicode character"),
            simple_type("String", "UTF-8 string"),
            simple_type("str", "String slice"),
            // Generic containers
            snippet_item(
                "Vec",
                "Vec<${1:T}>",
                "Dynamic array",
                CompletionItemKind::STRUCT,
            ),
            snippet_item(
                "Option",
                "Option<${1:T}>",
                "Optional value",
                CompletionItemKind::ENUM,
            ),
            snippet_item(
                "Result",
                "Result<${1:T}, ${2:E}>",
                "Result type for error handling",
                CompletionItemKind::ENUM,
            ),
            snippet_item(
                "HashMap",
                "HashMap<${1:K}, ${2:V}>",
                "Hash map",
                CompletionItemKind::STRUCT,
            ),
            snippet_item(
                "Box",
                "Box<${1:T}>",
                "Heap-allocated value",
                CompletionItemKind::STRUCT,
            ),
            // Unit types
            simple_type("()", "Unit type"),
        ];

        // Add user-defined types from cache
        for result in cache.values() {
            if let Some(ref ast) = result.ast {
                for item in &ast.items {
                    match item {
                        crate::ast::Item::Struct(s) => {
                            items.push(CompletionItem {
                                label: s.name.clone(),
                                kind: Some(CompletionItemKind::STRUCT),
                                detail: Some(format!("struct {}", s.name)),
                                ..Default::default()
                            });
                        }
                        crate::ast::Item::Enum(e) => {
                            items.push(CompletionItem {
                                label: e.name.clone(),
                                kind: Some(CompletionItemKind::ENUM),
                                detail: Some(format!("enum {}", e.name)),
                                ..Default::default()
                            });
                        }
                        crate::ast::Item::TypeAlias(t) => {
                            items.push(CompletionItem {
                                label: t.name.clone(),
                                kind: Some(CompletionItemKind::TYPE_PARAMETER),
                                detail: Some(format!("type {}", t.name)),
                                ..Default::default()
                            });
                        }
                        _ => {}
                    }
                }
            }
        }

        items
    }

    /// Expression completions
    fn expression_completions(&self, cache: &HashMap<Url, AnalysisResult>) -> Vec<CompletionItem> {
        let mut items = vec![
            // Control flow
            snippet_item(
                "let",
                "let ${1:name}: ${2:Type} = ${3:value}",
                "Immutable variable binding",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "let mut",
                "let mut ${1:name}: ${2:Type} = ${3:value}",
                "Mutable variable binding",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "if",
                "if ${1:condition} {\n\t$0\n}",
                "If expression",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "if else",
                "if ${1:condition} {\n\t$2\n} else {\n\t$0\n}",
                "If-else expression",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "match",
                "match ${1:expr} {\n\t${2:pattern} => $0,\n}",
                "Match expression",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "for",
                "for ${1:item} in ${2:iter} {\n\t$0\n}",
                "For loop",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "while",
                "while ${1:condition} {\n\t$0\n}",
                "While loop",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "loop",
                "loop {\n\t$0\n}",
                "Infinite loop",
                CompletionItemKind::KEYWORD,
            ),
            simple_keyword("return", "Return from function"),
            simple_keyword("break", "Break out of loop"),
            simple_keyword("continue", "Continue to next iteration"),
            simple_keyword("true", "Boolean true"),
            simple_keyword("false", "Boolean false"),
            // Effects
            snippet_item(
                "perform",
                "perform ${1:Effect}::${2:op}(${3:args})",
                "Perform effect operation",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "handle",
                "handle ${1:expr} with ${2:handler}",
                "Handle effects",
                CompletionItemKind::KEYWORD,
            ),
            snippet_item(
                "sample",
                "sample(${1:distribution})",
                "Sample from distribution",
                CompletionItemKind::FUNCTION,
            ),
            snippet_item(
                "observe",
                "observe(${1:distribution}, ${2:value})",
                "Condition on observation",
                CompletionItemKind::FUNCTION,
            ),
        ];

        // Add functions and variables from cache
        for result in cache.values() {
            if let Some(ref ast) = result.ast {
                for item in &ast.items {
                    if let crate::ast::Item::Function(f) = item {
                        items.push(CompletionItem {
                            label: f.name.clone(),
                            kind: Some(CompletionItemKind::FUNCTION),
                            detail: Some(format!("fn {}(...)", f.name)),
                            insert_text: Some(format!("{}($0)", f.name)),
                            insert_text_format: Some(InsertTextFormat::SNIPPET),
                            ..Default::default()
                        });
                    }
                }
            }
        }

        items
    }

    /// Effect completions
    fn effect_completions(&self) -> Vec<CompletionItem> {
        vec![
            effect_item("IO", "File, network, console I/O"),
            effect_item("Mut", "Mutable state"),
            effect_item("Alloc", "Heap allocation"),
            effect_item("Panic", "Recoverable failure (exceptions)"),
            effect_item("Async", "Asynchronous operations"),
            effect_item("GPU", "GPU kernel launch, device memory"),
            effect_item("Prob", "Probabilistic computation"),
            effect_item("Div", "Potential divergence (non-termination)"),
        ]
    }

    /// Unit completions
    fn unit_completions(&self) -> Vec<CompletionItem> {
        vec![
            // Mass
            unit_item("kg", "Kilogram (SI mass)"),
            unit_item("g", "Gram"),
            unit_item("mg", "Milligram"),
            unit_item("ug", "Microgram (μg)"),
            unit_item("ng", "Nanogram"),
            // Volume
            unit_item("L", "Liter"),
            unit_item("mL", "Milliliter"),
            unit_item("uL", "Microliter (μL)"),
            // Time
            unit_item("s", "Second (SI time)"),
            unit_item("ms", "Millisecond"),
            unit_item("min", "Minute"),
            unit_item("h", "Hour"),
            unit_item("d", "Day"),
            // Length
            unit_item("m", "Meter (SI length)"),
            unit_item("cm", "Centimeter"),
            unit_item("mm", "Millimeter"),
            unit_item("um", "Micrometer (μm)"),
            unit_item("nm", "Nanometer"),
            unit_item("km", "Kilometer"),
            // Derived units - Concentration
            unit_item("mg/mL", "Concentration (mass/volume)"),
            unit_item("ug/mL", "Concentration (microgram/milliliter)"),
            unit_item("mol/L", "Molarity"),
            unit_item("mmol/L", "Millimolar"),
            // Derived units - Rate
            unit_item("mL/min", "Flow rate / Clearance"),
            unit_item("L/h", "Flow rate (liters per hour)"),
            unit_item("mg/h", "Dose rate"),
            unit_item("mg/kg", "Dose per body weight"),
            // Temperature
            unit_item("K", "Kelvin (SI temperature)"),
            unit_item("C", "Celsius"),
            // Amount
            unit_item("mol", "Mole (SI amount)"),
            unit_item("mmol", "Millimole"),
        ]
    }

    /// Pattern completions
    fn pattern_completions(&self, cache: &HashMap<Url, AnalysisResult>) -> Vec<CompletionItem> {
        let mut items = vec![
            simple_keyword("_", "Wildcard pattern"),
            snippet_item(
                "Some",
                "Some(${1:value})",
                "Some variant",
                CompletionItemKind::ENUM_MEMBER,
            ),
            simple_keyword("None", "None variant"),
            snippet_item(
                "Ok",
                "Ok(${1:value})",
                "Ok variant",
                CompletionItemKind::ENUM_MEMBER,
            ),
            snippet_item(
                "Err",
                "Err(${1:error})",
                "Err variant",
                CompletionItemKind::ENUM_MEMBER,
            ),
        ];

        // Add enum variants from cache
        for result in cache.values() {
            if let Some(ref ast) = result.ast {
                for item in &ast.items {
                    if let crate::ast::Item::Enum(e) = item {
                        for variant in &e.variants {
                            items.push(CompletionItem {
                                label: format!("{}::{}", e.name, variant.name),
                                kind: Some(CompletionItemKind::ENUM_MEMBER),
                                detail: Some(format!("Variant of {}", e.name)),
                                ..Default::default()
                            });
                        }
                    }
                }
            }
        }

        items
    }

    /// Common method completions for member access
    fn common_methods(&self) -> Vec<CompletionItem> {
        vec![
            method_item("len", "len()", "Get length"),
            method_item("is_empty", "is_empty()", "Check if empty"),
            method_item("push", "push($0)", "Add element"),
            method_item("pop", "pop()", "Remove and return last element"),
            method_item("get", "get($0)", "Get element by index"),
            method_item("iter", "iter()", "Get iterator"),
            method_item("map", "map(|${1:x}| $0)", "Transform elements"),
            method_item("filter", "filter(|${1:x}| $0)", "Filter elements"),
            method_item(
                "fold",
                "fold(${1:init}, |${2:acc}, ${3:x}| $0)",
                "Fold/reduce",
            ),
            method_item("collect", "collect::<${1:Vec<_>>>()", "Collect iterator"),
            method_item("clone", "clone()", "Clone value"),
            method_item("to_string", "to_string()", "Convert to string"),
            method_item("unwrap", "unwrap()", "Unwrap Option/Result"),
            method_item("expect", "expect(\"${1:message}\")", "Unwrap with message"),
            method_item("unwrap_or", "unwrap_or(${1:default})", "Unwrap or default"),
            method_item("ok", "ok()", "Convert Result to Option"),
            method_item("err", "err()", "Get error from Result"),
        ]
    }
}

impl Default for CompletionProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Completion context
enum CompletionContext {
    TopLevel,
    Type,
    Expression,
    Effect,
    Unit,
    Member(String),
    Pattern,
}

/// Create a simple type completion
fn simple_type(name: &str, detail: &str) -> CompletionItem {
    CompletionItem {
        label: name.to_string(),
        kind: Some(CompletionItemKind::TYPE_PARAMETER),
        detail: Some(detail.to_string()),
        ..Default::default()
    }
}

/// Create a simple keyword completion
fn simple_keyword(name: &str, detail: &str) -> CompletionItem {
    CompletionItem {
        label: name.to_string(),
        kind: Some(CompletionItemKind::KEYWORD),
        detail: Some(detail.to_string()),
        ..Default::default()
    }
}

/// Create a snippet completion
fn snippet_item(
    label: &str,
    snippet: &str,
    detail: &str,
    kind: CompletionItemKind,
) -> CompletionItem {
    CompletionItem {
        label: label.to_string(),
        kind: Some(kind),
        detail: Some(detail.to_string()),
        insert_text: Some(snippet.to_string()),
        insert_text_format: Some(InsertTextFormat::SNIPPET),
        ..Default::default()
    }
}

/// Create an effect completion
fn effect_item(name: &str, detail: &str) -> CompletionItem {
    CompletionItem {
        label: name.to_string(),
        kind: Some(CompletionItemKind::TYPE_PARAMETER),
        detail: Some(detail.to_string()),
        documentation: Some(Documentation::MarkupContent(MarkupContent {
            kind: MarkupKind::Markdown,
            value: format!("**{}** effect\n\n{}", name, detail),
        })),
        ..Default::default()
    }
}

/// Create a unit completion
fn unit_item(name: &str, detail: &str) -> CompletionItem {
    CompletionItem {
        label: name.to_string(),
        kind: Some(CompletionItemKind::UNIT),
        detail: Some(detail.to_string()),
        ..Default::default()
    }
}

/// Create a method completion
fn method_item(name: &str, snippet: &str, detail: &str) -> CompletionItem {
    CompletionItem {
        label: name.to_string(),
        kind: Some(CompletionItemKind::METHOD),
        detail: Some(detail.to_string()),
        insert_text: Some(snippet.to_string()),
        insert_text_format: Some(InsertTextFormat::SNIPPET),
        ..Default::default()
    }
}

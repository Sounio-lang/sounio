//! Semantic analysis for LSP features
//!
//! Manages parsing, name resolution, and type checking with caching.

use std::collections::HashMap;
use tower_lsp::lsp_types::*;

use super::completion::CompletionProvider;
use super::definition::DefinitionProvider;
use super::diagnostics::DiagnosticsProvider;
use super::document::Document;
use super::hover::HoverProvider;
use super::references::ReferencesProvider;
use super::semantic_tokens::SemanticTokensProvider;

use crate::ast::Ast;
use crate::common::Span;
use crate::lexer;
use crate::parser;
use crate::resolve::SymbolTable;

/// Cached analysis result for a document
#[derive(Debug)]
pub struct AnalysisResult {
    /// Parsed AST
    pub ast: Option<Ast>,

    /// Symbol table from name resolution
    pub symbols: Option<SymbolTable>,

    /// Diagnostics from all phases
    pub diagnostics: Vec<Diagnostic>,

    /// Source version when analyzed
    pub version: i32,
}

impl Default for AnalysisResult {
    fn default() -> Self {
        Self {
            ast: None,
            symbols: None,
            diagnostics: Vec::new(),
            version: 0,
        }
    }
}

/// Analysis host manages incremental analysis and caching
pub struct AnalysisHost {
    /// Cached results per file
    cache: HashMap<Url, AnalysisResult>,

    /// Completion provider
    completion: CompletionProvider,

    /// Hover provider
    hover: HoverProvider,

    /// Definition provider
    definition: DefinitionProvider,

    /// References provider
    references: ReferencesProvider,

    /// Semantic tokens provider
    semantic_tokens: SemanticTokensProvider,

    /// Diagnostics provider
    diagnostics_provider: DiagnosticsProvider,
}

impl AnalysisHost {
    /// Create a new analysis host
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            completion: CompletionProvider::new(),
            hover: HoverProvider::new(),
            definition: DefinitionProvider::new(),
            references: ReferencesProvider::new(),
            semantic_tokens: SemanticTokensProvider::new(),
            diagnostics_provider: DiagnosticsProvider::new(),
        }
    }

    /// Analyze a document and return diagnostics
    pub fn analyze(&mut self, source: &str, uri: &Url) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let mut ast = None;
        let mut symbols = None;

        // Phase 1: Lexing
        match lexer::lex(source) {
            Ok(tokens) => {
                // Phase 2: Parsing
                match parser::parse(&tokens, source) {
                    Ok(parsed_ast) => {
                        ast = Some(parsed_ast.clone());

                        // Phase 3: Name resolution
                        let sym_table = SymbolTable::new();
                        // Note: Full resolution would involve more work
                        // For now, just create a basic symbol table
                        symbols = Some(sym_table);
                    }
                    Err(err) => {
                        diagnostics
                            .push(self.diagnostics_provider.miette_to_diagnostic(&err, source));
                    }
                }
            }
            Err(err) => {
                diagnostics.push(self.diagnostics_provider.miette_to_diagnostic(&err, source));
            }
        }

        // Cache the result
        self.cache.insert(
            uri.clone(),
            AnalysisResult {
                ast,
                symbols,
                diagnostics: diagnostics.clone(),
                version: 0,
            },
        );

        diagnostics
    }

    /// Get hover information at a position
    pub fn hover(&self, doc: &Document, pos: Position, uri: &Url) -> Option<Hover> {
        let (word, range) = doc.word_at(pos)?;

        // Check cached analysis
        if let Some(result) = self.cache.get(uri) {
            if let Some(ref symbols) = result.symbols {
                return self.hover.hover_for_symbol(&word, range, symbols);
            }
        }

        // Fallback to keyword hover
        self.hover.hover_for_keyword(&word, range)
    }

    /// Go to definition
    pub fn goto_definition(
        &self,
        doc: &Document,
        pos: Position,
        uri: &Url,
    ) -> Option<GotoDefinitionResponse> {
        let (word, _) = doc.word_at(pos)?;

        if let Some(result) = self.cache.get(uri) {
            if let Some(ref symbols) = result.symbols {
                return self.definition.find_definition(&word, symbols, uri);
            }
        }

        None
    }

    /// Find all references
    pub fn find_references(
        &self,
        doc: &Document,
        pos: Position,
        uri: &Url,
    ) -> Option<Vec<Location>> {
        let (word, _) = doc.word_at(pos)?;

        if let Some(result) = self.cache.get(uri) {
            if let Some(ref symbols) = result.symbols {
                return Some(self.references.find_references(&word, symbols, uri));
            }
        }

        None
    }

    /// Get completions at a position
    pub fn completions(&self, doc: &Document, pos: Position) -> Vec<CompletionItem> {
        let source = doc.text();
        let offset = doc.position_to_offset(pos).unwrap_or(0);
        let context = doc.get_context(pos, 50).unwrap_or_default();

        self.completion
            .complete(&source, offset, &context, &self.cache)
    }

    /// Get semantic tokens for highlighting
    pub fn semantic_tokens(&self, doc: &Document) -> SemanticTokens {
        let source = doc.text();
        self.semantic_tokens.tokenize(&source)
    }

    /// Get document symbols (outline)
    pub fn document_symbols(&self, doc: &Document, uri: &Url) -> Vec<DocumentSymbol> {
        if let Some(result) = self.cache.get(uri) {
            if let Some(ref ast) = result.ast {
                return self.extract_symbols(ast, &doc.text());
            }
        }

        Vec::new()
    }

    /// Get signature help
    pub fn signature_help(&self, doc: &Document, pos: Position) -> Option<SignatureHelp> {
        let source = doc.text();
        let offset = doc.position_to_offset(pos)?;

        // Find the function call context
        let before = &source[..offset];
        let paren_pos = before.rfind('(')?;

        // Count commas to determine active parameter
        let in_call = &source[paren_pos..offset];
        let active_param = in_call.matches(',').count();

        // Extract function name
        let func_end = before[..paren_pos].trim_end().len();
        let mut func_start = func_end;
        let bytes = before.as_bytes();
        while func_start > 0
            && (bytes[func_start - 1].is_ascii_alphanumeric() || bytes[func_start - 1] == b'_')
        {
            func_start -= 1;
        }

        let func_name = &before[func_start..func_end];

        // Look up function signature
        self.get_function_signature(func_name, active_param as u32)
    }

    /// Rename symbol
    pub fn rename(
        &self,
        doc: &Document,
        pos: Position,
        new_name: &str,
        uri: &Url,
    ) -> Option<WorkspaceEdit> {
        let (old_name, _) = doc.word_at(pos)?;

        // Find all references and create edits
        let locations = self.find_references(doc, pos, uri)?;

        let mut changes: HashMap<Url, Vec<TextEdit>> = HashMap::new();

        for loc in locations {
            let edit = TextEdit {
                range: loc.range,
                new_text: new_name.to_string(),
            };

            changes.entry(loc.uri).or_default().push(edit);
        }

        Some(WorkspaceEdit {
            changes: Some(changes),
            ..Default::default()
        })
    }

    /// Get code actions
    pub fn code_actions(
        &self,
        _doc: &Document,
        _range: Range,
        diagnostics: &[Diagnostic],
        _uri: &Url,
    ) -> CodeActionResponse {
        let mut actions = Vec::new();

        for diag in diagnostics {
            // Generate quick fixes based on diagnostic code
            if let Some(NumberOrString::String(code)) = &diag.code {
                match code.as_str() {
                    "E0002" => {
                        // Undefined variable
                        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                            title: "Add variable declaration".to_string(),
                            kind: Some(CodeActionKind::QUICKFIX),
                            diagnostics: Some(vec![diag.clone()]),
                            edit: None,
                            ..Default::default()
                        }));
                    }
                    "E0006" => {
                        // Effect not declared
                        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                            title: "Add effect to function signature".to_string(),
                            kind: Some(CodeActionKind::QUICKFIX),
                            diagnostics: Some(vec![diag.clone()]),
                            edit: None,
                            ..Default::default()
                        }));
                    }
                    _ => {}
                }
            }
        }

        actions
    }

    /// Get inlay hints
    pub fn inlay_hints(&self, doc: &Document, range: Range) -> Vec<InlayHint> {
        let mut hints = Vec::new();
        let source = doc.text();

        // Parse and find let bindings without explicit types
        if let Ok(tokens) = lexer::lex(&source) {
            let mut i = 0;
            while i < tokens.len() {
                // Look for "let" keyword
                if tokens[i].kind == crate::lexer::TokenKind::Let {
                    // Check if there's no type annotation
                    // Pattern: let <name> = <value>
                    if i + 3 < tokens.len() {
                        if let crate::lexer::TokenKind::Ident = tokens[i + 1].kind {
                            if tokens[i + 2].kind == crate::lexer::TokenKind::Eq {
                                // No type annotation, could add inlay hint
                                let pos = doc.offset_to_position(tokens[i + 1].span.end);
                                if pos.line >= range.start.line && pos.line <= range.end.line {
                                    hints.push(InlayHint {
                                        position: pos,
                                        label: InlayHintLabel::String(": <inferred>".to_string()),
                                        kind: Some(InlayHintKind::TYPE),
                                        text_edits: None,
                                        tooltip: Some(InlayHintTooltip::String(
                                            "Inferred type".to_string(),
                                        )),
                                        padding_left: Some(false),
                                        padding_right: Some(true),
                                        data: None,
                                    });
                                }
                            }
                        }
                    }
                }
                i += 1;
            }
        }

        hints
    }

    /// Format document
    pub fn format(&self, _doc: &Document) -> Option<Vec<TextEdit>> {
        // TODO: Implement formatter
        None
    }

    /// Get folding ranges
    pub fn folding_ranges(&self, doc: &Document) -> Vec<FoldingRange> {
        let mut ranges = Vec::new();
        let source = doc.text();

        // Track brace depth for folding
        if let Ok(tokens) = lexer::lex(&source) {
            let mut brace_stack: Vec<(u32, FoldingRangeKind)> = Vec::new();

            for token in &tokens {
                let pos = doc.offset_to_position(token.span.start);

                match token.kind {
                    crate::lexer::TokenKind::LBrace => {
                        brace_stack.push((pos.line, FoldingRangeKind::Region));
                    }
                    crate::lexer::TokenKind::RBrace => {
                        if let Some((start_line, kind)) = brace_stack.pop() {
                            if pos.line > start_line {
                                ranges.push(FoldingRange {
                                    start_line,
                                    start_character: None,
                                    end_line: pos.line,
                                    end_character: None,
                                    kind: Some(kind),
                                    collapsed_text: None,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        ranges
    }

    /// Extract document symbols from AST
    fn extract_symbols(&self, ast: &Ast, source: &str) -> Vec<DocumentSymbol> {
        let mut symbols = Vec::new();

        for item in &ast.items {
            match item {
                crate::ast::Item::Function(f) => {
                    let range = span_to_range(&f.span, source);
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: f.name.clone(),
                        detail: Some(format_fn_signature(f)),
                        kind: SymbolKind::FUNCTION,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::Struct(s) => {
                    let range = span_to_range(&s.span, source);
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: s.name.clone(),
                        detail: Some(format!("struct {}", s.name)),
                        kind: SymbolKind::STRUCT,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::Enum(e) => {
                    let range = span_to_range(&e.span, source);
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: e.name.clone(),
                        detail: Some(format!("enum {}", e.name)),
                        kind: SymbolKind::ENUM,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::TypeAlias(t) => {
                    let range = span_to_range(&t.span, source);
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: t.name.clone(),
                        detail: Some(format!("type {}", t.name)),
                        kind: SymbolKind::TYPE_PARAMETER,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::Effect(e) => {
                    let range = span_to_range(&e.span, source);
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: e.name.clone(),
                        detail: Some(format!("effect {}", e.name)),
                        kind: SymbolKind::INTERFACE,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::Trait(t) => {
                    let range = span_to_range(&t.span, source);
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: t.name.clone(),
                        detail: Some(format!("trait {}", t.name)),
                        kind: SymbolKind::INTERFACE,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                _ => {}
            }
        }

        symbols
    }

    /// Get function signature for signature help
    fn get_function_signature(&self, name: &str, active_param: u32) -> Option<SignatureHelp> {
        // Check cache for function definitions
        for result in self.cache.values() {
            if let Some(ref ast) = result.ast {
                for item in &ast.items {
                    if let crate::ast::Item::Function(f) = item {
                        if f.name == name {
                            let params: Vec<ParameterInformation> = f
                                .params
                                .iter()
                                .map(|p| {
                                    let param_name = match &p.pattern {
                                        crate::ast::Pattern::Binding { name, .. } => name.clone(),
                                        _ => "_".to_string(),
                                    };
                                    ParameterInformation {
                                        label: ParameterLabel::Simple(param_name),
                                        documentation: None,
                                    }
                                })
                                .collect();

                            return Some(SignatureHelp {
                                signatures: vec![SignatureInformation {
                                    label: format_fn_signature(f),
                                    documentation: None,
                                    parameters: Some(params),
                                    active_parameter: Some(active_param),
                                }],
                                active_signature: Some(0),
                                active_parameter: Some(active_param),
                            });
                        }
                    }
                }
            }
        }

        // Check for built-in functions
        get_builtin_signature(name, active_param)
    }
}

impl Default for AnalysisHost {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert Span to LSP Range
fn span_to_range(span: &Span, source: &str) -> Range {
    let (start_line, start_col) = offset_to_line_col(source, span.start);
    let (end_line, end_col) = offset_to_line_col(source, span.end);

    Range {
        start: Position {
            line: start_line as u32,
            character: start_col as u32,
        },
        end: Position {
            line: end_line as u32,
            character: end_col as u32,
        },
    }
}

/// Convert byte offset to line/column
fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(source.len());
    let mut line = 0;
    let mut col = 0;

    for (i, c) in source.char_indices() {
        if i >= offset {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    (line, col)
}

/// Format a function signature
fn format_fn_signature(f: &crate::ast::FnDef) -> String {
    let params: Vec<String> = f
        .params
        .iter()
        .map(|p| {
            let name = match &p.pattern {
                crate::ast::Pattern::Binding { name, .. } => name.clone(),
                _ => "_".to_string(),
            };
            format!("{}: {}", name, format_type(&p.ty))
        })
        .collect();

    let ret = f
        .return_type
        .as_ref()
        .map(|t| format!(" -> {}", format_type(t)))
        .unwrap_or_default();

    let effects = if f.effects.is_empty() {
        String::new()
    } else {
        let eff_names: Vec<String> = f.effects.iter().map(|e| e.name.to_string()).collect();
        format!(" with {}", eff_names.join(", "))
    };

    format!("fn {}({}){}{}", f.name, params.join(", "), ret, effects)
}

/// Format a type expression
fn format_type(ty: &crate::ast::TypeExpr) -> String {
    match ty {
        crate::ast::TypeExpr::Unit => "()".to_string(),
        crate::ast::TypeExpr::SelfType => "Self".to_string(),
        crate::ast::TypeExpr::Named { path, args, unit } => {
            let base = path.to_string();
            let type_args = if args.is_empty() {
                String::new()
            } else {
                let arg_strs: Vec<String> = args.iter().map(format_type).collect();
                format!("<{}>", arg_strs.join(", "))
            };
            let unit_suffix = unit
                .as_ref()
                .map(|u| format!("<{}>", u))
                .unwrap_or_default();
            format!("{}{}{}", base, type_args, unit_suffix)
        }
        crate::ast::TypeExpr::Reference { mutable, inner } => {
            if *mutable {
                format!("&mut {}", format_type(inner))
            } else {
                format!("&{}", format_type(inner))
            }
        }
        crate::ast::TypeExpr::Array { element, size } => {
            if let Some(_size) = size {
                format!("[{}; N]", format_type(element))
            } else {
                format!("[{}]", format_type(element))
            }
        }
        crate::ast::TypeExpr::Tuple(types) => {
            let inner: Vec<String> = types.iter().map(format_type).collect();
            format!("({})", inner.join(", "))
        }
        crate::ast::TypeExpr::Function {
            params,
            return_type,
            ..
        } => {
            let p: Vec<String> = params.iter().map(format_type).collect();
            format!("fn({}) -> {}", p.join(", "), format_type(return_type))
        }
        crate::ast::TypeExpr::Infer => "_".to_string(),
    }
}

/// Get built-in function signature
fn get_builtin_signature(name: &str, active_param: u32) -> Option<SignatureHelp> {
    let (label, params): (&str, Vec<&str>) = match name {
        "print" => ("fn print(value: T)", vec!["value: T"]),
        "println" => ("fn println(value: T)", vec!["value: T"]),
        "len" => ("fn len(collection: C) -> usize", vec!["collection: C"]),
        "push" => (
            "fn push(vec: &mut Vec<T>, value: T)",
            vec!["vec: &mut Vec<T>", "value: T"],
        ),
        "pop" => (
            "fn pop(vec: &mut Vec<T>) -> Option<T>",
            vec!["vec: &mut Vec<T>"],
        ),
        "sample" => (
            "fn sample(distribution: D) -> T with Prob",
            vec!["distribution: D"],
        ),
        "observe" => (
            "fn observe(distribution: D, value: T) with Prob",
            vec!["distribution: D", "value: T"],
        ),
        _ => return None,
    };

    Some(SignatureHelp {
        signatures: vec![SignatureInformation {
            label: label.to_string(),
            documentation: None,
            parameters: Some(
                params
                    .iter()
                    .map(|p| ParameterInformation {
                        label: ParameterLabel::Simple(p.to_string()),
                        documentation: None,
                    })
                    .collect(),
            ),
            active_parameter: Some(active_param),
        }],
        active_signature: Some(0),
        active_parameter: Some(active_param),
    })
}

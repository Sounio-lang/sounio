//! Sounio Language Server Protocol Implementation
//!
//! Provides LSP server for IDEs to offer advanced Sounio language features.
//! Supports epistemic types, scientific computing, and uncertainty-aware programming.

use tower_lsp::LspService;
use tower_lsp::Server;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::lsp_types::notification::Notification;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};

/// Sounio LSP Server State
#[derive(Debug)]
pub struct SounioLspServer {
    /// Document store
    documents: Arc<Mutex<DocumentStore>>,
    /// Symbol index
    symbol_index: Arc<Mutex<SymbolIndex>>,
    /// Compilation cache
    compilation_cache: Arc<Mutex<CompilationCache>>,
}

/// Document store for LSP
#[derive(Debug)]
pub struct DocumentStore {
    /// Opened documents
    documents: HashMap<Url, Document>,
}

/// Document representation for LSP
#[derive(Debug)]
pub struct Document {
    /// Document URL
    pub uri: Url,
    /// Document content
    pub content: String,
    /// Document version
    pub version: i32,
    /// Last compilation result
    pub compilation_result: Option<CompilationResult>,
}

/// Compilation result from Sounio compiler
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// Diagnostics
    pub diagnostics: Vec<Diagnostic>,
    /// Symbols found
    pub symbols: Vec<Symbol>,
    /// Semantic tokens
    pub tokens: Vec<SemanticToken>,
}

/// Symbol information
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Symbol name
    pub name: String,
    /// Symbol kind
    pub kind: SymbolKind,
    /// Symbol range
    pub range: Range,
    /// Symbol documentation
    pub documentation: Option<String>,
}

/// Semantic token information
#[derive(Debug, Clone)]
pub struct SemanticToken {
    /// Token range
    pub range: Range,
    /// Token type
    pub token_type: SemanticTokenType,
    /// Token modifiers
    pub modifiers: Vec<SemanticTokenModifier>,
}

/// Symbol index for fast lookups
#[derive(Debug)]
pub struct SymbolIndex {
    /// Global symbols
    global_symbols: HashMap<String, Symbol>,
    /// Document-specific symbols
    document_symbols: HashMap<Url, HashMap<String, Symbol>>,
}

/// Compilation cache for performance
#[derive(Debug)]
pub struct CompilationCache {
    /// Compiled modules cache
    modules: HashMap<Url, CompiledModule>,
    /// Type information cache
    types: HashMap<String, TypeInfo>,
}

/// Compiled module information
#[derive(Debug, Clone)]
pub struct CompiledModule {
    /// Module symbols
    pub symbols: Vec<Symbol>,
    /// Epistemic types
    pub epistemic_types: Vec<EpistemicType>,
    /// Import graph
    pub imports: Vec<Import>,
}

/// Epistemic type information
#[derive(Debug, Clone)]
pub struct EpistemicType {
    /// Type name
    pub name: String,
    /// Uncertainty bounds
    pub uncertainty: Option<UncertaintyBounds>,
    /// Confidence information
    pub confidence: Option<f64>,
    /// Provenance chain
    pub provenance: Vec<String>,
}

/// Uncertainty bounds
#[derive(Debug, Clone)]
pub struct UncertaintyBounds {
    /// Lower bound
    pub lower: f64,
    /// Upper bound
    pub upper: f64,
    /// Distribution type
    pub distribution: UncertaintyDistribution,
}

/// Uncertainty distribution types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyDistribution {
    /// Normal distribution
    Normal { mean: f64, stddev: f64 },
    /// Uniform distribution
    Uniform { min: f64, max: f64 },
    /// Custom distribution
    Custom { values: Vec<f64> },
}

/// Import information
#[derive(Debug, Clone)]
pub struct Import {
    /// Imported module
    pub module: String,
    /// Import range
    pub range: Range,
    /// Import type
    pub kind: ImportKind,
}

/// Import types
#[derive(Debug, Clone)]
pub enum ImportKind {
    /// Standard library import
    Stdlib,
    /// User module import
    User,
    /// External library import
    External,
}

/// Type information cache
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// Type name
    pub name: String,
    /// Type definition
    pub definition: String,
    /// Type examples
    pub examples: Vec<String>,
}

impl SounioLspServer {
    /// Create new LSP server
    pub fn new() -> Self {
        Self {
            documents: Arc::new(Mutex::new(DocumentStore {
                documents: HashMap::new(),
            })),
            symbol_index: Arc::new(Mutex::new(SymbolIndex {
                global_symbols: HashMap::new(),
                document_symbols: HashMap::new(),
            })),
            compilation_cache: Arc::new(Mutex::new(CompilationCache {
                modules: HashMap::new(),
                types: HashMap::new(),
            })),
        }
    }

    /// Initialize LSP server
    pub async fn run() -> Result<()> {
        let (service, socket) = LspService::new(SounioLspServer::new());
        Server::new(socket).serve(service).await;
        Ok(())
    }
}

#[tower_lsp::async_trait]
impl tower_lsp::LanguageServer for SounioLspServer {
    async fn initialize(&self, _params: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult::default())
    }

    async fn initialized(&self, _params: InitializedParams) {
        // Server initialized
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let document = Document {
            uri: params.text_document.uri.clone(),
            content: params.text_document.text,
            version: params.text_document.version,
            compilation_result: None,
        };

        let mut store = self.documents.lock().unwrap();
        store.documents.insert(params.text_document.uri.clone(), document);

        // Trigger compilation
        self.compile_document(&params.text_document.uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(mut document) = self.documents.lock().unwrap()
            .documents
            .get_mut(&params.text_document.uri) 
        {
            document.version = params.text_document.version;
            
            // Update content with changes
            for change in params.content_changes {
                if let Some(range) = change.range {
                    // Apply text change
                    let _ = self.apply_change(&mut document.content, range, change.text);
                } else {
                    document.content = change.text;
                }
            }
            
            // Recompile
            self.compile_document(&params.text_document.uri).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let mut store = self.documents.lock().unwrap();
        store.documents.remove(&params.text_document.uri);
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let documents = self.documents.lock().unwrap();
        let position = params.text_document_position.position;
        
        if let Some(document) = documents.documents.get(&params.text_document_position.text_document.uri) {
            let completions = self.get_completions(&document.content, position);
            Ok(Some(CompletionResponse::Array(completions)))
        } else {
            Ok(None)
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let documents = self.documents.lock().unwrap();
        let position = params.text_document_position_params.position;
        
        if let Some(document) = documents.documents.get(&params.text_document_position.text_document.uri.uri) {
            let hover = self.get_hover(&document.content, position);
            Ok(hover)
        } else {
            Ok(None)
        }
    }

    async fn goto_definition(&self, params: GotoDefinitionParams) -> Result<Option<GotoDefinitionResponse>> {
        let documents = self.documents.lock().unwrap();
        let position = params.text_document_position_params.position;
        
        if let Some(document) = documents.documents.get(&params.text_document_position_params.text_document.uri.uri) {
            let definition = self.get_definition(&document.content, position);
            Ok(definition)
        } else {
            Ok(None)
        }
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let documents = self.documents.lock().unwrap();
        let position = params.text_document_position_params.position;
        
        if let Some(document) = documents.documents.get(&params.text_document_position_params.text_document.uri.uri) {
            let references = self.find_references(&document.content, position);
            Ok(Some(references))
        } else {
            Ok(None)
        }
    }

    async fn document_symbol(&self, params: DocumentSymbolParams) -> Result<Option<DocumentSymbolResponse>> {
        let documents = self.documents.lock().unwrap();
        
        if let Some(document) = documents.documents.get(&params.text_document.uri) {
            let symbols = self.extract_document_symbols(&document.content);
            Ok(Some(DocumentSymbolResponse::Nested(symbols)))
        } else {
            Ok(None)
        }
    }

    async fn rename(&self, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
        let documents = self.documents.lock().unwrap();
        let new_name = params.new_name;
        let position = params.text_document_position_params.position;
        
        if let Some(document) = documents.documents.get(&params.text_document_position_params.text_document.uri.uri) {
            let edit = self.rename_symbol(&document.content, position, &new_name);
            Ok(edit)
        } else {
            Ok(None)
        }
    }

    async fn semantic_tokens_full(&self, params: SemanticTokensParams) -> Result<Option<SemanticTokensResponse>> {
        let documents = self.documents.lock().unwrap();
        
        if let Some(document) = documents.documents.get(&params.text_document.uri) {
            let tokens = self.get_semantic_tokens(&document.content);
            Ok(Some(SemanticTokensResponse::Tokens(SemanticTokens {
                result_id: None,
                data: tokens,
            })))
        } else {
            Ok(None)
        }
    }

    async fn code_action(&self, params: CodeActionParams) -> Result<Option<CodeActionResponse>> {
        let documents = self.documents.lock().unwrap();
        let range = params.range;
        
        if let Some(document) = documents.documents.get(&params.text_document.uri.uri) {
            let actions = self.get_code_actions(&document.content, range);
            Ok(Some(CodeActionResponse::from(actions)))
        } else {
            Ok(None)
        }
    }

    async fn formatting(&self, params: DocumentFormattingParams) -> Result<Option<Vec<TextEdit>>> {
        let documents = self.documents.lock().unwrap();
        
        if let Some(document) = documents.documents.get(&params.text_document.uri) {
            let edits = self.format_document(&document.content);
            Ok(Some(edits))
        } else {
            Ok(None)
        }
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let documents = self.documents.lock().unwrap();
        let range = params.range;
        
        if let Some(document) = documents.documents.get(&params.text_document.uri.uri) {
            let hints = self.get_inlay_hints(&document.content, range);
            Ok(Some(hints))
        } else {
            Ok(None)
        }
    }

    async fn inlay_hint_resolve(&self, params: InlayHint) -> Result<InlayHint> {
        Ok(params)
    }
}

impl SounioLspServer {
    /// Apply text change to document content
    fn apply_change(&self, content: &mut String, range: Range, text: String) {
        let _ = content.replace_range(
            (range.start.character as usize)..(range.end.character as usize),
            &text,
        );
    }

    /// Compile document and update cache
    async fn compile_document(&self, uri: &Url) {
        let documents = self.documents.lock().unwrap();
        if let Some(document) = documents.documents.get(uri) {
            // Compile with Sounio compiler
            let result = self.run_compilation(&document.content);
            
            // Update compilation result
            if let Ok(mut doc_store) = self.documents.lock() {
                if let Some(doc) = doc_store.documents.get_mut(uri) {
                    doc.compilation_result = Some(result.clone());
                }
            }
            
            // Update symbol index
            self.update_symbol_index(uri, &result);
        }
    }

    /// Run Sounio compilation
    fn run_compilation(&self, content: &str) -> CompilationResult {
        // Simplified compilation - in real implementation would use full Sounio compiler
        CompilationResult {
            diagnostics: self.analyze_diagnostics(content),
            symbols: self.extract_symbols(content),
            tokens: self.analyze_tokens(content),
        }
    }

    /// Analyze diagnostics in content
    fn analyze_diagnostics(&self, content: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        
        // Basic syntax analysis
        let lines: Vec<&str> = content.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let line_num = (i + 1) as u32;
            
            // Check for common issues
            if line.contains("Knowledge<T>") {
                if !line.contains("import") && !line.contains("struct") {
                    diagnostics.push(Diagnostic {
                        range: Range::new(
                            Position::new(line_num, 0),
                            Position::new(line_num, line.len() as u32),
                        ),
                        severity: Some(DiagnosticSeverity::Warning),
                        code: Some(NumberOrString::String("EPISTEMIC_001".to_string())),
                        source: Some("Sounio LSP".to_string()),
                        message: "Epistemic type Knowledge<T> requires import or type definition".to_string(),
                        related_information: None,
                        tags: None,
                    });
                }
            }
        }
        
        diagnostics
    }

    /// Extract symbols from content
    fn extract_symbols(&self, content: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        
        for (i, line) in lines.iter().enumerate() {
            let line_num = i as u32;
            
            // Function definitions
            if line.contains("fn ") {
                if let Some(name_start) = line.find("fn ") {
                    let name_end = line.find('(').unwrap_or(line.len());
                    let name = &line[name_start + 3..name_end];
                    
                    symbols.push(Symbol {
                        name: name.trim().to_string(),
                        kind: SymbolKind::Function,
                        range: Range::new(
                            Position::new(line_num, name_start as u32),
                            Position::new(line_num, name_end as u32),
                        ),
                        documentation: None,
                    });
                }
            }
            
            // Type definitions
            if line.contains("struct ") {
                if let Some(name_start) = line.find("struct ") {
                    let name_end = line.find('{').unwrap_or(line.len());
                    let name = &line[name_start + 7..name_end];
                    
                    symbols.push(Symbol {
                        name: name.trim().to_string(),
                        kind: SymbolKind::Struct,
                        range: Range::new(
                            Position::new(line_num, name_start as u32),
                            Position::new(line_num, name_end as u32),
                        ),
                        documentation: None,
                    });
                }
            }
            
            // Knowledge<T> types
            if line.contains("Knowledge<T>") {
                symbols.push(Symbol {
                    name: "Knowledge".to_string(),
                    kind: SymbolKind::TypeParameter,
                    range: Range::new(
                        Position::new(line_num, 0),
                        Position::new(line_num, 15),
                    ),
                    documentation: Some("Epistemic type that carries uncertainty and confidence information".to_string()),
                });
            }
        }
        
        symbols
    }

    /// Analyze semantic tokens
    fn analyze_tokens(&self, content: &str) -> Vec<SemanticToken> {
        let mut tokens = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        
        for (i, line) in lines.iter().enumerate() {
            let line_num = i as u32;
            
            // Keyword tokens
            if line.contains("fn ") {
                tokens.push(SemanticToken {
                    range: Range::new(
                        Position::new(line_num, 0),
                        Position::new(line_num, 2),
                    ),
                    token_type: SemanticTokenType::Keyword,
                    modifiers: vec![],
                });
            }
            
            if line.contains("struct ") {
                tokens.push(SemanticToken {
                    range: Range::new(
                        Position::new(line_num, 0),
                        Position::new(line_num, 6),
                    ),
                    token_type: SemanticTokenType::Keyword,
                    modifiers: vec![],
                });
            }
            
            // Epistemic type tokens
            if line.contains("Knowledge<T>") {
                tokens.push(SemanticToken {
                    range: Range::new(
                        Position::new(line_num, 0),
                        Position::new(line_num, 9),
                    ),
                    token_type: SemanticTokenType::Type,
                    modifiers: vec![SemanticTokenModifier::Epistemic],
                });
            }
        }
        
        tokens
    }

    /// Get completions at position
    fn get_completions(&self, content: &str, position: Position) -> Vec<CompletionItem> {
        let mut completions = Vec::new();
        
        // Basic completions
        completions.extend(vec![
            CompletionItem::new_simple("fn", "fn function_name(parameters) -> ReturnType"),
            CompletionItem::new_simple("struct", "struct Name { }"),
            CompletionItem::new_simple("import", "import stdlib.*"),
        ]);
        
        // Epistemic completions
        completions.extend(vec![
            CompletionItem::new_simple("Knowledge::new", "Knowledge::new(value, uncertainty, confidence, source)"),
            CompletionItem::new_simple("uncertainty", "uncertainty: f64"),
            CompletionItem::new_simple("confidence", "confidence: f64"),
        ]);
        
        completions
    }

    /// Get hover information
    fn get_hover(&self, content: &str, position: Position) -> Option<Hover> {
        let lines: Vec<&str> = content.lines().collect();
        
        if let Some(line) = lines.get(position.line as usize) {
            let line_part = &line[position.character as usize..];
            
            // Knowledge<T> hover
            if line_part.starts_with("Knowledge") {
                Some(Hover {
                    contents: HoverContents::Scalar(MarkUpContent::plain_markdown(
                        "Epistemic type that carries both value and uncertainty information\n\n\
                        **Usage:** `Knowledge<T>::new(value, uncertainty, confidence, source)`\n\n\
                        **Example:** `Knowledge::new(42.0, 0.1, 0.95, \"measurement\")`",
                    )),
                    range: Some(Range::new(position, position)),
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get definition location
    fn get_definition(&self, content: &str, position: Position) -> Option<GotoDefinitionResponse> {
        // Simplified - would find actual definition location
        Some(GotoDefinitionResponse::Scalar(Location::new(
            Url::parse("file:///example.sio").unwrap(),
            Range::new(Position::new(0, 0), Position::new(0, 0)),
        )))
    }

    /// Find references
    fn find_references(&self, content: &str, position: Position) -> Vec<Location> {
        // Would find all references to symbol at position
        vec![]
    }

    /// Extract document symbols
    fn extract_document_symbols(&self, content: &str) -> Vec<DocumentSymbol> {
        let mut symbols = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        
        for (i, line) in lines.iter().enumerate() {
            if line.contains("fn ") {
                symbols.push(DocumentSymbol::new(
                    "function".to_string(),
                    SymbolKind::Function,
                    Range::new(
                        Position::new(i as u32, 0),
                        Position::new(i as u32, line.len() as u32),
                    ),
                ));
            }
        }
        
        symbols
    }

    /// Rename symbol
    fn rename_symbol(&self, content: &str, position: Position, new_name: &str) -> Option<WorkspaceEdit> {
        // Would perform actual renaming
        Some(WorkspaceEdit::new(HashMap::new()))
    }

    /// Get semantic tokens
    fn get_semantic_tokens(&self, content: &str) -> Vec<SemanticToken> {
        self.analyze_tokens(content)
    }

    /// Get code actions
    fn get_code_actions(&self, content: str, range: Range) -> Vec<CodeAction> {
        let mut actions = Vec::new();
        
        // Add epistemic actions
        actions.push(CodeAction::new(
            "Add uncertainty bounds".to_string(),
            None,
            CodeActionKind::REFACTOR_INLINE,
        ));
        
        actions
    }

    /// Format document
    fn format_document(&self, content: &str) -> Vec<TextEdit> {
        // Basic formatting
        vec![]
    }

    /// Get inlay hints
    fn get_inlay_hints(&self, content: str, range: Range) -> Vec<InlayHint> {
        let mut hints = Vec::new();
        
        // Uncertainty hints
        hints.push(InlayHint::new(
            range.end,
            "uncertainty: f64",
            InlayHintLabel::String("uncertainty: f64".to_string()),
        ));
        
        hints
    }

    /// Update symbol index
    fn update_symbol_index(&self, uri: &Url, result: &CompilationResult) {
        let mut index = self.symbol_index.lock().unwrap();
        
        for symbol in &result.symbols {
            index.document_symbols
                .entry(uri.clone())
                .or_insert_with(HashMap::new)
                .insert(symbol.name.clone(), symbol.clone());
        }
    }
}

impl Default for SounioLspServer {
    fn default() -> Self {
        Self::new()
    }
}

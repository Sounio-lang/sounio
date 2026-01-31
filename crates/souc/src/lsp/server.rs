//! Sounio Language Server Protocol Implementation
//!
//! Provides LSP server for IDEs to offer advanced Sounio language features.
//! Supports epistemic types, scientific computing, and uncertainty-aware programming.

use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LspService, Server};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::analysis::AnalysisHost;
use super::document::Document;

/// Sounio LSP Server State
pub struct SounioLspServer {
    /// LSP client for sending notifications
    client: Client,
    /// Document store with rope-based editing
    documents: Arc<Mutex<HashMap<Url, Document>>>,
    /// Analysis host with query-based incremental analysis
    analysis: Arc<Mutex<AnalysisHost>>,
    /// Symbol index (legacy, kept for cross-file lookups)
    symbol_index: Arc<Mutex<SymbolIndex>>,
    /// Compilation cache (legacy, for backward compat)
    compilation_cache: Arc<Mutex<CompilationCache>>,
}

/// Type alias for the language server (consistent naming)
pub type SounioLanguageServer = SounioLspServer;

// Legacy types kept for cross-file symbol index (will be replaced in Phase 9)
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct LegacySymbol {
    name: String,
    kind: SymbolKind,
    range: Range,
}

/// Symbol index for cross-file lookups (legacy, will be migrated to query system)
#[derive(Debug, Default)]
pub struct SymbolIndex {
    /// Global symbols
    global_symbols: HashMap<String, LegacySymbol>,
    /// Document-specific symbols
    document_symbols: HashMap<Url, HashMap<String, LegacySymbol>>,
}

/// Compilation cache (legacy, will be replaced by query database)
#[derive(Debug, Default)]
pub struct CompilationCache {
    /// Placeholder - actual caching done by LspQueryDatabase
    _placeholder: (),
}

impl SounioLspServer {
    /// Create new LSP server with client
    pub fn new(client: Client) -> Self {
        Self {
            client,
            documents: Arc::new(Mutex::new(HashMap::new())),
            analysis: Arc::new(Mutex::new(AnalysisHost::new())),
            symbol_index: Arc::new(Mutex::new(SymbolIndex::default())),
            compilation_cache: Arc::new(Mutex::new(CompilationCache::default())),
        }
    }

    /// Initialize LSP server over stdin/stdout
    pub async fn run_stdio() {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();

        let (service, socket) = LspService::new(Self::new);

        Server::new(stdin, stdout, socket).serve(service).await;
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
        let uri = params.text_document.uri.clone();
        let text = params.text_document.text.clone();
        let version = params.text_document.version;

        // Create document with rope-based storage
        let document = Document::new(text.clone(), version);

        {
            let mut docs = self.documents.lock().expect("mutex poisoned");
            docs.insert(uri.clone(), document);
        }

        // Analyze using query database
        let diagnostics = {
            let mut analysis = self.analysis.lock().expect("mutex poisoned");
            analysis.analyze(&text, &uri)
        };

        // Publish diagnostics to client
        self.client
            .publish_diagnostics(uri, diagnostics, Some(version))
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        let version = params.text_document.version;

        // Apply incremental changes to document
        let text = {
            let mut docs = self.documents.lock().expect("mutex poisoned");
            if let Some(doc) = docs.get_mut(&uri) {
                for change in params.content_changes {
                    doc.apply_change(change, version);
                }
                doc.text()
            } else {
                return;
            }
        };

        // Re-analyze using query database (incremental)
        let diagnostics = {
            let mut analysis = self.analysis.lock().expect("mutex poisoned");
            analysis.analyze(&text, &uri)
        };

        // Publish diagnostics
        self.client
            .publish_diagnostics(uri, diagnostics, Some(version))
            .await;
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;

        // Remove from document store
        {
            let mut docs = self.documents.lock().expect("mutex poisoned");
            docs.remove(&uri);
        }

        // Notify analysis host
        {
            let mut analysis = self.analysis.lock().expect("mutex poisoned");
            analysis.file_closed(&uri);
        }

        // Clear diagnostics for closed file
        self.client.publish_diagnostics(uri, vec![], None).await;
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        let docs = self.documents.lock().expect("mutex poisoned");
        if let Some(doc) = docs.get(uri) {
            let analysis = self.analysis.lock().expect("mutex poisoned");
            let completions = analysis.completions(doc, position);
            Ok(Some(CompletionResponse::Array(completions)))
        } else {
            Ok(None)
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let docs = self.documents.lock().expect("mutex poisoned");
        if let Some(doc) = docs.get(uri) {
            let analysis = self.analysis.lock().expect("mutex poisoned");
            Ok(analysis.hover(doc, position, uri))
        } else {
            Ok(None)
        }
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let docs = self.documents.lock().expect("mutex poisoned");
        if let Some(doc) = docs.get(uri) {
            let analysis = self.analysis.lock().expect("mutex poisoned");
            Ok(analysis.goto_definition(doc, position, uri))
        } else {
            Ok(None)
        }
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        let docs = self.documents.lock().expect("mutex poisoned");
        if let Some(doc) = docs.get(uri) {
            let analysis = self.analysis.lock().expect("mutex poisoned");
            Ok(analysis.find_references(doc, position, uri))
        } else {
            Ok(None)
        }
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;

        let docs = self.documents.lock().expect("mutex poisoned");
        if let Some(doc) = docs.get(uri) {
            let analysis = self.analysis.lock().expect("mutex poisoned");
            let symbols = analysis.document_symbols(doc, uri);
            Ok(Some(DocumentSymbolResponse::Nested(symbols)))
        } else {
            Ok(None)
        }
    }

    async fn rename(&self, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let new_name = params.new_name;

        let docs = self.documents.lock().expect("mutex poisoned");
        if let Some(doc) = docs.get(uri) {
            let analysis = self.analysis.lock().expect("mutex poisoned");
            Ok(analysis.rename(doc, position, &new_name, uri))
        } else {
            Ok(None)
        }
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = &params.text_document.uri;

        let docs = self.documents.lock().expect("mutex poisoned");
        if let Some(doc) = docs.get(uri) {
            let analysis = self.analysis.lock().expect("mutex poisoned");
            let tokens = analysis.semantic_tokens(doc);
            Ok(Some(SemanticTokensResult::Tokens(tokens)))
        } else {
            Ok(None)
        }
    }

    async fn code_action(&self, params: CodeActionParams) -> Result<Option<CodeActionResponse>> {
        let uri = &params.text_document.uri;
        let range = params.range;
        let diagnostics = &params.context.diagnostics;

        let docs = self.documents.lock().expect("mutex poisoned");
        if let Some(doc) = docs.get(uri) {
            let analysis = self.analysis.lock().expect("mutex poisoned");
            let actions = analysis.code_actions(doc, range, diagnostics, uri);
            Ok(Some(actions))
        } else {
            Ok(None)
        }
    }

    async fn formatting(&self, params: DocumentFormattingParams) -> Result<Option<Vec<TextEdit>>> {
        let uri = &params.text_document.uri;

        let docs = self.documents.lock().expect("mutex poisoned");
        if let Some(doc) = docs.get(uri) {
            let analysis = self.analysis.lock().expect("mutex poisoned");
            Ok(analysis.format(doc))
        } else {
            Ok(None)
        }
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let uri = &params.text_document.uri;
        let range = params.range;

        let docs = self.documents.lock().expect("mutex poisoned");
        if let Some(doc) = docs.get(uri) {
            let analysis = self.analysis.lock().expect("mutex poisoned");
            let hints = analysis.inlay_hints(doc, range, uri);
            Ok(Some(hints))
        } else {
            Ok(None)
        }
    }

    async fn inlay_hint_resolve(&self, params: InlayHint) -> Result<InlayHint> {
        Ok(params)
    }
}

// Note: All document manipulation and analysis is delegated to AnalysisHost

// Note: Default cannot be implemented for SounioLspServer as it requires a Client

//! LSP Server implementation
//!
//! Main server struct implementing the Language Server Protocol.

use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

use super::analysis::AnalysisHost;
use super::document::Document;
use super::semantic_tokens::{semantic_token_modifiers, semantic_token_types};
use super::workspace::Workspace;

/// The Sounio Language Server
pub struct SounioLanguageServer {
    /// LSP client for sending notifications
    client: Client,

    /// Open documents (thread-safe)
    documents: DashMap<Url, Document>,

    /// Analysis host for semantic analysis
    analysis: Arc<RwLock<AnalysisHost>>,

    /// Multi-file workspace manager
    workspace: Arc<RwLock<Workspace>>,
}

impl SounioLanguageServer {
    /// Create a new language server instance
    pub fn new(client: Client) -> Self {
        let workspace = Arc::new(RwLock::new(Workspace::new()));
        let analysis = Arc::new(RwLock::new(AnalysisHost::with_workspace(workspace.clone())));

        Self {
            client,
            documents: DashMap::new(),
            analysis,
            workspace,
        }
    }

    /// Get server capabilities
    fn server_capabilities() -> ServerCapabilities {
        ServerCapabilities {
            // Text document sync
            text_document_sync: Some(TextDocumentSyncCapability::Options(
                TextDocumentSyncOptions {
                    open_close: Some(true),
                    change: Some(TextDocumentSyncKind::INCREMENTAL),
                    save: Some(TextDocumentSyncSaveOptions::SaveOptions(SaveOptions {
                        include_text: Some(true),
                    })),
                    ..Default::default()
                },
            )),

            // Hover
            hover_provider: Some(HoverProviderCapability::Simple(true)),

            // Completion
            completion_provider: Some(CompletionOptions {
                trigger_characters: Some(vec![
                    ".".to_string(),
                    ":".to_string(),
                    "<".to_string(),
                    "_".to_string(),
                ]),
                resolve_provider: Some(true),
                ..Default::default()
            }),

            // Go to definition
            definition_provider: Some(OneOf::Left(true)),

            // Find references
            references_provider: Some(OneOf::Left(true)),

            // Document symbols
            document_symbol_provider: Some(OneOf::Left(true)),

            // Workspace symbols
            workspace_symbol_provider: Some(OneOf::Left(true)),

            // Rename
            rename_provider: Some(OneOf::Right(RenameOptions {
                prepare_provider: Some(true),
                work_done_progress_options: Default::default(),
            })),

            // Code actions
            code_action_provider: Some(CodeActionProviderCapability::Options(CodeActionOptions {
                code_action_kinds: Some(vec![CodeActionKind::QUICKFIX, CodeActionKind::REFACTOR]),
                ..Default::default()
            })),

            // Semantic tokens
            semantic_tokens_provider: Some(
                SemanticTokensServerCapabilities::SemanticTokensOptions(SemanticTokensOptions {
                    legend: SemanticTokensLegend {
                        token_types: semantic_token_types(),
                        token_modifiers: semantic_token_modifiers(),
                    },
                    full: Some(SemanticTokensFullOptions::Bool(true)),
                    range: Some(true),
                    ..Default::default()
                }),
            ),

            // Inlay hints
            inlay_hint_provider: Some(OneOf::Left(true)),

            // Signature help
            signature_help_provider: Some(SignatureHelpOptions {
                trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
                retrigger_characters: Some(vec![",".to_string()]),
                ..Default::default()
            }),

            // Folding
            folding_range_provider: Some(FoldingRangeProviderCapability::Simple(true)),

            // Formatting
            document_formatting_provider: Some(OneOf::Left(true)),

            ..Default::default()
        }
    }

    /// Analyze a document and publish diagnostics
    async fn analyze_document(&self, uri: &Url) {
        if let Some(doc) = self.documents.get(uri) {
            let source = doc.text();
            let version = doc.version();

            // Update workspace index
            {
                let mut workspace = self.workspace.write().await;
                workspace.update_file(uri, source.clone());
            }

            // Run analysis
            let mut analysis = self.analysis.write().await;
            let diagnostics = analysis.analyze(&source, uri);

            self.client
                .publish_diagnostics(uri.clone(), diagnostics, Some(version))
                .await;
        }
    }

    /// Initialize workspace with root URI
    async fn initialize_workspace(&self, root_uri: Option<Url>) {
        let mut workspace = self.workspace.write().await;
        workspace.initialize(root_uri);

        // Scan for files
        let files = workspace.scan_workspace();

        // Log discovered files
        drop(workspace);

        self.client
            .log_message(
                MessageType::INFO,
                format!("Discovered {} source files in workspace", files.len()),
            )
            .await;

        // Index each file (in background)
        for path in files {
            if let Ok(uri) = Url::from_file_path(&path) {
                if let Ok(source) = std::fs::read_to_string(&path) {
                    let mut workspace = self.workspace.write().await;
                    workspace.update_file(&uri, source);
                }
            }
        }
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for SounioLanguageServer {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        // Extract root URI from params
        let root_uri = params
            .root_uri
            .or_else(|| params.root_path.as_ref().and_then(|p| Url::from_file_path(p).ok()));

        // Initialize workspace with root URI
        self.initialize_workspace(root_uri).await;

        Ok(InitializeResult {
            capabilities: Self::server_capabilities(),
            server_info: Some(ServerInfo {
                name: "sounio-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _params: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Sounio LSP initialized with workspace support")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    // === Document Synchronization ===

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let text = params.text_document.text;
        let version = params.text_document.version;

        let doc = Document::new(text, version);
        self.documents.insert(uri.clone(), doc);

        self.analyze_document(&uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;

        if let Some(mut doc) = self.documents.get_mut(&uri) {
            for change in params.content_changes {
                doc.apply_change(change, version);
            }
        }

        self.analyze_document(&uri).await;
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        self.analyze_document(&params.text_document.uri).await;
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        self.documents.remove(&params.text_document.uri);
        // Clear diagnostics for closed document
        self.client
            .publish_diagnostics(params.text_document.uri, vec![], None)
            .await;
    }

    // === Hover ===

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        if let Some(doc) = self.documents.get(uri) {
            let analysis = self.analysis.read().await;
            return Ok(analysis.hover(&doc, position, uri));
        }

        Ok(None)
    }

    // === Go to Definition ===

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        if let Some(doc) = self.documents.get(uri) {
            let analysis = self.analysis.read().await;
            // Use cross-file definition lookup
            return Ok(analysis.goto_definition_cross_file(&doc, position, uri).await);
        }

        Ok(None)
    }

    // === Find References ===

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let include_declaration = params.context.include_declaration;

        if let Some(doc) = self.documents.get(uri) {
            let analysis = self.analysis.read().await;
            // Use cross-file references lookup
            return Ok(analysis.find_references_cross_file(&doc, position, uri, include_declaration).await);
        }

        Ok(None)
    }

    // === Completion ===

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        if let Some(doc) = self.documents.get(uri) {
            let analysis = self.analysis.read().await;
            let items = analysis.completions(&doc, position);
            return Ok(Some(CompletionResponse::Array(items)));
        }

        Ok(None)
    }

    // === Semantic Tokens ===

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = &params.text_document.uri;

        if let Some(doc) = self.documents.get(uri) {
            let analysis = self.analysis.read().await;
            let tokens = analysis.semantic_tokens(&doc);
            return Ok(Some(SemanticTokensResult::Tokens(tokens)));
        }

        Ok(None)
    }

    // === Document Symbols ===

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;

        if let Some(doc) = self.documents.get(uri) {
            let analysis = self.analysis.read().await;
            let symbols = analysis.document_symbols(&doc, uri);
            return Ok(Some(DocumentSymbolResponse::Nested(symbols)));
        }

        Ok(None)
    }

    // === Signature Help ===

    async fn signature_help(&self, params: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        if let Some(doc) = self.documents.get(uri) {
            let analysis = self.analysis.read().await;
            return Ok(analysis.signature_help(&doc, position));
        }

        Ok(None)
    }

    // === Rename ===

    async fn rename(&self, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let new_name = &params.new_name;

        if let Some(doc) = self.documents.get(uri) {
            let analysis = self.analysis.read().await;
            return Ok(analysis.rename(&doc, position, new_name, uri));
        }

        Ok(None)
    }

    // === Code Actions ===

    async fn code_action(&self, params: CodeActionParams) -> Result<Option<CodeActionResponse>> {
        let uri = &params.text_document.uri;
        let range = params.range;
        let diagnostics = &params.context.diagnostics;

        if let Some(doc) = self.documents.get(uri) {
            let analysis = self.analysis.read().await;
            let actions = analysis.code_actions(&doc, range, diagnostics, uri);
            return Ok(Some(actions));
        }

        Ok(None)
    }

    // === Inlay Hints ===

    async fn inlay_hint(&self, params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let uri = &params.text_document.uri;
        let range = params.range;

        if let Some(doc) = self.documents.get(uri) {
            let analysis = self.analysis.read().await;
            return Ok(Some(analysis.inlay_hints(&doc, range)));
        }

        Ok(None)
    }

    // === Formatting ===

    async fn formatting(&self, params: DocumentFormattingParams) -> Result<Option<Vec<TextEdit>>> {
        let uri = &params.text_document.uri;

        if let Some(doc) = self.documents.get(uri) {
            let analysis = self.analysis.read().await;
            return Ok(analysis.format(&doc));
        }

        Ok(None)
    }

    // === Folding Ranges ===

    async fn folding_range(&self, params: FoldingRangeParams) -> Result<Option<Vec<FoldingRange>>> {
        let uri = &params.text_document.uri;

        if let Some(doc) = self.documents.get(uri) {
            let analysis = self.analysis.read().await;
            return Ok(Some(analysis.folding_ranges(&doc)));
        }

        Ok(None)
    }

    // === Workspace Symbols ===

    async fn symbol(
        &self,
        params: WorkspaceSymbolParams,
    ) -> Result<Option<Vec<SymbolInformation>>> {
        let query = &params.query;
        let analysis = self.analysis.read().await;
        let symbols = analysis.workspace_symbols(query).await;

        if symbols.is_empty() {
            Ok(None)
        } else {
            Ok(Some(symbols))
        }
    }
}

// Semantic token types and modifiers are now defined in semantic_tokens.rs

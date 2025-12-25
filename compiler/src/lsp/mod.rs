//! Language Server Protocol implementation for Sounio
//!
//! Provides IDE features:
//! - Real-time diagnostics
//! - Hover information
//! - Go to definition
//! - Find references
//! - Code completion
//! - Semantic highlighting
//!
//! # Architecture
//!
//! The LSP server is built on `tower-lsp` and provides:
//!
//! - `SounioLanguageServer` - Main server struct implementing LSP protocol
//! - `Document` - In-memory document representation using rope data structure
//! - `AnalysisHost` - Manages semantic analysis and caching
//! - Feature providers for hover, completion, definitions, etc.
//!
//! # Usage
//!
//! ```bash
//! # Build with LSP feature
//! cargo build --features lsp
//!
//! # Run the LSP server
//! sounio-lsp --stdio
//! ```
//!
//! # References
//!
//! - LSP Specification: <https://microsoft.github.io/language-server-protocol/>
//! - tower-lsp: <https://docs.rs/tower-lsp/>

pub mod analysis;
pub mod code_actions;
pub mod completion;
pub mod definition;
pub mod diagnostics;
pub mod document;
pub mod epistemic;
pub mod hover;
pub mod inlay_hints;
pub mod references;
pub mod semantic_tokens;
pub mod server;

pub use server::SounioLanguageServer;

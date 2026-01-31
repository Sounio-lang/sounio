//! Sounio Language Server Protocol Implementation
//!
//! Provides LSP server for IDEs to offer advanced Sounio features.
//!
//! # Architecture
//!
//! The LSP is built on a demand-driven incremental query system:
//!
//! ```text
//! ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
//! │   Editor    │────▶│  LSP Server  │────▶│ QueryBridge │
//! └─────────────┘     └──────────────┘     └──────┬──────┘
//!                                                 │
//!                     ┌───────────────────────────┼───────────────────────────┐
//!                     │                           ▼                           │
//!                     │  ┌─────────────────────────────────────────────────┐  │
//!                     │  │              QueryDatabase                       │  │
//!                     │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐          │  │
//!                     │  │  │FileText │─▶│  AST    │─▶│TypedAST │          │  │
//!                     │  │  └─────────┘  └─────────┘  └────┬────┘          │  │
//!                     │  │                                 │               │  │
//!                     │  │          ┌──────────┬──────────┬┴─────────┐     │  │
//!                     │  │          ▼          ▼          ▼          ▼     │  │
//!                     │  │     ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐  │  │
//!                     │  │     │Effects │ │ Units  │ │Epistemic│ │Refine │  │  │
//!                     │  │     └────────┘ └────────┘ └────────┘ └───────┘  │  │
//!                     │  └─────────────────────────────────────────────────┘  │
//!                     │                      Compiler                          │
//!                     └───────────────────────────────────────────────────────┘
//! ```

// Core modules
pub mod analysis;
pub mod code_actions;
pub mod completion;
pub mod definition;
pub mod diagnostics;
pub mod document;
pub mod epistemic;
pub mod hover;
pub mod inlay_hints;
pub mod query_bridge;
pub mod references;
pub mod rich_info;
pub mod semantic_tokens;
pub mod server;
pub mod workspace;

// Re-export key types
pub use analysis::AnalysisHost;
pub use document::Document;
pub use query_bridge::{
    EffectInfo, EpistemicInfo, LspAnalysisResult, LspQueryDatabase, RefinementInfo, UnitInfo,
};
pub use server::SounioLspServer;

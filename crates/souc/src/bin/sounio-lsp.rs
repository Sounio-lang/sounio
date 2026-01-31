//! Sounio Language Server Protocol Binary
//!
//! Entry point for LSP server that can be run as standalone executable.

use sounio::lsp::server::SounioLspServer;
use tower_lsp::{LspService, Server};

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    eprintln!("Starting Sounio LSP Server...");

    // Create LSP service with factory function
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(SounioLspServer::new);

    // Run server
    Server::new(stdin, stdout, socket).serve(service).await;

    eprintln!("Sounio LSP Server stopped");
}

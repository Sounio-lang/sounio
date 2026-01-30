//! Sounio Language Server Protocol Binary
//!
//! Entry point for LSP server that can be run as standalone executable.

use sounio::lsp::server::SounioLspServer;
use tower_lsp::Server;
use tower_lsp::LspService;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    eprintln!("Starting Sounio LSP Server...");

    // Create LSP service
    let (service, socket) = LspService::new(SounioLspServer::new());
    
    // Run server
    Server::new(socket).serve(service).await?;
    
    eprintln!("Sounio LSP Server stopped");
    
    Ok(())
}
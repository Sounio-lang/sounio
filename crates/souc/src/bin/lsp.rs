//! Sounio Language Server binary
//!
//! Starts the LSP server for IDE integration.
//!
//! # Usage
//!
//! ```bash
//! # Standard I/O mode (for editors)
//! sounio-lsp --stdio
//!
//! # Show version
//! sounio-lsp --version
//!
//! # Show help
//! sounio-lsp --help
//! ```

use tower_lsp::{LspService, Server};

use sounio::lsp::SounioLanguageServer;

/// Command line arguments
#[derive(Debug)]
struct Args {
    stdio: bool,
    version: bool,
    help: bool,
}

impl Args {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();

        Self {
            stdio: args.contains(&"--stdio".to_string()),
            version: args.contains(&"--version".to_string()) || args.contains(&"-V".to_string()),
            help: args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()),
        }
    }
}

fn print_help() {
    eprintln!(
        r#"Sounio Language Server

USAGE:
    sounio-lsp [OPTIONS]

OPTIONS:
    --stdio         Use stdio for communication (required for editors)
    --version, -V   Print version information
    --help, -h      Print this help message

DESCRIPTION:
    The Sounio Language Server provides IDE features through the
    Language Server Protocol (LSP). It communicates with editors via
    standard input/output.

SUPPORTED FEATURES:
    - Real-time diagnostics (syntax and type errors)
    - Hover information (types, documentation)
    - Go to definition
    - Find all references
    - Code completion
    - Semantic syntax highlighting
    - Signature help
    - Document symbols (outline)
    - Code actions (quick fixes)
    - Inlay hints (type annotations)
    - Rename symbol
    - Folding ranges

EDITOR INTEGRATION:
    VS Code:  Install the 'sounio-vscode' extension
    Neovim:   Configure with nvim-lspconfig
    Emacs:    Configure with lsp-mode or eglot
    Helix:    Add to languages.toml

EXAMPLES:
    # Start server for VS Code
    sounio-lsp --stdio

    # Check version
    sounio-lsp --version
"#
    );
}

fn print_version() {
    eprintln!(
        "sounio-lsp {} (Sounio Language Server)",
        env!("CARGO_PKG_VERSION")
    );
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    if args.help {
        print_help();
        return;
    }

    if args.version {
        print_version();
        return;
    }

    if args.stdio {
        // Standard I/O mode (for editors)
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();

        let (service, socket) = LspService::new(SounioLanguageServer::new);

        Server::new(stdin, stdout, socket).serve(service).await;
    } else {
        // No arguments - print usage
        eprintln!("Sounio Language Server v{}", env!("CARGO_PKG_VERSION"));
        eprintln!();
        eprintln!("This server communicates via Language Server Protocol over stdin/stdout.");
        eprintln!();
        eprintln!("Usage: sounio-lsp --stdio");
        eprintln!();
        eprintln!("For more information, run: sounio-lsp --help");
    }
}

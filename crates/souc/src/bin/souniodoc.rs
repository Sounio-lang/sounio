//! Sounio Documentation Generator
//!
//! Generate rustdoc-style HTML documentation from Sounio source code.
//!
//! # Commands
//!
//! - `generate`: Generate HTML documentation
//! - `check`: Validate documentation coverage
//! - `preview`: Serve documentation locally (TODO)
//!
//! # Examples
//!
//! Generate documentation for the stdlib:
//! ```bash
//! souniodoc generate stdlib --output target/doc
//! ```
//!
//! Check documentation coverage:
//! ```bash
//! souniodoc check --min-coverage 80
//! ```

use clap::{Parser as ClapParser, Subcommand};
use miette::{IntoDiagnostic, Result};
use sounio::doc::{DocExtractor, HtmlRenderer};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(ClapParser)]
#[command(
    name = "souniodoc",
    about = "Sounio documentation generator",
    version,
    author
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate HTML documentation
    Generate {
        /// Root source file or directory
        #[arg(default_value = "stdlib")]
        path: PathBuf,

        /// Output directory
        #[arg(short, long, default_value = "target/doc")]
        output: PathBuf,

        /// Include private items
        #[arg(long)]
        document_private: bool,

        /// Include source code pages
        #[arg(long, default_value_t = true)]
        include_source: bool,

        /// Enable dark mode
        #[arg(long, default_value_t = true)]
        dark_mode: bool,
    },

    /// Check documentation coverage
    Check {
        /// Minimum coverage percentage
        #[arg(short, long, default_value = "80")]
        min_coverage: u8,

        /// Fail on undocumented public items
        #[arg(long)]
        strict: bool,

        /// Path to check (defaults to stdlib)
        #[arg(default_value = "stdlib")]
        path: PathBuf,
    },

    /// Generate markdown reference (for /docs)
    GenerateMarkdown {
        /// Path to generate from
        #[arg(default_value = "stdlib")]
        path: PathBuf,

        /// Output markdown file
        #[arg(short, long, default_value = "docs/STDLIB_REFERENCE.md")]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    // Set up tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            path,
            output,
            document_private,
            include_source,
            dark_mode,
        } => {
            generate_docs(&path, &output, document_private, include_source, dark_mode)?;
        }
        Commands::Check {
            min_coverage,
            strict,
            path,
        } => {
            check_coverage(&path, min_coverage, strict)?;
        }
        Commands::GenerateMarkdown { path, output } => {
            generate_markdown(&path, &output)?;
        }
    }

    Ok(())
}

/// Generate HTML documentation
fn generate_docs(
    source_path: &Path,
    output_dir: &Path,
    document_private: bool,
    include_source: bool,
    dark_mode: bool,
) -> Result<()> {
    tracing::info!(
        "Generating documentation from {:?} to {:?}",
        source_path,
        output_dir
    );

    // Determine crate name and version
    let crate_name = source_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Create extractor
    let extractor = DocExtractor::new(&crate_name, "0.1.0")
        .document_private(document_private)
        .document_hidden(false);

    // Find all .sio files
    let source_files = find_sounio_files(source_path)?;

    if source_files.is_empty() {
        tracing::warn!("No .sio files found in {:?}", source_path);
        return Ok(());
    }

    tracing::info!("Found {} source files", source_files.len());

    // Parse all files and extract documentation
    let mut crate_doc = None;

    for file_path in &source_files {
        tracing::debug!("Processing {:?}", file_path);

        // Read source
        let source = fs::read_to_string(file_path).into_diagnostic()?;

        // Lex
        let tokens = match sounio::lexer::lex(&source) {
            Ok(tokens) => tokens,
            Err(e) => {
                tracing::error!("Failed to lex {:?}: {:?}", file_path, e);
                continue;
            }
        };

        // Parse
        let ast = match sounio::parser::parse(&tokens, &source) {
            Ok(ast) => ast,
            Err(e) => {
                tracing::error!("Failed to parse {:?}: {:?}", file_path, e);
                continue;
            }
        };

        // Extract documentation
        let doc = extractor.extract(&ast);

        // Merge into crate doc (first file becomes root)
        if crate_doc.is_none() {
            crate_doc = Some(doc);
        } else {
            // TODO: Implement merging of multiple modules
            tracing::warn!("Multi-file documentation merging not yet implemented");
        }
    }

    let crate_doc = crate_doc
        .ok_or_else(|| miette::miette!("No documentation extracted - all files failed to parse"))?;

    // Debug: Print what was extracted
    tracing::info!(
        "Extracted {} functions, {} types, {} traits",
        crate_doc.root_module.functions.len(),
        crate_doc.root_module.types.len(),
        crate_doc.root_module.traits.len()
    );

    // Generate HTML
    tracing::info!("Generating HTML documentation...");
    let renderer = HtmlRenderer::new(crate_doc, output_dir.to_path_buf())
        .with_dark_mode(dark_mode)
        .with_source(include_source);

    renderer.generate().into_diagnostic()?;

    tracing::info!("Documentation generated at {:?}/index.html", output_dir);

    Ok(())
}

/// Check documentation coverage
fn check_coverage(source_path: &Path, min_coverage: u8, strict: bool) -> Result<()> {
    tracing::info!("Checking documentation coverage in {:?}", source_path);

    let source_files = find_sounio_files(source_path)?;

    if source_files.is_empty() {
        tracing::warn!("No .sio files found");
        return Ok(());
    }

    let mut total_items = 0;
    let mut documented_items = 0;
    let mut undocumented: Vec<String> = Vec::new();

    for file_path in &source_files {
        let source = fs::read_to_string(file_path).into_diagnostic()?;

        let tokens = match sounio::lexer::lex(&source) {
            Ok(tokens) => tokens,
            Err(_) => continue,
        };

        let ast = match sounio::parser::parse(&tokens, &source) {
            Ok(ast) => ast,
            Err(_) => continue,
        };

        // Count public items
        for item in &ast.items {
            use sounio::ast::Item;

            let (is_public, has_doc, name) = match item {
                Item::Function(f) => {
                    let is_pub = matches!(f.visibility, sounio::ast::Visibility::Public);
                    // TODO: Check for doc comments via attrs
                    let has_doc = false; // Placeholder
                    (is_pub, has_doc, f.name.clone())
                }
                Item::Struct(s) => {
                    let is_pub = matches!(s.visibility, sounio::ast::Visibility::Public);
                    let has_doc = false;
                    (is_pub, has_doc, s.name.clone())
                }
                Item::Enum(e) => {
                    let is_pub = matches!(e.visibility, sounio::ast::Visibility::Public);
                    let has_doc = false;
                    (is_pub, has_doc, e.name.clone())
                }
                Item::Trait(t) => {
                    let is_pub = matches!(t.visibility, sounio::ast::Visibility::Public);
                    let has_doc = false;
                    (is_pub, has_doc, t.name.clone())
                }
                _ => continue,
            };

            if is_public {
                total_items += 1;
                if has_doc {
                    documented_items += 1;
                } else {
                    undocumented.push(format!("{}:{}", file_path.display(), name));
                }
            }
        }
    }

    let coverage = if total_items > 0 {
        (documented_items * 100) / total_items
    } else {
        100
    };

    tracing::info!(
        "Documentation coverage: {}/{} items documented ({}%)",
        documented_items,
        total_items,
        coverage
    );

    if !undocumented.is_empty() {
        tracing::warn!("Undocumented public items:");
        for item in &undocumented {
            tracing::warn!("  - {}", item);
        }
    }

    if coverage < min_coverage as usize {
        let msg = format!(
            "Documentation coverage {}% is below minimum {}%",
            coverage, min_coverage
        );
        if strict {
            return Err(miette::miette!(msg));
        } else {
            tracing::warn!("{}", msg);
        }
    }

    Ok(())
}

/// Generate markdown reference
fn generate_markdown(source_path: &Path, output_path: &Path) -> Result<()> {
    tracing::info!(
        "Generating markdown reference from {:?} to {:?}",
        source_path,
        output_path
    );

    // Similar to generate_docs but output markdown instead
    let crate_name = source_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let extractor = DocExtractor::new(&crate_name, "0.1.0");
    let source_files = find_sounio_files(source_path)?;

    if source_files.is_empty() {
        return Err(miette::miette!("No .sio files found"));
    }

    // Parse and extract
    let source = fs::read_to_string(&source_files[0]).into_diagnostic()?;
    let tokens = sounio::lexer::lex(&source)?;
    let ast = sounio::parser::parse(&tokens, &source)?;
    let crate_doc = extractor.extract(&ast);

    // Generate markdown (simple version)
    let mut markdown = String::from("# Sounio Standard Library Reference\n\n");
    markdown.push_str(&format!(
        "**Auto-generated from source code** ({})\n\n",
        chrono::Utc::now().format("%Y-%m-%d")
    ));

    // Add module documentation
    markdown.push_str(&format!("## {}\n\n", crate_doc.root_module.name));

    if let Some(doc) = &crate_doc.root_module.doc {
        markdown.push_str(doc);
        markdown.push_str("\n\n");
    }

    // List functions
    if !crate_doc.root_module.functions.is_empty() {
        markdown.push_str("### Functions\n\n");
        for func in &crate_doc.root_module.functions {
            markdown.push_str(&format!("#### `{}`\n\n", func.signature));
            if let Some(doc) = &func.doc {
                markdown.push_str(doc);
                markdown.push_str("\n\n");
            }
        }
    }

    // Write output
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).into_diagnostic()?;
    }
    fs::write(output_path, markdown).into_diagnostic()?;

    tracing::info!("Markdown reference generated at {:?}", output_path);

    Ok(())
}

/// Find all .sio files in a directory
fn find_sounio_files(path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if path.is_file() {
        if path.extension().and_then(|s| s.to_str()) == Some("sio") {
            files.push(path.to_path_buf());
        }
        return Ok(files);
    }

    // Walk directory
    for entry in WalkDir::new(path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("sio") {
            files.push(path.to_path_buf());
        }
    }

    Ok(files)
}

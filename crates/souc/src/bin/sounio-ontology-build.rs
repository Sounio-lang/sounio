//! Ontology Store Builder CLI
//!
//! ETL tool for building optimized ontology stores from source files.
//!
//! # Usage
//!
//! ```bash
//! # Build from a single file
//! dc-ontology-build --input chebi.obo --output ./store
//!
//! # Build from multiple files
//! dc-ontology-build --input *.obo --input *.owl --output ./store
//!
//! # Build with specific options
//! dc-ontology-build --input sources/ --output ./store --parallel 8 --no-compression
//! ```

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use sounio::ontology::build::{
    BuildConfig, BuildError, BuildResult, OntologyBuilder, ParserRegistry,
};

/// Ontology Store Builder - ETL pipeline for building optimized ontology stores
#[derive(Parser, Debug)]
#[command(name = "dc-ontology-build")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input files or directories to process
    #[arg(short, long, required_unless_present = "list_formats", num_args = 1..)]
    input: Vec<PathBuf>,

    /// Output directory for the ontology store
    #[arg(short, long, default_value = "./ontology-store")]
    output: PathBuf,

    /// Number of parallel workers (default: number of CPUs)
    #[arg(short, long)]
    parallel: Option<usize>,

    /// Disable prefix compression (faster build, larger output)
    #[arg(long)]
    no_compression: bool,

    /// Memory limit in megabytes
    #[arg(long, default_value = "1024")]
    memory_limit: usize,

    /// Minimum prefix frequency for compression
    #[arg(long, default_value = "100")]
    min_prefix_freq: usize,

    /// Output format
    #[arg(long, value_enum, default_value = "binary")]
    format: OutputFormat,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// List supported file formats and exit
    #[arg(long)]
    list_formats: bool,

    /// Dry run - parse files but don't write output
    #[arg(long)]
    dry_run: bool,
}

#[derive(ValueEnum, Clone, Debug)]
enum OutputFormat {
    /// Binary format (fastest load time)
    Binary,
    /// JSON format (human-readable)
    Json,
}

fn main() {
    let cli = Cli::parse();

    if cli.list_formats {
        list_supported_formats();
        return;
    }

    if let Err(e) = run(cli) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), BuildError> {
    let start = Instant::now();

    // Collect all input files
    let files = collect_input_files(&cli.input)?;

    if files.is_empty() {
        eprintln!("No input files found");
        return Ok(());
    }

    if cli.verbose {
        println!("Ontology Store Builder v{}", env!("CARGO_PKG_VERSION"));
        println!("----------------------------------------");
        println!("Input files: {}", files.len());
        println!("Output: {:?}", cli.output);
        println!("Compression: {}", !cli.no_compression);
        println!("Memory limit: {} MB", cli.memory_limit);
        println!();
    }

    // Configure the builder
    let config = BuildConfig {
        parallelism: cli.parallel.unwrap_or_else(num_cpus::get),
        memory_limit: cli.memory_limit * 1024 * 1024,
        prefix_compression: !cli.no_compression,
        min_prefix_frequency: cli.min_prefix_freq,
        output_dir: cli.output.clone(),
    };

    let mut builder = OntologyBuilder::with_config(config);

    // Build the store
    if cli.verbose {
        println!("Processing files...");
    }

    let file_refs: Vec<&PathBuf> = files.iter().collect();
    let result = builder.build(&file_refs)?;

    // Print results
    print_results(&result, &cli, start.elapsed());

    if cli.dry_run {
        println!("\n[Dry run - no files written]");
    }

    Ok(())
}

fn collect_input_files(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, BuildError> {
    let registry = ParserRegistry::new();
    let mut files = Vec::new();

    for input in inputs {
        if input.is_file() {
            if registry.parser_for(input).is_some() {
                files.push(input.clone());
            }
        } else if input.is_dir() {
            // Recursively find all supported files
            for entry in walkdir::WalkDir::new(input)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if path.is_file() && registry.parser_for(path).is_some() {
                    files.push(path.to_path_buf());
                }
            }
        } else {
            // Try glob pattern
            if let Some(pattern) = input.to_str() {
                for entry in glob::glob(pattern).map_err(|e| BuildError::Io(e.to_string()))? {
                    match entry {
                        Ok(path) => {
                            if path.is_file() && registry.parser_for(&path).is_some() {
                                files.push(path);
                            }
                        }
                        Err(e) => {
                            eprintln!("Warning: {}", e);
                        }
                    }
                }
            }
        }
    }

    Ok(files)
}

fn print_results(result: &BuildResult, cli: &Cli, duration: std::time::Duration) {
    println!("\nBuild Complete");
    println!("==============");
    println!();
    println!("Statistics:");
    println!("  Files processed:    {}", result.stats.files_processed);
    println!("  Terms parsed:       {}", result.stats.terms_parsed);
    println!("  Relations parsed:   {}", result.stats.relations_parsed);
    println!("  Strings interned:   {}", result.stats.strings_interned);
    println!();

    if !cli.no_compression {
        let saved_mb = result.stats.bytes_saved as f64 / (1024.0 * 1024.0);
        println!("Compression:");
        println!("  Bytes saved:        {:.2} MB", saved_mb);
    }

    println!();
    println!("Performance:");
    println!("  Duration:           {:.2}s", duration.as_secs_f64());
    println!(
        "  Terms/second:       {:.0}",
        result.stats.terms_parsed as f64 / duration.as_secs_f64()
    );

    if result.stats.parse_errors > 0 {
        println!();
        println!("Warnings:");
        println!("  Parse errors:       {}", result.stats.parse_errors);
    }

    if cli.verbose && !result.warnings.is_empty() {
        println!();
        println!("Details:");
        for (i, warning) in result.warnings.iter().take(10).enumerate() {
            println!("  {}. {}", i + 1, warning);
        }
        if result.warnings.len() > 10 {
            println!("  ... and {} more", result.warnings.len() - 10);
        }
    }

    println!();
    println!("Output: {:?}", result.output_path);
}

fn list_supported_formats() {
    println!("Supported File Formats");
    println!("======================");
    println!();

    let registry = ParserRegistry::new();

    for parser in registry.parsers() {
        println!("{}", parser.name());
        print!("  Extensions: ");
        println!("{}", parser.extensions().join(", "));
        println!();
    }

    println!("Use --input <file> to process files.");
}

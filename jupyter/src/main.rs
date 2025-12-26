//! Sounio Jupyter Kernel Entry Point
//!
//! This is the main executable for the Sounio Jupyter kernel.
//! It reads the connection file provided by Jupyter and starts the kernel.

use sounio_jupyter::{ConnectionInfo, SounioKernel};
use std::env;
use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("sounio_jupyter=info".parse().unwrap()),
        )
        .init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    // Find the connection file
    let connection_file = args.iter().find(|arg| arg.ends_with(".json"));

    let connection_file = match connection_file {
        Some(f) => f,
        None => {
            eprintln!("Usage: sounio-jupyter <connection-file.json>");
            eprintln!();
            eprintln!("The connection file is provided by Jupyter when launching the kernel.");
            eprintln!();
            eprintln!("To install the kernel, run:");
            eprintln!("  python jupyter/install.py");
            eprintln!();
            eprintln!("Then start Jupyter:");
            eprintln!("  jupyter notebook");
            eprintln!("  jupyter lab");
            return ExitCode::FAILURE;
        }
    };

    tracing::info!("Loading connection file: {}", connection_file);

    // Load connection info
    let conn_info = match ConnectionInfo::from_file(connection_file) {
        Ok(info) => info,
        Err(e) => {
            eprintln!("Failed to load connection file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    tracing::info!(
        "Connecting to shell: {}:{}",
        conn_info.ip,
        conn_info.shell_port
    );
    tracing::info!(
        "Connecting to iopub: {}:{}",
        conn_info.ip,
        conn_info.iopub_port
    );
    tracing::info!(
        "Connecting to stdin: {}:{}",
        conn_info.ip,
        conn_info.stdin_port
    );
    tracing::info!(
        "Connecting to control: {}:{}",
        conn_info.ip,
        conn_info.control_port
    );
    tracing::info!(
        "Connecting to heartbeat: {}:{}",
        conn_info.ip,
        conn_info.hb_port
    );

    // Create and run the kernel
    let kernel = SounioKernel::new(conn_info);

    match kernel.run().await {
        Ok(()) => {
            tracing::info!("Kernel shut down cleanly");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Kernel error: {}", e);
            ExitCode::FAILURE
        }
    }
}

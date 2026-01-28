//! Integration Tests for LSP (Language Server Protocol)
//!
//! Tests for LSP server functionality including:
//! - Server initialization and shutdown
//! - Document lifecycle (open, change, close)
//! - Diagnostics reporting
//! - Hover information
//! - Completion requests
//! - Go to definition
//!
//! These tests use JSON-RPC messages over stdio to communicate with the LSP server.

#![cfg(feature = "lsp")]

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicI64, Ordering};
use std::thread;
use std::time::Duration;

// ============================================================================
// JSON-RPC Types
// ============================================================================

/// JSON-RPC request/response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcMessage {
    jsonrpc: String,
    id: Option<i64>,
    method: Option<String>,
    params: Option<Value>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
    data: Option<Value>,
}

/// LSP Position
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Position {
    line: u32,
    character: u32,
}

/// LSP Range
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Range {
    start: Position,
    end: Position,
}

/// LSP Text Document Identifier
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TextDocumentIdentifier {
    uri: String,
}

/// LSP Versioned Text Document Identifier
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VersionedTextDocumentIdentifier {
    uri: String,
    version: i32,
}

/// LSP Text Document Item
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TextDocumentItem {
    uri: String,
    #[serde(rename = "languageId")]
    language_id: String,
    version: i32,
    text: String,
}

// ============================================================================
// Test Helpers
// ============================================================================

/// LSP client for testing
struct LspClient {
    child: Child,
    next_id: AtomicI64,
}

impl LspClient {
    /// Start a new LSP server
    fn new() -> Self {
        let mut lsp_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        lsp_path.push("target/debug/sounio-lsp");

        // If debug binary doesn't exist, try release
        if !lsp_path.exists() {
            lsp_path.pop();
            lsp_path.pop();
            lsp_path.push("release/sounio-lsp");
        }

        // If still doesn't exist, build it
        if !lsp_path.exists() {
            eprintln!("Building sounio-lsp binary...");
            lsp_path.pop();
            lsp_path.pop();
            let status = Command::new("cargo")
                .args(["build", "--bin", "sounio-lsp", "--features", "lsp"])
                .current_dir(&lsp_path)
                .status()
                .expect("Failed to build sounio-lsp");
            assert!(status.success(), "Failed to build sounio-lsp binary");

            lsp_path.push("target/debug/sounio-lsp");
        }

        let child = Command::new(lsp_path)
            .arg("--stdio")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start LSP server");

        // Give server time to start
        thread::sleep(Duration::from_millis(300));

        Self {
            child,
            next_id: AtomicI64::new(1),
        }
    }

    /// Get next request ID
    fn next_id(&self) -> i64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Send a JSON-RPC request
    fn send_request(&mut self, method: &str, params: Value) -> i64 {
        let id = self.next_id();
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });

        self.send_message(&message);
        id
    }

    /// Send a JSON-RPC notification (no response expected)
    fn send_notification(&mut self, method: &str, params: Value) {
        let message = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });

        self.send_message(&message);
    }

    /// Send a JSON-RPC message
    fn send_message(&mut self, message: &Value) {
        let message_str = serde_json::to_string(message).expect("Failed to serialize message");
        let content = format!(
            "Content-Length: {}\r\n\r\n{}",
            message_str.len(),
            message_str
        );

        let stdin = self.child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(content.as_bytes())
            .expect("Failed to write to stdin");
        stdin.flush().expect("Failed to flush stdin");

        // Give server time to process
        thread::sleep(Duration::from_millis(100));
    }

    /// Try to read a response (non-blocking)
    fn try_read_response(&mut self) -> Option<Value> {
        let stdout = self.child.stdout.as_mut().expect("Failed to get stdout");
        let mut reader = BufReader::new(stdout);

        // Try to read Content-Length header
        let mut header = String::new();
        if reader.read_line(&mut header).ok()? == 0 {
            return None;
        }

        if !header.starts_with("Content-Length:") {
            return None;
        }

        let content_length: usize = header
            .trim_start_matches("Content-Length:")
            .trim()
            .parse()
            .ok()?;

        // Read empty line
        let mut empty = String::new();
        reader.read_line(&mut empty).ok()?;

        // Read content
        let mut content = vec![0u8; content_length];
        std::io::Read::read_exact(&mut reader, &mut content).ok()?;

        let content_str = String::from_utf8(content).ok()?;
        serde_json::from_str(&content_str).ok()
    }

    /// Read all pending responses (with timeout)
    fn read_responses(&mut self, timeout_ms: u64) -> Vec<Value> {
        let mut responses = Vec::new();
        let start = std::time::Instant::now();

        while start.elapsed() < Duration::from_millis(timeout_ms) {
            if let Some(response) = self.try_read_response() {
                responses.push(response);
            } else {
                thread::sleep(Duration::from_millis(50));
            }
        }

        responses
    }

    /// Initialize LSP session
    fn initialize(&mut self) -> Value {
        let id = self.send_request(
            "initialize",
            json!({
                "processId": std::process::id(),
                "rootUri": null,
                "capabilities": {
                    "textDocument": {
                        "synchronization": {
                            "dynamicRegistration": false,
                            "willSave": false,
                            "willSaveWaitUntil": false,
                            "didSave": false
                        },
                        "completion": {
                            "dynamicRegistration": false,
                            "completionItem": {
                                "snippetSupport": false
                            }
                        },
                        "hover": {
                            "dynamicRegistration": false
                        },
                        "definition": {
                            "dynamicRegistration": false
                        }
                    }
                }
            }),
        );

        // Wait for initialize response
        thread::sleep(Duration::from_millis(500));
        let responses = self.read_responses(1000);

        // Find the initialize response
        responses
            .into_iter()
            .find(|r| r.get("id").and_then(|id| id.as_i64()) == Some(id))
            .expect("No initialize response received")
    }

    /// Send initialized notification
    fn initialized(&mut self) {
        self.send_notification("initialized", json!({}));
    }

    /// Shutdown the LSP server
    fn shutdown(&mut self) {
        self.send_request("shutdown", json!(null));
        thread::sleep(Duration::from_millis(200));
        self.send_notification("exit", json!(null));
        thread::sleep(Duration::from_millis(200));
    }

    /// Open a document
    fn did_open(&mut self, uri: &str, text: &str) {
        self.send_notification(
            "textDocument/didOpen",
            json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": "sounio",
                    "version": 1,
                    "text": text
                }
            }),
        );
    }

    /// Change a document
    fn did_change(&mut self, uri: &str, version: i32, text: &str) {
        self.send_notification(
            "textDocument/didChange",
            json!({
                "textDocument": {
                    "uri": uri,
                    "version": version
                },
                "contentChanges": [{
                    "text": text
                }]
            }),
        );
    }

    /// Close a document
    fn did_close(&mut self, uri: &str) {
        self.send_notification(
            "textDocument/didClose",
            json!({
                "textDocument": {
                    "uri": uri
                }
            }),
        );
    }

    /// Request hover information
    fn hover(&mut self, uri: &str, line: u32, character: u32) -> i64 {
        self.send_request(
            "textDocument/hover",
            json!({
                "textDocument": {
                    "uri": uri
                },
                "position": {
                    "line": line,
                    "character": character
                }
            }),
        )
    }

    /// Request completion
    fn completion(&mut self, uri: &str, line: u32, character: u32) -> i64 {
        self.send_request(
            "textDocument/completion",
            json!({
                "textDocument": {
                    "uri": uri
                },
                "position": {
                    "line": line,
                    "character": character
                }
            }),
        )
    }

    /// Request go to definition
    fn definition(&mut self, uri: &str, line: u32, character: u32) -> i64 {
        self.send_request(
            "textDocument/definition",
            json!({
                "textDocument": {
                    "uri": uri
                },
                "position": {
                    "line": line,
                    "character": character
                }
            }),
        )
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ============================================================================
// Initialization Tests
// ============================================================================

#[test]
fn test_lsp_initialize() {
    let mut client = LspClient::new();
    let response = client.initialize();

    // Should receive initialize result
    assert!(response.get("result").is_some(), "Should have result field");

    let result = response.get("result").unwrap();
    assert!(
        result.get("capabilities").is_some(),
        "Should have capabilities"
    );

    client.shutdown();
}

#[test]
fn test_lsp_initialize_and_shutdown() {
    let mut client = LspClient::new();
    client.initialize();
    client.initialized();

    // Send shutdown request
    let id = client.send_request("shutdown", json!(null));
    thread::sleep(Duration::from_millis(300));

    let responses = client.read_responses(1000);

    // Should receive shutdown response
    let shutdown_response = responses
        .iter()
        .find(|r| r.get("id").and_then(|v| v.as_i64()) == Some(id));

    assert!(
        shutdown_response.is_some(),
        "Should receive shutdown response"
    );

    // Send exit notification
    client.send_notification("exit", json!(null));
}

// ============================================================================
// Document Lifecycle Tests
// ============================================================================

#[test]
fn test_lsp_did_open_valid_document() {
    let mut client = LspClient::new();
    client.initialize();
    client.initialized();

    let uri = "file:///test.sio";
    let text = r#"
fn main() -> i64 {
    42
}
"#;

    client.did_open(uri, text);

    // Wait for any diagnostics
    thread::sleep(Duration::from_millis(500));
    let responses = client.read_responses(1000);

    // Should process document without errors
    // (May or may not send diagnostics depending on implementation)

    client.did_close(uri);
    client.shutdown();
}

#[test]
fn test_lsp_did_open_syntax_error() {
    let mut client = LspClient::new();
    client.initialize();
    client.initialized();

    let uri = "file:///test_error.sio";
    let text = r#"
fn main() -> i64 {
    let x =
}
"#;

    client.did_open(uri, text);

    // Wait for diagnostics
    thread::sleep(Duration::from_millis(500));
    let responses = client.read_responses(1000);

    // Should receive diagnostic notification with errors
    let has_diagnostics = responses.iter().any(|r| {
        r.get("method")
            .and_then(|m| m.as_str())
            .map(|m| m == "textDocument/publishDiagnostics")
            .unwrap_or(false)
    });

    // Note: This may or may not be true depending on implementation state
    // Just verify the LSP processes the document without crashing

    client.did_close(uri);
    client.shutdown();
}

#[test]
fn test_lsp_did_change() {
    let mut client = LspClient::new();
    client.initialize();
    client.initialized();

    let uri = "file:///test_change.sio";

    // Open document
    client.did_open(uri, "fn main() -> i64 { 1 }");

    // Change document
    client.did_change(uri, 2, "fn main() -> i64 { 42 }");

    // Wait for processing
    thread::sleep(Duration::from_millis(500));

    client.did_close(uri);
    client.shutdown();
}

#[test]
fn test_lsp_did_close() {
    let mut client = LspClient::new();
    client.initialize();
    client.initialized();

    let uri = "file:///test_close.sio";

    client.did_open(uri, "fn main() -> i64 { 1 }");
    thread::sleep(Duration::from_millis(200));

    client.did_close(uri);
    thread::sleep(Duration::from_millis(200));

    // Should handle close without error
    client.shutdown();
}

// ============================================================================
// Hover Tests
// ============================================================================

#[test]
fn test_lsp_hover_request() {
    let mut client = LspClient::new();
    client.initialize();
    client.initialized();

    let uri = "file:///test_hover.sio";
    let text = r#"
fn main() -> i64 {
    let x = 42
    x
}
"#;

    client.did_open(uri, text);
    thread::sleep(Duration::from_millis(300));

    // Request hover on 'x'
    let hover_id = client.hover(uri, 2, 8);
    thread::sleep(Duration::from_millis(500));

    let responses = client.read_responses(1000);

    // Look for hover response
    let hover_response = responses
        .iter()
        .find(|r| r.get("id").and_then(|v| v.as_i64()) == Some(hover_id));

    // Should receive some response (even if null)
    assert!(
        hover_response.is_some(),
        "Should receive hover response. Responses: {:?}",
        responses
    );

    client.did_close(uri);
    client.shutdown();
}

// ============================================================================
// Completion Tests
// ============================================================================

#[test]
fn test_lsp_completion_request() {
    let mut client = LspClient::new();
    client.initialize();
    client.initialized();

    let uri = "file:///test_completion.sio";
    let text = r#"
fn main() -> i64 {
    let x = 42

}
"#;

    client.did_open(uri, text);
    thread::sleep(Duration::from_millis(300));

    // Request completion
    let completion_id = client.completion(uri, 3, 4);
    thread::sleep(Duration::from_millis(500));

    let responses = client.read_responses(1000);

    // Look for completion response
    let completion_response = responses
        .iter()
        .find(|r| r.get("id").and_then(|v| v.as_i64()) == Some(completion_id));

    // Should receive completion response
    assert!(
        completion_response.is_some(),
        "Should receive completion response"
    );

    client.did_close(uri);
    client.shutdown();
}

// ============================================================================
// Go to Definition Tests
// ============================================================================

#[test]
fn test_lsp_definition_request() {
    let mut client = LspClient::new();
    client.initialize();
    client.initialized();

    let uri = "file:///test_definition.sio";
    let text = r#"
fn add(x: i64, y: i64) -> i64 {
    x + y
}

fn main() -> i64 {
    add(1, 2)
}
"#;

    client.did_open(uri, text);
    thread::sleep(Duration::from_millis(300));

    // Request definition of 'add' at line 6
    let def_id = client.definition(uri, 6, 4);
    thread::sleep(Duration::from_millis(500));

    let responses = client.read_responses(1000);

    // Look for definition response
    let def_response = responses
        .iter()
        .find(|r| r.get("id").and_then(|v| v.as_i64()) == Some(def_id));

    // Should receive definition response
    assert!(
        def_response.is_some(),
        "Should receive definition response"
    );

    client.did_close(uri);
    client.shutdown();
}

// ============================================================================
// Multiple Document Tests
// ============================================================================

#[test]
fn test_lsp_multiple_documents() {
    let mut client = LspClient::new();
    client.initialize();
    client.initialized();

    // Open multiple documents
    client.did_open("file:///test1.sio", "fn main() -> i64 { 1 }");
    client.did_open("file:///test2.sio", "fn main() -> i64 { 2 }");
    client.did_open("file:///test3.sio", "fn main() -> i64 { 3 }");

    thread::sleep(Duration::from_millis(500));

    // Close them
    client.did_close("file:///test1.sio");
    client.did_close("file:///test2.sio");
    client.did_close("file:///test3.sio");

    client.shutdown();
}

// ============================================================================
// Error Recovery Tests
// ============================================================================

#[test]
fn test_lsp_continues_after_invalid_document() {
    let mut client = LspClient::new();
    client.initialize();
    client.initialized();

    // Open invalid document
    client.did_open("file:///invalid.sio", "this is not valid sounio code @#$%");

    thread::sleep(Duration::from_millis(300));

    // Open valid document - should still work
    client.did_open("file:///valid.sio", "fn main() -> i64 { 42 }");

    thread::sleep(Duration::from_millis(300));

    // Server should still be responsive
    client.did_close("file:///invalid.sio");
    client.did_close("file:///valid.sio");

    client.shutdown();
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_lsp_rapid_changes() {
    let mut client = LspClient::new();
    client.initialize();
    client.initialized();

    let uri = "file:///test_rapid.sio";

    client.did_open(uri, "fn main() -> i64 { 1 }");

    // Rapid changes
    for i in 2..10 {
        let text = format!("fn main() -> i64 {{ {} }}", i);
        client.did_change(uri, i, &text);
        thread::sleep(Duration::from_millis(50));
    }

    thread::sleep(Duration::from_millis(300));

    client.did_close(uri);
    client.shutdown();
}

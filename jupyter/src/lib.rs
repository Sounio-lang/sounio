//! Jupyter Kernel for Sounio
//!
//! Implements the Jupyter wire protocol to provide an interactive Sounio
//! execution environment in Jupyter notebooks.
//!
//! # Architecture
//!
//! The kernel uses ZeroMQ sockets to communicate with the Jupyter frontend:
//! - Shell: Request/reply for execute_request, kernel_info_request, etc.
//! - IOPub: Publish/subscribe for execution results, stream output
//! - Stdin: Request input from frontend (not fully implemented)
//! - Control: Shutdown and interrupt requests
//! - Heartbeat: Simple echo for connection health

use chrono::Utc;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;
use uuid::Uuid;
use zeromq::{Socket, SocketRecv, SocketSend};

// Re-export for main.rs
pub use zeromq;

/// Version of the Jupyter messaging protocol we implement
pub const PROTOCOL_VERSION: &str = "5.3";

/// Delimiter in wire protocol messages
const MSG_DELIMITER: &[u8] = b"<IDS|MSG>";

/// Kernel errors
#[derive(Error, Debug)]
pub enum KernelError {
    #[error("ZeroMQ error: {0}")]
    Zmq(#[from] zeromq::ZmqError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HMAC error: {0}")]
    Hmac(String),

    #[error("Invalid message format: {0}")]
    InvalidMessage(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, KernelError>;

/// Connection information from the connection file
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConnectionInfo {
    pub shell_port: u16,
    pub iopub_port: u16,
    pub stdin_port: u16,
    pub control_port: u16,
    pub hb_port: u16,
    pub ip: String,
    pub key: String,
    pub transport: String,
    pub signature_scheme: String,
    #[serde(default)]
    pub kernel_name: String,
}

impl ConnectionInfo {
    /// Load connection info from a JSON file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    /// Build address string for a port
    pub fn address(&self, port: u16) -> String {
        format!("{}://{}:{}", self.transport, self.ip, port)
    }
}

/// Jupyter message header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub msg_id: String,
    pub session: String,
    pub username: String,
    pub date: String,
    pub msg_type: String,
    pub version: String,
}

impl Header {
    pub fn new(msg_type: &str, session: &str) -> Self {
        Self {
            msg_id: Uuid::new_v4().to_string(),
            session: session.to_string(),
            username: "sounio".to_string(),
            date: Utc::now().to_rfc3339(),
            msg_type: msg_type.to_string(),
            version: PROTOCOL_VERSION.to_string(),
        }
    }
}

/// Complete Jupyter message
#[derive(Debug, Clone)]
pub struct Message {
    pub identities: Vec<Vec<u8>>,
    pub header: Header,
    pub parent_header: Option<Header>,
    pub metadata: serde_json::Value,
    pub content: serde_json::Value,
    pub buffers: Vec<Vec<u8>>,
}

impl Message {
    /// Create a reply message
    pub fn reply(&self, msg_type: &str, content: serde_json::Value) -> Self {
        Self {
            identities: self.identities.clone(),
            header: Header::new(msg_type, &self.header.session),
            parent_header: Some(self.header.clone()),
            metadata: serde_json::json!({}),
            content,
            buffers: vec![],
        }
    }

    /// Sign message parts with HMAC
    fn sign(&self, key: &[u8]) -> Result<String> {
        if key.is_empty() {
            return Ok(String::new());
        }

        let mut mac = Hmac::<Sha256>::new_from_slice(key)
            .map_err(|e| KernelError::Hmac(e.to_string()))?;

        mac.update(serde_json::to_string(&self.header)?.as_bytes());
        mac.update(
            self.parent_header
                .as_ref()
                .map(|h| serde_json::to_string(h))
                .transpose()?
                .unwrap_or_else(|| "{}".to_string())
                .as_bytes(),
        );
        mac.update(self.metadata.to_string().as_bytes());
        mac.update(self.content.to_string().as_bytes());

        Ok(hex::encode(mac.finalize().into_bytes()))
    }

    /// Serialize message for sending
    pub fn serialize(&self, key: &[u8]) -> Result<Vec<Vec<u8>>> {
        let signature = self.sign(key)?;
        let header = serde_json::to_vec(&self.header)?;
        let parent_header = self
            .parent_header
            .as_ref()
            .map(|h| serde_json::to_vec(h))
            .transpose()?
            .unwrap_or_else(|| b"{}".to_vec());
        let metadata = serde_json::to_vec(&self.metadata)?;
        let content = serde_json::to_vec(&self.content)?;

        let mut frames = Vec::new();

        // Identities
        for id in &self.identities {
            frames.push(id.clone());
        }

        // Delimiter
        frames.push(MSG_DELIMITER.to_vec());

        // Signature
        frames.push(signature.into_bytes());

        // Message parts
        frames.push(header);
        frames.push(parent_header);
        frames.push(metadata);
        frames.push(content);

        // Buffers
        for buf in &self.buffers {
            frames.push(buf.clone());
        }

        Ok(frames)
    }

    /// Parse message from received frames
    pub fn parse(frames: Vec<Vec<u8>>, key: &[u8]) -> Result<Self> {
        // Find delimiter
        let delim_pos = frames
            .iter()
            .position(|f| f.as_slice() == MSG_DELIMITER)
            .ok_or_else(|| KernelError::InvalidMessage("Missing delimiter".to_string()))?;

        let identities = frames[..delim_pos].to_vec();

        if frames.len() < delim_pos + 6 {
            return Err(KernelError::InvalidMessage("Not enough message parts".to_string()));
        }

        let signature = String::from_utf8_lossy(&frames[delim_pos + 1]).to_string();
        let header: Header = serde_json::from_slice(&frames[delim_pos + 2])?;
        let parent_header: Option<Header> = {
            let ph = &frames[delim_pos + 3];
            if ph == b"{}" || ph.is_empty() {
                None
            } else {
                Some(serde_json::from_slice(ph)?)
            }
        };
        let metadata: serde_json::Value = serde_json::from_slice(&frames[delim_pos + 4])?;
        let content: serde_json::Value = serde_json::from_slice(&frames[delim_pos + 5])?;
        let buffers = frames[delim_pos + 6..].to_vec();

        let msg = Self {
            identities,
            header,
            parent_header,
            metadata,
            content,
            buffers,
        };

        // Verify signature
        if !key.is_empty() {
            let expected = msg.sign(key)?;
            if signature != expected {
                return Err(KernelError::InvalidMessage("Invalid signature".to_string()));
            }
        }

        Ok(msg)
    }
}

/// Kernel execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionState {
    Busy,
    Idle,
    Starting,
}

impl Serialize for ExecutionState {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ExecutionState::Busy => serializer.serialize_str("busy"),
            ExecutionState::Idle => serializer.serialize_str("idle"),
            ExecutionState::Starting => serializer.serialize_str("starting"),
        }
    }
}

/// Execution context that persists across cells
pub struct ExecutionContext {
    /// Accumulated function definitions
    functions: HashMap<String, String>,
    /// Accumulated type definitions
    types: HashMap<String, String>,
    /// Variable bindings from previous expressions
    bindings: HashMap<String, sounio::interp::Value>,
    /// Binding statements for type checker
    binding_stmts: Vec<(String, String)>,
    /// Execution counter
    execution_count: u32,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            types: HashMap::new(),
            bindings: HashMap::new(),
            binding_stmts: Vec::new(),
            execution_count: 0,
        }
    }

    /// Execute Sounio code and return the result
    pub fn execute(&mut self, code: &str) -> std::result::Result<ExecutionResult, String> {
        self.execution_count += 1;
        let input = code.trim();

        if input.is_empty() {
            return Ok(ExecutionResult {
                output: None,
                display_data: None,
            });
        }

        // Check if this is a definition
        if input.starts_with("fn ") {
            if let Some(name) = self.extract_fn_name(input) {
                self.functions.insert(name.clone(), input.to_string());
                return Ok(ExecutionResult {
                    output: Some(format!("Defined function: {}", name)),
                    display_data: None,
                });
            }
        }

        if input.starts_with("struct ") {
            if let Some(name) = self.extract_type_name(input, "struct") {
                self.types.insert(name.clone(), input.to_string());
                return Ok(ExecutionResult {
                    output: Some(format!("Defined struct: {}", name)),
                    display_data: None,
                });
            }
        }

        if input.starts_with("enum ") {
            if let Some(name) = self.extract_type_name(input, "enum") {
                self.types.insert(name.clone(), input.to_string());
                return Ok(ExecutionResult {
                    output: Some(format!("Defined enum: {}", name)),
                    display_data: None,
                });
            }
        }

        // Check if this is a let or var binding
        let is_let = input.starts_with("let ");
        let is_var = input.starts_with("var ");
        let is_binding = is_let || is_var;

        // Wrap expression for evaluation
        let wrapped = if is_binding {
            if let Some(name) = self.extract_binding_name(input) {
                format!("{}\n    {}", input, name)
            } else {
                input.to_string()
            }
        } else {
            input.to_string()
        };

        let source = self.build_source(&wrapped);

        // Parse and type-check
        let tokens = sounio::lexer::lex(&source).map_err(|e| format!("Lex error: {:?}", e))?;
        let ast =
            sounio::parser::parse(&tokens, &source).map_err(|e| format!("Parse error: {:?}", e))?;
        let hir = sounio::check::check(&ast).map_err(|e| format!("Type error: {}", e))?;

        // Execute with interpreter
        let mut interp = sounio::interp::Interpreter::new();

        // Pre-populate environment with existing bindings
        for (name, value) in &self.bindings {
            interp.env_mut().define(name.clone(), value.clone());
        }

        match interp.run(&hir) {
            Ok(value) => {
                let output = if is_binding {
                    if let Some(name) = self.extract_binding_name(input) {
                        // Store binding
                        self.binding_stmts.retain(|(n, _)| n != &name);
                        self.binding_stmts.push((name.clone(), input.to_string()));
                        self.bindings.insert(name.clone(), value.clone());
                        Some(format!("{} = {:?}", name, value))
                    } else {
                        Some(format!("{:?}", value))
                    }
                } else if value != sounio::interp::Value::Unit {
                    Some(format!("{:?}", value))
                } else {
                    None
                };

                Ok(ExecutionResult {
                    output,
                    display_data: None,
                })
            }
            Err(e) => Err(format!("Runtime error: {}", e)),
        }
    }

    fn build_source(&self, expr: &str) -> String {
        let mut source = String::new();

        // Add type definitions
        for def in self.types.values() {
            source.push_str(def);
            source.push('\n');
        }

        // Add function definitions
        for def in self.functions.values() {
            source.push_str(def);
            source.push('\n');
        }

        // Build main function with previous bindings
        source.push_str("fn main() -> i64 {\n");

        // Add previous binding statements
        for (_, stmt) in &self.binding_stmts {
            source.push_str("    ");
            source.push_str(stmt);
            source.push('\n');
        }

        // Add the new expression
        source.push_str("    ");
        source.push_str(expr);
        source.push_str("\n}\n");

        source
    }

    fn extract_fn_name(&self, input: &str) -> Option<String> {
        let input = input.strip_prefix("fn ")?.trim_start();
        let end = input.find('(')?;
        Some(input[..end].trim().to_string())
    }

    fn extract_type_name(&self, input: &str, prefix: &str) -> Option<String> {
        let input = input.strip_prefix(prefix)?.strip_prefix(' ')?.trim_start();
        let end = input.find(|c: char| c == '{' || c == '<' || c.is_whitespace())?;
        Some(input[..end].trim().to_string())
    }

    fn extract_binding_name(&self, input: &str) -> Option<String> {
        let input = if input.starts_with("let ") {
            input.strip_prefix("let ")?
        } else if input.starts_with("var ") {
            input.strip_prefix("var ")?
        } else {
            return None;
        };
        let input = input.trim_start();
        let end = input.find(|c: char| c == '=' || c == ':' || c.is_whitespace())?;
        Some(input[..end].trim().to_string())
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of executing code
pub struct ExecutionResult {
    pub output: Option<String>,
    pub display_data: Option<serde_json::Value>,
}

/// The Sounio Jupyter kernel
pub struct SounioKernel {
    /// Connection information
    conn_info: ConnectionInfo,
    /// Session ID
    session: String,
    /// HMAC key for signing
    key: Vec<u8>,
    /// Execution context
    context: Arc<Mutex<ExecutionContext>>,
    /// Execution counter
    execution_count: Arc<Mutex<u32>>,
}

impl SounioKernel {
    pub fn new(conn_info: ConnectionInfo) -> Self {
        let key = conn_info.key.as_bytes().to_vec();
        Self {
            conn_info,
            session: Uuid::new_v4().to_string(),
            key,
            context: Arc::new(Mutex::new(ExecutionContext::new())),
            execution_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Run the kernel
    pub async fn run(&self) -> Result<()> {
        tracing::info!("Starting Sounio Jupyter kernel");

        // Create sockets
        let mut shell = zeromq::RouterSocket::new();
        let mut iopub = zeromq::PubSocket::new();
        let mut stdin = zeromq::RouterSocket::new();
        let mut control = zeromq::RouterSocket::new();
        let mut heartbeat = zeromq::RepSocket::new();

        // Bind sockets
        shell.bind(&self.conn_info.address(self.conn_info.shell_port)).await?;
        iopub.bind(&self.conn_info.address(self.conn_info.iopub_port)).await?;
        stdin.bind(&self.conn_info.address(self.conn_info.stdin_port)).await?;
        control.bind(&self.conn_info.address(self.conn_info.control_port)).await?;
        heartbeat.bind(&self.conn_info.address(self.conn_info.hb_port)).await?;

        tracing::info!("Sockets bound successfully");

        // Spawn heartbeat handler
        let hb_handle = tokio::spawn(async move {
            loop {
                match heartbeat.recv().await {
                    Ok(msg) => {
                        if let Err(e) = heartbeat.send(msg).await {
                            tracing::error!("Heartbeat send error: {}", e);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Heartbeat recv error: {}", e);
                        break;
                    }
                }
            }
        });

        // Wrap sockets in Arc<Mutex> for sharing
        let iopub = Arc::new(Mutex::new(iopub));

        // Main message loop
        loop {
            tokio::select! {
                // Shell socket
                msg = shell.recv() => {
                    match msg {
                        Ok(msg) => {
                            let frames: Vec<Vec<u8>> = msg.into_iter().map(|f| f.to_vec()).collect();
                            if let Ok(msg) = Message::parse(frames, &self.key) {
                                if let Err(e) = self.handle_shell_message(msg, &mut shell, iopub.clone()).await {
                                    tracing::error!("Error handling shell message: {}", e);
                                }
                            }
                        }
                        Err(e) => tracing::error!("Shell recv error: {}", e),
                    }
                }

                // Control socket
                msg = control.recv() => {
                    match msg {
                        Ok(msg) => {
                            let frames: Vec<Vec<u8>> = msg.into_iter().map(|f| f.to_vec()).collect();
                            if let Ok(msg) = Message::parse(frames, &self.key) {
                                if msg.header.msg_type == "shutdown_request" {
                                    tracing::info!("Shutdown requested");
                                    // Send shutdown reply
                                    let reply = msg.reply("shutdown_reply", serde_json::json!({
                                        "status": "ok",
                                        "restart": false
                                    }));
                                    self.send_message(&mut control, reply).await?;
                                    break;
                                }
                            }
                        }
                        Err(e) => tracing::error!("Control recv error: {}", e),
                    }
                }
            }
        }

        hb_handle.abort();
        Ok(())
    }

    /// Handle a message on the shell socket
    async fn handle_shell_message(
        &self,
        msg: Message,
        shell: &mut zeromq::RouterSocket,
        iopub: Arc<Mutex<zeromq::PubSocket>>,
    ) -> Result<()> {
        tracing::debug!("Received shell message: {}", msg.header.msg_type);

        match msg.header.msg_type.as_str() {
            "kernel_info_request" => {
                self.handle_kernel_info_request(msg, shell).await?;
            }
            "execute_request" => {
                self.handle_execute_request(msg, shell, iopub).await?;
            }
            "is_complete_request" => {
                self.handle_is_complete_request(msg, shell).await?;
            }
            "complete_request" => {
                self.handle_complete_request(msg, shell).await?;
            }
            "comm_info_request" => {
                // Comm info request - return empty comms
                let reply = msg.reply("comm_info_reply", serde_json::json!({
                    "status": "ok",
                    "comms": {}
                }));
                self.send_message(shell, reply).await?;
            }
            other => {
                tracing::warn!("Unhandled message type: {}", other);
            }
        }

        Ok(())
    }

    /// Handle kernel_info_request
    async fn handle_kernel_info_request(
        &self,
        msg: Message,
        shell: &mut zeromq::RouterSocket,
    ) -> Result<()> {
        let content = serde_json::json!({
            "status": "ok",
            "protocol_version": PROTOCOL_VERSION,
            "implementation": "sounio-jupyter",
            "implementation_version": "0.1.0",
            "language_info": {
                "name": "sounio",
                "version": sounio::VERSION,
                "mimetype": "text/x-sounio",
                "file_extension": ".sio",
                "pygments_lexer": "rust",
                "codemirror_mode": "rust"
            },
            "banner": format!("Sounio {} - Epistemic computing at the horizon of certainty", sounio::VERSION),
            "help_links": [
                {"text": "Sounio Documentation", "url": "https://sounio-lang.org/docs"}
            ]
        });

        let reply = msg.reply("kernel_info_reply", content);
        self.send_message(shell, reply).await
    }

    /// Handle execute_request
    async fn handle_execute_request(
        &self,
        msg: Message,
        shell: &mut zeromq::RouterSocket,
        iopub: Arc<Mutex<zeromq::PubSocket>>,
    ) -> Result<()> {
        let code = msg.content["code"].as_str().unwrap_or("");
        let silent = msg.content["silent"].as_bool().unwrap_or(false);
        let store_history = msg.content["store_history"].as_bool().unwrap_or(true);

        // Publish busy status
        self.publish_status(ExecutionState::Busy, &msg, iopub.clone()).await?;

        // Execute the code
        let mut ctx = self.context.lock().await;
        let mut exec_count = self.execution_count.lock().await;

        if store_history {
            *exec_count += 1;
        }
        let current_count = *exec_count;

        // Publish execute_input
        if !silent {
            let input_msg = Message {
                identities: vec![b"execute_input".to_vec()],
                header: Header::new("execute_input", &self.session),
                parent_header: Some(msg.header.clone()),
                metadata: serde_json::json!({}),
                content: serde_json::json!({
                    "code": code,
                    "execution_count": current_count
                }),
                buffers: vec![],
            };
            let mut iopub_lock = iopub.lock().await;
            self.send_iopub_message(&mut *iopub_lock, input_msg).await?;
        }

        let result = ctx.execute(code);
        drop(ctx);

        match result {
            Ok(exec_result) => {
                // Publish output if any
                if let Some(output) = &exec_result.output {
                    if !silent {
                        let result_msg = Message {
                            identities: vec![b"execute_result".to_vec()],
                            header: Header::new("execute_result", &self.session),
                            parent_header: Some(msg.header.clone()),
                            metadata: serde_json::json!({}),
                            content: serde_json::json!({
                                "execution_count": current_count,
                                "data": {
                                    "text/plain": output
                                },
                                "metadata": {}
                            }),
                            buffers: vec![],
                        };
                        let mut iopub_lock = iopub.lock().await;
                        self.send_iopub_message(&mut *iopub_lock, result_msg).await?;
                    }
                }

                // Send reply
                let reply = msg.reply("execute_reply", serde_json::json!({
                    "status": "ok",
                    "execution_count": current_count,
                    "user_expressions": {}
                }));
                self.send_message(shell, reply).await?;
            }
            Err(error) => {
                // Publish error
                if !silent {
                    let error_msg = Message {
                        identities: vec![b"error".to_vec()],
                        header: Header::new("error", &self.session),
                        parent_header: Some(msg.header.clone()),
                        metadata: serde_json::json!({}),
                        content: serde_json::json!({
                            "ename": "SounioError",
                            "evalue": &error,
                            "traceback": [error.clone()]
                        }),
                        buffers: vec![],
                    };
                    let mut iopub_lock = iopub.lock().await;
                    self.send_iopub_message(&mut *iopub_lock, error_msg).await?;
                }

                // Send error reply
                let reply = msg.reply("execute_reply", serde_json::json!({
                    "status": "error",
                    "execution_count": current_count,
                    "ename": "SounioError",
                    "evalue": &error,
                    "traceback": [error]
                }));
                self.send_message(shell, reply).await?;
            }
        }

        // Publish idle status
        self.publish_status(ExecutionState::Idle, &msg, iopub).await?;

        Ok(())
    }

    /// Handle is_complete_request
    async fn handle_is_complete_request(
        &self,
        msg: Message,
        shell: &mut zeromq::RouterSocket,
    ) -> Result<()> {
        let code = msg.content["code"].as_str().unwrap_or("");

        // Simple heuristic: check if braces are balanced
        let open_braces = code.matches('{').count();
        let close_braces = code.matches('}').count();

        let status = if open_braces > close_braces {
            "incomplete"
        } else {
            "complete"
        };

        let content = serde_json::json!({
            "status": status,
            "indent": if status == "incomplete" { "    " } else { "" }
        });

        let reply = msg.reply("is_complete_reply", content);
        self.send_message(shell, reply).await
    }

    /// Handle complete_request (tab completion)
    async fn handle_complete_request(
        &self,
        msg: Message,
        shell: &mut zeromq::RouterSocket,
    ) -> Result<()> {
        let code = msg.content["code"].as_str().unwrap_or("");
        let cursor_pos = msg.content["cursor_pos"].as_u64().unwrap_or(0) as usize;

        // Get the word being completed
        let before_cursor = &code[..cursor_pos.min(code.len())];
        let word_start = before_cursor
            .rfind(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|i| i + 1)
            .unwrap_or(0);
        let prefix = &before_cursor[word_start..];

        // Keywords for completion
        let keywords = [
            "let", "var", "fn", "struct", "enum", "effect", "if", "else", "while", "for", "in",
            "return", "true", "false", "pub", "use", "mod", "match", "with", "linear", "kernel",
            "type", "import",
        ];

        let matches: Vec<&str> = keywords
            .iter()
            .filter(|kw| kw.starts_with(prefix))
            .copied()
            .collect();

        let content = serde_json::json!({
            "status": "ok",
            "matches": matches,
            "cursor_start": word_start,
            "cursor_end": cursor_pos,
            "metadata": {}
        });

        let reply = msg.reply("complete_reply", content);
        self.send_message(shell, reply).await
    }

    /// Publish execution status on iopub
    async fn publish_status(
        &self,
        state: ExecutionState,
        parent: &Message,
        iopub: Arc<Mutex<zeromq::PubSocket>>,
    ) -> Result<()> {
        let msg = Message {
            identities: vec![b"status".to_vec()],
            header: Header::new("status", &self.session),
            parent_header: Some(parent.header.clone()),
            metadata: serde_json::json!({}),
            content: serde_json::json!({
                "execution_state": state
            }),
            buffers: vec![],
        };

        let mut iopub_lock = iopub.lock().await;
        self.send_iopub_message(&mut *iopub_lock, msg).await
    }

    /// Send a message on a router socket
    async fn send_message(&self, socket: &mut zeromq::RouterSocket, msg: Message) -> Result<()> {
        let frames = msg.serialize(&self.key)?;
        let zmq_msg: zeromq::ZmqMessage = frames.into_iter().map(|f| f.into()).collect();
        socket.send(zmq_msg).await?;
        Ok(())
    }

    /// Send a message on the iopub socket
    async fn send_iopub_message(&self, socket: &mut zeromq::PubSocket, msg: Message) -> Result<()> {
        let frames = msg.serialize(&self.key)?;
        let zmq_msg: zeromq::ZmqMessage = frames.into_iter().map(|f| f.into()).collect();
        socket.send(zmq_msg).await?;
        Ok(())
    }
}

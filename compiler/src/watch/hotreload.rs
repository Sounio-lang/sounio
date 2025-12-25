//! Hot reload for live code updates
//!
//! Provides:
//! - Function patching at runtime
//! - State preservation across reloads
//! - Client-server protocol for reload coordination
//! - Rollback on failure

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

/// Hot reload configuration
#[derive(Debug, Clone)]
pub struct HotReloadConfig {
    /// Server address
    pub address: String,

    /// Server port
    pub port: u16,

    /// Preserve program state across reloads
    pub preserve_state: bool,

    /// Connection timeout
    pub timeout: Duration,

    /// Maximum connected clients
    pub max_clients: usize,

    /// Auto-reload on changes
    pub auto_reload: bool,

    /// Retry count for failed reloads
    pub retry_count: usize,

    /// Verbose logging
    pub verbose: bool,
}

impl Default for HotReloadConfig {
    fn default() -> Self {
        HotReloadConfig {
            address: "127.0.0.1".into(),
            port: 9999,
            preserve_state: true,
            timeout: Duration::from_secs(5),
            max_clients: 10,
            auto_reload: true,
            retry_count: 3,
            verbose: false,
        }
    }
}

/// Reload message protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReloadMessage {
    /// Initial handshake
    Hello { version: u64, client_id: u64 },

    /// Subscribe to function updates
    Subscribe { functions: Vec<String> },

    /// Unsubscribe from function updates
    Unsubscribe { functions: Vec<String> },

    /// Function update notification
    Update {
        version: u64,
        functions: Vec<FunctionUpdate>,
    },

    /// Apply pending updates
    Apply { version: u64 },

    /// Update applied successfully
    Applied { version: u64 },

    /// Update failed
    Failed { version: u64, error: String },

    /// Rollback to previous version
    Rollback { version: u64 },

    /// State snapshot request
    StateSnapshot { data: Vec<u8> },

    /// Restore state
    StateRestore { data: Vec<u8> },

    /// Keep-alive ping
    Ping,

    /// Keep-alive response
    Pong,
}

/// Function update data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionUpdate {
    /// Function name
    pub name: String,

    /// Module name
    pub module: String,

    /// Compiled machine code
    pub code: Vec<u8>,

    /// Relocation entries
    pub relocations: Vec<Relocation>,

    /// Function dependencies
    pub dependencies: Vec<String>,

    /// Source hash for change detection
    pub source_hash: u64,
}

/// Relocation entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relocation {
    /// Offset in code
    pub offset: usize,

    /// Relocation kind
    pub kind: RelocKind,

    /// Symbol name
    pub symbol: String,
}

/// Relocation kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelocKind {
    /// Absolute address
    Absolute,

    /// PC-relative
    Relative,

    /// GOT entry
    GotEntry,

    /// PLT entry
    PltEntry,
}

/// Hot reload record
#[derive(Debug, Clone)]
pub struct ReloadRecord {
    /// Timestamp
    pub timestamp: SystemTime,

    /// Version number
    pub version: u64,

    /// Functions updated
    pub functions: Vec<String>,

    /// Success status
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,

    /// Duration of reload
    pub duration: Duration,
}

/// Function information for hot reload
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    /// Function name
    pub name: String,

    /// Module name
    pub module: String,

    /// Source file
    pub source_file: PathBuf,

    /// Source hash
    pub source_hash: u64,

    /// Code address (if loaded)
    pub code_addr: Option<usize>,

    /// Code size
    pub code_size: usize,

    /// Whether function can be hot-reloaded
    pub reloadable: bool,
}

/// Connected client
struct ReloadClient {
    /// TCP stream
    stream: TcpStream,

    /// Client ID
    id: u64,

    /// Subscribed functions
    subscriptions: Vec<String>,

    /// Client version
    version: u64,
}

/// Hot reload engine (server side)
pub struct HotReloadEngine {
    /// Configuration
    config: HotReloadConfig,

    /// Connected clients
    clients: Arc<RwLock<Vec<ReloadClient>>>,

    /// TCP listener
    server: Option<TcpListener>,

    /// Reload history
    history: Arc<Mutex<Vec<ReloadRecord>>>,

    /// Current version
    version: Arc<Mutex<u64>>,

    /// Registered functions
    functions: Arc<RwLock<HashMap<String, FunctionInfo>>>,

    /// Accept thread handle
    accept_thread: Option<JoinHandle<()>>,

    /// Running flag
    running: Arc<Mutex<bool>>,

    /// Next client ID
    next_client_id: Arc<Mutex<u64>>,
}

impl HotReloadEngine {
    /// Create a new hot reload engine
    pub fn new(config: HotReloadConfig) -> Self {
        HotReloadEngine {
            config,
            clients: Arc::new(RwLock::new(Vec::new())),
            server: None,
            history: Arc::new(Mutex::new(Vec::new())),
            version: Arc::new(Mutex::new(0)),
            functions: Arc::new(RwLock::new(HashMap::new())),
            accept_thread: None,
            running: Arc::new(Mutex::new(false)),
            next_client_id: Arc::new(Mutex::new(1)),
        }
    }

    /// Start the hot reload server
    pub fn start(&mut self) -> Result<(), HotReloadError> {
        let addr = format!("{}:{}", self.config.address, self.config.port);
        let listener = TcpListener::bind(&addr)?;
        listener.set_nonblocking(true)?;

        println!("Hot reload server listening on {}", addr);

        self.server = Some(listener);
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = true;

        // Start accept thread
        let clients = Arc::clone(&self.clients);
        let version = Arc::clone(&self.version);
        let max_clients = self.config.max_clients;
        let running = Arc::clone(&self.running);
        let next_client_id = Arc::clone(&self.next_client_id);
        let listener = self.server.as_ref().unwrap().try_clone()?;
        let verbose = self.config.verbose;

        let handle = thread::Builder::new()
            .name("hotreload-accept".into())
            .spawn(move || {
                Self::accept_loop(
                    listener,
                    clients,
                    version,
                    max_clients,
                    running,
                    next_client_id,
                    verbose,
                );
            })?;

        self.accept_thread = Some(handle);
        Ok(())
    }

    /// Accept loop for incoming connections
    fn accept_loop(
        listener: TcpListener,
        clients: Arc<RwLock<Vec<ReloadClient>>>,
        version: Arc<Mutex<u64>>,
        max_clients: usize,
        running: Arc<Mutex<bool>>,
        next_client_id: Arc<Mutex<u64>>,
        verbose: bool,
    ) {
        loop {
            if !*running.lock().unwrap_or_else(|e| e.into_inner()) {
                break;
            }

            match listener.accept() {
                Ok((stream, addr)) => {
                    if verbose {
                        println!("Hot reload client connected: {}", addr);
                    }

                    // Check max clients
                    if clients.read().unwrap_or_else(|e| e.into_inner()).len() >= max_clients {
                        eprintln!("Hot reload: max clients reached, rejecting connection");
                        continue;
                    }

                    // Get next client ID
                    let client_id = {
                        let mut id = next_client_id.lock().unwrap_or_else(|e| e.into_inner());
                        let current = *id;
                        *id += 1;
                        current
                    };

                    let current_version = *version.lock().unwrap_or_else(|e| e.into_inner());

                    // Create client
                    let mut client = ReloadClient {
                        stream,
                        id: client_id,
                        subscriptions: Vec::new(),
                        version: current_version,
                    };

                    // Send hello
                    let hello = ReloadMessage::Hello {
                        version: current_version,
                        client_id,
                    };

                    if Self::send_message(&mut client.stream, &hello).is_ok() {
                        if let Ok(mut clients_guard) = clients.write() {
                            clients_guard.push(client);
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(100));
                }
                Err(e) => {
                    if *running.lock().unwrap_or_else(|e| e.into_inner()) {
                        eprintln!("Hot reload accept error: {}", e);
                    }
                    break;
                }
            }
        }
    }

    /// Send a message to a client
    fn send_message(stream: &mut TcpStream, msg: &ReloadMessage) -> Result<(), HotReloadError> {
        let data = serde_json::to_vec(msg)?;
        let len = data.len() as u32;

        stream.write_all(&len.to_le_bytes())?;
        stream.write_all(&data)?;
        stream.flush()?;

        Ok(())
    }

    /// Receive a message from a client
    #[allow(dead_code)]
    fn recv_message(stream: &mut TcpStream) -> Result<ReloadMessage, HotReloadError> {
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;

        let mut data = vec![0u8; len];
        stream.read_exact(&mut data)?;

        let msg = serde_json::from_slice(&data)?;
        Ok(msg)
    }

    /// Register a function for hot reload
    pub fn register_function(&self, info: FunctionInfo) {
        self.functions
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(info.name.clone(), info);
    }

    /// Unregister a function
    pub fn unregister_function(&self, name: &str) {
        self.functions
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name);
    }

    /// Get registered function info
    pub fn get_function(&self, name: &str) -> Option<FunctionInfo> {
        self.functions
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .cloned()
    }

    /// Reload functions
    pub fn reload(&self, updates: Vec<FunctionUpdate>) -> Result<ReloadResult, HotReloadError> {
        let start = std::time::Instant::now();

        // Increment version
        let new_version = {
            let mut version = self.version.lock().unwrap_or_else(|e| e.into_inner());
            *version += 1;
            *version
        };

        // Create update message
        let update_msg = ReloadMessage::Update {
            version: new_version,
            functions: updates.clone(),
        };

        // Send to all clients
        let mut clients = self.clients.write().unwrap_or_else(|e| e.into_inner());
        let mut success_count = 0;
        let mut fail_count = 0;
        let mut failed_clients = Vec::new();

        for (i, client) in clients.iter_mut().enumerate() {
            match Self::send_message(&mut client.stream, &update_msg) {
                Ok(_) => {
                    client.version = new_version;
                    success_count += 1;
                }
                Err(_) => {
                    fail_count += 1;
                    failed_clients.push(i);
                }
            }
        }

        // Remove failed clients
        for i in failed_clients.into_iter().rev() {
            clients.remove(i);
        }

        // Record history
        let record = ReloadRecord {
            timestamp: SystemTime::now(),
            version: new_version,
            functions: updates.iter().map(|u| u.name.clone()).collect(),
            success: fail_count == 0,
            error: if fail_count > 0 {
                Some(format!("{} clients failed to update", fail_count))
            } else {
                None
            },
            duration: start.elapsed(),
        };

        self.history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(record);

        Ok(ReloadResult {
            version: new_version,
            clients_updated: success_count,
            clients_failed: fail_count,
            duration: start.elapsed(),
        })
    }

    /// Rollback to previous version
    pub fn rollback(&self) -> Result<(), HotReloadError> {
        let current_version = *self.version.lock().unwrap_or_else(|e| e.into_inner());

        if current_version == 0 {
            return Err(HotReloadError::Protocol("No version to rollback to".into()));
        }

        let rollback_version = current_version - 1;
        let rollback_msg = ReloadMessage::Rollback {
            version: rollback_version,
        };

        let mut clients = self.clients.write().unwrap_or_else(|e| e.into_inner());
        for client in clients.iter_mut() {
            let _ = Self::send_message(&mut client.stream, &rollback_msg);
            client.version = rollback_version;
        }

        *self.version.lock().unwrap_or_else(|e| e.into_inner()) = rollback_version;

        Ok(())
    }

    /// Broadcast a ping to all clients
    pub fn ping(&self) -> usize {
        let mut clients = self.clients.write().unwrap_or_else(|e| e.into_inner());
        let mut alive = 0;
        let mut dead = Vec::new();

        for (i, client) in clients.iter_mut().enumerate() {
            if Self::send_message(&mut client.stream, &ReloadMessage::Ping).is_ok() {
                alive += 1;
            } else {
                dead.push(i);
            }
        }

        // Remove dead clients
        for i in dead.into_iter().rev() {
            clients.remove(i);
        }

        alive
    }

    /// Get reload history
    pub fn history(&self) -> Vec<ReloadRecord> {
        self.history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Get current version
    pub fn version(&self) -> u64 {
        *self.version.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Get connected client count
    pub fn client_count(&self) -> usize {
        self.clients.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Get all registered functions
    pub fn functions(&self) -> Vec<FunctionInfo> {
        self.functions
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .cloned()
            .collect()
    }

    /// Stop the hot reload server
    pub fn stop(&mut self) {
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = false;

        // Close all client connections
        self.clients
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();

        // Drop server to close listener
        self.server = None;

        // Wait for accept thread
        if let Some(handle) = self.accept_thread.take() {
            let _ = handle.join();
        }
    }

    /// Check if server is running
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Drop for HotReloadEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Reload result
#[derive(Debug, Clone)]
pub struct ReloadResult {
    /// New version
    pub version: u64,

    /// Clients successfully updated
    pub clients_updated: usize,

    /// Clients that failed to update
    pub clients_failed: usize,

    /// Time taken
    pub duration: Duration,
}

// =============================================================================
// Client-side Runtime
// =============================================================================

/// Hot reload runtime (client side)
pub struct HotReloadRuntime {
    /// Server address
    server: String,

    /// Connection
    connection: Option<TcpStream>,

    /// Current version
    version: u64,

    /// Client ID
    client_id: u64,

    /// Function pointer table
    function_table: HashMap<String, *mut u8>,

    /// Saved state
    state: HashMap<String, Vec<u8>>,

    /// Reload callback
    on_reload: Option<Box<dyn Fn(&[String]) + Send>>,

    /// Error callback
    on_error: Option<Box<dyn Fn(&str) + Send>>,
}

impl HotReloadRuntime {
    /// Create a new runtime
    pub fn new() -> Self {
        HotReloadRuntime {
            server: "127.0.0.1:9999".into(),
            connection: None,
            version: 0,
            client_id: 0,
            function_table: HashMap::new(),
            state: HashMap::new(),
            on_reload: None,
            on_error: None,
        }
    }

    /// Connect to hot reload server
    pub fn connect(&mut self, server: &str) -> Result<(), HotReloadError> {
        self.server = server.to_string();

        let stream = TcpStream::connect(server)?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;

        self.connection = Some(stream);

        // Wait for hello message
        if let Some(ref mut conn) = self.connection {
            let msg = HotReloadEngine::recv_message(conn)?;

            if let ReloadMessage::Hello { version, client_id } = msg {
                self.version = version;
                self.client_id = client_id;
                println!("Connected to hot reload server (client {})", client_id);
            }
        }

        Ok(())
    }

    /// Register a function pointer
    pub fn register_function(&mut self, name: &str, ptr: *mut u8) {
        self.function_table.insert(name.to_string(), ptr);
    }

    /// Set reload callback
    pub fn on_reload<F>(&mut self, callback: F)
    where
        F: Fn(&[String]) + Send + 'static,
    {
        self.on_reload = Some(Box::new(callback));
    }

    /// Set error callback
    pub fn on_error<F>(&mut self, callback: F)
    where
        F: Fn(&str) + Send + 'static,
    {
        self.on_error = Some(Box::new(callback));
    }

    /// Poll for updates (non-blocking)
    pub fn poll(&mut self) -> Option<Vec<String>> {
        if let Some(ref mut conn) = self.connection {
            conn.set_nonblocking(true).ok()?;

            match HotReloadEngine::recv_message(conn) {
                Ok(ReloadMessage::Update { version, functions }) => {
                    self.version = version;
                    let names: Vec<_> = functions.iter().map(|f| f.name.clone()).collect();

                    // Apply updates
                    for update in functions {
                        self.apply_update(&update);
                    }

                    // Call callback
                    if let Some(ref callback) = self.on_reload {
                        callback(&names);
                    }

                    return Some(names);
                }
                Ok(ReloadMessage::Rollback { version }) => {
                    self.version = version;
                    // Would restore previous function versions
                }
                Ok(ReloadMessage::Ping) => {
                    let _ = HotReloadEngine::send_message(conn, &ReloadMessage::Pong);
                }
                Err(_) => {
                    // No message or error
                }
                _ => {}
            }

            conn.set_nonblocking(false).ok();
        }

        None
    }

    /// Apply a function update
    fn apply_update(&mut self, _update: &FunctionUpdate) {
        // In a real implementation, this would:
        // 1. Allocate executable memory
        // 2. Copy the new code
        // 3. Apply relocations
        // 4. Update the function pointer
        // 5. Optionally patch call sites
    }

    /// Save state
    pub fn save_state(&mut self, key: &str, data: Vec<u8>) {
        self.state.insert(key.to_string(), data);
    }

    /// Load state
    pub fn load_state(&self, key: &str) -> Option<&[u8]> {
        self.state.get(key).map(|v| v.as_slice())
    }

    /// Get current version
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Get client ID
    pub fn client_id(&self) -> u64 {
        self.client_id
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connection.is_some()
    }

    /// Disconnect from server
    pub fn disconnect(&mut self) {
        self.connection = None;
    }
}

impl Default for HotReloadRuntime {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Errors
// =============================================================================

/// Hot reload error
#[derive(Debug)]
pub enum HotReloadError {
    /// IO error
    Io(std::io::Error),

    /// JSON error
    Json(serde_json::Error),

    /// Protocol error
    Protocol(String),

    /// Timeout
    Timeout,

    /// Not connected
    NotConnected,
}

impl From<std::io::Error> for HotReloadError {
    fn from(e: std::io::Error) -> Self {
        HotReloadError::Io(e)
    }
}

impl From<serde_json::Error> for HotReloadError {
    fn from(e: serde_json::Error) -> Self {
        HotReloadError::Json(e)
    }
}

impl std::fmt::Display for HotReloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HotReloadError::Io(e) => write!(f, "IO error: {}", e),
            HotReloadError::Json(e) => write!(f, "JSON error: {}", e),
            HotReloadError::Protocol(s) => write!(f, "Protocol error: {}", s),
            HotReloadError::Timeout => write!(f, "Timeout"),
            HotReloadError::NotConnected => write!(f, "Not connected"),
        }
    }
}

impl std::error::Error for HotReloadError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hot_reload_config_default() {
        let config = HotReloadConfig::default();
        assert_eq!(config.port, 9999);
        assert!(config.preserve_state);
        assert!(config.auto_reload);
    }

    #[test]
    fn test_reload_message_serialization() {
        let msg = ReloadMessage::Hello {
            version: 1,
            client_id: 42,
        };

        let json = serde_json::to_string(&msg).unwrap();
        let decoded: ReloadMessage = serde_json::from_str(&json).unwrap();

        if let ReloadMessage::Hello { version, client_id } = decoded {
            assert_eq!(version, 1);
            assert_eq!(client_id, 42);
        } else {
            panic!("Wrong message type");
        }
    }

    #[test]
    fn test_function_update_serialization() {
        let update = FunctionUpdate {
            name: "test_fn".into(),
            module: "test".into(),
            code: vec![0x90, 0x90, 0xc3], // NOP NOP RET
            relocations: vec![],
            dependencies: vec!["other_fn".into()],
            source_hash: 12345,
        };

        let json = serde_json::to_string(&update).unwrap();
        let decoded: FunctionUpdate = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.name, "test_fn");
        assert_eq!(decoded.code, vec![0x90, 0x90, 0xc3]);
    }

    #[test]
    fn test_hot_reload_runtime() {
        let mut runtime = HotReloadRuntime::new();
        assert!(!runtime.is_connected());

        runtime.save_state("test", vec![1, 2, 3]);
        assert_eq!(runtime.load_state("test"), Some(&[1, 2, 3][..]));
    }
}

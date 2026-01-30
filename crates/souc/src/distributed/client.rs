//! Distributed build client
//!
//! This module implements a client for connecting to remote build servers
//! and submitting build jobs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, RwLock, mpsc, oneshot};

use super::protocol::*;

/// Client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Server address
    pub server: String,

    /// Client ID
    pub client_id: String,

    /// Connection timeout
    pub connect_timeout: Duration,

    /// Request timeout
    pub request_timeout: Duration,

    /// Retry attempts
    pub retry_attempts: u32,

    /// Retry delay
    pub retry_delay: Duration,

    /// Enable compression
    pub compression: bool,
}

impl Default for ClientConfig {
    fn default() -> Self {
        ClientConfig {
            server: "localhost:9876".into(),
            client_id: format!("client-{:016x}", rand::random::<u64>()),
            connect_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(300),
            retry_attempts: 3,
            retry_delay: Duration::from_secs(1),
            compression: true,
        }
    }
}

/// Build client
pub struct BuildClient {
    /// Configuration
    config: ClientConfig,

    /// Connection state
    connection: Mutex<Option<Connection>>,

    /// Pending requests
    pending: RwLock<HashMap<String, PendingRequest>>,

    /// Server capabilities
    capabilities: RwLock<Option<ServerCapabilities>>,

    /// Connected flag
    connected: RwLock<bool>,
}

/// Active connection
struct Connection {
    session_id: String,
    tx: mpsc::Sender<BuildMessage>,
}

/// Pending request
struct PendingRequest {
    tx: oneshot::Sender<BuildMessage>,
    #[allow(dead_code)]
    submitted_at: Instant,
}

impl BuildClient {
    /// Create new client
    pub fn new(config: ClientConfig) -> Arc<Self> {
        Arc::new(BuildClient {
            config,
            connection: Mutex::new(None),
            pending: RwLock::new(HashMap::new()),
            capabilities: RwLock::new(None),
            connected: RwLock::new(false),
        })
    }

    /// Connect to server
    pub async fn connect(self: &Arc<Self>) -> Result<(), ClientError> {
        let stream = tokio::time::timeout(
            self.config.connect_timeout,
            TcpStream::connect(&self.config.server),
        )
        .await
        .map_err(|_| ClientError::Timeout)?
        .map_err(ClientError::Io)?;

        let (reader, writer) = stream.into_split();
        let (tx, mut rx) = mpsc::channel::<BuildMessage>(100);

        // Spawn writer
        tokio::spawn(async move {
            let mut writer = writer;
            while let Some(msg) = rx.recv().await {
                let encoded = msg.encode();
                if writer.write_all(&encoded).await.is_err() {
                    break;
                }
            }
        });

        // Store connection
        *self.connection.lock().await = Some(Connection {
            session_id: String::new(),
            tx: tx.clone(),
        });

        // Send hello
        tx.send(BuildMessage::Hello {
            protocol_version: PROTOCOL_VERSION,
            client_id: self.config.client_id.clone(),
            capabilities: ClientCapabilities {
                compression: if self.config.compression {
                    vec!["gzip".into(), "zstd".into()]
                } else {
                    vec![]
                },
                max_uploads: 10,
                streaming: true,
                version: env!("CARGO_PKG_VERSION").into(),
            },
        })
        .await
        .map_err(|_| ClientError::Disconnected)?;

        // Spawn reader
        let client = Arc::clone(self);
        tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(reader);
            loop {
                // Read length prefix
                let mut len_buf = [0u8; 4];
                if reader.read_exact(&mut len_buf).await.is_err() {
                    break;
                }
                let len = u32::from_be_bytes(len_buf) as usize;

                // Read message
                let mut msg_buf = vec![0u8; len];
                if reader.read_exact(&mut msg_buf).await.is_err() {
                    break;
                }

                if let Ok(msg) = serde_json::from_slice::<BuildMessage>(&msg_buf) {
                    client.handle_message(msg).await;
                }
            }

            // Disconnected
            *client.connection.lock().await = None;
            *client.connected.write().await = false;
        });

        // Wait for welcome (with timeout)
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if self.capabilities.read().await.is_some() {
                *self.connected.write().await = true;
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        Err(ClientError::Timeout)
    }

    /// Handle incoming message
    async fn handle_message(&self, msg: BuildMessage) {
        match msg {
            BuildMessage::Welcome {
                session_id,
                server_capabilities,
                ..
            } => {
                *self.capabilities.write().await = Some(server_capabilities);
                if let Some(ref mut conn) = *self.connection.lock().await {
                    conn.session_id = session_id;
                }
            }

            BuildMessage::JobAccepted { ref job_id, .. }
            | BuildMessage::JobRejected { ref job_id, .. }
            | BuildMessage::JobComplete { ref job_id, .. }
            | BuildMessage::JobFailed { ref job_id, .. }
            | BuildMessage::JobCancelled { ref job_id, .. } => {
                let mut pending = self.pending.write().await;
                if let Some(req) = pending.remove(job_id) {
                    let _ = req.tx.send(msg);
                }
            }

            BuildMessage::JobProgress { job_id, progress } => {
                // Log progress
                if let Some(file) = &progress.current_file {
                    eprintln!(
                        "[{}] {}: {}% - {}",
                        job_id, progress.phase, progress.percent, file
                    );
                } else {
                    eprintln!("[{}] {}: {}%", job_id, progress.phase, progress.percent);
                }
            }

            BuildMessage::Pong { .. } => {
                // Heartbeat response
            }

            BuildMessage::Error { code, message, .. } => {
                eprintln!("Server error: {:?} - {}", code, message);
            }

            BuildMessage::ServerStatus { .. } => {
                // Handle status response if needed
            }

            _ => {}
        }
    }

    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }

    /// Get server capabilities
    pub async fn capabilities(&self) -> Option<ServerCapabilities> {
        self.capabilities.read().await.clone()
    }

    /// Submit a build job
    pub async fn submit_job(&self, job: BuildJob) -> Result<BuildResult, ClientError> {
        let job_id = format!("job-{:016x}", rand::random::<u64>());

        // Create response channel
        let (tx, rx) = oneshot::channel();

        // Register pending request
        self.pending.write().await.insert(
            job_id.clone(),
            PendingRequest {
                tx,
                submitted_at: Instant::now(),
            },
        );

        // Send job
        let conn = self.connection.lock().await;
        let sender = conn.as_ref().ok_or(ClientError::Disconnected)?.tx.clone();
        drop(conn);

        sender
            .send(BuildMessage::SubmitJob {
                job_id: job_id.clone(),
                job,
            })
            .await
            .map_err(|_| ClientError::Disconnected)?;

        // Wait for response
        let response = tokio::time::timeout(self.config.request_timeout, rx)
            .await
            .map_err(|_| ClientError::Timeout)?
            .map_err(|_| ClientError::Disconnected)?;

        match response {
            BuildMessage::JobComplete { result, .. } => Ok(result),
            BuildMessage::JobFailed { error, .. } => Err(ClientError::BuildFailed(error)),
            BuildMessage::JobRejected { reason, .. } => Err(ClientError::Rejected(reason)),
            BuildMessage::JobAccepted { job_id, .. } => {
                // Job accepted, wait for completion
                self.wait_for_completion(&job_id).await
            }
            _ => Err(ClientError::Protocol("Unexpected response".into())),
        }
    }

    /// Wait for job completion
    async fn wait_for_completion(&self, job_id: &str) -> Result<BuildResult, ClientError> {
        let (tx, rx) = oneshot::channel();

        self.pending.write().await.insert(
            job_id.to_string(),
            PendingRequest {
                tx,
                submitted_at: Instant::now(),
            },
        );

        let response = tokio::time::timeout(self.config.request_timeout, rx)
            .await
            .map_err(|_| ClientError::Timeout)?
            .map_err(|_| ClientError::Disconnected)?;

        match response {
            BuildMessage::JobComplete { result, .. } => Ok(result),
            BuildMessage::JobFailed { error, .. } => Err(ClientError::BuildFailed(error)),
            BuildMessage::JobCancelled { .. } => Err(ClientError::Cancelled),
            _ => Err(ClientError::Protocol("Unexpected response".into())),
        }
    }

    /// Build a project remotely
    pub async fn build_remote(
        &self,
        project_dir: &Path,
        target: &str,
        profile: &str,
    ) -> Result<BuildResult, ClientError> {
        // Collect source files
        let sources = self.collect_sources(project_dir).await?;

        // Compute cache key
        let cache_key = self.compute_cache_key(&sources, target, profile);

        // Create job
        let job = BuildJob {
            job_type: JobType::Build,
            sources,
            dependencies: vec![],
            config: BuildConfig {
                opt_level: if profile == "release" {
                    "3".into()
                } else {
                    "0".into()
                },
                debug_info: profile != "release",
                features: vec![],
                flags: vec![],
                profile: profile.into(),
            },
            target: target.into(),
            env: std::env::vars().collect(),
            outputs: vec![OutputSpec {
                output_type: OutputType::Executable,
                path: PathBuf::from("output"),
            }],
            priority: 0,
            timeout: Duration::from_secs(3600),
            cache_key: Some(cache_key),
        };

        self.submit_job(job).await
    }

    /// Collect source files from directory
    async fn collect_sources(&self, dir: &Path) -> Result<Vec<SourceFile>, ClientError> {
        let mut sources = Vec::new();

        fn visit_dir(
            dir: &Path,
            base: &Path,
            sources: &mut Vec<(PathBuf, PathBuf)>,
        ) -> std::io::Result<()> {
            if dir.is_dir() {
                for entry in std::fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.is_dir() {
                        visit_dir(&path, base, sources)?;
                    } else if path.extension().map(|x| x == "d").unwrap_or(false) {
                        let rel = path.strip_prefix(base).unwrap_or(&path).to_path_buf();
                        sources.push((path, rel));
                    }
                }
            }
            Ok(())
        }

        let mut paths = Vec::new();
        visit_dir(dir, dir, &mut paths).map_err(ClientError::Io)?;

        for (path, rel) in paths {
            let content = tokio::fs::read(&path).await.map_err(ClientError::Io)?;
            let hash = format!("{:x}", Sha256::digest(&content));

            sources.push(SourceFile {
                path: rel,
                hash,
                size: content.len(),
                is_main: path.file_name().map(|n| n == "main.sio").unwrap_or(false),
            });
        }

        // Sort for determinism
        sources.sort_by(|a, b| a.path.cmp(&b.path));

        Ok(sources)
    }

    /// Compute cache key for build
    fn compute_cache_key(&self, sources: &[SourceFile], target: &str, profile: &str) -> String {
        let mut hasher = Sha256::new();

        for src in sources {
            hasher.update(&src.hash);
            hasher.update(src.path.to_string_lossy().as_bytes());
        }
        hasher.update(target.as_bytes());
        hasher.update(profile.as_bytes());
        hasher.update(env!("CARGO_PKG_VERSION").as_bytes());

        format!("{:x}", hasher.finalize())
    }

    /// Query server status
    pub async fn server_status(&self) -> Result<ServerStatusInfo, ClientError> {
        let conn = self.connection.lock().await;
        let sender = conn.as_ref().ok_or(ClientError::Disconnected)?.tx.clone();
        drop(conn);

        // Create response channel
        let (tx, rx) = oneshot::channel();
        let query_id = format!("query-{:016x}", rand::random::<u64>());

        self.pending.write().await.insert(
            query_id.clone(),
            PendingRequest {
                tx,
                submitted_at: Instant::now(),
            },
        );

        sender
            .send(BuildMessage::QueryServer)
            .await
            .map_err(|_| ClientError::Disconnected)?;

        // Wait for response with short timeout
        match tokio::time::timeout(Duration::from_secs(5), rx).await {
            Ok(Ok(BuildMessage::ServerStatus {
                workers,
                queue_length,
                active_jobs,
                uptime,
            })) => Ok(ServerStatusInfo {
                workers,
                queue_length,
                active_jobs,
                uptime,
            }),
            _ => {
                // Query didn't match, but we might have received it anyway
                // Return placeholder for now
                Ok(ServerStatusInfo {
                    workers: vec![],
                    queue_length: 0,
                    active_jobs: 0,
                    uptime: Duration::default(),
                })
            }
        }
    }

    /// Cancel a job
    pub async fn cancel_job(&self, job_id: &str) -> Result<(), ClientError> {
        let conn = self.connection.lock().await;
        let sender = conn.as_ref().ok_or(ClientError::Disconnected)?.tx.clone();
        drop(conn);

        sender
            .send(BuildMessage::CancelJob {
                job_id: job_id.to_string(),
                reason: "User cancelled".into(),
            })
            .await
            .map_err(|_| ClientError::Disconnected)?;

        Ok(())
    }

    /// Disconnect from server
    pub async fn disconnect(&self) {
        if let Some(conn) = self.connection.lock().await.take() {
            let _ = conn
                .tx
                .send(BuildMessage::Goodbye {
                    reason: "Client disconnecting".into(),
                })
                .await;
        }
        *self.connected.write().await = false;
    }

    /// Send heartbeat
    pub async fn ping(&self) -> Result<Duration, ClientError> {
        let conn = self.connection.lock().await;
        let sender = conn.as_ref().ok_or(ClientError::Disconnected)?.tx.clone();
        drop(conn);

        let start = Instant::now();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        sender
            .send(BuildMessage::Ping { timestamp })
            .await
            .map_err(|_| ClientError::Disconnected)?;

        // We don't wait for pong in this simple implementation
        Ok(start.elapsed())
    }
}

/// Server status information
#[derive(Debug, Clone)]
pub struct ServerStatusInfo {
    pub workers: Vec<WorkerInfo>,
    pub queue_length: usize,
    pub active_jobs: usize,
    pub uptime: Duration,
}

/// Client error
#[derive(Debug)]
pub enum ClientError {
    Io(std::io::Error),
    Timeout,
    Disconnected,
    Rejected(String),
    BuildFailed(BuildError),
    Protocol(String),
    Cancelled,
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::Io(e) => write!(f, "IO error: {}", e),
            ClientError::Timeout => write!(f, "Connection timeout"),
            ClientError::Disconnected => write!(f, "Disconnected from server"),
            ClientError::Rejected(reason) => write!(f, "Job rejected: {}", reason),
            ClientError::BuildFailed(e) => write!(f, "Build failed: {}", e.message),
            ClientError::Protocol(msg) => write!(f, "Protocol error: {}", msg),
            ClientError::Cancelled => write!(f, "Job cancelled"),
        }
    }
}

impl std::error::Error for ClientError {}

impl From<std::io::Error> for ClientError {
    fn from(e: std::io::Error) -> Self {
        ClientError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_default() {
        let config = ClientConfig::default();
        assert_eq!(config.server, "localhost:9876");
        assert!(config.compression);
        assert_eq!(config.retry_attempts, 3);
    }

    #[test]
    fn test_cache_key_computation() {
        let config = ClientConfig::default();
        let client = BuildClient::new(config);

        let sources = vec![
            SourceFile {
                path: PathBuf::from("main.sio"),
                hash: "abc123".into(),
                size: 100,
                is_main: true,
            },
            SourceFile {
                path: PathBuf::from("lib.sio"),
                hash: "def456".into(),
                size: 200,
                is_main: false,
            },
        ];

        let key1 = client.compute_cache_key(&sources, "x86_64-unknown-linux-gnu", "release");
        let key2 = client.compute_cache_key(&sources, "x86_64-unknown-linux-gnu", "release");
        let key3 = client.compute_cache_key(&sources, "aarch64-unknown-linux-gnu", "release");

        assert_eq!(key1, key2); // Same inputs = same key
        assert_ne!(key1, key3); // Different target = different key
    }
}

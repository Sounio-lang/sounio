//! Distributed build server
//!
//! This module implements a build server that accepts jobs from clients,
//! distributes them to workers, and manages the build queue.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, RwLock, broadcast, mpsc};

use super::protocol::*;

/// Build server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Listen address
    pub address: SocketAddr,

    /// Maximum concurrent connections
    pub max_connections: usize,

    /// Maximum queue size
    pub max_queue_size: usize,

    /// Job timeout
    pub job_timeout: Duration,

    /// Heartbeat interval
    pub heartbeat_interval: Duration,

    /// Cache enabled
    pub cache_enabled: bool,

    /// Cache server URL (if remote)
    pub cache_url: Option<String>,

    /// Authentication required
    pub auth_required: bool,

    /// Server name
    pub server_name: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            address: "0.0.0.0:9876".parse().unwrap(),
            max_connections: 100,
            max_queue_size: 1000,
            job_timeout: Duration::from_secs(3600),
            heartbeat_interval: Duration::from_secs(30),
            cache_enabled: true,
            cache_url: None,
            auth_required: false,
            server_name: "d-build-server".to_string(),
        }
    }
}

/// Build server
pub struct BuildServer {
    /// Configuration
    config: ServerConfig,

    /// Connected clients
    clients: RwLock<HashMap<String, ClientHandle>>,

    /// Registered workers
    workers: RwLock<HashMap<String, WorkerHandle>>,

    /// Job queue
    job_queue: Mutex<JobQueue>,

    /// Active jobs
    active_jobs: RwLock<HashMap<String, ActiveJob>>,

    /// Completed jobs (recent)
    completed_jobs: RwLock<HashMap<String, CompletedJob>>,

    /// Server statistics
    stats: RwLock<ServerStats>,

    /// Start time
    start_time: Instant,

    /// Shutdown signal
    shutdown: broadcast::Sender<()>,
}

/// Client connection handle
struct ClientHandle {
    id: String,
    session_id: String,
    addr: SocketAddr,
    connected_at: Instant,
    tx: mpsc::Sender<BuildMessage>,
    capabilities: ClientCapabilities,
}

/// Worker connection handle
struct WorkerHandle {
    id: String,
    name: String,
    targets: Vec<String>,
    status: WorkerStatus,
    tx: mpsc::Sender<WorkerMessage>,
    stats: WorkerStats,
    last_heartbeat: Instant,
}

/// Message to worker
#[derive(Debug, Clone)]
pub enum WorkerMessage {
    Execute(BuildJob, String), // Job and job_id
    Cancel(String),
    Shutdown,
}

/// Worker statistics
#[derive(Debug, Default)]
struct WorkerStats {
    jobs_completed: u64,
    jobs_failed: u64,
    total_duration: Duration,
}

/// Job queue
struct JobQueue {
    jobs: Vec<QueuedJob>,
    max_size: usize,
}

impl JobQueue {
    fn new(max_size: usize) -> Self {
        JobQueue {
            jobs: Vec::new(),
            max_size,
        }
    }

    fn push(&mut self, job: QueuedJob) -> Result<(), String> {
        if self.jobs.len() >= self.max_size {
            return Err("Queue full".into());
        }

        // Insert by priority (higher priority first)
        let pos = self
            .jobs
            .iter()
            .position(|j| j.priority < job.priority)
            .unwrap_or(self.jobs.len());
        self.jobs.insert(pos, job);

        Ok(())
    }

    fn pop(&mut self) -> Option<QueuedJob> {
        if self.jobs.is_empty() {
            None
        } else {
            Some(self.jobs.remove(0))
        }
    }

    fn pop_for_target(&mut self, target: &str) -> Option<QueuedJob> {
        let pos = self.jobs.iter().position(|j| j.job.target == target);
        pos.map(|i| self.jobs.remove(i))
    }

    fn remove(&mut self, job_id: &str) -> Option<QueuedJob> {
        let pos = self.jobs.iter().position(|j| j.job_id == job_id);
        pos.map(|i| self.jobs.remove(i))
    }

    fn len(&self) -> usize {
        self.jobs.len()
    }

    fn get(&self, job_id: &str) -> Option<&QueuedJob> {
        self.jobs.iter().find(|j| j.job_id == job_id)
    }
}

/// Queued job
struct QueuedJob {
    job_id: String,
    job: BuildJob,
    client_id: String,
    priority: i32,
    submitted_at: Instant,
}

/// Active job
struct ActiveJob {
    job_id: String,
    job: BuildJob,
    client_id: String,
    worker_id: String,
    started_at: Instant,
    progress: JobProgress,
}

/// Completed job (cached briefly)
struct CompletedJob {
    job_id: String,
    result: BuildResult,
    completed_at: Instant,
}

/// Server statistics
#[derive(Debug, Default)]
struct ServerStats {
    total_jobs: u64,
    completed_jobs: u64,
    failed_jobs: u64,
    cache_hits: u64,
    cache_misses: u64,
    bytes_transferred: u64,
    total_connections: u64,
}

impl BuildServer {
    /// Create new server
    pub fn new(config: ServerConfig) -> Arc<Self> {
        let (shutdown_tx, _) = broadcast::channel(1);

        Arc::new(BuildServer {
            config: config.clone(),
            clients: RwLock::new(HashMap::new()),
            workers: RwLock::new(HashMap::new()),
            job_queue: Mutex::new(JobQueue::new(config.max_queue_size)),
            active_jobs: RwLock::new(HashMap::new()),
            completed_jobs: RwLock::new(HashMap::new()),
            stats: RwLock::new(ServerStats::default()),
            start_time: Instant::now(),
            shutdown: shutdown_tx,
        })
    }

    /// Start the server
    pub async fn start(self: Arc<Self>) -> Result<(), ServerError> {
        let listener = TcpListener::bind(&self.config.address).await?;
        println!(
            "Build server '{}' listening on {}",
            self.config.server_name, self.config.address
        );

        let mut shutdown_rx = self.shutdown.subscribe();

        // Spawn cleanup task
        let cleanup_server = Arc::clone(&self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
                cleanup_server.cleanup_completed_jobs().await;
            }
        });

        loop {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, addr)) => {
                            let server = Arc::clone(&self);
                            self.stats.write().await.total_connections += 1;

                            tokio::spawn(async move {
                                if let Err(e) = server.handle_connection(stream, addr).await {
                                    eprintln!("Connection error from {}: {:?}", addr, e);
                                }
                            });
                        }
                        Err(e) => {
                            eprintln!("Accept error: {:?}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    println!("Server shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle a client connection
    async fn handle_connection(
        self: Arc<Self>,
        stream: TcpStream,
        addr: SocketAddr,
    ) -> Result<(), ServerError> {
        let (reader, writer) = stream.into_split();
        let (tx, mut rx) = mpsc::channel::<BuildMessage>(100);

        // Spawn writer task
        let writer_handle = tokio::spawn(async move {
            let mut writer = writer;
            while let Some(msg) = rx.recv().await {
                let encoded = msg.encode();
                if writer.write_all(&encoded).await.is_err() {
                    break;
                }
            }
        });

        // Read messages
        let mut reader = tokio::io::BufReader::new(reader);
        let mut client_id: Option<String> = None;

        loop {
            // Read length prefix
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf).await {
                Ok(_) => {}
                Err(_) => break,
            }
            let len = u32::from_be_bytes(len_buf) as usize;

            // Sanity check
            if len > 100 * 1024 * 1024 {
                let _ = tx
                    .send(BuildMessage::Error {
                        code: ErrorCode::InvalidProtocol,
                        message: "Message too large".into(),
                        details: None,
                    })
                    .await;
                break;
            }

            // Read message
            let mut msg_buf = vec![0u8; len];
            if reader.read_exact(&mut msg_buf).await.is_err() {
                break;
            }

            // Parse message
            let msg: BuildMessage = match serde_json::from_slice(&msg_buf) {
                Ok(m) => m,
                Err(e) => {
                    let _ = tx
                        .send(BuildMessage::Error {
                            code: ErrorCode::InvalidProtocol,
                            message: format!("Invalid message: {}", e),
                            details: None,
                        })
                        .await;
                    continue;
                }
            };

            // Handle message
            let response = self.handle_message(&msg, &tx, &mut client_id, addr).await;

            if let Some(resp) = response {
                let _ = tx.send(resp).await;
            }

            // Check for goodbye
            if matches!(msg, BuildMessage::Goodbye { .. }) {
                break;
            }
        }

        // Cleanup
        if let Some(id) = client_id {
            self.clients.write().await.remove(&id);
        }

        writer_handle.abort();
        Ok(())
    }

    /// Handle a single message
    async fn handle_message(
        &self,
        msg: &BuildMessage,
        tx: &mpsc::Sender<BuildMessage>,
        client_id: &mut Option<String>,
        addr: SocketAddr,
    ) -> Option<BuildMessage> {
        match msg {
            BuildMessage::Hello {
                protocol_version,
                client_id: cid,
                capabilities,
            } => {
                if *protocol_version != PROTOCOL_VERSION {
                    return Some(BuildMessage::Error {
                        code: ErrorCode::InvalidProtocol,
                        message: format!("Unsupported protocol version: {}", protocol_version),
                        details: None,
                    });
                }

                let session_id = format!("{:016x}", rand::random::<u64>());
                *client_id = Some(cid.clone());

                // Register client
                let handle = ClientHandle {
                    id: cid.clone(),
                    session_id: session_id.clone(),
                    addr,
                    connected_at: Instant::now(),
                    tx: tx.clone(),
                    capabilities: capabilities.clone(),
                };
                self.clients.write().await.insert(cid.clone(), handle);

                Some(BuildMessage::Welcome {
                    server_id: self.config.server_name.clone(),
                    server_capabilities: self.get_capabilities().await,
                    session_id,
                })
            }

            BuildMessage::Ping { timestamp } => Some(BuildMessage::Pong {
                timestamp: *timestamp,
            }),

            BuildMessage::SubmitJob { job_id, job } => {
                let cid = match client_id {
                    Some(id) => id.clone(),
                    None => {
                        return Some(BuildMessage::Error {
                            code: ErrorCode::Unauthorized,
                            message: "Not authenticated".into(),
                            details: None,
                        });
                    }
                };

                // Validate job
                if let Err(e) = self.validate_job(job).await {
                    return Some(BuildMessage::JobRejected {
                        job_id: job_id.clone(),
                        reason: e,
                    });
                }

                // Check cache
                if self.config.cache_enabled {
                    if let Some(ref cache_key) = job.cache_key {
                        if let Some(result) = self.check_cache(cache_key).await {
                            self.stats.write().await.cache_hits += 1;
                            return Some(BuildMessage::JobComplete {
                                job_id: job_id.clone(),
                                result,
                            });
                        }
                        self.stats.write().await.cache_misses += 1;
                    }
                }

                // Queue job
                let queued = QueuedJob {
                    job_id: job_id.clone(),
                    job: job.clone(),
                    client_id: cid,
                    priority: job.priority,
                    submitted_at: Instant::now(),
                };

                let mut queue = self.job_queue.lock().await;
                if let Err(e) = queue.push(queued) {
                    return Some(BuildMessage::JobRejected {
                        job_id: job_id.clone(),
                        reason: e,
                    });
                }
                drop(queue);

                self.stats.write().await.total_jobs += 1;

                // Try to schedule
                self.try_schedule().await;

                Some(BuildMessage::JobAccepted {
                    job_id: job_id.clone(),
                    estimated_duration: self.estimate_duration(job),
                    assigned_worker: None,
                })
            }

            BuildMessage::CancelJob { job_id, reason: _ } => {
                // Check if in queue
                let mut queue = self.job_queue.lock().await;
                if queue.remove(job_id).is_some() {
                    return Some(BuildMessage::JobCancelled {
                        job_id: job_id.clone(),
                    });
                }
                drop(queue);

                // Check if active
                let active = self.active_jobs.read().await;
                if let Some(job) = active.get(job_id) {
                    let workers = self.workers.read().await;
                    if let Some(worker) = workers.get(&job.worker_id) {
                        let _ = worker.tx.send(WorkerMessage::Cancel(job_id.clone())).await;
                    }
                    return Some(BuildMessage::JobCancelled {
                        job_id: job_id.clone(),
                    });
                }

                Some(BuildMessage::Error {
                    code: ErrorCode::NotFound,
                    message: format!("Job not found: {}", job_id),
                    details: None,
                })
            }

            BuildMessage::QueryJob { job_id } => {
                // Check queue
                let queue = self.job_queue.lock().await;
                if queue.get(job_id).is_some() {
                    return Some(BuildMessage::JobProgress {
                        job_id: job_id.clone(),
                        progress: JobProgress {
                            phase: "queued".into(),
                            percent: 0,
                            current_file: None,
                            diagnostics: vec![],
                        },
                    });
                }
                drop(queue);

                // Check active
                let active = self.active_jobs.read().await;
                if let Some(job) = active.get(job_id) {
                    return Some(BuildMessage::JobProgress {
                        job_id: job_id.clone(),
                        progress: job.progress.clone(),
                    });
                }
                drop(active);

                // Check completed
                let completed = self.completed_jobs.read().await;
                if let Some(job) = completed.get(job_id) {
                    return Some(BuildMessage::JobComplete {
                        job_id: job_id.clone(),
                        result: job.result.clone(),
                    });
                }

                Some(BuildMessage::Error {
                    code: ErrorCode::NotFound,
                    message: format!("Job not found: {}", job_id),
                    details: None,
                })
            }

            BuildMessage::QueryServer => {
                let workers = self.workers.read().await;
                let worker_infos: Vec<WorkerInfo> = workers
                    .values()
                    .map(|w| WorkerInfo {
                        id: w.id.clone(),
                        name: w.name.clone(),
                        targets: w.targets.clone(),
                        status: w.status,
                        current_job: None,
                        jobs_completed: w.stats.jobs_completed,
                        uptime: w.last_heartbeat.elapsed(),
                    })
                    .collect();

                let queue = self.job_queue.lock().await;
                let active = self.active_jobs.read().await;

                Some(BuildMessage::ServerStatus {
                    workers: worker_infos,
                    queue_length: queue.len(),
                    active_jobs: active.len(),
                    uptime: self.start_time.elapsed(),
                })
            }

            _ => None,
        }
    }

    /// Get server capabilities
    async fn get_capabilities(&self) -> ServerCapabilities {
        let workers = self.workers.read().await;
        let mut targets: Vec<String> = workers
            .values()
            .flat_map(|w| w.targets.iter().cloned())
            .collect();
        targets.sort();
        targets.dedup();

        // Add default targets if no workers
        if targets.is_empty() {
            targets = vec![
                "x86_64-unknown-linux-gnu".into(),
                "aarch64-unknown-linux-gnu".into(),
            ];
        }

        ServerCapabilities {
            targets,
            max_job_size: 100 * 1024 * 1024, // 100MB
            max_concurrent_jobs: self.config.max_queue_size,
            features: vec!["cache".into(), "streaming".into(), "compression".into()],
            cache_enabled: self.config.cache_enabled,
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    /// Validate a job
    async fn validate_job(&self, job: &BuildJob) -> Result<(), String> {
        // Check sources exist
        if job.sources.is_empty() {
            return Err("No source files".into());
        }

        // Validate timeout
        if job.timeout > self.config.job_timeout {
            return Err(format!(
                "Timeout exceeds maximum ({}s)",
                self.config.job_timeout.as_secs()
            ));
        }

        Ok(())
    }

    /// Check cache for result
    async fn check_cache(&self, _cache_key: &str) -> Option<BuildResult> {
        // TODO: Implement cache lookup
        None
    }

    /// Estimate job duration
    fn estimate_duration(&self, _job: &BuildJob) -> Option<Duration> {
        // TODO: ML-based estimation based on historical data
        Some(Duration::from_secs(60))
    }

    /// Try to schedule pending jobs
    async fn try_schedule(&self) {
        let mut queue = self.job_queue.lock().await;
        let mut workers = self.workers.write().await;

        // Find idle workers
        let idle_workers: Vec<String> = workers
            .iter()
            .filter(|(_, w)| w.status == WorkerStatus::Idle)
            .map(|(id, _)| id.clone())
            .collect();

        for worker_id in idle_workers {
            if let Some(worker) = workers.get_mut(&worker_id) {
                // Find job for this worker's targets
                for target in &worker.targets.clone() {
                    if let Some(job) = queue.pop_for_target(target) {
                        // Assign job
                        worker.status = WorkerStatus::Busy;
                        let _ = worker
                            .tx
                            .send(WorkerMessage::Execute(job.job.clone(), job.job_id.clone()))
                            .await;

                        // Track active job
                        let active = ActiveJob {
                            job_id: job.job_id.clone(),
                            job: job.job,
                            client_id: job.client_id,
                            worker_id: worker_id.clone(),
                            started_at: Instant::now(),
                            progress: JobProgress::default(),
                        };

                        drop(queue);
                        drop(workers);
                        self.active_jobs
                            .write()
                            .await
                            .insert(active.job_id.clone(), active);
                        return;
                    }
                }
            }
        }
    }

    /// Cleanup old completed jobs
    async fn cleanup_completed_jobs(&self) {
        let mut completed = self.completed_jobs.write().await;
        let cutoff = Instant::now() - Duration::from_secs(300); // 5 minutes

        completed.retain(|_, job| job.completed_at > cutoff);
    }

    /// Register a worker
    pub async fn register_worker(
        &self,
        id: String,
        name: String,
        targets: Vec<String>,
        tx: mpsc::Sender<WorkerMessage>,
    ) {
        let handle = WorkerHandle {
            id: id.clone(),
            name,
            targets,
            status: WorkerStatus::Idle,
            tx,
            stats: WorkerStats::default(),
            last_heartbeat: Instant::now(),
        };

        self.workers.write().await.insert(id, handle);
    }

    /// Report job completion
    pub async fn complete_job(&self, job_id: &str, result: BuildResult) {
        // Remove from active
        let active = self.active_jobs.write().await.remove(job_id);

        if let Some(job) = active {
            // Update worker status
            let mut workers = self.workers.write().await;
            if let Some(worker) = workers.get_mut(&job.worker_id) {
                worker.status = WorkerStatus::Idle;
                if result.success {
                    worker.stats.jobs_completed += 1;
                } else {
                    worker.stats.jobs_failed += 1;
                }
                worker.stats.total_duration += result.duration;
            }
            drop(workers);

            // Notify client
            let clients = self.clients.read().await;
            if let Some(client) = clients.get(&job.client_id) {
                let msg = if result.success {
                    BuildMessage::JobComplete {
                        job_id: job_id.to_string(),
                        result: result.clone(),
                    }
                } else {
                    BuildMessage::JobFailed {
                        job_id: job_id.to_string(),
                        error: BuildError {
                            code: "build_failed".into(),
                            message: "Build failed".into(),
                            diagnostics: result.diagnostics.clone(),
                            fatal: true,
                        },
                    }
                };
                let _ = client.tx.send(msg).await;
            }

            // Update stats
            let mut stats = self.stats.write().await;
            if result.success {
                stats.completed_jobs += 1;
            } else {
                stats.failed_jobs += 1;
            }

            // Cache result
            self.completed_jobs.write().await.insert(
                job_id.to_string(),
                CompletedJob {
                    job_id: job_id.to_string(),
                    result,
                    completed_at: Instant::now(),
                },
            );

            // Try to schedule more jobs
            self.try_schedule().await;
        }
    }

    /// Shutdown the server
    pub fn shutdown(&self) {
        let _ = self.shutdown.send(());
    }

    /// Get server statistics
    pub async fn get_stats(&self) -> ServerStatsSnapshot {
        let stats = self.stats.read().await;
        let queue = self.job_queue.lock().await;
        let active = self.active_jobs.read().await;
        let workers = self.workers.read().await;

        ServerStatsSnapshot {
            total_jobs: stats.total_jobs,
            completed_jobs: stats.completed_jobs,
            failed_jobs: stats.failed_jobs,
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
            queued_jobs: queue.len(),
            active_jobs: active.len(),
            connected_workers: workers.len(),
            uptime: self.start_time.elapsed(),
        }
    }
}

/// Server statistics snapshot
#[derive(Debug, Clone)]
pub struct ServerStatsSnapshot {
    pub total_jobs: u64,
    pub completed_jobs: u64,
    pub failed_jobs: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub queued_jobs: usize,
    pub active_jobs: usize,
    pub connected_workers: usize,
    pub uptime: Duration,
}

/// Server error
#[derive(Debug)]
pub enum ServerError {
    Io(std::io::Error),
    Protocol(String),
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerError::Io(e) => write!(f, "IO error: {}", e),
            ServerError::Protocol(msg) => write!(f, "Protocol error: {}", msg),
        }
    }
}

impl std::error::Error for ServerError {}

impl From<std::io::Error> for ServerError {
    fn from(e: std::io::Error) -> Self {
        ServerError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.max_connections, 100);
        assert_eq!(config.max_queue_size, 1000);
        assert!(config.cache_enabled);
    }

    #[test]
    fn test_job_queue() {
        let mut queue = JobQueue::new(10);

        let job1 = QueuedJob {
            job_id: "job1".into(),
            job: BuildJob {
                job_type: JobType::Build,
                sources: vec![],
                dependencies: vec![],
                config: BuildConfig::default(),
                target: "x86_64-unknown-linux-gnu".into(),
                env: HashMap::new(),
                outputs: vec![],
                priority: 0,
                timeout: Duration::from_secs(60),
                cache_key: None,
            },
            client_id: "client1".into(),
            priority: 0,
            submitted_at: Instant::now(),
        };

        let job2 = QueuedJob {
            job_id: "job2".into(),
            job: BuildJob {
                job_type: JobType::Build,
                sources: vec![],
                dependencies: vec![],
                config: BuildConfig::default(),
                target: "x86_64-unknown-linux-gnu".into(),
                env: HashMap::new(),
                outputs: vec![],
                priority: 10, // Higher priority
                timeout: Duration::from_secs(60),
                cache_key: None,
            },
            client_id: "client1".into(),
            priority: 10,
            submitted_at: Instant::now(),
        };

        queue.push(job1).unwrap();
        queue.push(job2).unwrap();

        // Higher priority should come first
        let popped = queue.pop().unwrap();
        assert_eq!(popped.job_id, "job2");
    }
}

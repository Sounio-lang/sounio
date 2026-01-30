//! Distributed build protocol definitions
//!
//! This module defines the protocol for communication between build clients,
//! servers, and workers in a distributed build system.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Protocol version
pub const PROTOCOL_VERSION: u32 = 1;

/// Message types for build protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BuildMessage {
    // =========================================================================
    // Connection Messages
    // =========================================================================
    /// Client hello (initiate connection)
    Hello {
        protocol_version: u32,
        client_id: String,
        capabilities: ClientCapabilities,
    },

    /// Server welcome (accept connection)
    Welcome {
        server_id: String,
        server_capabilities: ServerCapabilities,
        session_id: String,
    },

    /// Heartbeat (keep connection alive)
    Ping {
        timestamp: u64,
    },
    Pong {
        timestamp: u64,
    },

    /// Graceful disconnect
    Goodbye {
        reason: String,
    },

    // =========================================================================
    // Job Messages
    // =========================================================================
    /// Submit a build job
    SubmitJob {
        job_id: String,
        job: BuildJob,
    },

    /// Job accepted by server
    JobAccepted {
        job_id: String,
        estimated_duration: Option<Duration>,
        assigned_worker: Option<String>,
    },

    /// Job rejected by server
    JobRejected {
        job_id: String,
        reason: String,
    },

    /// Job status update
    JobProgress {
        job_id: String,
        progress: JobProgress,
    },

    /// Job completed successfully
    JobComplete {
        job_id: String,
        result: BuildResult,
    },

    /// Job failed
    JobFailed {
        job_id: String,
        error: BuildError,
    },

    /// Cancel a job
    CancelJob {
        job_id: String,
        reason: String,
    },

    /// Job was cancelled
    JobCancelled {
        job_id: String,
    },

    // =========================================================================
    // File Transfer Messages
    // =========================================================================
    /// Request file upload
    RequestUpload {
        transfer_id: String,
        files: Vec<FileInfo>,
    },

    /// Upload accepted
    UploadAccepted {
        transfer_id: String,
        upload_url: String,
    },

    /// File chunk
    FileChunk {
        transfer_id: String,
        file_index: usize,
        chunk_index: usize,
        data: Vec<u8>,
        is_last: bool,
    },

    /// Upload complete
    UploadComplete {
        transfer_id: String,
    },

    /// Request file download
    RequestDownload {
        transfer_id: String,
        artifact_ids: Vec<String>,
    },

    /// Download ready
    DownloadReady {
        transfer_id: String,
        download_url: String,
    },

    // =========================================================================
    // Query Messages
    // =========================================================================
    /// Query job status
    QueryJob {
        job_id: String,
    },

    /// Query server status
    QueryServer,

    /// Server status response
    ServerStatus {
        workers: Vec<WorkerInfo>,
        queue_length: usize,
        active_jobs: usize,
        uptime: Duration,
    },

    /// Error response
    Error {
        code: ErrorCode,
        message: String,
        details: Option<HashMap<String, String>>,
    },
}

/// Client capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Supported compression formats
    pub compression: Vec<String>,

    /// Maximum concurrent uploads
    pub max_uploads: usize,

    /// Supports streaming
    pub streaming: bool,

    /// Client version
    pub version: String,
}

/// Server capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Available targets
    pub targets: Vec<String>,

    /// Maximum job size (bytes)
    pub max_job_size: usize,

    /// Maximum concurrent jobs
    pub max_concurrent_jobs: usize,

    /// Supported features
    pub features: Vec<String>,

    /// Cache enabled
    pub cache_enabled: bool,

    /// Server version
    pub version: String,
}

/// Build job definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildJob {
    /// Job type
    pub job_type: JobType,

    /// Source files (content-addressed)
    pub sources: Vec<SourceFile>,

    /// Dependencies
    pub dependencies: Vec<Dependency>,

    /// Build configuration
    pub config: BuildConfig,

    /// Target specification
    pub target: String,

    /// Environment variables
    pub env: HashMap<String, String>,

    /// Requested outputs
    pub outputs: Vec<OutputSpec>,

    /// Priority (higher = more important)
    pub priority: i32,

    /// Timeout
    pub timeout: Duration,

    /// Cache key (for result caching)
    pub cache_key: Option<String>,
}

/// Job type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobType {
    /// Compile a single file
    Compile,

    /// Link object files
    Link,

    /// Full build (compile + link)
    Build,

    /// Run tests
    Test,

    /// Generate documentation
    Doc,

    /// Check (no codegen)
    Check,
}

/// Source file reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFile {
    /// Relative path
    pub path: PathBuf,

    /// Content hash (SHA-256)
    pub hash: String,

    /// File size
    pub size: usize,

    /// Is this the main file?
    pub is_main: bool,
}

/// Dependency reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    /// Dependency name
    pub name: String,

    /// Version
    pub version: String,

    /// Content hash
    pub hash: String,

    /// Dependency type
    pub dep_type: DependencyType,
}

/// Dependency type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyType {
    /// Compiled library
    Library,

    /// Source dependency
    Source,

    /// System library
    System,
}

/// Build configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Optimization level
    pub opt_level: String,

    /// Debug info
    pub debug_info: bool,

    /// Features enabled
    pub features: Vec<String>,

    /// Compiler flags
    pub flags: Vec<String>,

    /// Profile
    pub profile: String,
}

impl Default for BuildConfig {
    fn default() -> Self {
        BuildConfig {
            opt_level: "0".to_string(),
            debug_info: true,
            features: Vec::new(),
            flags: Vec::new(),
            profile: "debug".to_string(),
        }
    }
}

/// Output specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    /// Output type
    pub output_type: OutputType,

    /// Output path (relative)
    pub path: PathBuf,
}

/// Output type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputType {
    Object,
    Executable,
    Library,
    StaticLib,
    DynamicLib,
    Metadata,
    DebugInfo,
}

/// Job progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobProgress {
    /// Current phase
    pub phase: String,

    /// Progress percentage (0-100)
    pub percent: u8,

    /// Current file being processed
    pub current_file: Option<String>,

    /// Diagnostics emitted so far
    pub diagnostics: Vec<Diagnostic>,
}

impl Default for JobProgress {
    fn default() -> Self {
        JobProgress {
            phase: "pending".to_string(),
            percent: 0,
            current_file: None,
            diagnostics: Vec::new(),
        }
    }
}

/// Diagnostic message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

/// Build result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildResult {
    /// Success
    pub success: bool,

    /// Build duration
    pub duration: Duration,

    /// Output artifacts
    pub artifacts: Vec<Artifact>,

    /// Diagnostics
    pub diagnostics: Vec<Diagnostic>,

    /// Cache hit?
    pub cache_hit: bool,

    /// Build statistics
    pub stats: BuildStats,
}

/// Build artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Artifact ID (content hash)
    pub id: String,

    /// Output type
    pub output_type: OutputType,

    /// Relative path
    pub path: PathBuf,

    /// Size in bytes
    pub size: usize,

    /// Download URL (temporary)
    pub download_url: Option<String>,
}

/// Build statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildStats {
    /// Files compiled
    pub files_compiled: usize,

    /// Lines of code
    pub lines_of_code: usize,

    /// Cache hits
    pub cache_hits: usize,

    /// Cache misses
    pub cache_misses: usize,

    /// Peak memory usage
    pub peak_memory: usize,

    /// CPU time
    pub cpu_time: Duration,
}

/// Build error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildError {
    /// Error code
    pub code: String,

    /// Error message
    pub message: String,

    /// Diagnostics
    pub diagnostics: Vec<Diagnostic>,

    /// Is this a fatal error?
    pub fatal: bool,
}

/// File info for upload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    /// File path
    pub path: PathBuf,

    /// Content hash
    pub hash: String,

    /// File size
    pub size: usize,
}

/// Worker info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Worker ID
    pub id: String,

    /// Worker name
    pub name: String,

    /// Supported targets
    pub targets: Vec<String>,

    /// Current status
    pub status: WorkerStatus,

    /// Current job (if busy)
    pub current_job: Option<String>,

    /// Jobs completed
    pub jobs_completed: u64,

    /// Uptime
    pub uptime: Duration,
}

/// Worker status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerStatus {
    Idle,
    Busy,
    Draining,
    Offline,
}

/// Error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidProtocol,
    Unauthorized,
    NotFound,
    Timeout,
    Cancelled,
    InternalError,
    QuotaExceeded,
    InvalidJob,
    UnsupportedTarget,
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorCode::InvalidProtocol => write!(f, "invalid_protocol"),
            ErrorCode::Unauthorized => write!(f, "unauthorized"),
            ErrorCode::NotFound => write!(f, "not_found"),
            ErrorCode::Timeout => write!(f, "timeout"),
            ErrorCode::Cancelled => write!(f, "cancelled"),
            ErrorCode::InternalError => write!(f, "internal_error"),
            ErrorCode::QuotaExceeded => write!(f, "quota_exceeded"),
            ErrorCode::InvalidJob => write!(f, "invalid_job"),
            ErrorCode::UnsupportedTarget => write!(f, "unsupported_target"),
        }
    }
}

// =============================================================================
// Message Encoding/Decoding
// =============================================================================

impl BuildMessage {
    /// Encode message to bytes (length-prefixed JSON)
    pub fn encode(&self) -> Vec<u8> {
        let json = serde_json::to_vec(self).unwrap();
        let len = (json.len() as u32).to_be_bytes();
        let mut buf = Vec::with_capacity(4 + json.len());
        buf.extend_from_slice(&len);
        buf.extend_from_slice(&json);
        buf
    }

    /// Decode message from bytes
    pub fn decode(data: &[u8]) -> Result<Self, ProtocolError> {
        if data.len() < 4 {
            return Err(ProtocolError::IncompletMessage);
        }

        let len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < 4 + len {
            return Err(ProtocolError::IncompletMessage);
        }

        serde_json::from_slice(&data[4..4 + len]).map_err(ProtocolError::Json)
    }

    /// Get message length from header
    pub fn message_length(header: &[u8; 4]) -> usize {
        u32::from_be_bytes(*header) as usize
    }
}

/// Protocol error
#[derive(Debug)]
pub enum ProtocolError {
    IncompletMessage,
    Json(serde_json::Error),
    Io(std::io::Error),
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolError::IncompletMessage => write!(f, "incomplete message"),
            ProtocolError::Json(e) => write!(f, "JSON error: {}", e),
            ProtocolError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for ProtocolError {}

impl From<std::io::Error> for ProtocolError {
    fn from(e: std::io::Error) -> Self {
        ProtocolError::Io(e)
    }
}

impl From<serde_json::Error> for ProtocolError {
    fn from(e: serde_json::Error) -> Self {
        ProtocolError::Json(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_encode_decode() {
        let msg = BuildMessage::Hello {
            protocol_version: PROTOCOL_VERSION,
            client_id: "test-client".to_string(),
            capabilities: ClientCapabilities::default(),
        };

        let encoded = msg.encode();
        let decoded = BuildMessage::decode(&encoded).unwrap();

        match decoded {
            BuildMessage::Hello {
                protocol_version,
                client_id,
                ..
            } => {
                assert_eq!(protocol_version, PROTOCOL_VERSION);
                assert_eq!(client_id, "test-client");
            }
            _ => panic!("wrong message type"),
        }
    }

    #[test]
    fn test_job_type_serialization() {
        let job_type = JobType::Build;
        let json = serde_json::to_string(&job_type).unwrap();
        assert_eq!(json, "\"build\"");

        let parsed: JobType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, JobType::Build);
    }

    #[test]
    fn test_build_config_default() {
        let config = BuildConfig::default();
        assert_eq!(config.opt_level, "0");
        assert!(config.debug_info);
        assert_eq!(config.profile, "debug");
    }
}

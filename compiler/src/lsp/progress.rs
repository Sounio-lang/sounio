//! Progress reporting and cancellation support for LSP
//!
//! Provides infrastructure for:
//! - Work-done progress reporting to clients
//! - Cancellable long-running operations
//! - Task lifecycle management
//!
//! # Usage
//!
//! ```ignore
//! // Create a progress reporter
//! let progress = ProgressReporter::new(client.clone());
//!
//! // Start a cancellable operation
//! let token = progress.begin("Indexing workspace", Some("Scanning files...")).await;
//!
//! // Report progress updates
//! progress.report(&token, "Processing file.sio", Some(50)).await;
//!
//! // Complete the operation
//! progress.end(&token, Some("Indexed 150 files")).await;
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::{Mutex, RwLock};
use tower_lsp::lsp_types::*;
use tower_lsp::Client;

/// Unique token counter for progress operations
static TOKEN_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique progress token
fn generate_token() -> ProgressToken {
    let id = TOKEN_COUNTER.fetch_add(1, Ordering::SeqCst);
    ProgressToken::Number(id as i32)
}

/// Active progress entry tracking state
#[derive(Debug)]
struct ActiveProgress {
    /// The progress token
    token: ProgressToken,
    /// Title of the operation
    title: String,
    /// Whether the operation can be cancelled
    cancellable: bool,
    /// Current percentage (0-100)
    percentage: Option<u32>,
}

/// Progress reporter for sending work-done progress notifications to the LSP client.
///
/// Manages the lifecycle of progress operations, including:
/// - Creating progress tokens
/// - Sending begin/report/end notifications
/// - Tracking active operations
pub struct ProgressReporter {
    /// LSP client for sending notifications
    client: Client,
    /// Active progress operations
    active: Arc<Mutex<HashMap<String, ActiveProgress>>>,
}

impl ProgressReporter {
    /// Create a new progress reporter
    pub fn new(client: Client) -> Self {
        Self {
            client,
            active: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Begin a new progress operation.
    ///
    /// Creates a work-done progress token and sends the begin notification
    /// to the client. Returns the token for subsequent updates.
    ///
    /// # Arguments
    ///
    /// * `title` - Title of the operation (e.g., "Indexing workspace")
    /// * `message` - Optional initial message (e.g., "Scanning files...")
    /// * `cancellable` - Whether the operation can be cancelled by the client
    ///
    /// # Returns
    ///
    /// A ProgressToken that can be used for subsequent report/end calls
    pub async fn begin(
        &self,
        title: &str,
        message: Option<&str>,
        cancellable: bool,
    ) -> ProgressToken {
        let token = generate_token();

        // Request creation of the progress token
        // Note: Some clients require this before we can send progress notifications
        if let Err(e) = self
            .client
            .send_request::<request::WorkDoneProgressCreate>(WorkDoneProgressCreateParams {
                token: token.clone(),
            })
            .await
        {
            // Log but continue - some clients don't require this
            tracing::debug!("Failed to create progress token: {:?}", e);
        }

        // Send the begin notification
        self.client
            .send_notification::<notification::Progress>(ProgressParams {
                token: token.clone(),
                value: ProgressParamsValue::WorkDone(WorkDoneProgress::Begin(
                    WorkDoneProgressBegin {
                        title: title.to_string(),
                        cancellable: Some(cancellable),
                        message: message.map(|s| s.to_string()),
                        percentage: Some(0),
                    },
                )),
            })
            .await;

        // Track the active progress
        let key = token_to_string(&token);
        let mut active = self.active.lock().await;
        active.insert(
            key,
            ActiveProgress {
                token: token.clone(),
                title: title.to_string(),
                cancellable,
                percentage: Some(0),
            },
        );

        token
    }

    /// Report progress on an ongoing operation.
    ///
    /// # Arguments
    ///
    /// * `token` - The progress token from begin()
    /// * `message` - Current status message
    /// * `percentage` - Optional progress percentage (0-100)
    pub async fn report(&self, token: &ProgressToken, message: &str, percentage: Option<u32>) {
        // Clamp percentage to valid range
        let percentage = percentage.map(|p| p.min(100));

        self.client
            .send_notification::<notification::Progress>(ProgressParams {
                token: token.clone(),
                value: ProgressParamsValue::WorkDone(WorkDoneProgress::Report(
                    WorkDoneProgressReport {
                        cancellable: None, // Keep existing cancellable state
                        message: Some(message.to_string()),
                        percentage,
                    },
                )),
            })
            .await;

        // Update tracked state
        let key = token_to_string(token);
        let mut active = self.active.lock().await;
        if let Some(progress) = active.get_mut(&key) {
            progress.percentage = percentage;
        }
    }

    /// End a progress operation.
    ///
    /// # Arguments
    ///
    /// * `token` - The progress token from begin()
    /// * `message` - Optional completion message
    pub async fn end(&self, token: &ProgressToken, message: Option<&str>) {
        self.client
            .send_notification::<notification::Progress>(ProgressParams {
                token: token.clone(),
                value: ProgressParamsValue::WorkDone(WorkDoneProgress::End(WorkDoneProgressEnd {
                    message: message.map(|s| s.to_string()),
                })),
            })
            .await;

        // Remove from active tracking
        let key = token_to_string(token);
        let mut active = self.active.lock().await;
        active.remove(&key);
    }

    /// Check if a progress operation is still active
    pub async fn is_active(&self, token: &ProgressToken) -> bool {
        let key = token_to_string(token);
        let active = self.active.lock().await;
        active.contains_key(&key)
    }

    /// Get the number of active progress operations
    pub async fn active_count(&self) -> usize {
        let active = self.active.lock().await;
        active.len()
    }

    /// Cancel all active progress operations
    pub async fn cancel_all(&self) {
        let active = self.active.lock().await;
        for (_, progress) in active.iter() {
            self.client
                .send_notification::<notification::Progress>(ProgressParams {
                    token: progress.token.clone(),
                    value: ProgressParamsValue::WorkDone(WorkDoneProgress::End(
                        WorkDoneProgressEnd {
                            message: Some("Cancelled".to_string()),
                        },
                    )),
                })
                .await;
        }
    }
}

/// Convert a progress token to a string key for HashMap storage
fn token_to_string(token: &ProgressToken) -> String {
    match token {
        ProgressToken::Number(n) => format!("number:{}", n),
        ProgressToken::String(s) => format!("string:{}", s),
    }
}

/// Cancellation token for cooperative cancellation of async tasks.
///
/// This is a simple wrapper around an atomic boolean that tasks can
/// poll to check if they should stop execution.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    /// The underlying cancellation flag
    cancelled: Arc<std::sync::atomic::AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Request cancellation
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Create a child token that is cancelled when this token is cancelled
    pub fn child_token(&self) -> Self {
        // For simplicity, child shares the same cancellation state
        Self {
            cancelled: self.cancelled.clone(),
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// A cancellable async task that wraps a future with cancellation support.
///
/// The task can be polled for completion while periodically checking
/// the cancellation token.
pub struct CancellableTask<T> {
    /// Cancellation token for cooperative cancellation
    cancel_token: CancellationToken,
    /// The underlying task future
    task: Pin<Box<dyn Future<Output = T> + Send>>,
}

impl<T> CancellableTask<T> {
    /// Create a new cancellable task
    pub fn new<F>(cancel_token: CancellationToken, future: F) -> Self
    where
        F: Future<Output = T> + Send + 'static,
    {
        Self {
            cancel_token,
            task: Box::pin(future),
        }
    }

    /// Get the cancellation token
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    /// Check if the task has been cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Cancel the task
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }
}

/// Result of a cancellable operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CancellableResult<T> {
    /// The operation completed successfully
    Completed(T),
    /// The operation was cancelled
    Cancelled,
}

impl<T> CancellableResult<T> {
    /// Check if the operation was cancelled
    pub fn is_cancelled(&self) -> bool {
        matches!(self, CancellableResult::Cancelled)
    }

    /// Check if the operation completed
    pub fn is_completed(&self) -> bool {
        matches!(self, CancellableResult::Completed(_))
    }

    /// Get the result if completed, or None if cancelled
    pub fn into_option(self) -> Option<T> {
        match self {
            CancellableResult::Completed(v) => Some(v),
            CancellableResult::Cancelled => None,
        }
    }

    /// Map the result value
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> CancellableResult<U> {
        match self {
            CancellableResult::Completed(v) => CancellableResult::Completed(f(v)),
            CancellableResult::Cancelled => CancellableResult::Cancelled,
        }
    }
}

/// Manager for tracking cancellable operations by request ID.
///
/// Used to implement LSP's `$/cancelRequest` notification.
pub struct CancellationManager {
    /// Map of request ID to cancellation token
    tokens: RwLock<HashMap<RequestId, CancellationToken>>,
}

/// Request ID type (matches tower-lsp's internal representation)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RequestId {
    /// Integer request ID
    Number(i64),
    /// String request ID
    String(String),
}

impl From<NumberOrString> for RequestId {
    fn from(value: NumberOrString) -> Self {
        match value {
            NumberOrString::Number(n) => RequestId::Number(n as i64),
            NumberOrString::String(s) => RequestId::String(s),
        }
    }
}

impl CancellationManager {
    /// Create a new cancellation manager
    pub fn new() -> Self {
        Self {
            tokens: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new cancellable request
    pub async fn register(&self, id: RequestId) -> CancellationToken {
        let token = CancellationToken::new();
        let mut tokens = self.tokens.write().await;
        tokens.insert(id, token.clone());
        token
    }

    /// Cancel a request by ID
    pub async fn cancel(&self, id: &RequestId) -> bool {
        let tokens = self.tokens.read().await;
        if let Some(token) = tokens.get(id) {
            token.cancel();
            true
        } else {
            false
        }
    }

    /// Unregister a request (called when request completes)
    pub async fn unregister(&self, id: &RequestId) {
        let mut tokens = self.tokens.write().await;
        tokens.remove(id);
    }

    /// Check if a request has been cancelled
    pub async fn is_cancelled(&self, id: &RequestId) -> bool {
        let tokens = self.tokens.read().await;
        tokens.get(id).map(|t| t.is_cancelled()).unwrap_or(false)
    }

    /// Get the number of registered requests
    pub async fn count(&self) -> usize {
        let tokens = self.tokens.read().await;
        tokens.len()
    }
}

impl Default for CancellationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper trait for running operations with progress reporting
pub trait WithProgress {
    /// Run an operation with progress reporting
    fn with_progress<'a, T, F, Fut>(
        &'a self,
        title: &'a str,
        operation: F,
    ) -> Pin<Box<dyn Future<Output = T> + Send + 'a>>
    where
        F: FnOnce(ProgressContext<'a>) -> Fut + Send + 'a,
        Fut: Future<Output = T> + Send + 'a,
        T: Send + 'a;
}

/// Context passed to operations run with progress reporting
pub struct ProgressContext<'a> {
    /// The progress reporter
    pub reporter: &'a ProgressReporter,
    /// The progress token for this operation
    pub token: ProgressToken,
    /// Cancellation token
    pub cancel_token: CancellationToken,
}

impl<'a> ProgressContext<'a> {
    /// Report progress
    pub async fn report(&self, message: &str, percentage: Option<u32>) {
        self.reporter.report(&self.token, message, percentage).await;
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Check cancellation and return early if cancelled
    pub fn check_cancelled(&self) -> Result<(), Cancelled> {
        if self.is_cancelled() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Error type indicating an operation was cancelled
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "operation cancelled")
    }
}

impl std::error::Error for Cancelled {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_child() {
        let parent = CancellationToken::new();
        let child = parent.child_token();

        assert!(!child.is_cancelled());

        parent.cancel();
        assert!(child.is_cancelled());
    }

    #[test]
    fn test_cancellable_result() {
        let completed: CancellableResult<i32> = CancellableResult::Completed(42);
        assert!(completed.is_completed());
        assert!(!completed.is_cancelled());
        assert_eq!(completed.into_option(), Some(42));

        let cancelled: CancellableResult<i32> = CancellableResult::Cancelled;
        assert!(!cancelled.is_completed());
        assert!(cancelled.is_cancelled());
        assert_eq!(cancelled.into_option(), None);
    }

    #[test]
    fn test_cancellable_result_map() {
        let result: CancellableResult<i32> = CancellableResult::Completed(21);
        let mapped = result.map(|x| x * 2);
        assert_eq!(mapped, CancellableResult::Completed(42));

        let cancelled: CancellableResult<i32> = CancellableResult::Cancelled;
        let mapped = cancelled.map(|x| x * 2);
        assert_eq!(mapped, CancellableResult::Cancelled);
    }

    #[test]
    fn test_token_to_string() {
        let num_token = ProgressToken::Number(42);
        assert_eq!(token_to_string(&num_token), "number:42");

        let str_token = ProgressToken::String("test".to_string());
        assert_eq!(token_to_string(&str_token), "string:test");
    }

    #[test]
    fn test_request_id_from_number_or_string() {
        let num = NumberOrString::Number(42);
        let id: RequestId = num.into();
        assert_eq!(id, RequestId::Number(42));

        let str = NumberOrString::String("test".to_string());
        let id: RequestId = str.into();
        assert_eq!(id, RequestId::String("test".to_string()));
    }

    #[tokio::test]
    async fn test_cancellation_manager() {
        let manager = CancellationManager::new();

        let token = manager.register(RequestId::Number(1)).await;
        assert_eq!(manager.count().await, 1);
        assert!(!manager.is_cancelled(&RequestId::Number(1)).await);

        manager.cancel(&RequestId::Number(1)).await;
        assert!(manager.is_cancelled(&RequestId::Number(1)).await);
        assert!(token.is_cancelled());

        manager.unregister(&RequestId::Number(1)).await;
        assert_eq!(manager.count().await, 0);
    }
}

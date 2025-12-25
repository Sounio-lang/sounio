//! Cross-platform file system watcher
//!
//! Provides file change detection with:
//! - Platform-specific backends (polling as fallback)
//! - Event debouncing
//! - Pattern-based filtering
//! - Recursive watching

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// File system event
#[derive(Debug, Clone)]
pub struct FsEvent {
    /// Affected path
    pub path: PathBuf,

    /// Event kind
    pub kind: FsEventKind,

    /// Timestamp
    pub timestamp: Instant,

    /// Additional attributes
    pub attrs: FsEventAttrs,
}

/// File system event kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FsEventKind {
    /// File or directory created
    Create,

    /// File modified
    Modify,

    /// File or directory deleted
    Delete,

    /// File or directory renamed
    Rename,

    /// Metadata changed (permissions, etc.)
    Metadata,

    /// Access (read)
    Access,

    /// Other/unknown
    Other,
}

/// Additional event attributes
#[derive(Debug, Clone, Default)]
pub struct FsEventAttrs {
    /// For rename events, the new path
    pub rename_to: Option<PathBuf>,

    /// Process ID that caused the event (if available)
    pub pid: Option<u32>,

    /// Is this a directory?
    pub is_dir: bool,
}

/// Watcher configuration
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Paths to watch
    pub paths: Vec<PathBuf>,

    /// Watch recursively
    pub recursive: bool,

    /// Debounce duration
    pub debounce: Duration,

    /// Event kinds to watch
    pub events: HashSet<FsEventKind>,

    /// Patterns to include (glob)
    pub include: Vec<String>,

    /// Patterns to exclude (glob)
    pub exclude: Vec<String>,

    /// Follow symlinks
    pub follow_symlinks: bool,

    /// Poll interval (for polling backend)
    pub poll_interval: Duration,
}

impl Default for WatchConfig {
    fn default() -> Self {
        WatchConfig {
            paths: vec![PathBuf::from(".")],
            recursive: true,
            debounce: Duration::from_millis(100),
            events: [
                FsEventKind::Create,
                FsEventKind::Modify,
                FsEventKind::Delete,
                FsEventKind::Rename,
            ]
            .into_iter()
            .collect(),
            include: vec!["**/*.sio".to_string(), "**/d.toml".to_string()],
            exclude: vec![
                "**/target/**".to_string(),
                "**/.git/**".to_string(),
                "**/node_modules/**".to_string(),
                "**/*.tmp".to_string(),
                "**/*~".to_string(),
            ],
            follow_symlinks: false,
            poll_interval: Duration::from_secs(1),
        }
    }
}

impl WatchConfig {
    /// Create config for watching a single path
    pub fn for_path(path: PathBuf) -> Self {
        WatchConfig {
            paths: vec![path],
            ..Default::default()
        }
    }

    /// Add an include pattern
    pub fn include(mut self, pattern: &str) -> Self {
        self.include.push(pattern.to_string());
        self
    }

    /// Add an exclude pattern
    pub fn exclude(mut self, pattern: &str) -> Self {
        self.exclude.push(pattern.to_string());
        self
    }

    /// Set debounce duration
    pub fn debounce(mut self, duration: Duration) -> Self {
        self.debounce = duration;
        self
    }

    /// Set recursive mode
    pub fn recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }
}

/// File system watcher
pub struct Watcher {
    /// Configuration
    config: WatchConfig,

    /// Event receiver
    rx: Receiver<FsEvent>,

    /// Internal sender (for debouncer)
    tx: Sender<FsEvent>,

    /// Raw event sender (from backend)
    raw_tx: Sender<FsEvent>,

    /// Raw event receiver
    raw_rx: Arc<Mutex<Receiver<FsEvent>>>,

    /// Debouncer
    debouncer: Arc<Mutex<Debouncer>>,

    /// Watcher thread handle
    thread: Option<JoinHandle<()>>,

    /// Backend thread handle
    backend_thread: Option<JoinHandle<()>>,

    /// Stop signal
    stop: Arc<Mutex<bool>>,

    /// Compiled include patterns
    include_patterns: Vec<glob::Pattern>,

    /// Compiled exclude patterns
    exclude_patterns: Vec<glob::Pattern>,
}

impl Watcher {
    /// Create a new watcher with default config
    pub fn new() -> Result<Self, WatchError> {
        Self::with_config(WatchConfig::default())
    }

    /// Create a new watcher with custom config
    pub fn with_config(config: WatchConfig) -> Result<Self, WatchError> {
        let (tx, rx) = channel();
        let (raw_tx, raw_rx) = channel();
        let debouncer = Arc::new(Mutex::new(Debouncer::new(config.debounce)));
        let stop = Arc::new(Mutex::new(false));

        // Compile glob patterns
        let include_patterns: Vec<_> = config
            .include
            .iter()
            .filter_map(|p| glob::Pattern::new(p).ok())
            .collect();

        let exclude_patterns: Vec<_> = config
            .exclude
            .iter()
            .filter_map(|p| glob::Pattern::new(p).ok())
            .collect();

        Ok(Watcher {
            config,
            rx,
            tx,
            raw_tx,
            raw_rx: Arc::new(Mutex::new(raw_rx)),
            debouncer,
            thread: None,
            backend_thread: None,
            stop,
            include_patterns,
            exclude_patterns,
        })
    }

    /// Start watching
    pub fn start(&mut self) -> Result<(), WatchError> {
        // Start polling backend
        let paths = self.config.paths.clone();
        let recursive = self.config.recursive;
        let poll_interval = self.config.poll_interval;
        let raw_tx = self.raw_tx.clone();
        let stop = Arc::clone(&self.stop);

        let backend_handle = thread::Builder::new()
            .name("file-watcher-backend".into())
            .spawn(move || {
                let mut backend = PollBackend::new(paths, recursive, poll_interval);
                backend.run(raw_tx, stop);
            })?;

        self.backend_thread = Some(backend_handle);

        // Start debouncer thread
        let tx = self.tx.clone();
        let debouncer = Arc::clone(&self.debouncer);
        let raw_rx = Arc::clone(&self.raw_rx);
        let stop = Arc::clone(&self.stop);
        let include_patterns = self.include_patterns.clone();
        let exclude_patterns = self.exclude_patterns.clone();
        let debounce_duration = self.config.debounce;

        let handle = thread::Builder::new()
            .name("file-watcher-debouncer".into())
            .spawn(move || {
                Self::debounce_loop(
                    tx,
                    debouncer,
                    raw_rx,
                    stop,
                    include_patterns,
                    exclude_patterns,
                    debounce_duration,
                );
            })?;

        self.thread = Some(handle);
        Ok(())
    }

    /// Debounce loop
    fn debounce_loop(
        tx: Sender<FsEvent>,
        debouncer: Arc<Mutex<Debouncer>>,
        raw_rx: Arc<Mutex<Receiver<FsEvent>>>,
        stop: Arc<Mutex<bool>>,
        include_patterns: Vec<glob::Pattern>,
        exclude_patterns: Vec<glob::Pattern>,
        debounce_duration: Duration,
    ) {
        loop {
            // Use unwrap_or_else to recover from poisoned mutex instead of panicking
            if *stop.lock().unwrap_or_else(|e| e.into_inner()) {
                break;
            }

            // Receive raw events
            {
                let rx = raw_rx.lock().unwrap_or_else(|e| e.into_inner());
                while let Ok(event) = rx.try_recv() {
                    if Self::should_include(&event.path, &include_patterns, &exclude_patterns) {
                        debouncer
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .add(event);
                    }
                }
            }

            // Flush debounced events
            let events = debouncer.lock().unwrap_or_else(|e| e.into_inner()).flush();
            for event in events {
                let _ = tx.send(event);
            }

            thread::sleep(debounce_duration / 2);
        }
    }

    /// Check if path should be included
    fn should_include(
        path: &Path,
        include_patterns: &[glob::Pattern],
        exclude_patterns: &[glob::Pattern],
    ) -> bool {
        let path_str = path.to_string_lossy();

        // Check excludes first
        for pattern in exclude_patterns {
            if pattern.matches(&path_str) {
                return false;
            }
        }

        // Check includes
        if include_patterns.is_empty() {
            return true;
        }

        for pattern in include_patterns {
            if pattern.matches(&path_str) {
                return true;
            }
        }

        false
    }

    /// Stop watching
    pub fn stop(&mut self) {
        *self.stop.lock().unwrap_or_else(|e| e.into_inner()) = true;

        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }

        if let Some(handle) = self.backend_thread.take() {
            let _ = handle.join();
        }
    }

    /// Get next event (blocking)
    pub fn next(&self) -> Option<FsEvent> {
        self.rx.recv().ok()
    }

    /// Get next event with timeout
    pub fn next_timeout(&self, timeout: Duration) -> Option<FsEvent> {
        self.rx.recv_timeout(timeout).ok()
    }

    /// Try to get next event (non-blocking)
    pub fn try_next(&self) -> Option<FsEvent> {
        self.rx.try_recv().ok()
    }

    /// Get all pending events
    pub fn drain(&self) -> Vec<FsEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            events.push(event);
        }
        events
    }

    /// Add a path to watch
    pub fn watch(&mut self, path: &Path) -> Result<(), WatchError> {
        if !path.exists() {
            return Err(WatchError::PathNotFound(path.to_path_buf()));
        }
        self.config.paths.push(path.to_path_buf());
        Ok(())
    }

    /// Stop watching a path
    pub fn unwatch(&mut self, path: &Path) -> Result<(), WatchError> {
        self.config.paths.retain(|p| p != path);
        Ok(())
    }

    /// Check if watcher is running
    pub fn is_running(&self) -> bool {
        !*self.stop.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Default for Watcher {
    fn default() -> Self {
        Self::new().expect("Failed to create watcher")
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        self.stop();
    }
}

// =============================================================================
// Event Debouncer
// =============================================================================

/// Event debouncer to coalesce rapid changes
struct Debouncer {
    /// Debounce duration
    duration: Duration,

    /// Pending events by path
    pending: HashMap<PathBuf, (FsEvent, Instant)>,
}

impl Debouncer {
    fn new(duration: Duration) -> Self {
        Debouncer {
            duration,
            pending: HashMap::new(),
        }
    }

    /// Add an event (replaces previous for same path)
    fn add(&mut self, event: FsEvent) {
        self.pending
            .insert(event.path.clone(), (event, Instant::now()));
    }

    /// Flush events that have exceeded debounce duration
    fn flush(&mut self) -> Vec<FsEvent> {
        let now = Instant::now();
        let mut ready = Vec::new();

        self.pending.retain(|_, (event, added)| {
            if now.duration_since(*added) >= self.duration {
                ready.push(event.clone());
                false
            } else {
                true
            }
        });

        ready
    }

    /// Clear all pending events
    #[allow(dead_code)]
    fn clear(&mut self) {
        self.pending.clear();
    }
}

// =============================================================================
// Polling Backend
// =============================================================================

/// File state for change detection
#[derive(Clone)]
struct FileState {
    mtime: std::time::SystemTime,
    size: u64,
}

/// Polling-based file watcher backend
struct PollBackend {
    /// Watched paths
    paths: Vec<PathBuf>,

    /// Watch recursively
    recursive: bool,

    /// Poll interval
    interval: Duration,

    /// File states
    states: HashMap<PathBuf, FileState>,
}

impl PollBackend {
    fn new(paths: Vec<PathBuf>, recursive: bool, interval: Duration) -> Self {
        PollBackend {
            paths,
            recursive,
            interval,
            states: HashMap::new(),
        }
    }

    fn run(&mut self, tx: Sender<FsEvent>, stop: Arc<Mutex<bool>>) {
        // Initial scan
        for path in self.paths.clone() {
            let _ = self.scan_recursive(&path);
        }

        while !*stop.lock().unwrap_or_else(|e| e.into_inner()) {
            for path in self.paths.clone() {
                self.poll_path(&path, &tx);
            }
            thread::sleep(self.interval);
        }
    }

    fn scan_recursive(&mut self, path: &Path) -> Result<(), WatchError> {
        if path.is_file() {
            self.add_file(path)?;
        } else if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let entry_path = entry.path();

                    if entry_path.is_file() {
                        self.add_file(&entry_path)?;
                    } else if entry_path.is_dir() && self.recursive {
                        self.scan_recursive(&entry_path)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn add_file(&mut self, path: &Path) -> Result<(), WatchError> {
        if let Ok(metadata) = std::fs::metadata(path) {
            self.states.insert(
                path.to_path_buf(),
                FileState {
                    mtime: metadata
                        .modified()
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                    size: metadata.len(),
                },
            );
        }
        Ok(())
    }

    fn poll_path(&mut self, path: &Path, tx: &Sender<FsEvent>) {
        // Check existing files
        let paths: Vec<_> = self.states.keys().cloned().collect();

        for file_path in paths {
            match std::fs::metadata(&file_path) {
                Ok(metadata) => {
                    let mtime = metadata
                        .modified()
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    let size = metadata.len();

                    if let Some(state) = self.states.get_mut(&file_path) {
                        if mtime != state.mtime || size != state.size {
                            state.mtime = mtime;
                            state.size = size;
                            let _ = tx.send(FsEvent {
                                path: file_path,
                                kind: FsEventKind::Modify,
                                timestamp: Instant::now(),
                                attrs: FsEventAttrs::default(),
                            });
                        }
                    }
                }
                Err(_) => {
                    // File deleted
                    if self.states.remove(&file_path).is_some() {
                        let _ = tx.send(FsEvent {
                            path: file_path,
                            kind: FsEventKind::Delete,
                            timestamp: Instant::now(),
                            attrs: FsEventAttrs::default(),
                        });
                    }
                }
            }
        }

        // Check for new files in directories
        self.check_new_files(path, tx);
    }

    fn check_new_files(&mut self, path: &Path, tx: &Sender<FsEvent>) {
        if !path.is_dir() {
            return;
        }

        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let file_path = entry.path();

                if file_path.is_file() && !self.states.contains_key(&file_path) {
                    if let Ok(metadata) = std::fs::metadata(&file_path) {
                        self.states.insert(
                            file_path.clone(),
                            FileState {
                                mtime: metadata
                                    .modified()
                                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                                size: metadata.len(),
                            },
                        );
                        let _ = tx.send(FsEvent {
                            path: file_path,
                            kind: FsEventKind::Create,
                            timestamp: Instant::now(),
                            attrs: FsEventAttrs::default(),
                        });
                    }
                } else if file_path.is_dir() && self.recursive {
                    self.check_new_files(&file_path, tx);
                }
            }
        }
    }
}

// =============================================================================
// Errors
// =============================================================================

/// Watch error
#[derive(Debug)]
pub enum WatchError {
    /// IO error
    Io(std::io::Error),

    /// Path not found
    PathNotFound(PathBuf),

    /// Too many watches
    TooManyWatches,

    /// Permission denied
    PermissionDenied(PathBuf),

    /// Backend error
    Backend(String),

    /// Pattern error
    Pattern(glob::PatternError),
}

impl From<std::io::Error> for WatchError {
    fn from(e: std::io::Error) -> Self {
        WatchError::Io(e)
    }
}

impl From<glob::PatternError> for WatchError {
    fn from(e: glob::PatternError) -> Self {
        WatchError::Pattern(e)
    }
}

impl std::fmt::Display for WatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WatchError::Io(e) => write!(f, "IO error: {}", e),
            WatchError::PathNotFound(p) => write!(f, "Path not found: {}", p.display()),
            WatchError::TooManyWatches => write!(f, "Too many watches"),
            WatchError::PermissionDenied(p) => write!(f, "Permission denied: {}", p.display()),
            WatchError::Backend(s) => write!(f, "Backend error: {}", s),
            WatchError::Pattern(e) => write!(f, "Pattern error: {}", e),
        }
    }
}

impl std::error::Error for WatchError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_watch_config_default() {
        let config = WatchConfig::default();
        assert!(config.recursive);
        assert!(!config.include.is_empty());
        assert!(!config.exclude.is_empty());
    }

    #[test]
    fn test_watch_config_builder() {
        let config = WatchConfig::default()
            .include("**/*.rs")
            .exclude("**/test/**")
            .debounce(Duration::from_millis(200))
            .recursive(false);

        assert!(!config.recursive);
        assert!(config.include.contains(&"**/*.rs".to_string()));
        assert!(config.exclude.contains(&"**/test/**".to_string()));
        assert_eq!(config.debounce, Duration::from_millis(200));
    }

    #[test]
    fn test_debouncer() {
        let mut debouncer = Debouncer::new(Duration::from_millis(10));

        debouncer.add(FsEvent {
            path: PathBuf::from("test.sio"),
            kind: FsEventKind::Modify,
            timestamp: Instant::now(),
            attrs: FsEventAttrs::default(),
        });

        // Immediate flush should return nothing
        let events = debouncer.flush();
        assert!(events.is_empty());

        // Wait and flush should return events
        thread::sleep(Duration::from_millis(20));
        let events = debouncer.flush();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_watcher_creation() {
        let watcher = Watcher::new();
        assert!(watcher.is_ok());
    }

    #[test]
    fn test_should_include() {
        let include = vec![glob::Pattern::new("**/*.sio").unwrap()];
        let exclude = vec![glob::Pattern::new("**/target/**").unwrap()];

        assert!(Watcher::should_include(
            Path::new("src/main.sio"),
            &include,
            &exclude
        ));
        assert!(!Watcher::should_include(
            Path::new("target/main.sio"),
            &include,
            &exclude
        ));
        assert!(!Watcher::should_include(
            Path::new("src/main.rs"),
            &include,
            &exclude
        ));
    }

    #[test]
    fn test_poll_backend_file_detection() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.sio");

        fs::write(&test_file, "// test").unwrap();

        let mut backend = PollBackend::new(
            vec![temp_dir.path().to_path_buf()],
            true,
            Duration::from_millis(50),
        );

        let _ = backend.scan_recursive(temp_dir.path());
        assert!(backend.states.contains_key(&test_file));
    }
}

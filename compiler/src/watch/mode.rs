//! Watch mode for continuous compilation
//!
//! Provides:
//! - Automatic rebuilds on file changes
//! - Keyboard controls (quit, rebuild, pause)
//! - Status display
//! - Optional test running and command execution

use std::collections::HashSet;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use super::watcher::{FsEvent, FsEventKind, WatchConfig, WatchError, Watcher};

/// Watch mode configuration
#[derive(Debug, Clone)]
pub struct WatchModeConfig {
    /// Watch configuration
    pub watch: WatchConfig,

    /// Clear screen before each build
    pub clear_screen: bool,

    /// Show notifications (platform-dependent)
    pub notifications: bool,

    /// Run tests after successful build
    pub run_tests: bool,

    /// Run command after successful build
    pub exec: Option<String>,

    /// Delay before rebuilding
    pub delay: Duration,

    /// Ignore initial build
    pub no_initial_build: bool,

    /// Poll mode (force polling backend)
    pub poll: bool,

    /// Shell to use for exec
    pub shell: String,

    /// Verbose output
    pub verbose: bool,
}

impl Default for WatchModeConfig {
    fn default() -> Self {
        WatchModeConfig {
            watch: WatchConfig::default(),
            clear_screen: true,
            notifications: false,
            run_tests: false,
            exec: None,
            delay: Duration::from_millis(100),
            no_initial_build: false,
            poll: false,
            shell: if cfg!(windows) {
                "cmd".into()
            } else {
                "sh".into()
            },
            verbose: false,
        }
    }
}

/// Watch mode state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchState {
    /// Waiting for file changes
    Idle,

    /// Building
    Building,

    /// Running tests
    Testing,

    /// Running command
    Running,

    /// Paused (not watching)
    Paused,
}

impl WatchState {
    /// Get display string for state
    pub fn display(&self) -> &'static str {
        match self {
            WatchState::Idle => "Watching for changes...",
            WatchState::Building => "Building...",
            WatchState::Testing => "Running tests...",
            WatchState::Running => "Running command...",
            WatchState::Paused => "[PAUSED] Press 'p' to resume",
        }
    }

    /// Get color for state (ANSI color code)
    pub fn color(&self) -> &'static str {
        match self {
            WatchState::Idle => "\x1b[32m",     // Green
            WatchState::Building => "\x1b[33m", // Yellow
            WatchState::Testing => "\x1b[34m",  // Blue
            WatchState::Running => "\x1b[36m",  // Cyan
            WatchState::Paused => "\x1b[35m",   // Magenta
        }
    }
}

/// Build result
#[derive(Debug, Clone)]
pub struct BuildResult {
    /// Build succeeded
    pub success: bool,

    /// Number of errors
    pub errors: usize,

    /// Number of warnings
    pub warnings: usize,

    /// Build duration
    pub duration: Duration,

    /// Changed files that triggered build
    pub changed_files: Vec<PathBuf>,
}

impl Default for BuildResult {
    fn default() -> Self {
        BuildResult {
            success: true,
            errors: 0,
            warnings: 0,
            duration: Duration::ZERO,
            changed_files: Vec::new(),
        }
    }
}

/// Watch mode statistics
#[derive(Debug, Clone, Default)]
pub struct WatchStats {
    /// Total builds
    pub build_count: usize,

    /// Successful builds
    pub success_count: usize,

    /// Failed builds
    pub error_count: usize,

    /// Total build time
    pub total_build_time: Duration,

    /// Files changed
    pub files_changed: usize,

    /// Start time
    pub start_time: Option<Instant>,
}

impl WatchStats {
    /// Get average build time
    pub fn avg_build_time(&self) -> Duration {
        if self.build_count > 0 {
            self.total_build_time / self.build_count as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.build_count > 0 {
            self.success_count as f64 / self.build_count as f64 * 100.0
        } else {
            100.0
        }
    }
}

/// Watch mode controller
pub struct WatchMode {
    /// Configuration
    config: WatchModeConfig,

    /// File watcher
    watcher: Watcher,

    /// Current state
    state: Arc<Mutex<WatchState>>,

    /// Statistics
    stats: Arc<Mutex<WatchStats>>,

    /// Last build result
    last_result: Arc<Mutex<Option<BuildResult>>>,

    /// Changed files pending build
    changed_files: Arc<Mutex<HashSet<PathBuf>>>,

    /// Stop signal
    stop: Arc<AtomicBool>,

    /// Build function
    build_fn: Option<Box<dyn Fn(&[PathBuf]) -> BuildResult + Send + Sync>>,
}

impl WatchMode {
    /// Create new watch mode
    pub fn new(config: WatchModeConfig) -> Result<Self, WatchError> {
        let watcher = Watcher::with_config(config.watch.clone())?;

        Ok(WatchMode {
            config,
            watcher,
            state: Arc::new(Mutex::new(WatchState::Idle)),
            stats: Arc::new(Mutex::new(WatchStats::default())),
            last_result: Arc::new(Mutex::new(None)),
            changed_files: Arc::new(Mutex::new(HashSet::new())),
            stop: Arc::new(AtomicBool::new(false)),
            build_fn: None,
        })
    }

    /// Set custom build function
    pub fn set_build_fn<F>(&mut self, f: F)
    where
        F: Fn(&[PathBuf]) -> BuildResult + Send + Sync + 'static,
    {
        self.build_fn = Some(Box::new(f));
    }

    /// Run watch mode (blocking)
    pub fn run(&mut self) -> Result<(), WatchModeError> {
        // Initialize stats
        self.stats.lock().unwrap().start_time = Some(Instant::now());

        // Start watcher
        self.watcher.start()?;

        // Initial build
        if !self.config.no_initial_build {
            self.trigger_build()?;
        }

        self.print_status();
        self.print_help_hint();

        // Main loop
        loop {
            if self.stop.load(Ordering::SeqCst) {
                break;
            }

            // Check keyboard input (non-blocking)
            if let Some(action) = self.check_keyboard() {
                match action {
                    KeyAction::Quit => break,
                    KeyAction::Rebuild => self.trigger_build()?,
                    KeyAction::Pause => self.toggle_pause(),
                    KeyAction::Clear => self.clear_screen(),
                    KeyAction::Help => self.print_help(),
                    KeyAction::Stats => self.print_stats(),
                    KeyAction::None => {}
                }
            }

            // Check file changes
            if *self.state.lock().unwrap() != WatchState::Paused {
                while let Some(event) = self.watcher.try_next() {
                    self.handle_event(event)?;
                }

                let should_rebuild = !self.changed_files.lock().unwrap().is_empty();

                if should_rebuild {
                    // Wait for more changes to coalesce
                    thread::sleep(self.config.delay);

                    // Drain any additional changes
                    while let Some(event) = self.watcher.try_next() {
                        self.handle_event(event)?;
                    }

                    self.trigger_build()?;
                }
            }

            // Small sleep to avoid busy waiting
            thread::sleep(Duration::from_millis(50));
        }

        self.print_final_stats();
        Ok(())
    }

    /// Handle a file system event
    fn handle_event(&mut self, event: FsEvent) -> Result<(), WatchModeError> {
        match event.kind {
            FsEventKind::Create | FsEventKind::Modify | FsEventKind::Delete => {
                if self.config.verbose {
                    println!(
                        "  {} {:?}: {}",
                        match event.kind {
                            FsEventKind::Create => "+",
                            FsEventKind::Modify => "~",
                            FsEventKind::Delete => "-",
                            _ => "?",
                        },
                        event.kind,
                        event.path.display()
                    );
                }
                self.changed_files.lock().unwrap().insert(event.path);
            }
            FsEventKind::Rename => {
                self.changed_files
                    .lock()
                    .unwrap()
                    .insert(event.path.clone());
                if let Some(new_path) = event.attrs.rename_to {
                    self.changed_files.lock().unwrap().insert(new_path);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Check for keyboard input
    fn check_keyboard(&self) -> Option<KeyAction> {
        // Simple non-blocking stdin check
        // In a full implementation, we'd use crossterm or similar
        #[cfg(unix)]
        {
            use std::io::Read;

            // Set stdin to non-blocking temporarily
            let mut stdin = io::stdin();
            let mut buf = [0u8; 1];

            // Try to read without blocking
            if let Ok(n) = stdin.read(&mut buf)
                && n > 0
            {
                return Some(match buf[0] {
                    b'q' | 27 => KeyAction::Quit,    // 27 = ESC
                    b'r' | 13 => KeyAction::Rebuild, // 13 = Enter
                    b'p' => KeyAction::Pause,
                    b'c' => KeyAction::Clear,
                    b'h' | b'?' => KeyAction::Help,
                    b's' => KeyAction::Stats,
                    _ => KeyAction::None,
                });
            }
        }

        None
    }

    /// Trigger a build
    fn trigger_build(&mut self) -> Result<(), WatchModeError> {
        let changed: Vec<_> = self.changed_files.lock().unwrap().drain().collect();
        *self.state.lock().unwrap() = WatchState::Building;

        if self.config.clear_screen {
            self.clear_screen();
        }

        let start = Instant::now();

        // Print build header
        println!("\x1b[1m=== Building ===\x1b[0m");
        if !changed.is_empty() {
            println!("Changed files: {}", changed.len());
        }
        println!();

        // Run build
        let result = if let Some(ref build_fn) = self.build_fn {
            build_fn(&changed)
        } else {
            self.default_build(&changed)
        };

        let result = BuildResult {
            duration: start.elapsed(),
            changed_files: changed,
            ..result
        };

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.build_count += 1;
            stats.total_build_time += result.duration;
            stats.files_changed += result.changed_files.len();

            if result.success {
                stats.success_count += 1;
            } else {
                stats.error_count += 1;
            }
        }

        // Print result
        self.print_build_result(&result);

        // Store result
        *self.last_result.lock().unwrap() = Some(result.clone());

        // Run tests if configured
        if result.success && self.config.run_tests {
            *self.state.lock().unwrap() = WatchState::Testing;
            self.run_tests()?;
        }

        // Run command if configured
        if result.success
            && let Some(ref cmd) = self.config.exec
        {
            *self.state.lock().unwrap() = WatchState::Running;
            self.run_command(cmd)?;
        }

        *self.state.lock().unwrap() = WatchState::Idle;
        self.print_status();

        Ok(())
    }

    /// Default build implementation
    fn default_build(&self, _changed: &[PathBuf]) -> BuildResult {
        // In a real implementation, this would invoke the build system
        // For now, return a mock success
        BuildResult {
            success: true,
            errors: 0,
            warnings: 0,
            duration: Duration::ZERO,
            changed_files: Vec::new(),
        }
    }

    /// Run tests
    fn run_tests(&self) -> Result<(), WatchModeError> {
        println!("\n\x1b[1m=== Running Tests ===\x1b[0m\n");

        let status = if cfg!(windows) {
            std::process::Command::new("cmd")
                .args(["/C", "souc", "test"])
                .status()
        } else {
            std::process::Command::new("sh")
                .args(["-c", "souc test"])
                .status()
        };

        if let Err(e) = status {
            eprintln!("Failed to run tests: {}", e);
        }

        Ok(())
    }

    /// Run a command
    fn run_command(&self, cmd: &str) -> Result<(), WatchModeError> {
        println!("\n\x1b[1m=== Running: {} ===\x1b[0m\n", cmd);

        let status = if cfg!(windows) {
            std::process::Command::new("cmd").args(["/C", cmd]).status()
        } else {
            std::process::Command::new("sh").args(["-c", cmd]).status()
        };

        if let Err(e) = status {
            eprintln!("Failed to run command: {}", e);
        }

        Ok(())
    }

    /// Toggle pause state
    fn toggle_pause(&self) {
        let mut state = self.state.lock().unwrap();
        *state = if *state == WatchState::Paused {
            println!("\n\x1b[32mResumed watching\x1b[0m");
            WatchState::Idle
        } else {
            println!("\n\x1b[35mPaused\x1b[0m");
            WatchState::Paused
        };
    }

    /// Clear the screen
    fn clear_screen(&self) {
        print!("\x1b[2J\x1b[1;1H");
        let _ = io::stdout().flush();
    }

    /// Print current status
    fn print_status(&self) {
        let state = *self.state.lock().unwrap();
        let stats = self.stats.lock().unwrap();

        println!(
            "\n{}[watch]\x1b[0m {} [builds: {}, errors: {}]",
            state.color(),
            state.display(),
            stats.build_count,
            stats.error_count
        );
    }

    /// Print help hint
    fn print_help_hint(&self) {
        println!("\x1b[90mPress 'h' for help, 'q' to quit\x1b[0m");
    }

    /// Print help
    fn print_help(&self) {
        println!("\n\x1b[1mWatch Mode Help\x1b[0m");
        println!("===============");
        println!();
        println!("  q, Esc     - Quit watch mode");
        println!("  r, Enter   - Force rebuild");
        println!("  p          - Pause/resume watching");
        println!("  c          - Clear screen");
        println!("  s          - Show statistics");
        println!("  h, ?       - Show this help");
        println!();
    }

    /// Print statistics
    fn print_stats(&self) {
        let stats = self.stats.lock().unwrap();

        println!("\n\x1b[1mWatch Statistics\x1b[0m");
        println!("================");
        println!();
        println!("  Builds:        {}", stats.build_count);
        println!("  Successful:    {}", stats.success_count);
        println!("  Failed:        {}", stats.error_count);
        println!("  Success rate:  {:.1}%", stats.success_rate());
        println!("  Files changed: {}", stats.files_changed);
        println!(
            "  Avg build:     {:.2}s",
            stats.avg_build_time().as_secs_f64()
        );
        println!(
            "  Total build:   {:.2}s",
            stats.total_build_time.as_secs_f64()
        );

        if let Some(start) = stats.start_time {
            let elapsed = start.elapsed();
            println!("  Watch time:    {:.0}s", elapsed.as_secs_f64());
        }

        println!();
    }

    /// Print build result
    fn print_build_result(&self, result: &BuildResult) {
        println!();

        if result.success {
            println!(
                "\x1b[32m[SUCCESS]\x1b[0m Build completed in {:.2}s",
                result.duration.as_secs_f64()
            );
        } else {
            println!(
                "\x1b[31m[FAILED]\x1b[0m Build failed with {} error(s)",
                result.errors
            );
        }

        if result.warnings > 0 {
            println!("\x1b[33m         {} warning(s)\x1b[0m", result.warnings);
        }
    }

    /// Print final statistics
    fn print_final_stats(&self) {
        let stats = self.stats.lock().unwrap();

        println!("\n\x1b[1mSession Summary\x1b[0m");
        println!("---------------");
        println!(
            "  {} builds ({} successful, {} failed)",
            stats.build_count, stats.success_count, stats.error_count
        );

        if let Some(start) = stats.start_time {
            println!("  Watch time: {:.0}s", start.elapsed().as_secs_f64());
        }

        println!();
    }

    /// Stop watch mode
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Get current state
    pub fn state(&self) -> WatchState {
        *self.state.lock().unwrap()
    }

    /// Get statistics
    pub fn stats(&self) -> WatchStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get last build result
    pub fn last_result(&self) -> Option<BuildResult> {
        self.last_result.lock().unwrap().clone()
    }
}

/// Key action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyAction {
    None,
    Quit,
    Rebuild,
    Pause,
    Clear,
    Help,
    Stats,
}

/// Watch mode error
#[derive(Debug)]
pub enum WatchModeError {
    /// Watcher error
    Watcher(WatchError),

    /// IO error
    Io(io::Error),

    /// Build error
    Build(String),
}

impl From<WatchError> for WatchModeError {
    fn from(e: WatchError) -> Self {
        WatchModeError::Watcher(e)
    }
}

impl From<io::Error> for WatchModeError {
    fn from(e: io::Error) -> Self {
        WatchModeError::Io(e)
    }
}

impl std::fmt::Display for WatchModeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WatchModeError::Watcher(e) => write!(f, "Watcher error: {}", e),
            WatchModeError::Io(e) => write!(f, "IO error: {}", e),
            WatchModeError::Build(s) => write!(f, "Build error: {}", s),
        }
    }
}

impl std::error::Error for WatchModeError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watch_mode_config_default() {
        let config = WatchModeConfig::default();
        assert!(config.clear_screen);
        assert!(!config.run_tests);
        assert!(config.exec.is_none());
    }

    #[test]
    fn test_watch_state_display() {
        assert!(!WatchState::Idle.display().is_empty());
        assert!(!WatchState::Building.display().is_empty());
        assert!(!WatchState::Paused.display().is_empty());
    }

    #[test]
    fn test_watch_stats() {
        let mut stats = WatchStats::default();
        stats.build_count = 10;
        stats.success_count = 8;
        stats.error_count = 2;
        stats.total_build_time = Duration::from_secs(30);

        assert_eq!(stats.avg_build_time(), Duration::from_secs(3));
        assert_eq!(stats.success_rate(), 80.0);
    }

    #[test]
    fn test_build_result_default() {
        let result = BuildResult::default();
        assert!(result.success);
        assert_eq!(result.errors, 0);
        assert_eq!(result.warnings, 0);
    }
}

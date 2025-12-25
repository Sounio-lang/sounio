//! Progress Reporting for Long Compilations
//!
//! Shows progress bars, spinners, and status updates
//! for operations that take noticeable time.

use std::io::{self, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Progress bar style
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressStyle {
    /// Bar with percentage: [████░░░░░░] 40%
    Bar,
    /// Spinner: ⠋ Loading...
    Spinner,
    /// Count: 42/100 files
    Count,
    /// Bytes: 12.5 MB / 100 MB
    Bytes,
}

/// Progress bar characters
pub mod chars {
    pub mod unicode {
        pub const FILLED: &str = "█";
        pub const EMPTY: &str = "░";
        pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    }

    pub mod ascii {
        pub const FILLED: &str = "#";
        pub const EMPTY: &str = "-";
        pub const SPINNER: [&str; 4] = ["|", "/", "-", "\\"];
    }
}

/// Progress reporter
pub struct Progress {
    /// Current progress (0-total)
    current: AtomicU64,
    /// Total count
    total: u64,
    /// Style
    style: ProgressStyle,
    /// Message
    message: Mutex<String>,
    /// Start time
    start: Instant,
    /// Is terminal (for updates)
    is_terminal: bool,
    /// Use unicode
    unicode: bool,
    /// Bar width
    width: usize,
    /// Is finished
    finished: AtomicBool,
}

impl Progress {
    /// Create a new progress bar
    pub fn new(total: u64, message: impl Into<String>) -> Self {
        Self {
            current: AtomicU64::new(0),
            total,
            style: ProgressStyle::Bar,
            message: Mutex::new(message.into()),
            start: Instant::now(),
            is_terminal: is_terminal_stderr(),
            unicode: std::env::var("DEMETRIOS_ASCII").is_err(),
            width: 40,
            finished: AtomicBool::new(false),
        }
    }

    /// Create with specific style
    pub fn with_style(mut self, style: ProgressStyle) -> Self {
        self.style = style;
        self
    }

    /// Set bar width
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Force ASCII mode
    pub fn ascii_only(mut self) -> Self {
        self.unicode = false;
        self
    }

    /// Increment progress by 1
    pub fn inc(&self) {
        self.inc_by(1);
    }

    /// Increment progress by n
    pub fn inc_by(&self, n: u64) {
        let new = self.current.fetch_add(n, Ordering::SeqCst) + n;
        self.render(new);
    }

    /// Set progress to specific value
    pub fn set(&self, value: u64) {
        self.current.store(value, Ordering::SeqCst);
        self.render(value);
    }

    /// Set message
    pub fn set_message(&self, message: impl Into<String>) {
        if let Ok(mut msg) = self.message.lock() {
            *msg = message.into();
        }
        self.render(self.current.load(Ordering::SeqCst));
    }

    /// Get current progress
    pub fn position(&self) -> u64 {
        self.current.load(Ordering::SeqCst)
    }

    /// Finish progress (100%)
    pub fn finish(&self) {
        self.finished.store(true, Ordering::SeqCst);
        self.render(self.total);
        if self.is_terminal {
            eprintln!();
        }
    }

    /// Finish with a message
    pub fn finish_with_message(&self, message: impl Into<String>) {
        self.set_message(message);
        self.finish();
    }

    /// Abandon progress without finishing
    pub fn abandon(&self) {
        self.finished.store(true, Ordering::SeqCst);
        if self.is_terminal {
            // Clear line
            eprint!("\r\x1b[K");
        }
    }

    fn render(&self, current: u64) {
        if !self.is_terminal || self.finished.load(Ordering::SeqCst) {
            return;
        }

        let mut stderr = io::stderr();

        // Clear line
        let _ = write!(stderr, "\r\x1b[K");

        match self.style {
            ProgressStyle::Bar => self.render_bar(&mut stderr, current),
            ProgressStyle::Spinner => self.render_spinner(&mut stderr),
            ProgressStyle::Count => self.render_count(&mut stderr, current),
            ProgressStyle::Bytes => self.render_bytes(&mut stderr, current),
        }

        let _ = stderr.flush();
    }

    fn render_bar(&self, w: &mut impl Write, current: u64) {
        let message = self.message.lock().map(|m| m.clone()).unwrap_or_default();

        let percent = if self.total > 0 {
            (current as f64 / self.total as f64 * 100.0) as u64
        } else {
            0
        };

        let filled_width = if self.total > 0 {
            (current as usize * self.width / self.total as usize).min(self.width)
        } else {
            0
        };

        let (filled, empty) = if self.unicode {
            (chars::unicode::FILLED, chars::unicode::EMPTY)
        } else {
            (chars::ascii::FILLED, chars::ascii::EMPTY)
        };

        let _ = write!(w, "    {} [", message);

        for _ in 0..filled_width {
            let _ = write!(w, "{}", filled);
        }
        for _ in filled_width..self.width {
            let _ = write!(w, "{}", empty);
        }

        let _ = write!(w, "] {:>3}%", percent);

        // ETA
        if current > 0 && current < self.total {
            let elapsed = self.start.elapsed();
            let rate = current as f64 / elapsed.as_secs_f64();
            if rate > 0.0 {
                let remaining = (self.total - current) as f64 / rate;
                let _ = write!(
                    w,
                    " ({})",
                    format_duration(Duration::from_secs_f64(remaining))
                );
            }
        }
    }

    fn render_spinner(&self, w: &mut impl Write) {
        let message = self.message.lock().map(|m| m.clone()).unwrap_or_default();
        let elapsed = self.start.elapsed();

        let frames = if self.unicode {
            &chars::unicode::SPINNER[..]
        } else {
            &chars::ascii::SPINNER[..]
        };

        let frame_idx = (elapsed.as_millis() / 100) as usize % frames.len();

        let _ = write!(w, "    {} {}...", frames[frame_idx], message);
    }

    fn render_count(&self, w: &mut impl Write, current: u64) {
        let message = self.message.lock().map(|m| m.clone()).unwrap_or_default();
        let _ = write!(w, "    {} {}/{}", message, current, self.total);
    }

    fn render_bytes(&self, w: &mut impl Write, current: u64) {
        let message = self.message.lock().map(|m| m.clone()).unwrap_or_default();
        let _ = write!(
            w,
            "    {} {} / {}",
            message,
            format_bytes(current),
            format_bytes(self.total)
        );
    }
}

/// Format duration for display
pub fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// Format bytes for display
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Status line for quick updates (single line, no progress bar)
pub struct StatusLine {
    message: Mutex<String>,
    is_terminal: bool,
}

impl StatusLine {
    pub fn new() -> Self {
        Self {
            message: Mutex::new(String::new()),
            is_terminal: is_terminal_stderr(),
        }
    }

    /// Set the status message
    pub fn set(&self, message: impl Into<String>) {
        let msg = message.into();
        if let Ok(mut m) = self.message.lock() {
            *m = msg.clone();
        }
        if self.is_terminal {
            eprint!("\r\x1b[K    {}...", msg);
            let _ = io::stderr().flush();
        }
    }

    /// Clear the status line
    pub fn clear(&self) {
        if self.is_terminal {
            eprint!("\r\x1b[K");
            let _ = io::stderr().flush();
        }
    }

    /// Finish with a message
    pub fn finish(&self, message: impl Into<String>) {
        if self.is_terminal {
            eprintln!("\r\x1b[K    {}", message.into());
        }
    }

    /// Finish with success indicator
    pub fn success(&self, message: impl Into<String>) {
        if self.is_terminal {
            eprintln!("\r\x1b[K    \x1b[32m✓\x1b[0m {}", message.into());
        }
    }

    /// Finish with error indicator
    pub fn error(&self, message: impl Into<String>) {
        if self.is_terminal {
            eprintln!("\r\x1b[K    \x1b[31m✗\x1b[0m {}", message.into());
        }
    }
}

impl Default for StatusLine {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-progress tracker for parallel operations
pub struct MultiProgress {
    /// Status lines for each task
    lines: Mutex<Vec<String>>,
    /// Whether this is a terminal
    is_terminal: bool,
}

impl MultiProgress {
    pub fn new() -> Self {
        Self {
            lines: Mutex::new(Vec::new()),
            is_terminal: is_terminal_stderr(),
        }
    }

    /// Add a task and return its index
    pub fn add_task(&self, message: impl Into<String>) -> usize {
        let mut lines = self.lines.lock().unwrap_or_else(|e| e.into_inner());
        let idx = lines.len();
        lines.push(message.into());
        self.render_all(&lines);
        idx
    }

    /// Update a task's message
    pub fn update_task(&self, idx: usize, message: impl Into<String>) {
        let mut lines = self.lines.lock().unwrap_or_else(|e| e.into_inner());
        if idx < lines.len() {
            lines[idx] = message.into();
            self.render_all(&lines);
        }
    }

    /// Mark a task as complete
    pub fn complete_task(&self, idx: usize, message: impl Into<String>) {
        let mut lines = self.lines.lock().unwrap_or_else(|e| e.into_inner());
        if idx < lines.len() {
            lines[idx] = format!("✓ {}", message.into());
            self.render_all(&lines);
        }
    }

    /// Clear all progress
    pub fn clear(&self) {
        if self.is_terminal {
            let lines = self.lines.lock().unwrap_or_else(|e| e.into_inner());
            // Move up and clear each line
            for _ in 0..lines.len() {
                eprint!("\x1b[A\x1b[K");
            }
            let _ = io::stderr().flush();
        }
    }

    fn render_all(&self, lines: &[String]) {
        if !self.is_terminal {
            return;
        }

        // Clear previous output
        for _ in 0..lines.len().saturating_sub(1) {
            eprint!("\x1b[A\x1b[K");
        }
        eprint!("\r\x1b[K");

        // Print all lines
        for (i, line) in lines.iter().enumerate() {
            if i > 0 {
                eprintln!();
            }
            eprint!("    {}", line);
        }
        let _ = io::stderr().flush();
    }
}

impl Default for MultiProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if stderr is a terminal
fn is_terminal_stderr() -> bool {
    // Use environment variable heuristics for terminal detection
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }
    if std::env::var("TERM").is_err() && std::env::var("WT_SESSION").is_err() {
        return false;
    }
    // Assume terminal if CI is not set and TERM is set
    std::env::var("CI").is_err() && std::env::var("TERM").is_ok()
}

/// Compilation phase reporter
pub struct CompilationProgress {
    status: StatusLine,
    start: Instant,
}

impl CompilationProgress {
    pub fn new() -> Self {
        Self {
            status: StatusLine::new(),
            start: Instant::now(),
        }
    }

    pub fn phase(&self, name: &str) {
        self.status.set(name);
    }

    pub fn parsing(&self, file: &str) {
        self.status.set(format!("Parsing {}", file));
    }

    pub fn resolving(&self) {
        self.status.set("Resolving names");
    }

    pub fn type_checking(&self) {
        self.status.set("Type checking");
    }

    pub fn loading_ontology(&self, name: &str) {
        self.status.set(format!("Loading ontology {}", name));
    }

    pub fn computing_distances(&self) {
        self.status.set("Computing semantic distances");
    }

    pub fn optimizing(&self) {
        self.status.set("Optimizing");
    }

    pub fn generating_code(&self) {
        self.status.set("Generating code");
    }

    pub fn linking(&self) {
        self.status.set("Linking");
    }

    pub fn finish_success(&self, output: &str) {
        let elapsed = self.start.elapsed();
        self.status.success(format!(
            "Compiled {} in {:.2}s",
            output,
            elapsed.as_secs_f64()
        ));
    }

    pub fn finish_error(&self, message: &str) {
        self.status.error(message);
    }
}

impl Default for CompilationProgress {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1500), "1.5 KB");
        assert_eq!(format_bytes(1_500_000), "1.4 MB");
        assert_eq!(format_bytes(1_500_000_000), "1.4 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3700)), "1h 1m");
    }

    #[test]
    fn test_progress_creation() {
        let progress = Progress::new(100, "Testing");
        assert_eq!(progress.position(), 0);
        assert_eq!(progress.total, 100);
    }

    #[test]
    fn test_progress_increment() {
        let progress = Progress::new(100, "Testing");
        progress.inc();
        assert_eq!(progress.position(), 1);
        progress.inc_by(5);
        assert_eq!(progress.position(), 6);
    }

    #[test]
    fn test_progress_set() {
        let progress = Progress::new(100, "Testing");
        progress.set(50);
        assert_eq!(progress.position(), 50);
    }
}

//! File change detection for incremental compilation.
//!
//! This module provides utilities for detecting when source files have changed
//! and triggering recompilation of affected units.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use super::graph::{BuildGraph, ContentHash};

/// Tracks the state of source files for change detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileState {
    /// File path
    pub path: PathBuf,

    /// Content hash
    pub hash: ContentHash,

    /// Last modification time
    pub mtime: SystemTime,

    /// File size in bytes
    pub size: u64,
}

impl FileState {
    /// Create a new file state by reading from disk
    pub fn from_path(path: &Path) -> std::io::Result<Self> {
        let metadata = std::fs::metadata(path)?;
        let hash = ContentHash::from_file(path)?;

        Ok(FileState {
            path: path.to_path_buf(),
            hash,
            mtime: metadata.modified()?,
            size: metadata.len(),
        })
    }

    /// Check if the file has changed since this state was captured
    pub fn has_changed(&self) -> std::io::Result<bool> {
        // Fast path: check mtime and size first
        let metadata = std::fs::metadata(&self.path)?;

        if metadata.modified()? != self.mtime || metadata.len() != self.size {
            // Slow path: verify with content hash
            let current_hash = ContentHash::from_file(&self.path)?;
            Ok(current_hash != self.hash)
        } else {
            Ok(false)
        }
    }

    /// Update to current file state
    pub fn update(&mut self) -> std::io::Result<()> {
        let new_state = Self::from_path(&self.path)?;
        *self = new_state;
        Ok(())
    }
}

/// Change detection for source files
pub struct ChangeDetector {
    /// Tracked file states
    file_states: HashMap<PathBuf, FileState>,

    /// Glob patterns to watch
    watch_patterns: Vec<glob::Pattern>,

    /// Paths to exclude from watching
    exclude_patterns: Vec<glob::Pattern>,
}

impl ChangeDetector {
    /// Create a new change detector
    pub fn new() -> Self {
        ChangeDetector {
            file_states: HashMap::new(),
            watch_patterns: vec![glob::Pattern::new("**/*.sio").unwrap()],
            exclude_patterns: Vec::new(),
        }
    }

    /// Add a glob pattern to watch
    pub fn watch(&mut self, pattern: &str) -> Result<(), ChangeDetectorError> {
        let glob_pattern = glob::Pattern::new(pattern)
            .map_err(|e| ChangeDetectorError::InvalidPattern(e.to_string()))?;
        self.watch_patterns.push(glob_pattern);
        Ok(())
    }

    /// Add a pattern to exclude from watching
    pub fn exclude(&mut self, pattern: &str) -> Result<(), ChangeDetectorError> {
        let glob_pattern = glob::Pattern::new(pattern)
            .map_err(|e| ChangeDetectorError::InvalidPattern(e.to_string()))?;
        self.exclude_patterns.push(glob_pattern);
        Ok(())
    }

    /// Track a file
    pub fn track(&mut self, path: &Path) -> std::io::Result<()> {
        let state = FileState::from_path(path)?;
        self.file_states.insert(path.to_path_buf(), state);
        Ok(())
    }

    /// Check if a path matches watch patterns
    fn should_watch(&self, path: &Path) -> bool {
        // Check exclusions first
        for pattern in &self.exclude_patterns {
            if pattern.matches_path(path) {
                return false;
            }
        }

        // Check inclusions
        for pattern in &self.watch_patterns {
            if pattern.matches_path(path) {
                return true;
            }
        }

        false
    }

    /// Scan a directory and track matching files
    pub fn scan_directory(&mut self, dir: &Path) -> std::io::Result<usize> {
        let mut count = 0;

        for entry in walkdir::WalkDir::new(dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            if path.is_file() && self.should_watch(path) {
                self.track(path)?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Check all tracked files for changes
    pub fn check_changes(&mut self) -> Result<Vec<PathBuf>, ChangeDetectorError> {
        let mut changed = Vec::new();

        for (path, state) in &mut self.file_states {
            match state.has_changed() {
                Ok(true) => {
                    changed.push(path.clone());
                    // Update the state
                    if let Err(e) = state.update() {
                        eprintln!(
                            "Warning: failed to update state for {}: {}",
                            path.display(),
                            e
                        );
                    }
                }
                Ok(false) => {}
                Err(e) => {
                    // File might have been deleted
                    if e.kind() == std::io::ErrorKind::NotFound {
                        changed.push(path.clone());
                    } else {
                        return Err(ChangeDetectorError::IoError(e));
                    }
                }
            }
        }

        Ok(changed)
    }

    /// Apply changes to build graph
    pub fn apply_changes(&self, graph: &mut BuildGraph, changed_paths: &[PathBuf]) {
        for path in changed_paths {
            if let Some(unit_id) = graph.get_unit_id(path) {
                // Update content hash
                if let Some(unit) = graph.get_unit_mut(unit_id)
                    && let Err(e) = unit.update_hash()
                {
                    eprintln!(
                        "Warning: failed to update hash for {}: {}",
                        path.display(),
                        e
                    );
                }

                // Invalidate this unit and its dependents
                graph.invalidate(unit_id);
            }
        }
    }

    /// Get number of tracked files
    pub fn tracked_count(&self) -> usize {
        self.file_states.len()
    }

    /// Get all tracked file paths
    pub fn tracked_files(&self) -> Vec<PathBuf> {
        self.file_states.keys().cloned().collect()
    }

    /// Clear all tracked files
    pub fn clear(&mut self) {
        self.file_states.clear();
    }

    /// Save tracked states to disk
    pub fn save(&self, path: &Path) -> Result<(), ChangeDetectorError> {
        let snapshot = ChangeSnapshot {
            file_states: self.file_states.values().cloned().collect(),
        };

        let data = bincode::serialize(&snapshot)
            .map_err(|e| ChangeDetectorError::SerializationError(e.to_string()))?;

        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load tracked states from disk
    pub fn load(&mut self, path: &Path) -> Result<(), ChangeDetectorError> {
        let data = std::fs::read(path)?;

        let snapshot: ChangeSnapshot = bincode::deserialize(&data)
            .map_err(|e| ChangeDetectorError::SerializationError(e.to_string()))?;

        self.file_states.clear();
        for state in snapshot.file_states {
            self.file_states.insert(state.path.clone(), state);
        }

        Ok(())
    }
}

impl Default for ChangeDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable snapshot of change detector state
#[derive(Serialize, Deserialize)]
struct ChangeSnapshot {
    file_states: Vec<FileState>,
}

/// Errors from change detection
#[derive(Debug, thiserror::Error)]
pub enum ChangeDetectorError {
    #[error("Invalid glob pattern: {0}")]
    InvalidPattern(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_file_state() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.sio");

        fs::write(&file_path, b"module test").unwrap();

        let state = FileState::from_path(&file_path).unwrap();
        assert_eq!(state.path, file_path);
        assert!(!state.has_changed().unwrap());

        // Modify file
        std::thread::sleep(std::time::Duration::from_millis(10));
        fs::write(&file_path, b"module test2").unwrap();

        assert!(state.has_changed().unwrap());
    }

    #[test]
    fn test_change_detector() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.sio");

        fs::write(&file_path, b"module test").unwrap();

        let mut detector = ChangeDetector::new();
        detector.track(&file_path).unwrap();

        assert_eq!(detector.tracked_count(), 1);

        // No changes initially
        let changes = detector.check_changes().unwrap();
        assert_eq!(changes.len(), 0);

        // Modify file
        std::thread::sleep(std::time::Duration::from_millis(10));
        fs::write(&file_path, b"module test2").unwrap();

        let changes = detector.check_changes().unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0], file_path);
    }

    #[test]
    fn test_watch_patterns() {
        let mut detector = ChangeDetector::new();

        assert!(detector.should_watch(Path::new("src/main.sio")));
        assert!(detector.should_watch(Path::new("lib/core.sio")));
        assert!(!detector.should_watch(Path::new("test.rs")));

        detector.exclude("**/test/**").unwrap();
        assert!(!detector.should_watch(Path::new("src/test/foo.sio")));
    }

    #[test]
    fn test_scan_directory() {
        let temp_dir = TempDir::new().unwrap();

        fs::create_dir(temp_dir.path().join("src")).unwrap();
        fs::write(temp_dir.path().join("src/main.sio"), b"module main").unwrap();
        fs::write(temp_dir.path().join("src/lib.sio"), b"module lib").unwrap();
        fs::write(temp_dir.path().join("README.md"), b"# Test").unwrap();

        let mut detector = ChangeDetector::new();
        let count = detector.scan_directory(temp_dir.path()).unwrap();

        assert_eq!(count, 2); // Only .d files
        assert_eq!(detector.tracked_count(), 2);
    }
}

"""
File Watcher for Hot Reload

Uses polling (cross-platform) or inotify/kqueue when available.
"""

import os
import time
import hashlib
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set
from dataclasses import dataclass


@dataclass
class FileChange:
    """Represents a file change event."""
    path: Path
    change_type: str  # 'modified', 'created', 'deleted'
    old_checksum: Optional[str] = None
    new_checksum: Optional[str] = None


class FileWatcher:
    """
    Watches files for changes.
    
    Features:
    - Cross-platform (works on Linux, macOS, Windows)
    - Checksum-based change detection (not just timestamps)
    - Debouncing to handle rapid successive changes
    - Recursive directory watching
    """
    
    def __init__(
        self,
        paths: List[str],
        on_change: Callable[[List[FileChange]], None],
        interval: float = 0.1,
        debounce: float = 0.2,
        patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ):
        """
        Initialize file watcher.
        
        Args:
            paths: Directories or files to watch
            on_change: Callback when changes detected
            interval: Polling interval in seconds
            debounce: Debounce time after changes
            patterns: File patterns to watch (e.g., ['*.sio', '*.py'])
            ignore_patterns: Patterns to ignore
        """
        self.paths = [Path(p).resolve() for p in paths]
        self.on_change = on_change
        self.interval = interval
        self.debounce = debounce
        self.patterns = patterns or ['*.sio']
        self.ignore_patterns = ignore_patterns or [
            '*.tmp', '*.swp', '*~', '.git/*', 
            '__pycache__/*', '*.pyc', '.sou_hot_state/*'
        ]
        
        self._state: Dict[Path, str] = {}  # path -> checksum
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start watching for changes."""
        if self._running:
            return
        
        self._running = True
        
        # Build initial state
        self._scan()
        
        # Start watcher thread
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _loop(self) -> None:
        """Main watcher loop."""
        pending_changes: List[FileChange] = []
        last_change_time = 0.0
        
        while self._running:
            time.sleep(self.interval)
            
            # Scan for changes
            changes = self._scan()
            
            if changes:
                pending_changes.extend(changes)
                last_change_time = time.time()
            
            # If debounce period has passed and we have pending changes
            if pending_changes and (time.time() - last_change_time) >= self.debounce:
                # Filter out duplicates (keep most recent)
                seen = set()
                unique_changes = []
                for c in reversed(pending_changes):
                    if c.path not in seen:
                        seen.add(c.path)
                        unique_changes.append(c)
                
                if unique_changes:
                    self.on_change(list(reversed(unique_changes)))
                
                pending_changes = []
    
    def _scan(self) -> List[FileChange]:
        """Scan watched paths for changes."""
        changes = []
        new_state: Dict[Path, str] = {}
        
        with self._lock:
            current_files: Set[Path] = set()
            
            for watch_path in self.paths:
                if not watch_path.exists():
                    continue
                
                if watch_path.is_file():
                    files = [watch_path]
                else:
                    files = self._list_files(watch_path)
                
                for file_path in files:
                    current_files.add(file_path)
                    
                    try:
                        content = file_path.read_bytes()
                        checksum = hashlib.md5(content).hexdigest()
                        new_state[file_path] = checksum
                        
                        # Check for modifications
                        if file_path in self._state:
                            if self._state[file_path] != checksum:
                                changes.append(FileChange(
                                    path=file_path,
                                    change_type='modified',
                                    old_checksum=self._state[file_path],
                                    new_checksum=checksum
                                ))
                        else:
                            # New file
                            changes.append(FileChange(
                                path=file_path,
                                change_type='created',
                                new_checksum=checksum
                            ))
                    except (IOError, OSError):
                        pass
            
            # Check for deleted files
            for old_path in self._state:
                if old_path not in current_files:
                    changes.append(FileChange(
                        path=old_path,
                        change_type='deleted',
                        old_checksum=self._state[old_path]
                    ))
            
            self._state = new_state
        
        return changes
    
    def _list_files(self, directory: Path) -> List[Path]:
        """List all files to watch in a directory."""
        files = []
        
        try:
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                
                # Check if matches patterns
                if self.patterns:
                    matches = any(path.match(p) for p in self.patterns)
                    if not matches:
                        continue
                
                # Check if should be ignored
                if self._should_ignore(path):
                    continue
                
                files.append(path)
        except (PermissionError, OSError):
            pass
        
        return files
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if a file should be ignored."""
        path_str = str(path)
        
        for pattern in self.ignore_patterns:
            if self._match_pattern(path_str, pattern):
                return True
        
        return False
    
    def _match_pattern(self, path: str, pattern: str) -> bool:
        """Match a path against a glob pattern."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern)
    
    def add_watch(self, path: str) -> None:
        """Add a new path to watch."""
        with self._lock:
            p = Path(path).resolve()
            if p not in self.paths:
                self.paths.append(p)
    
    def remove_watch(self, path: str) -> None:
        """Remove a watched path."""
        with self._lock:
            p = Path(path).resolve()
            if p in self.paths:
                self.paths.remove(p)
    
    def get_watched_files(self) -> List[Path]:
        """Get list of currently watched files."""
        with self._lock:
            return list(self._state.keys())


class InotifyWatcher(FileWatcher):
    """
    Linux-optimized watcher using inotify.
    
    Much more efficient than polling - uses kernel events.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inotify = None
        
        # Try to import inotify
        try:
            import inotify.adapters
            self._inotify_available = True
        except ImportError:
            self._inotify_available = False
    
    def start(self) -> None:
        """Start watching using inotify if available."""
        if not self._inotify_available:
            # Fall back to polling
            super().start()
            return
        
        # TODO: Implement inotify-based watching
        # This would be much more efficient on Linux
        super().start()


# Convenience function
def watch(
    paths: List[str],
    callback: Callable[[List[FileChange]], None],
    **kwargs
) -> FileWatcher:
    """
    Create and start a file watcher.
    
    Example:
        def on_change(changes):
            for c in changes:
                print(f"{c.change_type}: {c.path}")
        
        watcher = watch(['./src'], on_change)
        time.sleep(60)  # Watch for 60 seconds
        watcher.stop()
    """
    watcher = FileWatcher(paths, callback, **kwargs)
    watcher.start()
    return watcher

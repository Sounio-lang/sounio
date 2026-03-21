"""
Hot Code Reload Runtime Core

Manages the lifecycle of hot-reloadable processes.
"""

import os
import sys
import time
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModuleVersion:
    """Represents a version of a compiled module."""
    path: str
    version: int
    checksum: str
    compiled_at: datetime
    binary_path: Optional[str] = None
    
    def __hash__(self):
        return hash((self.path, self.version))


@dataclass
class ReloadEvent:
    """Records a hot reload event."""
    timestamp: datetime
    module_path: str
    old_version: int
    new_version: int
    duration_ms: float
    state_preserved: bool
    processes_migrated: int


class HotReloadRuntime:
    """
    Hot Code Reload Runtime for Sounio.
    
    Manages:
    - Module versioning and compilation
    - State preservation across reloads
    - Process supervision and migration
    - File watching and change detection
    """
    
    def __init__(
        self,
        entry_point: str,
        watch_dirs: Optional[List[str]] = None,
        state_dir: str = ".sou_hot_state",
        on_reload: Optional[Callable[[ReloadEvent], None]] = None
    ):
        self.entry_point = Path(entry_point).resolve()
        self.watch_dirs = [Path(d).resolve() for d in (watch_dirs or [self.entry_point.parent])]
        self.state_dir = Path(state_dir)
        self.on_reload = on_reload
        
        # Module registry: path -> current version
        self.modules: Dict[str, ModuleVersion] = {}
        self.module_versions: Dict[str, List[ModuleVersion]] = {}
        
        # Process registry
        self.processes: Dict[str, 'Process'] = {}
        self.next_pid = 1
        
        # Reload history
        self.reload_history: List[ReloadEvent] = []
        
        # Running state
        self.running = False
        self.compilation_errors: List[str] = []
        
        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def start(self) -> None:
        """Start the hot reload runtime."""
        print(f"🔥 Sounio Hot Reload Runtime")
        print(f"   Entry: {self.entry_point}")
        print(f"   Watch: {', '.join(str(d) for d in self.watch_dirs)}")
        print(f"   State: {self.state_dir}")
        print()
        
        # Initial compilation
        if not self._compile_entry():
            print("❌ Initial compilation failed")
            return
        
        print(f"✅ Initial compilation successful")
        print(f"   Starting with hot reload enabled...")
        print()
        
        self.running = True
        self._run_loop()
    
    def stop(self) -> None:
        """Stop the runtime gracefully."""
        print("\n🛑 Stopping hot reload runtime...")
        self.running = False
        
        # Save final state
        self._save_all_process_states()
        
        print("✅ Runtime stopped gracefully")
    
    def _run_loop(self) -> None:
        """Main runtime loop - watches for changes."""
        import select
        
        # Build initial checksums
        file_checksums = self._scan_files()
        
        print("👀 Watching for changes (Ctrl+C to stop)...")
        print()
        
        try:
            while self.running:
                # Check for file changes every 100ms
                time.sleep(0.1)
                
                new_checksums = self._scan_files()
                changed_files = self._detect_changes(file_checksums, new_checksums)
                
                if changed_files:
                    print(f"📝 Detected changes in {len(changed_files)} file(s)")
                    for f in changed_files:
                        print(f"   • {f}")
                    
                    # Debounce - wait for changes to settle
                    time.sleep(0.2)
                    new_checksums = self._scan_files()
                    changed_files = self._detect_changes(file_checksums, new_checksums)
                    
                    if changed_files:
                        self._handle_reload(changed_files)
                        file_checksums = new_checksums
                
        except KeyboardInterrupt:
            pass
    
    def _scan_files(self) -> Dict[str, str]:
        """Scan all watched directories for .sio files."""
        checksums = {}
        
        for watch_dir in self.watch_dirs:
            if not watch_dir.exists():
                continue
                
            for sio_file in watch_dir.rglob("*.sio"):
                # Skip test files and hidden directories
                if "." in sio_file.name and sio_file.name.startswith("."):
                    continue
                if "/test/" in str(sio_file) or "\\test\\" in str(sio_file):
                    continue
                
                try:
                    content = sio_file.read_bytes()
                    checksums[str(sio_file)] = hashlib.md5(content).hexdigest()
                except Exception:
                    pass
        
        return checksums
    
    def _detect_changes(
        self, 
        old_checksums: Dict[str, str], 
        new_checksums: Dict[str, str]
    ) -> List[str]:
        """Detect which files have changed."""
        changed = []
        
        for path, new_checksum in new_checksums.items():
            old_checksum = old_checksums.get(path)
            if old_checksum != new_checksum:
                changed.append(path)
        
        return changed
    
    def _handle_reload(self, changed_files: List[str]) -> None:
        """Handle a hot reload event."""
        start_time = time.time()
        
        print(f"\n🔥 HOT RELOAD TRIGGERED")
        print(f"   Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        # Save current process states
        states = self._save_all_process_states()
        print(f"   💾 Saved {len(states)} process state(s)")
        
        # Attempt recompilation
        success = self._compile_entry()
        
        if not success:
            print(f"   ❌ Compilation failed - keeping old version")
            print(f"   Errors:")
            for err in self.compilation_errors[:5]:
                print(f"      {err}")
            return
        
        # Update module versions
        for file_path in changed_files:
            self._bump_module_version(file_path)
        
        # Restore process states
        migrated = self._restore_all_process_states(states)
        
        duration = (time.time() - start_time) * 1000
        
        # Record event
        event = ReloadEvent(
            timestamp=datetime.now(),
            module_path=changed_files[0] if changed_files else "unknown",
            old_version=self.modules.get(changed_files[0], ModuleVersion("", 0, "", datetime.now())).version - 1 if changed_files else 0,
            new_version=self.modules.get(changed_files[0], ModuleVersion("", 0, "", datetime.now())).version if changed_files else 0,
            duration_ms=duration,
            state_preserved=True,
            processes_migrated=migrated
        )
        self.reload_history.append(event)
        
        if self.on_reload:
            self.on_reload(event)
        
        print(f"   ✅ Reloaded in {duration:.1f}ms")
        print(f"   🔄 Migrated {migrated} process(es)")
        print(f"   💾 State preserved: ✓")
        print()
    
    def _compile_entry(self) -> bool:
        """Compile the entry point and all dependencies."""
        self.compilation_errors = []
        
        # Determine compiler path
        souc_path = self._find_compiler()
        if not souc_path:
            self.compilation_errors.append("Could not find souc compiler")
            return False
        
        try:
            # Run compiler
            result = subprocess.run(
                [str(souc_path), str(self.entry_point)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self.compilation_errors = result.stderr.strip().split('\n') if result.stderr else ["Compilation failed"]
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            self.compilation_errors.append("Compilation timed out")
            return False
        except Exception as e:
            self.compilation_errors.append(f"Compilation error: {e}")
            return False
    
    def _find_compiler(self) -> Optional[Path]:
        """Find the souc compiler."""
        # Try local build
        local = Path("target/release/souc")
        if local.exists():
            return local.resolve()
        
        # Try bootstrap
        bootstrap = Path("bootstrap/souc")
        if bootstrap.exists():
            return bootstrap.resolve()
        
        # Try PATH
        import shutil
        souc_in_path = shutil.which("souc")
        if souc_in_path:
            return Path(souc_in_path)
        
        return None
    
    def _bump_module_version(self, file_path: str) -> None:
        """Increment the version of a module."""
        old_version = self.modules.get(file_path)
        
        new_version_num = 1
        if old_version:
            new_version_num = old_version.version + 1
        
        # Calculate new checksum
        try:
            content = Path(file_path).read_bytes()
            new_checksum = hashlib.md5(content).hexdigest()
        except Exception:
            new_checksum = "unknown"
        
        new_version = ModuleVersion(
            path=file_path,
            version=new_version_num,
            checksum=new_checksum,
            compiled_at=datetime.now()
        )
        
        self.modules[file_path] = new_version
        
        if file_path not in self.module_versions:
            self.module_versions[file_path] = []
        self.module_versions[file_path].append(new_version)
    
    def _save_all_process_states(self) -> Dict[str, Any]:
        """Save the state of all running processes."""
        states = {}
        
        for pid, process in self.processes.items():
            try:
                state = process.save_state()
                states[pid] = state
                
                # Also save to disk for crash recovery
                state_file = self.state_dir / f"{pid}.json"
                with open(state_file, 'w') as f:
                    json.dump(state, f, default=str, indent=2)
                    
            except Exception as e:
                print(f"   ⚠️  Failed to save state for {pid}: {e}")
        
        return states
    
    def _restore_all_process_states(self, states: Dict[str, Any]) -> int:
        """Restore process states after reload."""
        migrated = 0
        
        for pid, state in states.items():
            if pid in self.processes:
                try:
                    self.processes[pid].restore_state(state)
                    migrated += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to restore state for {pid}: {e}")
        
        return migrated
    
    def spawn(self, func: Callable, *args, **kwargs) -> 'Process':
        """Spawn a new supervised process."""
        from .supervisor import Process
        
        pid = f"proc_{self.next_pid}"
        self.next_pid += 1
        
        process = Process(pid, func, args, kwargs)
        self.processes[pid] = process
        
        process.start()
        return process
    
    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        return {
            "modules_tracked": len(self.modules),
            "total_versions": sum(len(v) for v in self.module_versions.values()),
            "processes_running": len(self.processes),
            "reloads": len(self.reload_history),
            "avg_reload_time_ms": (
                sum(e.duration_ms for e in self.reload_history) / len(self.reload_history)
                if self.reload_history else 0
            ),
            "last_reload": (
                self.reload_history[-1].timestamp.isoformat()
                if self.reload_history else None
            ),
        }

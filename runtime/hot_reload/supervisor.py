"""
Process Supervisor for Hot Reload

Manages the lifecycle of processes with state preservation.
"""

import threading
import time
from typing import Callable, Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class ProcessState(Enum):
    """States a process can be in."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"      # For hot reload
    MIGRATING = "migrating"
    RESTARTING = "restarting"
    STOPPED = "stopped"
    CRASHED = "crashed"


@dataclass
class Process:
    """
    A supervised process that can be hot-reloaded.
    
    Each process has:
    - A unique PID
    - State that can be saved/restored
    - Lifecycle managed by the supervisor
    """
    pid: str
    func: Callable
    args: tuple
    kwargs: dict
    
    # Runtime state
    state: ProcessState = field(default=ProcessState.PENDING)
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    
    # User-defined state (preserved across reloads)
    user_state: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    started_at: Optional[float] = field(default=None)
    reload_count: int = field(default=0)
    message_count: int = field(default=0)
    
    # State preservation hooks
    on_save_state: Optional[Callable[[], Dict]] = field(default=None, repr=False)
    on_restore_state: Optional[Callable[[Dict], None]] = field(default=None, repr=False)
    
    def start(self) -> 'Process':
        """Start the process."""
        if self.state != ProcessState.PENDING:
            raise RuntimeError(f"Cannot start process in state {self.state}")
        
        self.state = ProcessState.RUNNING
        self.started_at = time.time()
        
        # Start in a thread
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
        return self
    
    def _run(self) -> None:
        """Main process loop."""
        try:
            # Call the process function
            result = self.func(*self.args, **self.kwargs, _process=self)
            
            # If function returns a generator, run it
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                for _ in result:
                    if self.state == ProcessState.STOPPED:
                        break
        except Exception as e:
            print(f"   💥 Process {self.pid} crashed: {e}")
            self.state = ProcessState.CRASHED
            raise
    
    def pause(self) -> Dict[str, Any]:
        """Pause the process for hot reload."""
        if self.state not in (ProcessState.RUNNING, ProcessState.RESTARTING):
            return {}
        
        self.state = ProcessState.PAUSED
        
        # Save state
        state = self.save_state()
        
        return state
    
    def resume(self, state: Optional[Dict[str, Any]] = None) -> 'Process':
        """Resume the process after hot reload."""
        if state:
            self.restore_state(state)
        
        self.state = ProcessState.RUNNING
        self.reload_count += 1
        
        return self
    
    def stop(self) -> 'Process':
        """Stop the process gracefully."""
        self.state = ProcessState.STOPPED
        
        if self.thread and self.thread.is_alive():
            # Give it a moment to shut down
            self.thread.join(timeout=1.0)
        
        return self
    
    def save_state(self) -> Dict[str, Any]:
        """Save the process state for hot reload."""
        state = {
            "pid": self.pid,
            "state": self.state.value,
            "user_state": self.user_state.copy(),
            "started_at": self.started_at,
            "reload_count": self.reload_count,
            "message_count": self.message_count,
        }
        
        # Call custom save hook if provided
        if self.on_save_state:
            try:
                custom_state = self.on_save_state()
                state["custom"] = custom_state
            except Exception as e:
                print(f"   ⚠️  Custom save state failed for {self.pid}: {e}")
        
        return state
    
    def restore_state(self, state: Dict[str, Any]) -> 'Process':
        """Restore process state after hot reload."""
        # Restore user state
        if "user_state" in state:
            self.user_state.update(state["user_state"])
        
        # Restore metrics
        self.started_at = state.get("started_at", self.started_at)
        self.reload_count = state.get("reload_count", 0) + 1
        self.message_count = state.get("message_count", 0)
        
        # Call custom restore hook if provided
        if self.on_restore_state and "custom" in state:
            try:
                self.on_restore_state(state["custom"])
            except Exception as e:
                print(f"   ⚠️  Custom restore state failed for {self.pid}: {e}")
        
        return self
    
    def set_state_hook(
        self, 
        save_fn: Callable[[], Dict], 
        restore_fn: Callable[[Dict], None]
    ) -> 'Process':
        """Set custom state preservation hooks."""
        self.on_save_state = save_fn
        self.on_restore_state = restore_fn
        return self
    
    def is_alive(self) -> bool:
        """Check if the process is still running."""
        if self.state == ProcessState.STOPPED:
            return False
        if self.thread:
            return self.thread.is_alive()
        return False
    
    def uptime_seconds(self) -> float:
        """Get process uptime."""
        if self.started_at:
            return time.time() - self.started_at
        return 0.0
    
    def __repr__(self) -> str:
        return f"Process({self.pid}, state={self.state.value}, reloads={self.reload_count})"


class Supervisor:
    """
    Manages a collection of supervised processes.
    
    Features:
    - Automatic restart on crash
    - Hot reload state migration
    - Process grouping
    - Health monitoring
    """
    
    def __init__(self, max_restarts: int = 5, restart_window: int = 60):
        self.processes: Dict[str, Process] = {}
        self.max_restarts = max_restarts
        self.restart_window = restart_window
        self._restart_history: Dict[str, list] = {}
        self._lock = threading.RLock()
    
    def spawn(
        self, 
        func: Callable, 
        *args, 
        name: Optional[str] = None,
        **kwargs
    ) -> Process:
        """Spawn a new supervised process."""
        with self._lock:
            pid = name or f"proc_{len(self.processes) + 1}"
            
            process = Process(pid, func, args, kwargs)
            self.processes[pid] = process
            
            process.start()
            return process
    
    def kill(self, pid: str) -> bool:
        """Kill a process."""
        with self._lock:
            if pid in self.processes:
                self.processes[pid].stop()
                del self.processes[pid]
                return True
            return False
    
    def kill_all(self) -> None:
        """Kill all processes."""
        with self._lock:
            for process in self.processes.values():
                process.stop()
            self.processes.clear()
    
    def pause_all(self) -> Dict[str, Dict[str, Any]]:
        """Pause all processes for hot reload."""
        with self._lock:
            states = {}
            for pid, process in self.processes.items():
                states[pid] = process.pause()
            return states
    
    def resume_all(self, states: Dict[str, Dict[str, Any]]) -> None:
        """Resume all processes after hot reload."""
        with self._lock:
            for pid, state in states.items():
                if pid in self.processes:
                    self.processes[pid].resume(state)
    
    def get_process(self, pid: str) -> Optional[Process]:
        """Get a process by PID."""
        return self.processes.get(pid)
    
    def list_processes(self) -> list:
        """List all processes."""
        return list(self.processes.values())
    
    def count_by_state(self, state: ProcessState) -> int:
        """Count processes in a given state."""
        return sum(1 for p in self.processes.values() if p.state == state)
    
    def restart_failed(self) -> int:
        """Restart any crashed processes."""
        restarted = 0
        
        with self._lock:
            for pid, process in self.processes.items():
                if process.state == ProcessState.CRASHED:
                    if self._can_restart(pid):
                        # Restart the process
                        process.state = ProcessState.PENDING
                        process.start()
                        restarted += 1
                        
                        # Track restart
                        if pid not in self._restart_history:
                            self._restart_history[pid] = []
                        self._restart_history[pid].append(time.time())
        
        return restarted
    
    def _can_restart(self, pid: str) -> bool:
        """Check if a process can be restarted (rate limiting)."""
        if pid not in self._restart_history:
            return True
        
        now = time.time()
        recent_restarts = [
            t for t in self._restart_history[pid]
            if now - t < self.restart_window
        ]
        
        return len(recent_restarts) < self.max_restarts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get supervisor statistics."""
        return {
            "total_processes": len(self.processes),
            "running": self.count_by_state(ProcessState.RUNNING),
            "paused": self.count_by_state(ProcessState.PAUSED),
            "crashed": self.count_by_state(ProcessState.CRASHED),
            "stopped": self.count_by_state(ProcessState.STOPPED),
        }

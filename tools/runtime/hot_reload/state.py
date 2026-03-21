"""
State Preservation for Hot Reload

Handles serializing and deserializing process state.
"""

import json
import pickle
from typing import Any, Dict, Optional, Callable
from pathlib import Path


class StatePreserver:
    """
    Manages state preservation for hot reload.
    
    Supports multiple serialization strategies:
    - JSON: Human-readable, limited types
    - Pickle: Full Python object support
    - Custom: User-defined serialization
    """
    
    SERIALIZERS = {
        "json": (json.dumps, json.loads),
        "pickle": (lambda x: pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL), pickle.loads),
    }
    
    def __init__(self, format: str = "json", state_dir: str = ".sou_hot_state"):
        self.format = format
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom serializers
        self._custom_serializers: Dict[str, tuple] = {}
        
        # State transformers for migration
        self._migrations: Dict[str, list] = {}
    
    def save(self, pid: str, state: Any, version: int = 1) -> Path:
        """Save state to disk."""
        state_file = self.state_dir / f"{pid}_v{version}.{self.format}"
        
        serializer = self._get_serializer()
        serialized = serializer(state)
        
        if isinstance(serialized, str):
            state_file.write_text(serialized)
        else:
            state_file.write_bytes(serialized)
        
        # Also save metadata
        meta = {
            "pid": pid,
            "version": version,
            "format": self.format,
        }
        meta_file = self.state_dir / f"{pid}_v{version}.meta"
        meta_file.write_text(json.dumps(meta, indent=2))
        
        return state_file
    
    def load(self, pid: str, version: Optional[int] = None) -> Any:
        """Load state from disk."""
        if version is None:
            # Find latest version
            version = self._find_latest_version(pid)
        
        state_file = self.state_dir / f"{pid}_v{version}.{self.format}"
        
        if not state_file.exists():
            raise FileNotFoundError(f"No state file for {pid} v{version}")
        
        deserializer = self._get_deserializer()
        
        if self.format in ("json",):
            data = state_file.read_text()
        else:
            data = state_file.read_bytes()
        
        return deserializer(data)
    
    def _find_latest_version(self, pid: str) -> int:
        """Find the latest version of saved state."""
        versions = []
        for f in self.state_dir.glob(f"{pid}_v*.{self.format}"):
            try:
                version = int(f.stem.split("_v")[-1])
                versions.append(version)
            except ValueError:
                pass
        
        if not versions:
            raise FileNotFoundError(f"No state found for {pid}")
        
        return max(versions)
    
    def _get_serializer(self):
        """Get the serialize function for current format."""
        if self.format in self.SERIALIZERS:
            return self.SERIALIZERS[self.format][0]
        elif self.format in self._custom_serializers:
            return self._custom_serializers[self.format][0]
        else:
            raise ValueError(f"Unknown format: {self.format}")
    
    def _get_deserializer(self):
        """Get the deserialize function for current format."""
        if self.format in self.SERIALIZERS:
            return self.SERIALIZERS[self.format][1]
        elif self.format in self._custom_serializers:
            return self._custom_serializers[self.format][1]
        else:
            raise ValueError(f"Unknown format: {self.format}")
    
    def register_serializer(
        self, 
        name: str, 
        serialize: Callable[[Any], bytes], 
        deserialize: Callable[[bytes], Any]
    ) -> None:
        """Register a custom serializer."""
        self._custom_serializers[name] = (serialize, deserialize)
    
    def migrate(
        self, 
        pid: str, 
        from_version: int, 
        to_version: int,
        transformer: Callable[[Any], Any]
    ) -> Any:
        """
        Migrate state from one version to another.
        
        This allows schema evolution - old state can be transformed
        to match new code expectations.
        """
        old_state = self.load(pid, from_version)
        new_state = transformer(old_state)
        self.save(pid, new_state, to_version)
        return new_state
    
    def cleanup(self, keep_versions: int = 3) -> int:
        """Clean up old state files, keeping only recent versions."""
        cleaned = 0
        
        # Group files by pid
        by_pid: Dict[str, list] = {}
        for f in self.state_dir.glob(f"*_v*.*"):
            if f.suffix in (".json", ".pickle", ".meta"):
                try:
                    pid = f.stem.split("_v")[0]
                    version = int(f.stem.split("_v")[-1])
                    
                    if pid not in by_pid:
                        by_pid[pid] = []
                    by_pid[pid].append((version, f))
                except ValueError:
                    pass
        
        # Keep only the most recent versions
        for pid, files in by_pid.items():
            files.sort(key=lambda x: x[0], reverse=True)
            for version, f in files[keep_versions:]:
                f.unlink()
                cleaned += 1
        
        return cleaned
    
    def list_saved(self) -> Dict[str, list]:
        """List all saved states."""
        saved = {}
        
        for f in self.state_dir.glob("*.meta"):
            try:
                meta = json.loads(f.read_text())
                pid = meta["pid"]
                version = meta["version"]
                
                if pid not in saved:
                    saved[pid] = []
                saved[pid].append(version)
            except Exception:
                pass
        
        # Sort versions
        for pid in saved:
            saved[pid].sort()
        
        return saved


class StateBuilder:
    """
    Helper for building complex state objects.
    
    Example:
        state = (StateBuilder()
            .with_data(users)
            .with_connections(sockets)
            .with_counters(requests=100, errors=0)
            .build())
    """
    
    def __init__(self):
        self._data = {}
    
    def with_data(self, **kwargs) -> 'StateBuilder':
        """Add data to state."""
        self._data.update(kwargs)
        return self
    
    def with_connections(self, connections: list) -> 'StateBuilder':
        """Add connection state."""
        self._data["_connections"] = [
            {"id": c.id, "addr": c.addr} 
            for c in connections
        ]
        return self
    
    def with_counters(self, **kwargs) -> 'StateBuilder':
        """Add counter state."""
        if "_counters" not in self._data:
            self._data["_counters"] = {}
        self._data["_counters"].update(kwargs)
        return self
    
    def with_version(self, version: int) -> 'StateBuilder':
        """Add version info for migration."""
        self._data["_state_version"] = version
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the state object."""
        return self._data.copy()


def auto_preserve(state_keys: list):
    """
    Decorator for automatic state preservation.
    
    Automatically saves/restores specified attributes.
    
    Example:
        @auto_preserve(['users', 'connections'])
        class MyServer:
            def __init__(self):
                self.users = []
                self.connections = []
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._state_keys = state_keys
        
        cls.__init__ = new_init
        
        # Add save/restore methods
        def save_state(self) -> Dict:
            return {k: getattr(self, k) for k in self._state_keys}
        
        def restore_state(self, state: Dict) -> None:
            for k in self._state_keys:
                if k in state:
                    setattr(self, k, state[k])
        
        cls.save_state = save_state
        cls.restore_state = restore_state
        
        return cls
    
    return decorator

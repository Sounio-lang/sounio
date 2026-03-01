#!/usr/bin/env python3
"""
sou hot - Hot Code Reload for Sounio

Usage:
    sou hot <file.sio>              # Run with hot reload
    sou hot <file.sio> --watch ./   # Watch specific directories
    sou hot <file.sio> --state .sou # State directory
    sou hot --stats                 # Show reload statistics

Examples:
    sou hot server.sio
    sou hot game.sio --watch ./src --watch ./assets
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add runtime to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "runtime"))

from hot_reload import HotReloadRuntime


def main():
    parser = argparse.ArgumentParser(
        description="Sounio Hot Code Reload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sou hot server.sio                    # Basic usage
  sou hot server.sio --watch ./src      # Watch specific dir
  sou hot server.sio --port 8080        # Pass args to program
  sou hot --stats                       # Show statistics
        """
    )
    
    parser.add_argument("file", nargs="?", help="Entry point .sio file")
    parser.add_argument(
        "--watch", "-w", 
        action="append", 
        dest="watch_dirs",
        help="Additional directories to watch (can specify multiple)"
    )
    parser.add_argument(
        "--state", "-s",
        default=".sou_hot_state",
        help="State preservation directory (default: .sou_hot_state)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show reload statistics from previous runs"
    )
    parser.add_argument(
        "--clear-state",
        action="store_true",
        help="Clear saved state before starting"
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Run without hot reload (for testing)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    # Capture remaining args to pass to the Sounio program
    args, program_args = parser.parse_known_args()
    
    # Handle stats command
    if args.stats:
        show_stats(args.state)
        return
    
    # Validate file
    if not args.file:
        parser.print_help()
        sys.exit(1)
    
    entry_path = Path(args.file)
    if not entry_path.exists():
        print(f"❌ Error: File not found: {entry_path}")
        sys.exit(1)
    
    if not entry_path.suffix == ".sio":
        print(f"⚠️  Warning: File doesn't have .sio extension")
    
    # Clear state if requested
    if args.clear_state:
        import shutil
        state_path = Path(args.state)
        if state_path.exists():
            shutil.rmtree(state_path)
            print(f"🗑️  Cleared state directory: {state_path}")
    
    # Build watch directories
    watch_dirs = [str(entry_path.parent)]
    if args.watch_dirs:
        watch_dirs.extend(args.watch_dirs)
    
    # Print banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║                 🔥 SOUNIO HOT RELOAD 🔥                      ║
║                                                              ║
║  Edit your code and see changes instantly!                   ║
║  State is preserved across reloads.                          ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    if args.verbose:
        print(f"Entry: {entry_path}")
        print(f"Watch: {', '.join(watch_dirs)}")
        print(f"State: {args.state}")
        print()
    
    # Create and start runtime
    runtime = HotReloadRuntime(
        entry_point=str(entry_path),
        watch_dirs=watch_dirs,
        state_dir=args.state,
        on_reload=lambda e: print(f"   📊 Total reloads: {len(runtime.reload_history)}")
    )
    
    try:
        runtime.start()
    except KeyboardInterrupt:
        runtime.stop()


def show_stats(state_dir: str):
    """Show statistics from previous hot reload sessions."""
    state_path = Path(state_dir)
    
    print(f"📊 Hot Reload Statistics")
    print(f"   State directory: {state_path}")
    print()
    
    if not state_path.exists():
        print("   No state directory found.")
        return
    
    # List saved states
    saved_states = list(state_path.glob("*.json"))
    
    if not saved_states:
        print("   No saved process states found.")
    else:
        print(f"   Saved process states: {len(saved_states)}")
        for f in saved_states[:10]:  # Show first 10
            try:
                data = json.loads(f.read_text())
                pid = data.get("pid", "unknown")
                reloads = data.get("reload_count", 0)
                print(f"      • {pid}: {reloads} reload(s)")
            except Exception:
                print(f"      • {f.name} (unreadable)")
        
        if len(saved_states) > 10:
            print(f"      ... and {len(saved_states) - 10} more")
    
    print()
    
    # Try to load reload history
    history_file = state_path / "reload_history.json"
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text())
            print(f"   Total reloads: {len(history)}")
            
            if history:
                # Calculate average reload time
                times = [h.get("duration_ms", 0) for h in history]
                avg_time = sum(times) / len(times)
                print(f"   Average reload time: {avg_time:.1f}ms")
                
                # Show recent reloads
                print()
                print("   Recent reloads:")
                for h in history[-5:]:
                    time_str = h.get("timestamp", "unknown")
                    if "T" in str(time_str):
                        time_str = time_str.split("T")[1].split(".")[0]
                    module = Path(h.get("module_path", "unknown")).name
                    duration = h.get("duration_ms", 0)
                    print(f"      {time_str}  {module:30}  {duration:6.1f}ms")
                    
        except Exception as e:
            print(f"   Could not load history: {e}")
    else:
        print("   No reload history found.")


if __name__ == "__main__":
    main()

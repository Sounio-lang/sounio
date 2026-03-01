#!/usr/bin/env python3
"""
Hot Reload Demo Server

This demonstrates hot code reloading with state preservation.

Run:
    python3 examples/hot_reload/demo_server.py

Then edit this file and save - watch it reload without losing state!
"""

import sys
import time
import random
from pathlib import Path

# Add runtime to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "runtime"))

from hot_reload import HotReloadRuntime, Supervisor, Process


# Global state (this will be preserved!)
SERVER_STATE = {
    "connections": [],
    "request_count": 0,
    "error_count": 0,
    "start_time": time.time(),
    "version": 1,
}


def handle_request(process, request_type="normal"):
    """Simulate handling a request."""
    SERVER_STATE["request_count"] += 1
    
    # Simulate occasional errors
    if random.random() < 0.1:  # 10% error rate
        SERVER_STATE["error_count"] += 1
        return f"❌ Error processing {request_type} request"
    
    return f"✅ Handled {request_type} request #{SERVER_STATE['request_count']}"


def server_loop(_process=None):
    """Main server loop - runs continuously."""
    print(f"🚀 Server running (version {SERVER_STATE['version']})")
    print(f"   Requests: {SERVER_STATE['request_count']}")
    print(f"   Errors: {SERVER_STATE['error_count']}")
    print()
    
    counter = 0
    while True:
        time.sleep(2)
        counter += 1
        
        # Simulate request handling
        result = handle_request(_process, "normal")
        print(f"[{counter}] {result}")
        print(f"     State: {SERVER_STATE['request_count']} total, "
              f"{SERVER_STATE['error_count']} errors")


def background_worker(_process=None):
    """Background task - also preserved!"""
    worker_id = random.randint(1000, 9999)
    iteration = 0
    
    print(f"⚙️  Worker {worker_id} started")
    
    while True:
        time.sleep(3)
        iteration += 1
        
        # Simulate work
        work = random.choice(["cleanup", "metrics", "sync", "backup"])
        print(f"⚙️  Worker {worker_id}: {work} (iteration {iteration})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hot Reload Demo Server")
    parser.add_argument("--no-hot", action="store_true", help="Run without hot reload")
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║              🔥 HOT RELOAD DEMO SERVER 🔥                    ║
║                                                              ║
║  This server demonstrates hot code reloading!                ║
║                                                              ║
║  Try editing this file while it's running:                   ║
║  1. Change the version number                                ║
║  2. Add a print statement                                    ║
║  3. Modify request handling                                  ║
║                                                              ║
║  The state (request_count, etc.) will be preserved!          ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    if args.no_hot:
        # Run without hot reload
        print("Running WITHOUT hot reload (use --no-hot to disable)")
        server_loop()
    else:
        # Run with hot reload
        this_file = Path(__file__)
        
        runtime = HotReloadRuntime(
            entry_point=str(this_file),
            watch_dirs=[str(this_file.parent)],
            state_dir=".sou_hot_state_demo",
        )
        
        # Spawn multiple processes to demonstrate supervision
        runtime.spawn(server_loop)
        runtime.spawn(background_worker)
        runtime.spawn(background_worker)
        
        try:
            runtime.start()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            runtime.stop()
            
            # Show final stats
            stats = runtime.get_stats()
            print("\n📊 Final Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")

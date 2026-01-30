#!/usr/bin/env python3
"""
Sounio Development Logger
Logs all actions for analysis
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

PROJECT_DIR = os.environ.get('CLAUDE_PROJECT_DIR', os.getcwd())
LOG_DIR = Path(PROJECT_DIR) / '.claude' / 'logs'
LOG_FILE = LOG_DIR / 'full-activity.jsonl'

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        input_data = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        input_data = {}

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'tool_name': input_data.get('tool_name', 'unknown'),
        'tool_input': input_data.get('tool_input', {}),
        'session_id': os.environ.get('CLAUDE_SESSION_ID', 'unknown'),
    }

    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

if __name__ == '__main__':
    main()

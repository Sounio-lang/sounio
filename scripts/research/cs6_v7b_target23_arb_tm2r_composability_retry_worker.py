#!/usr/bin/env python3
"""Run the composability worker with a larger, explicitly bound split budget."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import cs6_v7b_target23_arb_tm2r_composability_carrier_worker as worker


MAX_EVENT_SPLIT_DEPTH = 12
MAX_EVENT_SPLIT_NODES_PER_TILE = 255


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    worker.transport.chain.MAX_EVENT_SPLIT_DEPTH = MAX_EVENT_SPLIT_DEPTH
    worker.transport.chain.MAX_EVENT_SPLIT_NODES_PER_TILE = (
        MAX_EVENT_SPLIT_NODES_PER_TILE
    )
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        worker.main()
    payload = json.loads(captured.getvalue())
    payload["execution_profile"] = "EXTENDED_SPLIT_BUDGET_V1"
    payload["execution_wrapper_source_sha256"] = sha256(Path(__file__))
    payload["max_event_split_depth"] = MAX_EVENT_SPLIT_DEPTH
    payload["max_event_split_nodes_per_tile"] = MAX_EVENT_SPLIT_NODES_PER_TILE
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

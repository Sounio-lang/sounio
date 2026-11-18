"""Timing-file grammar shared with the Spark observer. Sounio decides the gain."""
import os

BLOCKS = 30
VECTORS = 16384
REPLAYS = 32
GRAPH_WARMUP = 8


def row(block, candidate_ns, control_ns):
    if not isinstance(block, int) or block < 0 or block >= BLOCKS:
        raise ValueError("block")
    if not isinstance(candidate_ns, int) or not isinstance(control_ns, int):
        raise ValueError("duration")
    if min(candidate_ns, control_ns) <= 0 or max(candidate_ns, control_ns) >= 1000000000:
        raise ValueError("duration")
    return f"{block} {block % 2} {candidate_ns} {control_ns}\n"


def duration_ns(elapsed_ms):
    """CUDA event milliseconds to integer nanoseconds. Accepts a torch/numpy scalar."""
    if isinstance(elapsed_ms, bool) or elapsed_ms is None:
        raise ValueError("duration")
    item = getattr(elapsed_ms, "item", None)
    if callable(item) and not isinstance(elapsed_ms, (int, float, str, bytes)):
        try:
            elapsed_ms = item()
        except Exception as exc:
            raise ValueError("duration") from exc
    if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, (int, float)):
        raise ValueError("duration")
    return int(round(float(elapsed_ms) * 1000000))


def append_timing_block(log, block, candidate_ns, control_ns):
    log.write(row(block, candidate_ns, control_ns))
    log.flush()
    os.fsync(log.fileno())

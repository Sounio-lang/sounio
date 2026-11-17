"""Timing-file grammar shared with the Spark observer. Sounio decides the gain."""

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

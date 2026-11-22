"""Deterministic lowering plans. The formula is the one in cycle.py generate()."""

UNROLL = (1, 2, 4, 8, 16)


def lane_plan(index, round_id):
    if index < 0 or round_id not in (0, 1, 2):
        raise ValueError("plan index")
    return {
        "lane_stride": 1 + 2 * (index % 8),
        "lane_offset": (index // 8 + round_id) % 16,
        "load": (index // 2) % 2,
        "layout": index % 2,
        "unroll": UNROLL[index % 5],
    }


def phase_a():
    """Round 0, indices 0..7. Same generator inputs as the sm_121 baseline."""
    return [{"round": 0, "index": index, **lane_plan(index, 0)} for index in range(8)]


def phase_b():
    """32 further plans across rounds 0, 1 and 2, excluding the Phase A eight.

    cycle.py accepts budget 8 or 32 and one round per invocation. Index 17 needs
    budget 32. The kept (round, index) pairs are the committed Phase B list.
    """
    rows = []
    for index in range(8, 18):
        rows.append({"round": 0, "index": index, **lane_plan(index, 0)})
    for index in range(11):
        rows.append({"round": 1, "index": index, **lane_plan(index, 1)})
    for index in range(11):
        rows.append({"round": 2, "index": index, **lane_plan(index, 2)})
    return rows

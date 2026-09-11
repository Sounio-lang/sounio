"""Exclusive per-rank host-decode journals; no model or CUDA dependencies."""
import hashlib
import json
from pathlib import Path

class HostDecodeJournal:
    def __init__(self, directory, job, rank, pid):
        if not str(job).isdigit() or str(rank) not in ("0", "1") or type(pid) is not int or pid <= 0:
            raise ValueError("invalid journal identity")
        self.job, self.rank, self.pid = str(job), str(rank), pid
        self.path = Path(directory) / f"host-decode-{job}-{rank}.jsonl"
        # Exclusive creation also rejects pre-existing symlinks. Parent is the
        # existing receipt directory; do not silently invent a fallback path.
        self.stream = self.path.open("xb", buffering=0)
        self.count = 0
        self.failed = False

    def write(self, row):
        if self.failed or self.stream.closed:
            raise ValueError("journal unavailable")
        expected = (self.count // 2 + 1, ("HOST_DECODE_BEGIN", "HOST_DECODE_END")[self.count % 2])
        if (self.count >= 30 or row.get("job") != self.job or row.get("rank") != self.rank
                or row.get("pid") != self.pid or row.get("index") != 0
                or (row.get("step"), row.get("stage")) != expected):
            self.failed = True
            self.stream.close()
            raise ValueError("journal scope/order")
        raw = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
        try:
            remaining = memoryview(raw)
            while remaining:
                try:
                    written = self.stream.write(remaining)
                except InterruptedError:
                    continue
                if type(written) is not int or written <= 0 or written > len(remaining):
                    raise OSError("journal made no write progress")
                remaining = remaining[written:]
        except BaseException:
            self.failed = True
            self.stream.close()
            raise
        self.count += 1
        # Small stdout receipt is informational. Journal bytes are authoritative.
        return dict(stage="HOST_DECODE_JOURNAL_RECORD", job=self.job, rank=self.rank,
                    pid=self.pid, sequence=self.count, record_sha256=hashlib.sha256(raw).hexdigest())

    def close(self):
        self.stream.close()

def read_journal(path, expected_sha256, job, rank, pid):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("journal digest")
    if raw and not raw.endswith(b"\n"):
        raise ValueError("incomplete journal line")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) > 30:
        raise ValueError("journal record limit")
    for n, row in enumerate(rows):
        if not isinstance(row, dict) or (row.get("job"), row.get("rank"), row.get("pid"), row.get("index"),
                row.get("step"), row.get("stage")) != (str(job), str(rank), pid, 0, n // 2 + 1,
                    ("HOST_DECODE_BEGIN", "HOST_DECODE_END")[n % 2]):
            raise ValueError("journal identity/order")
    # Empty/truncated-at-record-boundary journals remain partial, never complete.
    return rows

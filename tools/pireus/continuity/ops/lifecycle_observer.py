"""Diagnostic observations only; independent of the memory guardian."""
import json
import os
from pathlib import Path
import threading
import time

HOST_KEYS = ("MemAvailable", "Cached", "SReclaimable", "Shmem", "Slab", "Unevictable")

def fields(path, names):
    try:
        lines = Path(path).read_text().splitlines()
        parsed = {p[0].rstrip(":"): int(p[1])*1024 for line in lines
                  if len(p := line.split()) == 3 and p[2] == "kB"}
        return {name: parsed.get(name) for name in names}, None
    except (OSError, ValueError) as exc:
        return {name: None for name in names}, type(exc).__name__

class Observer:
    def __init__(self, path, job, rank, cuda, interval=1.0):
        if not 0.1 <= interval <= 10:
            raise ValueError("observer interval outside declared bounds")
        self.out = Path(path).open("x")
        self.job, self.rank, self.cuda = str(job), str(rank), cuda
        self.interval = interval
        self.lock = threading.Lock()
        self.context = (None, None)
        self.stop = threading.Event()
        self.thread = None
        self.record("OBSERVER_START", extra={"interval_seconds": interval,
                    "cgroup_memory": None, "device_memory": None,
                    "unavailable": ["cgroup_memory", "device_memory"],
                    "peaks_scope": "process lifetime; no counter reset"})
    def record(self, stage, index=None, token_index=None, process=False, extra=None):
        started = time.monotonic_ns()
        row = dict(schema="pireus-lifecycle-observation-v1", stage=stage,
                   job=self.job, rank=self.rank, pid=os.getpid(),
                   monotonic_ns=started, index=index, token_index=token_index)
        row["host"], row["host_error"] = fields("/proc/meminfo", HOST_KEYS)
        if process:
            row["process"], row["process_error"] = fields("/proc/self/smaps_rollup",
                                                        ("Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem"))
            try:
                pending, seen, total = [os.getpid()], set(), 0
                while pending:
                    parent = pending.pop()
                    children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
                    for child in children:
                        if child in seen:
                            continue
                        seen.add(child)
                        pending.append(int(child))
                        data, error = fields(f"/proc/{child}/status", ("VmRSS",))
                        if error or data["VmRSS"] is None:
                            raise OSError("descendant accounting unavailable")
                        total += data["VmRSS"]
                row["owned_child_rss_bytes"] = total
                row["owned_child_error"] = None
            except OSError as exc:
                row["owned_child_rss_bytes"] = None
                row["owned_child_error"] = type(exc).__name__
            row["cuda"] = {}
            for key, method in (("allocated", "memory_allocated"), ("reserved", "memory_reserved"),
                                ("peak_allocated", "max_memory_allocated"),
                                ("peak_reserved", "max_memory_reserved")):
                try:
                    row["cuda"][key] = getattr(self.cuda, method)()
                except Exception as exc:
                    row["cuda"][key] = None
                    row["cuda"][key+"_error"] = type(exc).__name__
        row.update(extra or {})
        with self.lock:
            row["observation_duration_ns"] = time.monotonic_ns()-started
            self.out.write(json.dumps(row, sort_keys=True)+"\n")
            self.out.flush()
    def mark(self, stage, index, token_index=None):
        with self.lock:
            self.context = (index, token_index)
        self.record(stage, index, token_index, process=True)
    def _sample(self):
        previous = time.monotonic_ns()
        while not self.stop.wait(self.interval):
            now = time.monotonic_ns()
            with self.lock:
                index, token = self.context
            self.record("HOST_SAMPLE", index, token,
                        extra={"sample_gap_ns": now-previous,
                               "context_scope": "last runtime observation"})
            previous = now
    def start(self):
        if self.thread is not None:
            raise ValueError("observer already started")
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2*self.interval+1)
            if self.thread.is_alive():
                raise RuntimeError("observer did not stop")
        self.record("OBSERVER_END")
        self.out.close()

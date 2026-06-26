#!/usr/bin/env python3
"""Fetch a small real external corpus slice for the solver Level-3 pilot.

SAT uses SAT Competition 2024 through GBD's track_main_2024.uri endpoint. SMT
and OPB are intentionally conservative: the script records explicit blockers
when a small directly-downloadable subset or usable solver is unavailable rather
than inventing generated fixtures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html.parser
import lzma
import os
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


GBD_MAIN_2024_PAGE = "https://benchmark-database.de/?track=main_2024&context=cnf"
GBD_MAIN_2024_URI = "https://benchmark-database.de/getinstances?query=track%3Dmain_2024&context=cnf"
SATCOMP_2024_PAGE = "https://satcompetition.github.io/2024/"
SMTLIB_BENCHMARKS_PAGE = "https://smt-lib.org/benchmarks.shtml"
SMTLIB_2024_QFLIA_ARCHIVE = "https://zenodo.org/api/records/11061097/files/QF_LIA.tar.zst/content"
PB25_PAGE = "https://www.cril.univ-artois.fr/PB25/"
PB_SAMPLE_URL = "https://www.cril.univ-artois.fr/PB24/benchs/SampleByIntSize.tar"
DEFAULT_SMT_QFLIA_MANIFEST = "benchmarks/solver/qflia_pinned_manifest.csv"


@dataclass(frozen=True)
class GbdRow:
    file_id: str
    md5: str
    family: str
    contributor: str
    expected: str
    source_url: str
    verified: str
    filename: str
    track: str
    download_url: str


@dataclass(frozen=True)
class CnfStats:
    n_vars: int
    n_clauses: int
    max_clause_width: int


class GbdTableParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_td = False
        self.in_tr = False
        self.current: list[str] = []
        self.rows: list[list[str]] = []
        self.buf: list[str] = []
        self.file_href = ""
        self.href_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "tr":
            self.in_tr = True
            self.current = []
        if tag == "td" and self.in_tr:
            self.in_td = True
            self.buf = []
            self.file_href = ""
        if tag == "a" and self.in_td:
            href = attrs_dict.get("href") or ""
            if href.startswith("/file/"):
                self.file_href = href

    def handle_data(self, data: str) -> None:
        if self.in_td:
            self.buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.in_td:
            text = " ".join("".join(self.buf).split())
            if self.file_href:
                text = urllib.parse.urljoin("https://benchmark-database.de", self.file_href)
            self.current.append(text)
            self.in_td = False
        if tag == "tr" and self.in_tr:
            if len(self.current) >= 9:
                self.rows.append(self.current)
            self.in_tr = False


def urlopen_read(url: str, timeout: int = 30, limit: int | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sounio-external-corpus-pilot/1"})
    with urllib.request.urlopen(req, timeout=timeout, context=tls_context()) as r:
        if limit is None:
            return r.read()
        return r.read(limit)


def tls_context() -> ssl.SSLContext | None:
    if os.environ.get("SOUNIO_SOLVER_ALLOW_UNVERIFIED_TLS", "0") in {"1", "true", "TRUE", "yes", "YES"}:
        return ssl._create_unverified_context()
    return None


def head_length(url: str, timeout: int = 15) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "sounio-external-corpus-pilot/1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=tls_context()) as r:
            value = r.headers.get("content-length")
            return int(value) if value else None
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_root() -> Path:
    return Path(os.environ.get("SOUNIO_SOLVER_TOOL_ROOT", "~/.cache/sounio/solver-tools")).expanduser()


def which_solver(name: str) -> str | None:
    for candidate in (tool_root() / "bin" / name, tool_root() / name):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(name)


def parse_status(text: str) -> str:
    m = re.search(r"\(set-info\s+:status\s+([a-zA-Z]+)\)", text)
    if not m:
        return "UNKNOWN"
    value = m.group(1).upper()
    return value if value in {"SAT", "UNSAT", "UNKNOWN"} else "UNKNOWN"


def read_pinned_smt_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def open_zstd_tar_stream(url: str) -> tarfile.TarFile:
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("python_zstandard_missing_for_zstd_archive_streaming") from exc
    req = urllib.request.Request(url, headers={"User-Agent": "sounio-external-corpus-pilot/1"})
    response = urllib.request.urlopen(req, timeout=60, context=tls_context())
    reader = zstandard.ZstdDecompressor().stream_reader(response)
    return tarfile.open(fileobj=reader, mode="r|")


def z3_expected(path: Path, timeout_sec: int) -> str:
    z3 = which_solver("z3")
    if not z3:
        return "UNKNOWN"
    try:
        completed = subprocess.run([z3, str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
    except (OSError, subprocess.TimeoutExpired):
        return "UNKNOWN"
    text = (completed.stdout + completed.stderr).strip().splitlines()
    if not text:
        return "UNKNOWN"
    first = text[0].strip().upper()
    return first if first in {"SAT", "UNSAT", "UNKNOWN"} else "UNKNOWN"


def fetch_smt_archive_members(args: argparse.Namespace, root: Path, manifest_rows: list[dict[str, str]], dry_run: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    wanted = [row for row in manifest_rows if row.get("archive_member")]
    if not wanted:
        return rows
    wanted_by_member = {row["archive_member"]: row for row in wanted[: args.smt_count]}
    smt_dir = root / "smt" / "qflia"
    archive_url = wanted[0].get("archive_url") or SMTLIB_2024_QFLIA_ARCHIVE
    if dry_run:
        for member, item in wanted_by_member.items():
            rows.append(
                {
                    "domain": "smt",
                    "instance_id": item.get("instance_id") or Path(member).name,
                    "source": item.get("source", "SMT-LIB 2024 QF_LIA Zenodo archive member"),
                    "source_url": item.get("source_url", "https://zenodo.org/records/11061097"),
                    "download_url": f"{archive_url}#{member}",
                    "sha256": item.get("sha256", ""),
                    "family": item.get("family", "qflia"),
                    "expected": item.get("expected", "UNKNOWN").upper(),
                    "status": "DRY_RUN_PLANNED",
                    "n_vars": "",
                    "n_clauses": "",
                    "max_clause_width": "",
                    "size_bytes": "",
                    "local_path": str(smt_dir / Path(member).name),
                }
            )
        return rows
    try:
        tar = open_zstd_tar_stream(archive_url)
        for member in tar:
            if not member.isfile() or member.name not in wanted_by_member:
                continue
            item = wanted_by_member[member.name]
            if member.size > args.max_smt_file_bytes:
                rows.append(row_for_blocker("smt", item.get("instance_id") or Path(member.name).name, f"archive_member_too_large_{member.size}", archive_url))
                continue
            fin = tar.extractfile(member)
            if fin is None:
                continue
            data = fin.read()
            if b"(set-logic QF_LIA)" not in data:
                rows.append(row_for_blocker("smt", item.get("instance_id") or Path(member.name).name, "archive_member_not_qflia", archive_url))
                continue
            digest = hashlib.sha256(data).hexdigest()
            expected_sha = item.get("sha256", "")
            if expected_sha and digest != expected_sha:
                rows.append(row_for_blocker("smt", item.get("instance_id") or Path(member.name).name, f"archive_member_sha256_mismatch_{digest}", archive_url))
                continue
            local = smt_dir / Path(member.name).name
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            expected = item.get("expected", "").upper() or parse_status(data.decode("utf-8", "replace"))
            if expected == "UNKNOWN":
                expected = z3_expected(local, args.smt_probe_timeout_sec)
            rows.append(
                {
                    "domain": "smt",
                    "instance_id": item.get("instance_id") or Path(member.name).name,
                    "source": item.get("source", "SMT-LIB 2024 QF_LIA Zenodo archive member"),
                    "source_url": item.get("source_url", "https://zenodo.org/records/11061097"),
                    "download_url": f"{archive_url}#{member.name}",
                    "sha256": digest,
                    "family": item.get("family", "qflia"),
                    "expected": expected,
                    "status": "OK",
                    "n_vars": "",
                    "n_clauses": "",
                    "max_clause_width": "",
                    "size_bytes": str(local.stat().st_size),
                    "local_path": str(local),
                }
            )
            if len([row for row in rows if row["status"] == "OK"]) >= args.smt_count:
                break
    except Exception as exc:
        rows.append(row_for_blocker("smt", "smtlib-2024-qflia-stream", f"qflia_archive_stream_failed_{type(exc).__name__}_{exc}", archive_url))
    missing = set(wanted_by_member) - {row["download_url"].split("#", 1)[-1] for row in rows if row["status"] == "OK"}
    for member in sorted(missing):
        item = wanted_by_member[member]
        if not any(row["instance_id"] == (item.get("instance_id") or Path(member).name) for row in rows):
            rows.append(row_for_blocker("smt", item.get("instance_id") or Path(member).name, "archive_member_not_found_before_stream_end", archive_url))
    return rows


def parse_gbd_rows() -> list[GbdRow]:
    html = urlopen_read(GBD_MAIN_2024_PAGE, timeout=30).decode("utf-8", "replace")
    uri_lines = urlopen_read(GBD_MAIN_2024_URI, timeout=30).decode("utf-8", "replace").splitlines()
    downloads: dict[str, str] = {}
    for line in uri_lines:
        line = line.strip()
        m = re.search(r"/file/([0-9a-f]{32})", line)
        if m:
            downloads[m.group(1)] = line.replace("http://", "https://", 1)

    parser = GbdTableParser()
    parser.feed(html)
    rows: list[GbdRow] = []
    for cols in parser.rows:
        file_url = cols[0]
        m = re.search(r"/file/([0-9a-f]{32})", file_url)
        if not m:
            continue
        file_id = m.group(1)
        if file_id not in downloads:
            continue
        rows.append(
            GbdRow(
                file_id=file_id,
                md5=cols[1],
                family=cols[2],
                contributor=cols[3],
                expected=cols[4].upper(),
                source_url=cols[5],
                verified=cols[6],
                filename=cols[8],
                track=cols[9] if len(cols) > 9 else "",
                download_url=downloads[file_id],
            )
        )
    return rows


def download(url: str, path: Path, timeout: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "sounio-external-corpus-pilot/1"})
    with urllib.request.urlopen(req, timeout=timeout, context=tls_context()) as r, path.open("wb") as f:
        shutil.copyfileobj(r, f)


def download_limited(url: str, path: Path, max_bytes: int, timeout: int = 60) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "sounio-external-corpus-pilot/1"})
    total = 0
    with urllib.request.urlopen(req, timeout=timeout, context=tls_context()) as r, path.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"download_exceeds_limit_{max_bytes}")
            f.write(chunk)
    return total


def decompress_xz(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(src, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout)


def cnf_stats(path: Path) -> CnfStats:
    n_vars = 0
    declared_clauses = 0
    seen_clauses = 0
    max_width = 0
    with path.open("rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("c"):
                continue
            if stripped.startswith("p "):
                parts = stripped.split()
                if len(parts) >= 4 and parts[1] == "cnf":
                    n_vars = int(parts[2])
                    declared_clauses = int(parts[3])
                continue
            lits = [int(x) for x in stripped.split() if x != "0"]
            if lits:
                seen_clauses += 1
                max_width = max(max_width, len(lits))
    return CnfStats(n_vars=n_vars, n_clauses=declared_clauses or seen_clauses, max_clause_width=max_width)


def row_for_blocker(domain: str, instance_id: str, blocker: str, source_url: str) -> dict[str, str]:
    return {
        "domain": domain,
        "instance_id": instance_id,
        "source": "blocker",
        "source_url": source_url,
        "download_url": "",
        "sha256": "",
        "family": "",
        "expected": "BLOCKED",
        "status": blocker,
        "n_vars": "",
        "n_clauses": "",
        "max_clause_width": "",
        "size_bytes": "",
        "local_path": "",
    }


def fetch_sat(args: argparse.Namespace, root: Path, dry_run: bool) -> list[dict[str, str]]:
    if args.sat_count <= 0:
        return []
    gbd_rows = parse_gbd_rows()
    selected: list[dict[str, str]] = []
    by_status = {"SAT": 0, "UNSAT": 0, "UNKNOWN": 0}
    cnf_dir = root / "sat" / "cnf"
    compressed_dir = root / "sat" / "compressed"
    candidates = 0
    considered = 0
    max_compressed = args.max_sat_compressed_bytes

    for row in gbd_rows:
        if len(selected) >= args.sat_count:
            break
        considered += 1
        if considered > args.max_sat_candidates:
            break
        if row.expected not in {"SAT", "UNSAT"}:
            continue
        if by_status[row.expected] > by_status["SAT" if row.expected == "UNSAT" else "UNSAT"] + 1:
            continue
        length = None
        if args.sat_use_head:
            length = head_length(row.download_url, timeout=args.sat_head_timeout_sec)
            if length is not None and length > max_compressed:
                continue
        candidates += 1
        compressed_path = compressed_dir / f"{row.file_id}-{row.filename}"
        cnf_name = row.filename[:-3] if row.filename.endswith(".xz") else f"{row.filename}.cnf"
        cnf_path = cnf_dir / cnf_name
        if dry_run:
            selected.append(
                {
                    "domain": "sat",
                    "instance_id": row.file_id,
                    "source": "SATComp2024/GBD track_main_2024.uri",
                    "source_url": SATCOMP_2024_PAGE,
                    "download_url": row.download_url,
                    "sha256": "",
                    "family": row.family,
                    "expected": row.expected,
                    "status": "DRY_RUN_PLANNED",
                    "n_vars": "",
                    "n_clauses": "",
                    "max_clause_width": "",
                    "size_bytes": str(length or ""),
                    "local_path": str(cnf_path),
                }
            )
            by_status[row.expected] += 1
            continue
        try:
            actual_compressed = download_limited(row.download_url, compressed_path, max_compressed, timeout=args.sat_download_timeout_sec)
            if length is None:
                length = actual_compressed
            if compressed_path.suffix == ".xz":
                decompress_xz(compressed_path, cnf_path)
            else:
                shutil.copyfile(compressed_path, cnf_path)
            stats = cnf_stats(cnf_path)
        except (OSError, urllib.error.URLError, lzma.LZMAError, ValueError) as exc:
            continue
        if stats.n_vars > args.max_sat_vars or stats.n_clauses > args.max_sat_clauses or stats.max_clause_width > args.max_sat_clause_width:
            try:
                cnf_path.unlink()
            except OSError:
                pass
            continue
        selected.append(
            {
                "domain": "sat",
                "instance_id": row.file_id,
                "source": "SATComp2024/GBD track_main_2024.uri",
                "source_url": SATCOMP_2024_PAGE,
                "download_url": row.download_url,
                "sha256": sha256_file(cnf_path),
                "family": row.family,
                "expected": row.expected,
                "status": "OK",
                "n_vars": str(stats.n_vars),
                "n_clauses": str(stats.n_clauses),
                "max_clause_width": str(stats.max_clause_width),
                "size_bytes": str(cnf_path.stat().st_size),
                "local_path": str(cnf_path),
            }
        )
        by_status[row.expected] += 1

    if len(selected) < args.sat_count and not dry_run:
        selected.append(
            row_for_blocker(
                "sat",
                "satcomp2024-small-filter",
                f"only_{len(selected)}_supported_cnf_after_filtering_{candidates}_small_download_candidates_{considered}_gbd_rows_considered",
                GBD_MAIN_2024_URI,
            )
        )
    return selected


def fetch_smt(args: argparse.Namespace, root: Path, dry_run: bool) -> list[dict[str, str]]:
    if args.smt_count <= 0:
        return []
    manifest_path = args.smt_manifest or Path(DEFAULT_SMT_QFLIA_MANIFEST)
    pinned = read_pinned_smt_manifest(manifest_path)
    if not pinned:
        return [
            row_for_blocker(
                "smt",
                "smtlib-qflia-small-subset",
                f"no_pinned_qflia_manifest_entries_at_{manifest_path}",
                SMTLIB_BENCHMARKS_PAGE,
            )
        ]
    archive_rows = fetch_smt_archive_members(args, root, pinned, dry_run)
    if archive_rows:
        return archive_rows
    rows: list[dict[str, str]] = []
    smt_dir = root / "smt" / "qflia"
    for item in pinned[: args.smt_count]:
        url = item.get("download_url", "")
        instance_id = item.get("instance_id") or Path(urllib.parse.urlparse(url).path).name
        if not url:
            rows.append(row_for_blocker("smt", instance_id or "qflia-missing-url", "pinned_manifest_row_missing_download_url", str(manifest_path)))
            continue
        local = smt_dir / f"{instance_id}.smt2"
        if dry_run:
            rows.append(
                {
                    "domain": "smt",
                    "instance_id": instance_id,
                    "source": item.get("source", "pinned SMT-LIB QF_LIA file"),
                    "source_url": item.get("source_url", SMTLIB_BENCHMARKS_PAGE),
                    "download_url": url,
                    "sha256": item.get("sha256", ""),
                    "family": item.get("family", "qflia"),
                    "expected": item.get("expected", "UNKNOWN").upper(),
                    "status": "DRY_RUN_PLANNED",
                    "n_vars": "",
                    "n_clauses": "",
                    "max_clause_width": "",
                    "size_bytes": "",
                    "local_path": str(local),
                }
            )
            continue
        try:
            download(url, local, timeout=60)
            text = local.read_text(encoding="utf-8", errors="replace")
            if "(set-logic QF_LIA)" not in text:
                raise ValueError("pinned SMT file is not QF_LIA")
            digest = sha256_file(local)
            expected_sha = item.get("sha256", "")
            if expected_sha and digest != expected_sha:
                raise ValueError(f"sha256 mismatch expected {expected_sha} got {digest}")
            expected = item.get("expected", "").upper() or parse_status(text)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            rows.append(row_for_blocker("smt", instance_id, f"pinned_qflia_fetch_failed_{type(exc).__name__}_{exc}", url))
            continue
        rows.append(
            {
                "domain": "smt",
                "instance_id": instance_id,
                "source": item.get("source", "pinned SMT-LIB QF_LIA file"),
                "source_url": item.get("source_url", SMTLIB_BENCHMARKS_PAGE),
                "download_url": url,
                "sha256": digest,
                "family": item.get("family", "qflia"),
                "expected": expected,
                "status": "OK",
                "n_vars": "",
                "n_clauses": "",
                "max_clause_width": "",
                "size_bytes": str(local.stat().st_size),
                "local_path": str(local),
            }
        )
    if not any(row["status"] == "OK" for row in rows):
        rows.append(row_for_blocker("smt", "smtlib-qflia-small-subset", "no_pinned_qflia_entries_downloaded_cleanly", str(manifest_path)))
    return rows


def fetch_opb(args: argparse.Namespace, root: Path, dry_run: bool) -> list[dict[str, str]]:
    if args.opb_count <= 0:
        return []
    if not any(which_solver(name) for name in ("roundingsat", "open-wbo", "open-wbo_static", "scip", "sat4j-pb", "minisat+", "minisatp", "pbsolver")):
        return [row_for_blocker("opb", "pb25-solver", "no_pb_solver_available_on_path_or_user_cache", PB25_PAGE)]
    size = head_length(PB_SAMPLE_URL)
    if size is not None and size > args.max_opb_archive_bytes:
        return [row_for_blocker("opb", "pb-sample-by-int-size", f"archive_too_large_{size}_bytes", PB_SAMPLE_URL)]
    if dry_run:
        return [
            {
                "domain": "opb",
                "instance_id": "pb-sample-by-int-size",
                "source": "PB25-listed SampleByIntSize archive",
                "source_url": PB25_PAGE,
                "download_url": PB_SAMPLE_URL,
                "sha256": "",
                "family": "sample_by_intsize",
                "expected": "UNKNOWN",
                "status": "DRY_RUN_PLANNED",
                "n_vars": "",
                "n_clauses": "",
                "max_clause_width": "",
                "size_bytes": str(size or ""),
                "local_path": str(root / "opb"),
            }
        ]
    tar_path = root / "opb" / "SampleByIntSize.tar"
    download(PB_SAMPLE_URL, tar_path, timeout=300)
    rows: list[dict[str, str]] = []
    extract_dir = root / "opb" / "instances"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        members = [
            m
            for m in tar.getmembers()
            if m.isfile() and (m.name.endswith(".opb") or m.name.endswith(".opb.xz")) and m.size < args.max_opb_file_bytes
        ]
        for member in sorted(members, key=lambda m: (m.size, m.name))[: args.opb_count]:
            safe_name = Path(member.name).name
            if safe_name.endswith(".xz"):
                safe_name = safe_name[:-3]
            out = extract_dir / safe_name
            compressed_out = out.with_suffix(out.suffix + ".xz")
            with tar.extractfile(member) as fin, compressed_out.open("wb") as fout:  # type: ignore[arg-type]
                if fin is None:
                    continue
                shutil.copyfileobj(fin, fout)
            if member.name.endswith(".xz"):
                decompress_xz(compressed_out, out)
            else:
                shutil.copyfile(compressed_out, out)
            rows.append(
                {
                    "domain": "opb",
                    "instance_id": safe_name,
                    "source": "PB25-listed SampleByIntSize archive",
                    "source_url": PB25_PAGE,
                    "download_url": PB_SAMPLE_URL,
                    "sha256": sha256_file(out),
                    "family": "sample_by_intsize",
                    "expected": "UNKNOWN",
                    "status": "OK",
                    "n_vars": "",
                    "n_clauses": "",
                    "max_clause_width": "",
                    "size_bytes": str(out.stat().st_size),
                    "local_path": str(out),
                }
            )
    if not rows:
        rows.append(row_for_blocker("opb", "pb-sample-by-int-size", "no_small_opb_members_selected", PB_SAMPLE_URL))
    return rows


def write_manifest(rows: list[dict[str, str]], out_dir: Path, stamp: str) -> Path:
    manifest_csv = out_dir / "corpus_manifest.csv"
    fields = ["domain", "instance_id", "source", "source_url", "download_url", "sha256", "family", "expected", "status", "n_vars", "n_clauses", "max_clause_width", "size_bytes", "local_path"]
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest_txt = out_dir / "manifest.txt"
    manifest_txt.write_text(
        "\n".join(
            [
                "schema=sounio.solver.external_corpus_pilot.fetch.v1",
                f"timestamp_utc={stamp}",
                f"corpus_manifest={manifest_csv}",
                f"instances={len(rows)}",
                f"tls_verification={'disabled' if tls_context() is not None else 'default'}",
                "note=Real external corpus pilot; blockers are explicit rows, not generated replacements.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest_txt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sat-count", type=int, default=12)
    parser.add_argument("--smt-count", type=int, default=12)
    parser.add_argument("--opb-count", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-sat-vars", type=int, default=1024)
    parser.add_argument("--max-sat-clauses", type=int, default=4096)
    parser.add_argument("--max-sat-clause-width", type=int, default=8)
    parser.add_argument("--max-sat-compressed-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-sat-candidates", type=int, default=140)
    parser.add_argument("--sat-head-timeout-sec", type=int, default=3)
    parser.add_argument("--sat-download-timeout-sec", type=int, default=30)
    parser.add_argument("--sat-use-head", action="store_true", help="Use HEAD prefiltering before SAT downloads")
    parser.add_argument("--max-opb-archive-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-opb-file-bytes", type=int, default=512 * 1024)
    parser.add_argument("--smt-manifest", type=Path, default=None, help="Pinned CSV of small QF_LIA SMT-LIB files")
    parser.add_argument("--max-smt-file-bytes", type=int, default=256 * 1024)
    parser.add_argument("--smt-probe-timeout-sec", type=int, default=5)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.out_dir or Path(os.environ.get("SOLVER_EXTERNAL_CORPUS_ROOT", f"/tmp/sounio-solver-external-corpus-{stamp}"))
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(fetch_sat(args, root, args.dry_run))
    rows.extend(fetch_smt(args, root, args.dry_run))
    rows.extend(fetch_opb(args, root, args.dry_run))
    manifest = write_manifest(rows, root, stamp)
    print(manifest.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

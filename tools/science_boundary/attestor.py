#!/usr/bin/env python3
"""Evaluate Sounio science-ring boundaries and emit deterministic receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


RECEIPT_SCHEMA = "sounio.package-boundary-receipt.v1"
CLAIM_SCHEMA = "sounio.claim-contract.v1"
SCIENCE_SCHEMA = "sounio.science-manifest.v1"
TSV_COLUMNS = (
    "path",
    "ring",
    "evidence_status",
    "context_of_use",
    "visibility",
    "enforcement",
    "next_gate",
    "allowed_claim_classes",
    "evidence_refs",
    "declared_by",
    "declared_at",
    "review_state",
)
CONCLUSIVE_RINGS = {"pl-core", "scientific-package", "research"}
AUDITABLE_RINGS = {
    "scientific-package-candidate",
    "mixed-unresolved",
    "unclassified",
}
VISIBILITIES = {"public", "protected", "embargoed"}
ENFORCEMENTS = {"off", "advisory", "strict"}
ALLOWED_DEPENDENCIES = {
    "pl-core": {"pl-core"},
    "scientific-package": {"scientific-package", "pl-core"},
    "research": {"research", "scientific-package", "pl-core"},
}
EMPIRICAL_CLASSES = {"empirical", "clinical", "clinical-validation"}
EMPIRICAL_EVIDENCE = {"dataset", "split", "diagnostics", "gate", "review"}
GUM_CLASSES = {"gum", "gum-uncertainty", "gum-evaluation"}
GUM_EVIDENCE = {"method", "witness"}
LEGACY_EVIDENCE_TYPES = {
    "epistemic-score",
    "score",
    "regulatory-ready",
    "provenance-level",
    "gum-compliant",
    "validation-coverage",
}
EXIT_REJECT = 20
EXIT_UNKNOWN = 21


def is_sha256(value: Any, *, allow_empty: bool = False) -> bool:
    text = str(value)
    if allow_empty and not text:
        return True
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str
    message: str
    path: str = ""
    dependency: str = ""

    def to_dict(self) -> dict[str, str]:
        result = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.path:
            result["path"] = self.path
        if self.dependency:
            result["dependency"] = self.dependency
        return result


@dataclass(frozen=True)
class PolicyRow:
    path: str
    ring: str
    evidence_status: str
    context_of_use: str
    visibility: str
    enforcement: str
    next_gate: str
    allowed_claim_classes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    declared_by: str
    declared_at: str
    review_state: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "ring": self.ring,
            "evidence_status": self.evidence_status,
            "context_of_use": self.context_of_use,
            "visibility": self.visibility,
            "enforcement": self.enforcement,
            "next_gate": self.next_gate,
            "allowed_claim_classes": list(self.allowed_claim_classes),
            "evidence_refs": list(self.evidence_refs),
            "declared_by": self.declared_by,
            "declared_at": self.declared_at,
            "review_state": self.review_state,
        }


@dataclass
class Policy:
    root: Path
    source: Path | None
    kind: str
    rows: list[PolicyRow] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    legacy_present: bool = False

    @property
    def valid(self) -> bool:
        return not any(d.code == "E-SRB-000" for d in self.diagnostics)

    @property
    def declared(self) -> bool:
        return bool(self.rows)


@dataclass(frozen=True)
class ImportSpec:
    segments: tuple[str, ...]
    line: int

    @property
    def display(self) -> str:
        return "::".join(self.segments)


@dataclass
class Closure:
    root: Path
    source: Path
    nodes: list[Path] = field(default_factory=list)
    edges: list[tuple[Path, Path]] = field(default_factory=list)
    unresolved: list[tuple[Path, str]] = field(default_factory=list)
    parse_failures: list[tuple[Path, str]] = field(default_factory=list)
    saturated: bool = False
    capacity: int = 256
    collector: str = "sounio-host-syntax-v1"
    report_sha256: str = ""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def receipt_identity(receipt: dict[str, Any]) -> str:
    payload = json.loads(json.dumps(receipt))
    payload.setdefault("hashes", {}).pop("receipt_identity_sha256", None)
    return sha256_bytes(canonical_json(payload))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.setdefault("hashes", {})["receipt_identity_sha256"] = receipt_identity(payload)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def split_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(sorted({str(item).strip() for item in value if str(item).strip()}))
    if value is None:
        return ()
    text = str(value).strip()
    if not text:
        return ()
    separator = "|" if "|" in text else ","
    return tuple(sorted({item.strip() for item in text.split(separator) if item.strip()}))


def safe_relative_text(value: str) -> str | None:
    raw = value.strip().replace("\\", "/")
    if not raw:
        return None
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        return None
    normalized = path.as_posix()
    return "." if normalized in {"", "."} else normalized.rstrip("/")


def receipt_safe_ref(value: str) -> str:
    safe = safe_relative_text(value)
    if safe is not None:
        return safe
    name = Path(value).name
    return name if name and name not in {".", ".."} else "invalid-ref"


def relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix() or "."


def within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def unknown_diagnostic(mode: str, message: str, path: str = "") -> Diagnostic:
    return Diagnostic(
        "E-SRB-000",
        "error" if mode == "strict" else "warning",
        message,
        path,
    )


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise ValueError("top-level TOML value must be a table")
    return value


def validate_policy_row(row: PolicyRow, root: Path, mode: str) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if row.ring not in CONCLUSIVE_RINGS | AUDITABLE_RINGS:
        diagnostics.append(unknown_diagnostic(mode, f"unknown ring: {row.ring}", row.path))
    if row.visibility not in VISIBILITIES:
        diagnostics.append(unknown_diagnostic(mode, f"unknown visibility: {row.visibility}", row.path))
    if row.enforcement not in ENFORCEMENTS:
        diagnostics.append(unknown_diagnostic(mode, f"unknown enforcement: {row.enforcement}", row.path))
    if row.ring in CONCLUSIVE_RINGS and not row.evidence_refs:
        diagnostics.append(
            Diagnostic(
                "E-SRB-003",
                "error",
                "conclusive ring declaration requires named evidence references",
                row.path,
            )
        )
    for field_name, value in (
        ("evidence_status", row.evidence_status),
        ("context_of_use", row.context_of_use),
        ("next_gate", row.next_gate),
        ("declared_by", row.declared_by),
        ("declared_at", row.declared_at),
        ("review_state", row.review_state),
    ):
        if not value:
            diagnostics.append(unknown_diagnostic(mode, f"missing {field_name}", row.path))
    absolute = (root / row.path).resolve()
    if not within_root(absolute, root):
        diagnostics.append(unknown_diagnostic(mode, "policy path escapes its root", row.path))
    elif not absolute.exists():
        diagnostics.append(unknown_diagnostic(mode, "policy path does not exist", row.path))
    return diagnostics


def load_tsv_policy(path: Path, mode: str) -> Policy:
    root = path.parent.resolve()
    policy = Policy(root=root, source=path.resolve(), kind="science-rings-tsv")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(
                (line for line in handle if not line.startswith("#")),
                delimiter="\t",
            )
            if tuple(reader.fieldnames or ()) != TSV_COLUMNS:
                policy.diagnostics.append(
                    unknown_diagnostic(mode, "science-rings.tsv header does not match schema v1")
                )
                return policy
            seen: set[str] = set()
            for line_number, values in enumerate(reader, start=2):
                path_text = safe_relative_text(values.get("path", ""))
                if path_text is None:
                    policy.diagnostics.append(
                        unknown_diagnostic(mode, f"invalid policy path on line {line_number}")
                    )
                    continue
                if path_text in seen:
                    policy.diagnostics.append(
                        unknown_diagnostic(mode, f"duplicate policy path on line {line_number}", path_text)
                    )
                    continue
                seen.add(path_text)
                row = PolicyRow(
                    path=path_text,
                    ring=str(values.get("ring") or "").strip(),
                    evidence_status=str(values.get("evidence_status") or "").strip(),
                    context_of_use=str(values.get("context_of_use") or "").strip(),
                    visibility=str(values.get("visibility") or "").strip(),
                    enforcement=str(values.get("enforcement") or "").strip(),
                    next_gate=str(values.get("next_gate") or "").strip(),
                    allowed_claim_classes=split_list(values.get("allowed_claim_classes")),
                    evidence_refs=split_list(values.get("evidence_refs")),
                    declared_by=str(values.get("declared_by") or "").strip(),
                    declared_at=str(values.get("declared_at") or "").strip(),
                    review_state=str(values.get("review_state") or "").strip(),
                )
                policy.rows.append(row)
                policy.diagnostics.extend(validate_policy_row(row, root, mode))
    except (OSError, UnicodeError, csv.Error) as error:
        policy.diagnostics.append(unknown_diagnostic(mode, f"cannot parse science-rings.tsv: {error}"))
    if not policy.rows:
        policy.diagnostics.append(unknown_diagnostic(mode, "science-rings.tsv has no declarations"))
    return policy


def example_evidence_refs(example: dict[str, Any]) -> tuple[str, ...]:
    return split_list(example.get("evidence-refs", example.get("evidence_refs")))


def validate_scientific_examples(
    data: dict[str, Any], root: Path, mode: str
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    examples = data.get("example", [])
    if not isinstance(examples, list):
        return [unknown_diagnostic(mode, "[[example]] must be an array of tables")]
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            diagnostics.append(unknown_diagnostic(mode, f"example {index} is not a table"))
            continue
        name = str(example.get("name", f"example-{index}"))
        maturity = str(example.get("maturity", "")).strip()
        context = str(example.get("context-of-use", example.get("context_of_use", ""))).strip()
        refs = example_evidence_refs(example)
        for field_name, value in (
            ("maturity", maturity),
            ("context-of-use", context),
            ("evidence-refs", refs),
        ):
            if not value:
                diagnostics.append(
                    unknown_diagnostic(mode, f"scientific example {name} is missing {field_name}")
                )
        example_path = safe_relative_text(str(example.get("path", "")))
        if example_path is None or not within_root(root / (example_path or ""), root):
            diagnostics.append(unknown_diagnostic(mode, f"scientific example {name} has invalid path"))
        if maturity in {"calibrated", "validated"}:
            evidence_types = {ref.split(":", 1)[0] for ref in refs if ":" in ref}
            missing = sorted(EMPIRICAL_EVIDENCE - evidence_types)
            if missing:
                diagnostics.append(
                    unknown_diagnostic(
                        mode,
                        f"scientific example {name} maturity={maturity} lacks typed evidence: {','.join(missing)}",
                    )
                )
    return diagnostics


def load_toml_policy(path: Path, mode: str) -> Policy:
    root = path.parent.resolve()
    policy = Policy(root=root, source=path.resolve(), kind="sounio-toml")
    try:
        data = load_toml(path)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValueError) as error:
        policy.diagnostics.append(unknown_diagnostic(mode, f"cannot parse sounio.toml: {error}"))
        return policy
    if isinstance(data.get("epistemic"), dict):
        policy.legacy_present = True
        policy.diagnostics.append(
            Diagnostic(
                "W-SRB-LEGACY-001",
                "warning",
                "legacy [epistemic] metadata is read for compatibility and grants no boundary or claim authority",
                "sounio.toml",
            )
        )
    science = data.get("science")
    if not isinstance(science, dict):
        return policy
    schema = str(science.get("schema", "")).strip()
    if schema != SCIENCE_SCHEMA:
        policy.diagnostics.append(
            unknown_diagnostic(mode, f"[science].schema must be {SCIENCE_SCHEMA}", "sounio.toml")
        )
    row = PolicyRow(
        path=".",
        ring=str(science.get("ring", "")).strip(),
        evidence_status=str(science.get("evidence-status", science.get("evidence_status", ""))).strip(),
        context_of_use=str(science.get("context-of-use", science.get("context_of_use", ""))).strip(),
        visibility=str(science.get("visibility", "")).strip(),
        enforcement="advisory",
        next_gate=str(science.get("next-gate", science.get("next_gate", "package-boundary-receipt"))).strip(),
        allowed_claim_classes=split_list(
            science.get("allowed-claim-classes", science.get("allowed_claim_classes"))
        ),
        evidence_refs=split_list(science.get("evidence-refs", science.get("evidence_refs"))),
        declared_by=str(science.get("declared-by", science.get("declared_by", "sounio.toml"))).strip(),
        declared_at=str(science.get("declared-at", science.get("declared_at", "manifest-version"))).strip(),
        review_state=str(science.get("review-state", science.get("review_state", "draft"))).strip(),
    )
    policy.rows.append(row)
    policy.diagnostics.extend(validate_policy_row(row, root, mode))
    policy.diagnostics.extend(validate_scientific_examples(data, root, mode))
    return policy


def policy_has_science(path: Path) -> bool:
    try:
        return isinstance(load_toml(path).get("science"), dict)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValueError):
        return True


def discover_policy(source: Path, explicit: str, mode: str) -> Policy | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            return Policy(
                root=path.parent,
                source=path,
                kind="missing",
                diagnostics=[unknown_diagnostic(mode, "science manifest does not exist", path.name)],
            )
        if path.name == "sounio.toml" or path.suffix == ".toml":
            return load_toml_policy(path, mode)
        return load_tsv_policy(path, mode)

    start = source.resolve().parent
    for directory in (start, *start.parents):
        rings = directory / "science-rings.tsv"
        manifest = directory / "sounio.toml"
        has_rings = rings.is_file()
        has_manifest = manifest.is_file() and policy_has_science(manifest)
        if has_rings and has_manifest:
            return Policy(
                root=directory.resolve(),
                source=None,
                kind="conflicting-declarations",
                diagnostics=[
                    unknown_diagnostic(
                        mode,
                        "science-rings.tsv conflicts with [science] in sounio.toml at the same root",
                    )
                ],
            )
        if has_rings:
            return load_tsv_policy(rings, mode)
        if has_manifest:
            return load_toml_policy(manifest, mode)
    return None


def tokenize_import_surface(text: str) -> tuple[list[tuple[str, str, int]], list[str]]:
    tokens: list[tuple[str, str, int]] = []
    errors: list[str] = []
    i = 0
    line = 1
    length = len(text)
    while i < length:
        char = text[i]
        if char in " \t\r":
            i += 1
            continue
        if char == "\n":
            tokens.append(("newline", "\n", line))
            line += 1
            i += 1
            continue
        if text.startswith("//", i) or char == "#":
            end = text.find("\n", i)
            i = length if end < 0 else end
            continue
        if text.startswith("/*", i):
            start_line = line
            depth = 1
            i += 2
            while i < length and depth:
                if text.startswith("/*", i):
                    depth += 1
                    i += 2
                elif text.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    if text[i] == "\n":
                        line += 1
                    i += 1
            if depth:
                errors.append(f"unterminated block comment starting on line {start_line}")
            continue
        if char in {'"', "'"}:
            quote = char
            start_line = line
            i += 1
            escaped = False
            while i < length:
                current = text[i]
                if current == "\n":
                    line += 1
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == quote:
                    i += 1
                    break
                i += 1
            else:
                errors.append(f"unterminated string starting on line {start_line}")
            continue
        if char.isalpha() or char == "_":
            start = i
            i += 1
            while i < length and (text[i].isalnum() or text[i] == "_"):
                i += 1
            tokens.append(("ident", text[start:i], line))
            continue
        if text.startswith("::", i):
            tokens.append(("symbol", "::", line))
            i += 2
            continue
        tokens.append(("symbol", char, line))
        i += 1
    return tokens, errors


def parse_imports(path: Path) -> tuple[list[ImportSpec], list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [], [f"cannot read source: {error}"]
    tokens, errors = tokenize_import_surface(text)
    imports: list[ImportSpec] = []
    depth = 0
    index = 0
    while index < len(tokens):
        kind, value, line = tokens[index]
        import_index = index
        if depth == 0 and value == "pub" and index + 1 < len(tokens):
            if tokens[index + 1][1] in {"use", "import"}:
                import_index = index + 1
            else:
                import_index = index
        if depth == 0 and tokens[import_index][1] in {"use", "import"}:
            cursor = import_index + 1
            segments: list[str] = []
            expect_ident = True
            while cursor < len(tokens):
                token_kind, token_value, _ = tokens[cursor]
                if token_value in {"\n", ";", "{", "*", ","}:
                    break
                if expect_ident and token_kind == "ident":
                    segments.append(token_value)
                    expect_ident = False
                elif not expect_ident and token_value == "::":
                    expect_ident = True
                else:
                    break
                cursor += 1
            if segments:
                imports.append(ImportSpec(tuple(segments), line))
            else:
                errors.append(f"invalid import on line {line}")
            while cursor < len(tokens) and tokens[cursor][1] not in {"\n", ";"}:
                cursor += 1
            index = cursor + 1
            continue
        if value == "{":
            depth += 1
        elif value == "}" and depth > 0:
            depth -= 1
        index += 1
    return imports, errors


def candidate_import_paths(source: Path, spec: ImportSpec, root: Path) -> Iterable[Path]:
    relative = Path(*spec.segments).with_suffix(".sio")
    module_relative = Path(*spec.segments) / "mod.sio"
    current = source.parent.resolve()
    seen: set[Path] = set()
    while True:
        for suffix in (relative, module_relative):
            candidate = (current / suffix).resolve()
            if candidate not in seen:
                seen.add(candidate)
                yield candidate
        if current == root or not within_root(current.parent, root):
            break
        current = current.parent
    stdlib_root = os.environ.get("SOUNIO_STDLIB_PATH", "")
    if stdlib_root:
        for suffix in (relative, module_relative):
            yield (Path(stdlib_root) / suffix).resolve()
    for prefix in (root / "stdlib",):
        for suffix in (relative, module_relative):
            yield (prefix / suffix).resolve()
    package_name = spec.segments[0].replace("_", "-")
    package_root = root / "packages" / package_name / "src"
    if len(spec.segments) == 1:
        yield (package_root / "lib.sio").resolve()
    else:
        package_suffix = Path(*spec.segments[1:]).with_suffix(".sio")
        yield (package_root / package_suffix).resolve()


def resolve_import(source: Path, spec: ImportSpec, root: Path) -> Path | None:
    for candidate in candidate_import_paths(source, spec, root):
        if candidate.is_file():
            return candidate
    return None


def collect_closure(source: Path, root: Path) -> Closure:
    capacity_text = os.environ.get("SOUNIO_SCIENCE_BOUNDARY_MAX_NODES", "256")
    try:
        capacity = max(1, min(int(capacity_text), 4096))
    except ValueError:
        capacity = 256
    closure = Closure(root=root, source=source, capacity=capacity)
    queue = [source.resolve()]
    visited: set[Path] = set()
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        if len(visited) >= capacity:
            closure.saturated = True
            break
        visited.add(current)
        closure.nodes.append(current)
        imports, errors = parse_imports(current)
        closure.parse_failures.extend((current, error) for error in errors)
        for spec in imports:
            resolved = resolve_import(current, spec, root)
            if resolved is None:
                closure.unresolved.append((current, f"{spec.display}@{spec.line}"))
                continue
            if not within_root(resolved, root):
                closure.unresolved.append((current, f"root-escape:{spec.display}@{spec.line}"))
                continue
            closure.edges.append((current, resolved))
            if resolved not in visited and resolved not in queue:
                if len(visited) + len(queue) >= capacity:
                    closure.saturated = True
                else:
                    queue.append(resolved)
    closure.nodes.sort(key=lambda item: relative_to_root(item, root))
    closure.edges = sorted(
        set(closure.edges),
        key=lambda edge: (relative_to_root(edge[0], root), relative_to_root(edge[1], root)),
    )
    closure.unresolved.sort(key=lambda item: (relative_to_root(item[0], root), item[1]))
    closure.parse_failures.sort(key=lambda item: (relative_to_root(item[0], root), item[1]))
    return closure


def closure_report_path(value: str, root: Path) -> Path | None:
    raw = Path(value)
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    return candidate if within_root(candidate, root) else None


def load_raw_ast_closure_report(path: Path, source: Path, root: Path) -> tuple[Closure, list[Diagnostic]]:
    closure = Closure(
        root=root,
        source=source,
        collector="madaros-raw-ast-v1",
        report_sha256=sha256_file(path),
    )
    diagnostics: list[Diagnostic] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        closure.parse_failures.append((source, f"cannot read raw AST closure report: {error}"))
        return closure, diagnostics
    try:
        header_index = lines.index("SOUNIO_BOUNDARY_CLOSURE_V1")
    except ValueError:
        closure.parse_failures.append((source, "invalid raw AST closure report header"))
        return closure, diagnostics
    lines = lines[header_index:]

    status = ""
    declared_nodes: list[Path] = []
    declared_edges: list[tuple[Path, Path]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        kind = fields[0] if fields else ""
        if kind == "status" and len(fields) == 2:
            status = fields[1]
        elif kind == "capacity" and len(fields) == 2:
            try:
                closure.capacity = max(1, min(int(fields[1]), 4096))
            except ValueError:
                closure.parse_failures.append((source, f"invalid capacity on report line {line_number}"))
        elif kind == "saturated" and len(fields) == 2:
            closure.saturated = fields[1] == "true"
        elif kind == "parse_failed" and len(fields) == 2:
            if fields[1] == "true":
                closure.parse_failures.append((source, "raw AST parser reported failure"))
        elif kind == "node" and len(fields) == 2:
            node = closure_report_path(fields[1], root)
            if node is None or not node.is_file():
                closure.parse_failures.append((source, f"invalid raw AST node on report line {line_number}"))
            else:
                declared_nodes.append(node)
        elif kind == "edge" and len(fields) == 3:
            caller = closure_report_path(fields[1], root)
            dependency = closure_report_path(fields[2], root)
            if caller is None or dependency is None:
                closure.parse_failures.append((source, f"raw AST edge escapes root on report line {line_number}"))
            else:
                declared_edges.append((caller, dependency))
        elif kind == "unresolved" and len(fields) == 3:
            caller = closure_report_path(fields[1], root)
            if caller is None:
                closure.parse_failures.append((source, f"raw AST unresolved caller escapes root on report line {line_number}"))
            else:
                closure.unresolved.append((caller, fields[2]))
        elif line:
            closure.parse_failures.append((source, f"invalid raw AST record on report line {line_number}"))

    node_set = set(declared_nodes)
    if source.resolve() not in node_set:
        closure.parse_failures.append((source, "raw AST closure does not contain the root source"))
    for caller, dependency in declared_edges:
        if caller not in node_set or dependency not in node_set:
            closure.parse_failures.append((source, "raw AST edge references an undeclared node"))
    if status not in {"complete", "incomplete"}:
        closure.parse_failures.append((source, "raw AST closure status is missing or invalid"))
    elif status == "incomplete" and not closure.parse_failures:
        closure.parse_failures.append((source, "raw AST closure is incomplete"))
    if status == "complete" and (closure.saturated or closure.unresolved):
        closure.parse_failures.append((source, "raw AST closure claims complete with unresolved or saturated state"))

    closure.nodes = sorted(node_set, key=lambda item: relative_to_root(item, root))
    closure.edges = sorted(
        set(declared_edges),
        key=lambda edge: (relative_to_root(edge[0], root), relative_to_root(edge[1], root)),
    )
    closure.unresolved.sort(key=lambda item: (relative_to_root(item[0], root), item[1]))
    return closure, diagnostics


def classify_path(path: Path, policy: Policy) -> PolicyRow | None:
    matches: list[tuple[int, PolicyRow]] = []
    for row in policy.rows:
        absolute = (policy.root / row.path).resolve()
        if path.resolve() == absolute or within_root(path, absolute):
            matches.append((len(absolute.parts), row))
    if not matches:
        return None
    matches.sort(key=lambda value: (-value[0], value[1].path))
    return matches[0][1]


def load_claim_contract(path_text: str, mode: str) -> tuple[dict[str, Any] | None, list[Diagnostic]]:
    if not path_text:
        return None, []
    path = Path(path_text).expanduser().resolve()
    diagnostics: list[Diagnostic] = []
    try:
        contract = load_toml(path)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValueError) as error:
        return None, [unknown_diagnostic(mode, f"cannot parse claim contract: {error}", path.name)]
    if contract.get("schema") != CLAIM_SCHEMA:
        diagnostics.append(unknown_diagnostic(mode, f"claim contract schema must be {CLAIM_SCHEMA}", path.name))
    for key in ("claim-id", "requested-class", "context-of-use", "root-artifact"):
        if not str(contract.get(key, "")).strip():
            diagnostics.append(unknown_diagnostic(mode, f"claim contract is missing {key}", path.name))
    evidence = contract.get("evidence", [])
    if not isinstance(evidence, list):
        diagnostics.append(unknown_diagnostic(mode, "claim contract evidence must be an array of tables", path.name))
    else:
        for index, item in enumerate(evidence):
            digest = str(item.get("sha256", "")).strip() if isinstance(item, dict) else ""
            if (
                not isinstance(item, dict)
                or not str(item.get("type", "")).strip()
                or not str(item.get("ref", "")).strip()
                or not is_sha256(digest)
            ):
                diagnostics.append(
                    unknown_diagnostic(
                        mode,
                        f"claim evidence {index} requires type, ref, and lowercase SHA-256",
                        path.name,
                    )
                )
    return contract, diagnostics


def claim_diagnostics(
    contract: dict[str, Any] | None,
    root_row: PolicyRow | None,
    source_rel: str,
    source: Path,
    policy: Policy | None,
    compiler_path: str,
) -> list[Diagnostic]:
    if contract is None:
        return []
    diagnostics: list[Diagnostic] = []
    requested = str(contract.get("requested-class", "")).strip()
    context = str(contract.get("context-of-use", "")).strip()
    root_artifact = safe_relative_text(str(contract.get("root-artifact", "")))
    evidence = contract.get("evidence", [])
    evidence_types = {
        str(item.get("type", "")).strip()
        for item in evidence
        if isinstance(item, dict) and str(item.get("type", "")).strip()
    }
    root = policy.root if policy is not None else source.parent
    expected_source_sha = sha256_file(source)
    expected_policy_ref = ""
    expected_policy_sha = ""
    if policy is not None and policy.source is not None and policy.source.is_file():
        expected_policy_ref = relative_to_root(policy.source, policy.root)
        expected_policy_sha = sha256_file(policy.source)
    expected_compiler_sha = ""
    if compiler_path and Path(compiler_path).is_file():
        expected_compiler_sha = sha256_file(Path(compiler_path))

    for item in evidence:
        if not isinstance(item, dict):
            continue
        evidence_type = str(item.get("type", "")).strip()
        evidence_ref = str(item.get("ref", "")).strip()
        evidence_sha = str(item.get("sha256", "")).strip()
        if safe_relative_text(evidence_ref) is None:
            valid_binding = False
        elif evidence_type == "source":
            valid_binding = safe_relative_text(evidence_ref) == source_rel and evidence_sha == expected_source_sha
        elif evidence_type == "package":
            valid_binding = evidence_ref == expected_policy_ref and evidence_sha == expected_policy_sha
        elif evidence_type == "compiler":
            valid_binding = bool(expected_compiler_sha) and evidence_sha == expected_compiler_sha
        else:
            relative_ref = safe_relative_text(evidence_ref)
            evidence_path = (root / relative_ref).resolve() if relative_ref else root.parent / "invalid"
            valid_binding = (
                relative_ref is not None
                and within_root(evidence_path, root)
                and evidence_path.is_file()
                and sha256_file(evidence_path) == evidence_sha
            )
        if not valid_binding:
            display_ref = receipt_safe_ref(evidence_ref)
            diagnostics.append(
                Diagnostic(
                    "E-SRB-006",
                    "error",
                    f"claim evidence is not bound to verified content: {evidence_type}:{display_ref}",
                    source_rel,
                )
            )
    legacy = sorted(evidence_types & LEGACY_EVIDENCE_TYPES)
    if legacy:
        diagnostics.append(
            Diagnostic(
                "E-SRB-004",
                "error",
                f"legacy scalar metadata cannot serve as claim evidence: {','.join(legacy)}",
                source_rel,
            )
        )
    if root_artifact != source_rel:
        diagnostics.append(
            Diagnostic(
                "E-SRB-006",
                "error",
                f"claim root-artifact must bind the compiled source ({source_rel})",
                source_rel,
            )
        )
    required = {"source", "package", "compiler"}
    if requested in EMPIRICAL_CLASSES:
        required |= {"model", "data-manifest"}
    missing_provenance = sorted(required - evidence_types)
    if missing_provenance:
        diagnostics.append(
            Diagnostic(
                "E-SRB-006",
                "error",
                f"claim lacks provenance bindings: {','.join(missing_provenance)}",
                source_rel,
            )
        )
    if requested in GUM_CLASSES:
        missing_gum = sorted(GUM_EVIDENCE - evidence_types)
        if missing_gum:
            diagnostics.append(
                Diagnostic(
                    "E-SRB-004",
                    "error",
                    f"GUM claim requires named method and witness evidence; missing {','.join(missing_gum)}",
                    source_rel,
                )
            )
    if requested in EMPIRICAL_CLASSES:
        missing_empirical = sorted(EMPIRICAL_EVIDENCE - evidence_types)
        if missing_empirical:
            diagnostics.append(
                Diagnostic(
                    "E-SRB-005",
                    "error",
                    f"{requested} claim cannot be supported by compile/runtime evidence alone; missing {','.join(missing_empirical)}",
                    source_rel,
                )
            )
    if root_row is None or requested not in root_row.allowed_claim_classes:
        diagnostics.append(
            Diagnostic(
                "E-SRB-007",
                "error",
                f"requested claim class is not authorized by the root ring: {requested}",
                source_rel,
            )
        )
    if root_row is not None and context != root_row.context_of_use:
        diagnostics.append(
            Diagnostic(
                "E-SRB-004",
                "error",
                "claim context-of-use does not match the root declaration",
                source_rel,
            )
        )
    return diagnostics


def source_bundle_hash(nodes: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for node in sorted(nodes, key=lambda item: item["path"]):
        digest.update(node["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(node["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_claim_evidence_bindings(
    receipt: dict[str, Any],
    root: Path,
    compiler_path: Path | None,
) -> None:
    claim = receipt.get("claim_contract")
    if not isinstance(claim, dict):
        return
    for binding in claim.get("evidence_bindings", []):
        evidence_type = str(binding.get("type", ""))
        evidence_ref = safe_relative_text(str(binding.get("ref", "")))
        expected = str(binding.get("sha256", ""))
        if evidence_type == "compiler":
            if compiler_path is None or sha256_file(compiler_path) != expected:
                raise ValueError("claim compiler evidence hash mismatch")
            continue
        if evidence_ref is None:
            raise ValueError(f"claim evidence path is invalid: {evidence_type}")
        evidence_path = (root / evidence_ref).resolve()
        if not within_root(evidence_path, root) or sha256_file(evidence_path) != expected:
            raise ValueError(f"claim evidence hash mismatch: {evidence_type}:{evidence_ref}")


def build_receipt(
    *,
    source: Path,
    policy: Policy | None,
    requested_mode: str,
    manifest_path: str,
    claim_path: str,
    compiler_path: str,
    engine_identity: str,
    closure_report_path: str = "",
    elf_path: str = "",
    artifact_label: str = "",
) -> tuple[dict[str, Any], str, list[Diagnostic]]:
    effective_mode = requested_mode
    if requested_mode == "auto":
        has_declaration = (
            (policy is not None and (policy.declared or bool(manifest_path)))
            or bool(claim_path)
        )
        effective_mode = "advisory" if has_declaration else "off"
    root = (policy.root if policy is not None else source.resolve().parent).resolve()
    diagnostics = list(policy.diagnostics if policy is not None else [])
    if effective_mode != "off" and policy is None:
        diagnostics.append(unknown_diagnostic(effective_mode, "no scientific declaration was discovered"))
    closure: Closure | None = None
    rows_by_path: dict[str, PolicyRow | None] = {}
    graph_nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, str]] = []
    graph_unresolved: list[dict[str, str]] = []
    source_rel = source.name
    if source.exists() and within_root(source, root):
        source_rel = relative_to_root(source, root)
    elif effective_mode != "off":
        diagnostics.append(unknown_diagnostic(effective_mode, "source path escapes policy root", source.name))

    if effective_mode != "off" and policy is not None and policy.valid and within_root(source, root):
        if closure_report_path:
            report_path = Path(closure_report_path).expanduser().resolve()
            try:
                closure, report_diagnostics = load_raw_ast_closure_report(
                    report_path,
                    source.resolve(),
                    root,
                )
                diagnostics.extend(report_diagnostics)
            except OSError as error:
                closure = collect_closure(source.resolve(), root)
                diagnostics.append(
                    unknown_diagnostic(effective_mode, f"cannot hash raw AST closure report: {error}")
                )
        else:
            closure = collect_closure(source.resolve(), root)
            diagnostics.append(
                unknown_diagnostic(
                    effective_mode,
                    "raw Madaros AST closure report is required for an authoritative verdict; host syntax audit is non-authoritative",
                )
            )
        for current, message in closure.parse_failures:
            diagnostics.append(
                unknown_diagnostic(effective_mode, f"closure parser incomplete: {message}", relative_to_root(current, root))
            )
        for current, imported in closure.unresolved:
            diagnostics.append(
                unknown_diagnostic(
                    effective_mode,
                    f"unresolved import in authoritative closure: {imported}",
                    relative_to_root(current, root),
                )
            )
        if closure.saturated:
            diagnostics.append(
                unknown_diagnostic(effective_mode, f"module closure reached capacity {closure.capacity}")
            )
        for node in closure.nodes:
            rel = relative_to_root(node, root)
            row = classify_path(node, policy)
            rows_by_path[rel] = row
            if row is None:
                diagnostics.append(unknown_diagnostic(effective_mode, "module is unclassified", rel))
                ring_data = {
                    "ring": "unclassified",
                    "evidence_status": "unknown",
                    "context_of_use": "undeclared",
                    "visibility": "protected",
                    "review_state": "unreviewed",
                }
            else:
                ring_data = {
                    "ring": row.ring,
                    "evidence_status": row.evidence_status,
                    "context_of_use": row.context_of_use,
                    "visibility": row.visibility,
                    "review_state": row.review_state,
                }
                if row.ring not in CONCLUSIVE_RINGS:
                    diagnostics.append(unknown_diagnostic(effective_mode, f"non-conclusive ring: {row.ring}", rel))
            graph_nodes.append({"path": rel, "sha256": sha256_file(node), **ring_data})
        for caller, dependency in closure.edges:
            caller_rel = relative_to_root(caller, root)
            dependency_rel = relative_to_root(dependency, root)
            caller_row = rows_by_path.get(caller_rel)
            dependency_row = rows_by_path.get(dependency_rel)
            graph_edges.append({"caller": caller_rel, "dependency": dependency_rel})
            if caller_row is None or dependency_row is None:
                continue
            if (
                closure.collector == "madaros-raw-ast-v1"
                and caller_row.ring in CONCLUSIVE_RINGS
                and dependency_row.ring in CONCLUSIVE_RINGS
            ):
                if dependency_row.ring not in ALLOWED_DEPENDENCIES[caller_row.ring]:
                    diagnostics.append(
                        Diagnostic(
                            "E-SRB-001",
                            "error",
                            f"ring dependency is forbidden: {caller_row.ring} -> {dependency_row.ring}",
                            caller_rel,
                            dependency_rel,
                        )
                    )
            if (
                closure.collector == "madaros-raw-ast-v1"
                and caller_row.visibility == "public"
                and dependency_row.visibility in {"protected", "embargoed"}
            ):
                # Research examples may call scientific-package-candidate surfaces
                # that remain protected until ring promotion. This is the allowlist
                # for construction verticals (e.g. examples/particle_physics →
                # stdlib/particle_physics) without flipping all of stdlib public.
                research_to_candidate = (
                    caller_row.ring == "research"
                    and dependency_row.ring == "scientific-package-candidate"
                )
                if not research_to_candidate:
                    diagnostics.append(
                        Diagnostic(
                            "E-SRB-002",
                            "error",
                            f"public module cannot depend on {dependency_row.visibility} module",
                            caller_rel,
                            dependency_rel,
                        )
                    )
        graph_unresolved = [
            {"caller": relative_to_root(item[0], root), "import": item[1]}
            for item in closure.unresolved
        ]
    elif source.exists():
        graph_nodes.append(
            {
                "path": source_rel,
                "sha256": sha256_file(source),
                "ring": "unclassified",
                "evidence_status": "not-evaluated",
                "context_of_use": "not-evaluated",
                "visibility": "protected",
                "review_state": "not-evaluated",
            }
        )

    contract, claim_parse_diagnostics = load_claim_contract(claim_path, effective_mode)
    diagnostics.extend(claim_parse_diagnostics)
    if claim_path and not within_root(Path(claim_path).expanduser().resolve(), root):
        diagnostics.append(unknown_diagnostic(effective_mode, "claim contract path escapes policy root"))
    root_row = rows_by_path.get(source_rel)
    if effective_mode != "off" and not claim_parse_diagnostics and source.is_file():
        diagnostics.extend(
            claim_diagnostics(
                contract,
                root_row,
                source_rel,
                source,
                policy,
                compiler_path,
            )
        )

    compiler_sha = ""
    if compiler_path:
        compiler = Path(compiler_path).expanduser().resolve()
        try:
            compiler_sha = sha256_file(compiler)
        except OSError as error:
            diagnostics.append(unknown_diagnostic(effective_mode, f"cannot hash compiler: {error}"))
    elif effective_mode == "strict":
        diagnostics.append(unknown_diagnostic(effective_mode, "strict receipt requires compiler identity"))
    if effective_mode == "strict" and not engine_identity.strip():
        diagnostics.append(unknown_diagnostic(effective_mode, "strict receipt requires compiler engine identity"))

    reject = any(d.code in {f"E-SRB-{index:03d}" for index in range(1, 8)} for d in diagnostics)
    unknown = any(d.code == "E-SRB-000" for d in diagnostics)
    if effective_mode == "off":
        verdict = "UNKNOWN"
    elif unknown:
        verdict = "UNKNOWN"
    elif reject:
        verdict = "REJECT"
    else:
        verdict = "OK"

    policy_sha = ""
    policy_source = ""
    policy_kind = "none"
    if policy is not None and policy.source is not None and policy.source.is_file():
        policy_sha = sha256_file(policy.source)
        policy_source = relative_to_root(policy.source, policy.root)
        policy_kind = policy.kind
    claim_sha = ""
    claim_summary: dict[str, Any] | None = None
    if claim_path and Path(claim_path).is_file():
        claim_sha = sha256_file(Path(claim_path))
    if contract is not None:
        claim_source = Path(claim_path).expanduser().resolve()
        claim_summary = {
            "source": relative_to_root(claim_source, root) if within_root(claim_source, root) else claim_source.name,
            "claim_id": str(contract.get("claim-id", "")),
            "requested_class": str(contract.get("requested-class", "")),
            "context_of_use": str(contract.get("context-of-use", "")),
            "root_artifact": str(contract.get("root-artifact", "")),
            "evidence_types": sorted(
                {
                    str(item.get("type", ""))
                    for item in contract.get("evidence", [])
                    if isinstance(item, dict) and str(item.get("type", ""))
                }
            ),
            "evidence_bindings": sorted(
                [
                    {
                        "type": str(item.get("type", "")),
                        "ref": receipt_safe_ref(str(item.get("ref", ""))),
                        "sha256": str(item.get("sha256", "")),
                    }
                    for item in contract.get("evidence", [])
                    if isinstance(item, dict)
                ],
                key=lambda item: (item["type"], item["ref"], item["sha256"]),
            ),
        }

    graph_nodes.sort(key=lambda item: item["path"])
    graph_edges.sort(key=lambda item: (item["caller"], item["dependency"]))
    diagnostics.sort(key=lambda item: (item.code, item.path, item.dependency, item.message))
    hashes = {
        "source_bundle_sha256": source_bundle_hash(graph_nodes),
        "policy_sha256": policy_sha,
        "claim_contract_sha256": claim_sha,
        "compiler_sha256": compiler_sha,
        "closure_report_sha256": closure.report_sha256 if closure is not None else "",
        "elf_sha256": sha256_file(Path(elf_path)) if elf_path else "",
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "verdict": verdict,
        "mode": effective_mode,
        "identity_only": True,
        "graph": {
            "root_artifact": source_rel,
            "capacity": closure.capacity if closure is not None else 0,
            "saturated": closure.saturated if closure is not None else False,
            "nodes": graph_nodes,
            "edges": graph_edges,
            "unresolved_imports": graph_unresolved,
        },
        "diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics],
        "hashes": hashes,
        "engine": {
            "name": "madaros",
            "identity": engine_identity.strip(),
            "boundary_collector": closure.collector if closure is not None else "none",
        },
        "policy": {"kind": policy_kind, "source": policy_source},
        "claim_contract": claim_summary,
        "artifact": {
            "kind": "native-elf" if elf_path else "not-emitted",
            "path": safe_relative_text(artifact_label) or (Path(artifact_label).name if artifact_label else ""),
        },
        "assurance_level": "identity-only",
        "limitations": [
            "does_not_assert_scientific_truth",
            "does_not_assert_clinical_validation_or_clinical_authority",
            "does_not_assert_security_sandboxing",
            "does_not_assert_public_registry_status",
            "does_not_assert_attested_execution_or_independent_replay",
        ]
        + (
            ["host_syntax_closure_is_advisory_only"]
            if closure is not None and closure.collector == "sounio-host-syntax-v1"
            else ["raw_ast_closure_report_is_not_attested_execution"]
        ),
    }
    return receipt, verdict, diagnostics


def print_diagnostics(diagnostics: list[Diagnostic], verdict: str, mode: str) -> None:
    for diagnostic in diagnostics:
        location = f" {diagnostic.path}" if diagnostic.path else ""
        dependency = f" -> {diagnostic.dependency}" if diagnostic.dependency else ""
        print(
            f"{diagnostic.severity}[{diagnostic.code}]:{location}{dependency} {diagnostic.message}",
            file=sys.stderr,
        )
    if mode != "off":
        print(f"science-boundary: mode={mode} verdict={verdict}", file=sys.stderr)


def evaluate_command(args: argparse.Namespace) -> int:
    source = args.source.expanduser().resolve()
    policy = discover_policy(source, args.manifest, args.mode)
    receipt, verdict, diagnostics = build_receipt(
        source=source,
        policy=policy,
        requested_mode=args.mode,
        manifest_path=args.manifest,
        claim_path=args.claim_contract,
        compiler_path=args.compiler,
        engine_identity=args.engine_identity,
        closure_report_path=args.closure_report,
        elf_path=args.elf,
        artifact_label=args.artifact_label,
    )
    if args.receipt:
        write_json_atomic(args.receipt.expanduser().resolve(), receipt)
    print_diagnostics(diagnostics, verdict, receipt["mode"])
    if verdict == "REJECT":
        return EXIT_REJECT
    if verdict == "UNKNOWN" and receipt["mode"] != "off":
        return EXIT_UNKNOWN
    return 0


def verify_preflight_identity(receipt: dict[str, Any]) -> None:
    expected = receipt.get("hashes", {}).get("receipt_identity_sha256", "")
    if not expected or expected != receipt_identity(receipt):
        raise ValueError("receipt identity hash mismatch")


def validate_receipt_structure(receipt: dict[str, Any]) -> None:
    required = {
        "schema",
        "verdict",
        "mode",
        "identity_only",
        "graph",
        "diagnostics",
        "hashes",
        "engine",
        "policy",
        "claim_contract",
        "artifact",
        "assurance_level",
        "limitations",
    }
    if set(receipt) != required:
        raise ValueError("receipt fields do not match schema v1")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("bad receipt schema")
    if receipt.get("verdict") not in {"OK", "REJECT", "UNKNOWN"}:
        raise ValueError("bad receipt verdict")
    if receipt.get("mode") not in {"off", "advisory", "strict"}:
        raise ValueError("bad receipt mode")
    if receipt.get("identity_only") is not True or receipt.get("assurance_level") != "identity-only":
        raise ValueError("bad receipt assurance level")

    graph = receipt.get("graph")
    if not isinstance(graph, dict) or set(graph) != {
        "root_artifact",
        "capacity",
        "saturated",
        "nodes",
        "edges",
        "unresolved_imports",
    }:
        raise ValueError("bad receipt graph")
    if safe_relative_text(str(graph.get("root_artifact", ""))) is None:
        raise ValueError("bad receipt root artifact")
    if not isinstance(graph.get("capacity"), int) or graph["capacity"] < 0:
        raise ValueError("bad receipt graph capacity")
    if not isinstance(graph.get("saturated"), bool):
        raise ValueError("bad receipt graph saturation")
    for node in graph.get("nodes", []):
        if (
            not isinstance(node, dict)
            or safe_relative_text(str(node.get("path", ""))) is None
            or not is_sha256(node.get("sha256", ""))
        ):
            raise ValueError("bad receipt graph node")
    for edge in graph.get("edges", []):
        if (
            not isinstance(edge, dict)
            or safe_relative_text(str(edge.get("caller", ""))) is None
            or safe_relative_text(str(edge.get("dependency", ""))) is None
        ):
            raise ValueError("bad receipt graph edge")
    for unresolved in graph.get("unresolved_imports", []):
        if (
            not isinstance(unresolved, dict)
            or safe_relative_text(str(unresolved.get("caller", ""))) is None
            or not str(unresolved.get("import", ""))
        ):
            raise ValueError("bad receipt unresolved import")

    diagnostics = receipt.get("diagnostics")
    if not isinstance(diagnostics, list):
        raise ValueError("bad receipt diagnostics")
    for diagnostic in diagnostics:
        if (
            not isinstance(diagnostic, dict)
            or not str(diagnostic.get("code", ""))
            or diagnostic.get("severity") not in {"warning", "error"}
            or not str(diagnostic.get("message", ""))
        ):
            raise ValueError("bad receipt diagnostic")

    hashes = receipt.get("hashes")
    hash_fields = {
        "source_bundle_sha256",
        "policy_sha256",
        "claim_contract_sha256",
        "compiler_sha256",
        "closure_report_sha256",
        "elf_sha256",
        "receipt_identity_sha256",
    }
    if not isinstance(hashes, dict) or set(hashes) != hash_fields:
        raise ValueError("bad receipt hashes")
    if not is_sha256(hashes.get("source_bundle_sha256", "")):
        raise ValueError("bad source bundle hash")
    if not is_sha256(hashes.get("receipt_identity_sha256", "")):
        raise ValueError("bad receipt identity hash")
    for key in hash_fields - {"source_bundle_sha256", "receipt_identity_sha256"}:
        if not is_sha256(hashes.get(key, ""), allow_empty=True):
            raise ValueError(f"bad receipt hash: {key}")

    engine = receipt.get("engine")
    if (
        not isinstance(engine, dict)
        or set(engine) != {"name", "identity", "boundary_collector"}
        or engine.get("name") != "madaros"
        or engine.get("boundary_collector") not in {
            "none",
            "madaros-raw-ast-v1",
            "sounio-host-syntax-v1",
        }
    ):
        raise ValueError("bad receipt engine")
    policy = receipt.get("policy")
    if not isinstance(policy, dict) or set(policy) != {"kind", "source"}:
        raise ValueError("bad receipt policy")
    if policy.get("source") and safe_relative_text(str(policy["source"])) is None:
        raise ValueError("bad receipt policy path")
    claim = receipt.get("claim_contract")
    if claim is not None:
        claim_fields = {
            "source",
            "claim_id",
            "requested_class",
            "context_of_use",
            "root_artifact",
            "evidence_types",
            "evidence_bindings",
        }
        if (
            not isinstance(claim, dict)
            or set(claim) != claim_fields
            or safe_relative_text(str(claim.get("source", ""))) is None
            or not str(claim.get("claim_id", ""))
            or not str(claim.get("requested_class", ""))
            or not str(claim.get("context_of_use", ""))
            or safe_relative_text(str(claim.get("root_artifact", ""))) is None
            or not isinstance(claim.get("evidence_types"), list)
            or any(not isinstance(item, str) or not item for item in claim["evidence_types"])
            or len(set(claim["evidence_types"])) != len(claim["evidence_types"])
        ):
            raise ValueError("bad receipt claim contract")
        for binding in claim.get("evidence_bindings", []):
            if (
                not isinstance(binding, dict)
                or set(binding) != {"type", "ref", "sha256"}
                or not str(binding.get("type", ""))
                or safe_relative_text(str(binding.get("ref", ""))) is None
                or not is_sha256(binding.get("sha256", ""))
            ):
                raise ValueError("bad receipt claim evidence binding")
    artifact = receipt.get("artifact")
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"kind", "path"}
        or artifact.get("kind") not in {"not-emitted", "native-elf"}
        or (artifact.get("path") and safe_relative_text(str(artifact["path"])) is None)
    ):
        raise ValueError("bad receipt artifact")
    if artifact.get("kind") == "native-elf" and not is_sha256(hashes.get("elf_sha256", "")):
        raise ValueError("native ELF receipt lacks an ELF hash")
    limitations = receipt.get("limitations")
    if not isinstance(limitations, list) or not all(isinstance(item, str) and item for item in limitations):
        raise ValueError("bad receipt limitations")


def finalize_command(args: argparse.Namespace) -> int:
    try:
        receipt = json.loads(args.preflight_receipt.read_text(encoding="ascii"))
        verify_preflight_identity(receipt)
        validate_receipt_structure(receipt)
        if receipt.get("mode") == "strict" and receipt.get("verdict") != "OK":
            raise ValueError("strict receipt finalization requires an OK preflight")
        source = args.source.expanduser().resolve()
        policy = discover_policy(source, args.manifest, receipt.get("mode", "advisory"))
        root = (policy.root if policy is not None else source.parent).resolve()
        nodes = receipt.get("graph", {}).get("nodes", [])
        for node in nodes:
            current = (root / node["path"]).resolve()
            if not within_root(current, root) or sha256_file(current) != node["sha256"]:
                raise ValueError(f"source changed after boundary preflight: {node['path']}")
        if source_bundle_hash(nodes) != receipt.get("hashes", {}).get("source_bundle_sha256"):
            raise ValueError("source bundle hash mismatch")
        expected_policy_sha = receipt.get("hashes", {}).get("policy_sha256", "")
        if expected_policy_sha:
            if policy is None or policy.source is None:
                raise ValueError("policy disappeared after boundary preflight")
            if sha256_file(policy.source) != expected_policy_sha:
                raise ValueError("policy changed after boundary preflight")
        if args.claim_contract:
            if sha256_file(Path(args.claim_contract)) != receipt.get("hashes", {}).get("claim_contract_sha256"):
                raise ValueError("claim contract changed after boundary preflight")
        elif receipt.get("hashes", {}).get("claim_contract_sha256"):
            raise ValueError("claim contract is required for receipt finalization")
        if args.compiler:
            if sha256_file(Path(args.compiler)) != receipt.get("hashes", {}).get("compiler_sha256"):
                raise ValueError("compiler changed after boundary preflight")
        verify_claim_evidence_bindings(
            receipt,
            root,
            Path(args.compiler).expanduser().resolve() if args.compiler else None,
        )
        elf = args.elf.expanduser().resolve()
        if not elf.is_file() or elf.stat().st_size == 0:
            raise ValueError("compiler artifact is missing or empty")
        receipt["hashes"]["elf_sha256"] = sha256_file(elf)
        receipt["artifact"] = {
            "kind": "native-elf",
            "path": safe_relative_text(args.artifact_label) or Path(args.artifact_label).name,
        }
        write_json_atomic(args.receipt.expanduser().resolve(), receipt)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error[E-SRB-000]: receipt finalization failed: {error}", file=sys.stderr)
        return 2
    return 0


def verify_command(args: argparse.Namespace) -> int:
    try:
        receipt = json.loads(args.receipt.read_text(encoding="ascii"))
        verify_preflight_identity(receipt)
        validate_receipt_structure(receipt)
        root = args.root.expanduser().resolve()
        nodes = receipt.get("graph", {}).get("nodes", [])
        for node in nodes:
            path = (root / node["path"]).resolve()
            if not within_root(path, root) or sha256_file(path) != node["sha256"]:
                raise ValueError(f"source hash mismatch: {node['path']}")
        if source_bundle_hash(nodes) != receipt.get("hashes", {}).get("source_bundle_sha256"):
            raise ValueError("source bundle hash mismatch")
        policy_source = receipt.get("policy", {}).get("source", "")
        if policy_source:
            policy_path = (root / policy_source).resolve()
            if not within_root(policy_path, root):
                raise ValueError("policy path escapes verification root")
            if sha256_file(policy_path) != receipt.get("hashes", {}).get("policy_sha256"):
                raise ValueError("policy hash mismatch")
        claim_source = (receipt.get("claim_contract") or {}).get("source", "")
        if claim_source:
            claim_path = (root / claim_source).resolve()
            if not within_root(claim_path, root):
                raise ValueError("claim path escapes verification root")
            if sha256_file(claim_path) != receipt.get("hashes", {}).get("claim_contract_sha256"):
                raise ValueError("claim contract hash mismatch")
        if receipt.get("hashes", {}).get("compiler_sha256") and not args.compiler:
            raise ValueError("compiler is required to verify this receipt")
        if args.compiler:
            if sha256_file(args.compiler) != receipt.get("hashes", {}).get("compiler_sha256"):
                raise ValueError("compiler hash mismatch")
        verify_claim_evidence_bindings(receipt, root, args.compiler.expanduser().resolve() if args.compiler else None)
        if receipt.get("hashes", {}).get("elf_sha256") and not args.elf:
            raise ValueError("ELF is required to verify this receipt")
        if args.elf:
            if sha256_file(args.elf) != receipt.get("hashes", {}).get("elf_sha256"):
                raise ValueError("ELF hash mismatch")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"PACKAGE_BOUNDARY_RECEIPT_VERIFY_FAIL reason={error}", file=sys.stderr)
        return 1
    print(
        "PACKAGE_BOUNDARY_RECEIPT_VERIFY_PASS "
        f"verdict={receipt['verdict']} mode={receipt['mode']}"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-science-boundary")
    subparsers = result.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--source", required=True, type=Path)
    evaluate.add_argument("--mode", choices=("auto", "off", "advisory", "strict"), default="auto")
    evaluate.add_argument("--manifest", default="")
    evaluate.add_argument("--claim-contract", default="")
    evaluate.add_argument("--receipt", type=Path)
    evaluate.add_argument("--compiler", default="")
    evaluate.add_argument("--engine-identity", default="")
    evaluate.add_argument("--closure-report", default="")
    evaluate.add_argument("--elf", default="")
    evaluate.add_argument("--artifact-label", default="")
    evaluate.set_defaults(handler=evaluate_command)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--preflight-receipt", required=True, type=Path)
    finalize.add_argument("--source", required=True, type=Path)
    finalize.add_argument("--manifest", default="")
    finalize.add_argument("--claim-contract", default="")
    finalize.add_argument("--compiler", default="")
    finalize.add_argument("--elf", required=True, type=Path)
    finalize.add_argument("--artifact-label", required=True)
    finalize.add_argument("--receipt", required=True, type=Path)
    finalize.set_defaults(handler=finalize_command)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", required=True, type=Path)
    verify.add_argument("--root", type=Path, default=Path.cwd())
    verify.add_argument("--compiler", type=Path)
    verify.add_argument("--elf", type=Path)
    verify.set_defaults(handler=verify_command)
    return result


def main(argv: list[str]) -> int:
    args = parser().parse_args(argv[1:])
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

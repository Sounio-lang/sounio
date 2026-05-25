#!/usr/bin/env python3
"""Compiler-in-the-loop evaluation harness for Sounio code models.

The harness is intentionally dependency-light. It can evaluate generated
completions from a directory, call a user-provided generator command, or smoke
test the judge with repository reference files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = REPO_ROOT / "datasets" / "sounio-ai-eval" / "prompts.v1.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "sounio-ai-eval"
DEFAULT_MODELS = [
    "base=Qwen/Qwen2.5-Coder-1.5B",
    "lora=chiuratto-AIgourakis/sounio-qwen25-coder-1p5b-lora",
]


FOREIGN_SYNTAX_PATTERNS = {
    "rust_mut_ref": re.compile(r"&mut\b"),
    "rust_let_mut": re.compile(r"\blet\s+mut\b"),
    "rust_macro": re.compile(r"\b(?:println|print|assert|vec|format)!\s*\("),
    "rust_vec_type": re.compile(r"\bVec\s*<"),
    "rust_string_type": re.compile(r"\bString\b"),
    "python_def": re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE),
    "python_import": re.compile(r"^\s*(?:from\s+\w+\s+import|import\s+\w+)", re.MULTILINE),
    "c_include": re.compile(r"^\s*#\s*include\b", re.MULTILINE),
    "markdown_fence": re.compile(r"```"),
}

SOUNIO_API_MISFIRES = {
    "read_line": re.compile(r"\bread_line\s*\("),
    "parse_method": re.compile(r"\.parse\s*<|\bparse::<"),
    "len_method": re.compile(r"\.len\s*\("),
    "push_method": re.compile(r"\.push\s*\("),
    "swap_method": re.compile(r"\.swap\s*\("),
    "unwrap_method": re.compile(r"\.unwrap\s*\("),
}


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    category: str
    prompt: str
    source_path: str | None
    expected_stdout: str | None
    run: bool
    difficulty: str | None
    tags: list[str]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="KEY=MODEL_ID",
        help="Model to evaluate. Repeatable. Defaults to Qwen base and Sounio LoRA ids.",
    )
    parser.add_argument(
        "--generator-command",
        help=(
            "Command template that prints one completion to stdout. The prompt is sent "
            "on stdin. Available placeholders: {model_key}, {model_id}, {prompt_id}."
        ),
    )
    parser.add_argument(
        "--completions-dir",
        type=Path,
        help="Read precomputed completions from COMPLETIONS_DIR/<model-key>/<prompt-id>.sio.",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help=(
            "Judge N samples per prompt. With --completions-dir, samples are read from "
            "<prompt-id>.sampleN.sio when N > 1."
        ),
    )
    parser.add_argument(
        "--use-reference-completions",
        action="store_true",
        help="Use each prompt's source_path as the completion. This is only for harness smoke tests.",
    )
    parser.add_argument(
        "--reference-skip-tag",
        default="reference_smoke_skip",
        help=(
            "When --use-reference-completions is set, skip prompts carrying this tag. "
            "This keeps stale repository reference files out of harness smoke tests "
            "without changing model-evaluation prompt coverage."
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run programs after they pass `souc check` when the prompt marks run=true.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N prompts.",
    )
    parser.add_argument(
        "--souc",
        type=Path,
        help="Explicit souc binary or wrapper. Defaults to SOUC_BIN or repo bin/souc.",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def load_prompts(path: Path, limit: int | None) -> list[Prompt]:
    prompts: list[Prompt] = []
    for record in read_jsonl(path):
        prompts.append(
            Prompt(
                prompt_id=str(record["id"]),
                category=str(record.get("category", "uncategorized")),
                prompt=str(record["prompt"]),
                source_path=record.get("source_path"),
                expected_stdout=record.get("expected_stdout"),
                run=bool(record.get("run", False)),
                difficulty=record.get("difficulty"),
                tags=list(record.get("tags", [])),
            )
        )
    if limit is not None:
        return prompts[:limit]
    return prompts


def parse_model_specs(items: list[str]) -> list[ModelSpec]:
    raw = items or DEFAULT_MODELS
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"invalid --model {item!r}; expected KEY=MODEL_ID")
        key, model_id = item.split("=", 1)
        key = key.strip()
        model_id = model_id.strip()
        if not key or not model_id:
            raise SystemExit(f"invalid --model {item!r}; expected KEY=MODEL_ID")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise SystemExit(f"invalid model key {key!r}; use letters, numbers, dot, dash, underscore")
        if key in seen:
            raise SystemExit(f"duplicate model key {key!r}")
        seen.add(key)
        specs.append(ModelSpec(key=key, model_id=model_id))
    return specs


def resolve_souc(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit
    elif os.environ.get("SOUC_BIN"):
        candidate = Path(os.environ["SOUC_BIN"])
    else:
        candidate = REPO_ROOT / "bin" / "souc"
    if not candidate.exists():
        raise SystemExit(f"souc not found at {candidate}; pass --souc or set SOUC_BIN")
    if not os.access(candidate, os.X_OK):
        raise SystemExit(f"souc is not executable at {candidate}")
    return candidate


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    completions_dir = output_dir / "completions"
    raw_completions_dir = output_dir / "raw-completions"
    results_dir = output_dir / "results"
    completions_dir.mkdir(parents=True, exist_ok=True)
    raw_completions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return completions_dir, raw_completions_dir, results_dir


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip() + "\n"


def extract_candidate_source(text: str) -> tuple[str, dict]:
    """Extract the source file candidate that should be judged by `souc`.

    Raw generation is preserved and scored separately so syntax-purity metrics
    still penalize Markdown fences or chat-role transcripts.
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    fenced = re.search(r"```(?:sounio|sio)?[ \t]*\n(?P<body>.*?)\n```", normalized, re.DOTALL | re.IGNORECASE)
    if fenced:
        body = fenced.group("body").strip()
        return body + ("\n" if body else ""), {
            "candidate_extraction": "first_markdown_fence",
            "candidate_extracted_from_fence": True,
        }

    candidate = normalized.strip()
    extraction = "raw_stripped"
    role_markers = [
        "\nassistant\n",
        "\nassistant:",
        "### Response",
        "<|im_start|>assistant",
        "<|assistant|>",
    ]
    marker_positions = [(candidate.rfind(marker), marker) for marker in role_markers]
    marker_positions = [(pos, marker) for pos, marker in marker_positions if pos >= 0]
    if marker_positions:
        pos, marker = max(marker_positions, key=lambda item: item[0])
        candidate = candidate[pos + len(marker) :].strip()
        extraction = "after_assistant_marker"

    lines = candidate.splitlines()
    while lines and lines[0].strip().lower() in {"sounio", "sio"}:
        lines = lines[1:]
        extraction = "dropped_language_label"
    candidate = "\n".join(lines).strip()

    sounio_decl_pattern = (
        r"(?m)^\s*(?:(?:async|kernel|extern)\s+fn|(?:linear\s+)?struct|"
        r"fn|effect|enum|type|import|use|let|var|const|extern)\b"
    )
    first_code_line = re.search(sounio_decl_pattern, candidate)
    prefix = candidate[: first_code_line.start()].strip() if first_code_line else ""
    prefix_lines = [line.strip() for line in prefix.splitlines() if line.strip()]
    prefix_is_commentary = prefix_lines and all(
        line.startswith(("//", "/*", "*")) or line.endswith("*/") for line in prefix_lines
    )
    prefix_has_sounio_declaration = bool(re.search(sounio_decl_pattern, prefix))
    if first_code_line and prefix and not prefix_is_commentary and not prefix_has_sounio_declaration:
        candidate = candidate[first_code_line.start() :].strip()
        extraction = "from_first_sounio_like_line"

    return candidate + ("\n" if candidate else ""), {
        "candidate_extraction": extraction,
        "candidate_extracted_from_fence": False,
    }


def run_subprocess(
    args: list[str],
    *,
    cwd: Path,
    timeout: float,
    stdin: str | None = None,
) -> tuple[int, str, str, float]:
    def normalize_stream(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    start = time.monotonic()
    try:
        proc = subprocess.run(
            args,
            input=stdin,
            text=True,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        return proc.returncode, proc.stdout, proc.stderr, elapsed_ms
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        stdout = normalize_stream(exc.stdout)
        stderr = normalize_stream(exc.stderr) or f"timeout after {timeout}s"
        return 124, stdout, stderr, elapsed_ms


def render_generator_command(template: str, model: ModelSpec, prompt: Prompt) -> list[str]:
    rendered = template.format(
        model_key=model.key,
        model_id=model.model_id,
        prompt_id=prompt.prompt_id,
    )
    return shlex.split(rendered)


def completion_from_reference(prompt: Prompt) -> str:
    if not prompt.source_path:
        raise RuntimeError(f"{prompt.prompt_id}: no source_path for reference completion")
    path = REPO_ROOT / prompt.source_path
    if not path.exists():
        raise RuntimeError(f"{prompt.prompt_id}: reference source missing: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def completion_filename(prompt: Prompt, sample_index: int, samples_per_prompt: int, suffix: str) -> str:
    sample_suffix = f".sample{sample_index}" if samples_per_prompt > 1 else ""
    return f"{prompt.prompt_id}{sample_suffix}{suffix}"


def completion_from_dir(
    root: Path,
    model: ModelSpec,
    prompt: Prompt,
    sample_index: int,
    samples_per_prompt: int,
) -> str:
    path = root / model.key / completion_filename(prompt, sample_index, samples_per_prompt, ".sio")
    if sample_index == 0 and not path.exists():
        legacy_path = root / model.key / f"{prompt.prompt_id}.sio"
        if legacy_path.exists():
            path = legacy_path
    if not path.exists():
        raise RuntimeError(f"missing completion file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def completion_from_command(
    template: str,
    model: ModelSpec,
    prompt: Prompt,
    timeout: float,
) -> tuple[str, dict]:
    args = render_generator_command(template, model, prompt)
    rc, stdout, stderr, elapsed_ms = run_subprocess(
        args,
        cwd=REPO_ROOT,
        timeout=timeout,
        stdin=prompt.prompt,
    )
    info = {
        "generator_command": args,
        "generator_rc": rc,
        "generator_stderr": stderr[-4000:],
        "generator_elapsed_ms": round(elapsed_ms, 3),
    }
    if rc != 0:
        raise RuntimeError(f"generator failed rc={rc}: {stderr[-4000:]}")
    return stdout, info


def get_completion(
    args: argparse.Namespace,
    model: ModelSpec,
    prompt: Prompt,
    sample_index: int,
) -> tuple[str, dict]:
    if args.use_reference_completions:
        return completion_from_reference(prompt), {"completion_source": "reference"}
    if args.completions_dir is not None:
        return completion_from_dir(
            args.completions_dir,
            model,
            prompt,
            sample_index,
            args.samples_per_prompt,
        ), {
            "completion_source": "precomputed",
        }
    if args.generator_command:
        text, info = completion_from_command(args.generator_command, model, prompt, args.timeout)
        info["completion_source"] = "generator_command"
        return text, info
    raise SystemExit(
        "no completion source selected; pass --generator-command, --completions-dir, "
        "or --use-reference-completions for a smoke test"
    )


def syntax_metrics(source: str) -> dict:
    foreign_hits: dict[str, int] = {}
    for name, pattern in FOREIGN_SYNTAX_PATTERNS.items():
        count = len(pattern.findall(source))
        if count:
            foreign_hits[name] = count
    api_hits: dict[str, int] = {}
    for name, pattern in SOUNIO_API_MISFIRES.items():
        count = len(pattern.findall(source))
        if count:
            api_hits[name] = count
    semicolon_count = sum(1 for line in source.splitlines() if line.strip().endswith(";"))
    if semicolon_count:
        foreign_hits["semicolon_line_end"] = semicolon_count
    return {
        "syntax_pure": not foreign_hits,
        "foreign_syntax_hits": foreign_hits,
        "hallucinated_api_hits": api_hits,
        "hallucinated_api_clean": not api_hits,
    }


def write_completion(
    completions_root: Path,
    model: ModelSpec,
    prompt: Prompt,
    source: str,
    sample_index: int,
    samples_per_prompt: int,
) -> Path:
    model_dir = completions_root / model.key
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / completion_filename(prompt, sample_index, samples_per_prompt, ".sio")
    out.write_text(source, encoding="utf-8")
    return out


def write_raw_completion(
    raw_completions_root: Path,
    model: ModelSpec,
    prompt: Prompt,
    source: str,
    sample_index: int,
    samples_per_prompt: int,
) -> Path:
    model_dir = raw_completions_root / model.key
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / completion_filename(prompt, sample_index, samples_per_prompt, ".txt")
    out.write_text(source, encoding="utf-8")
    return out


def judge_completion(
    souc: Path,
    source_path: Path,
    prompt: Prompt,
    timeout: float,
    do_run: bool,
) -> dict:
    check_rc, check_stdout, check_stderr, check_elapsed_ms = run_subprocess(
        [str(souc), "check", str(source_path)],
        cwd=REPO_ROOT,
        timeout=timeout,
    )
    result = {
        "check_rc": check_rc,
        "check_pass": check_rc == 0,
        "check_stdout": check_stdout[-4000:],
        "check_stderr": check_stderr[-4000:],
        "check_elapsed_ms": round(check_elapsed_ms, 3),
        "run_attempted": False,
        "run_rc": None,
        "run_pass": None,
        "run_stdout": "",
        "run_stderr": "",
        "run_elapsed_ms": None,
        "output_match": None,
    }
    if check_rc != 0 or not do_run or not prompt.run:
        return result
    run_rc, run_stdout, run_stderr, run_elapsed_ms = run_subprocess(
        [str(souc), "run", str(source_path)],
        cwd=REPO_ROOT,
        timeout=timeout,
    )
    result.update(
        {
            "run_attempted": True,
            "run_rc": run_rc,
            "run_pass": run_rc == 0,
            "run_stdout": run_stdout[-4000:],
            "run_stderr": run_stderr[-4000:],
            "run_elapsed_ms": round(run_elapsed_ms, 3),
        }
    )
    if prompt.expected_stdout:
        result["output_match"] = prompt.expected_stdout in run_stdout
    return result


def summarize(records: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = {}
    for record in records:
        by_model.setdefault(record["model_key"], []).append(record)
    summary: dict[str, dict] = {}
    for model_key, items in by_model.items():
        total_samples = len(items)
        by_prompt: dict[str, list[dict]] = {}
        for item in items:
            by_prompt.setdefault(str(item.get("prompt_id")), []).append(item)
        prompt_items = []
        for prompt_id, prompt_records in by_prompt.items():
            ordered = sorted(prompt_records, key=lambda item: int(item.get("sample_index", 0)))
            prompt_items.append((prompt_id, ordered))
        total_prompts = len(prompt_items)
        max_samples = max((int(item.get("sample_index", 0)) for item in items), default=0) + 1

        def sample_at(ordered: list[dict], index: int) -> dict | None:
            for item in ordered:
                if int(item.get("sample_index", 0)) == index:
                    return item
            return None

        first_samples = [sample_at(ordered, 0) for _prompt_id, ordered in prompt_items]
        first_samples = [item for item in first_samples if item is not None]

        def compile_at(k: int) -> float:
            if total_prompts == 0:
                return 0.0
            passed = 0
            for _prompt_id, ordered in prompt_items:
                if any(int(item.get("sample_index", 0)) < k and item.get("check_pass") for item in ordered):
                    passed += 1
            return passed / total_prompts

        selector_records: list[dict] = []
        first_passing_sample_counts: dict[str, int] = {}
        for _prompt_id, ordered in prompt_items:
            passing = [item for item in ordered if item.get("check_pass")]
            if not passing:
                continue
            selected = min(passing, key=lambda item: int(item.get("sample_index", 0)))
            selector_records.append(selected)
            sample_key = str(int(selected.get("sample_index", 0)))
            first_passing_sample_counts[sample_key] = first_passing_sample_counts.get(sample_key, 0) + 1

        sample_check_pass = sum(1 for item in items if item.get("check_pass"))
        check_pass_at_1 = sum(1 for item in first_samples if item.get("check_pass"))
        run_attempted = sum(1 for item in selector_records if item.get("run_attempted"))
        run_pass = sum(1 for item in selector_records if item.get("run_pass"))
        output_checks = [item for item in selector_records if item.get("output_match") is not None]
        output_match = sum(1 for item in output_checks if item.get("output_match"))
        syntax_pure_at_1 = sum(1 for item in first_samples if item.get("syntax_pure"))
        candidate_syntax_pure_at_1 = sum(
            1 for item in first_samples if item.get("candidate_syntax_pure")
        )
        hallucinated_api_clean_at_1 = sum(
            1 for item in first_samples if item.get("hallucinated_api_clean")
        )
        candidate_hallucinated_api_clean_at_1 = sum(
            1 for item in first_samples if item.get("candidate_hallucinated_api_clean")
        )
        categories: dict[str, dict[str, int]] = {}
        for _prompt_id, ordered in prompt_items:
            first = sample_at(ordered, 0) or ordered[0]
            cat = first.get("category", "uncategorized")
            bucket = categories.setdefault(
                cat,
                {
                    "total": 0,
                    "check_pass": 0,
                    "check_pass_at_3": 0,
                    "check_pass_at_5": 0,
                    "run_pass": 0,
                },
            )
            bucket["total"] += 1
            bucket["check_pass"] += int(bool(first.get("check_pass")))
            bucket["check_pass_at_3"] += int(
                any(int(item.get("sample_index", 0)) < 3 and item.get("check_pass") for item in ordered)
            )
            bucket["check_pass_at_5"] += int(
                any(int(item.get("sample_index", 0)) < 5 and item.get("check_pass") for item in ordered)
            )
            selected = next((item for item in selector_records if item.get("prompt_id") == first.get("prompt_id")), None)
            bucket["run_pass"] += int(bool(selected and selected.get("run_pass")))
        summary[model_key] = {
            "total": total_prompts,
            "sample_total": total_samples,
            "samples_per_prompt": max_samples,
            "sample_compile_rate": sample_check_pass / total_samples if total_samples else 0.0,
            "compile_rate": check_pass_at_1 / total_prompts if total_prompts else 0.0,
            "compile_at_1": check_pass_at_1 / total_prompts if total_prompts else 0.0,
            "compile_at_3": compile_at(3),
            "compile_at_5": compile_at(5),
            "runtime_pass_rate": run_pass / run_attempted if run_attempted else None,
            "runtime_attempted": run_attempted,
            "selector_compile_count": len(selector_records),
            "first_passing_sample_counts": first_passing_sample_counts,
            "output_match_rate": output_match / len(output_checks) if output_checks else None,
            "output_checks": len(output_checks),
            "syntax_purity_rate": syntax_pure_at_1 / total_prompts if total_prompts else 0.0,
            "candidate_syntax_purity_rate": (
                candidate_syntax_pure_at_1 / total_prompts if total_prompts else 0.0
            ),
            "hallucinated_api_clean_rate": (
                hallucinated_api_clean_at_1 / total_prompts if total_prompts else 0.0
            ),
            "candidate_hallucinated_api_clean_rate": (
                candidate_hallucinated_api_clean_at_1 / total_prompts if total_prompts else 0.0
            ),
            "categories": categories,
        }
    if len(summary) >= 2 and "base" in summary and "lora" in summary:
        summary["base_vs_lora_delta"] = {
            "compile_rate": summary["lora"]["compile_rate"] - summary["base"]["compile_rate"],
            "syntax_purity_rate": summary["lora"]["syntax_purity_rate"]
            - summary["base"]["syntax_purity_rate"],
            "candidate_syntax_purity_rate": summary["lora"]["candidate_syntax_purity_rate"]
            - summary["base"]["candidate_syntax_purity_rate"],
            "candidate_hallucinated_api_clean_rate": summary["lora"][
                "candidate_hallucinated_api_clean_rate"
            ]
            - summary["base"]["candidate_hallucinated_api_clean_rate"],
        }
        if (
            summary["base"]["runtime_pass_rate"] is not None
            and summary["lora"]["runtime_pass_rate"] is not None
        ):
            summary["base_vs_lora_delta"]["runtime_pass_rate"] = (
                summary["lora"]["runtime_pass_rate"] - summary["base"]["runtime_pass_rate"]
            )
    return summary


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def copy_prompt_manifest(args: argparse.Namespace, output_dir: Path) -> None:
    if args.prompts.exists():
        shutil.copyfile(args.prompts, output_dir / "prompts.jsonl")


def main() -> None:
    args = parse_args()
    if args.samples_per_prompt < 1:
        raise SystemExit("--samples-per-prompt must be >= 1")
    prompts = load_prompts(args.prompts, args.limit)
    if args.use_reference_completions and args.reference_skip_tag:
        before = len(prompts)
        prompts = [prompt for prompt in prompts if args.reference_skip_tag not in prompt.tags]
        skipped = before - len(prompts)
        if skipped:
            print(f"skipped {skipped} reference prompts tagged {args.reference_skip_tag!r}")
    models = parse_model_specs(args.model)
    souc = resolve_souc(args.souc)
    completions_root, raw_completions_root, results_dir = ensure_output_dirs(args.output_dir)
    copy_prompt_manifest(args, results_dir)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    records: list[dict] = []
    for model in models:
        for prompt in prompts:
            for sample_index in range(args.samples_per_prompt):
                record = {
                    "run_id": run_id,
                    "model_key": model.key,
                    "model_id": model.model_id,
                    "prompt_id": prompt.prompt_id,
                    "sample_index": sample_index,
                    "samples_per_prompt": args.samples_per_prompt,
                    "category": prompt.category,
                    "difficulty": prompt.difficulty,
                    "tags": prompt.tags,
                    "source_path": prompt.source_path,
                    "expected_stdout": prompt.expected_stdout,
                }
                try:
                    raw_source, completion_info = get_completion(args, model, prompt, sample_index)
                    source, extraction_info = extract_candidate_source(raw_source)
                    raw_completion_path = write_raw_completion(
                        raw_completions_root,
                        model,
                        prompt,
                        raw_source,
                        sample_index,
                        args.samples_per_prompt,
                    )
                    completion_path = write_completion(
                        completions_root,
                        model,
                        prompt,
                        source,
                        sample_index,
                        args.samples_per_prompt,
                    )
                    raw_metrics = syntax_metrics(raw_source)
                    candidate_metrics = syntax_metrics(source)
                    record.update(completion_info)
                    record.update(extraction_info)
                    record["raw_completion_path"] = display_path(raw_completion_path)
                    record["completion_path"] = display_path(completion_path)
                    record.update(raw_metrics)
                    record.update(
                        {
                            f"candidate_{key}": value
                            for key, value in candidate_metrics.items()
                        }
                    )
                    record.update(
                        judge_completion(
                            souc=souc,
                            source_path=completion_path,
                            prompt=prompt,
                            timeout=args.timeout,
                            do_run=args.run,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - result rows should capture failures.
                    record.update(
                        {
                            "completion_source": "error",
                            "candidate_extraction": "error",
                            "candidate_extracted_from_fence": False,
                            "raw_completion_path": None,
                            "completion_path": None,
                            "syntax_pure": False,
                            "foreign_syntax_hits": {},
                            "hallucinated_api_hits": {},
                            "hallucinated_api_clean": False,
                            "candidate_syntax_pure": False,
                            "candidate_foreign_syntax_hits": {},
                            "candidate_hallucinated_api_hits": {},
                            "candidate_hallucinated_api_clean": False,
                            "check_rc": None,
                            "check_pass": False,
                            "check_stdout": "",
                            "check_stderr": str(exc)[-4000:],
                            "run_attempted": False,
                            "run_rc": None,
                            "run_pass": None,
                            "run_stdout": "",
                            "run_stderr": "",
                            "output_match": None,
                        }
                    )
                records.append(record)
                print(
                    f"{model.key:>8} {prompt.prompt_id:<40} sample={sample_index:<2} "
                    f"check={record.get('check_pass')} run={record.get('run_pass')} "
                    f"syntax={record.get('syntax_pure')}",
                    flush=True,
                )

    results_path = results_dir / f"{run_id}.jsonl"
    summary_path = results_dir / f"{run_id}.summary.json"
    write_jsonl(results_path, records)
    summary = summarize(records)
    summary.update(
        {
            "run_id": run_id,
            "prompt_count": len(prompts),
            "model_count": len(models),
            "samples_per_prompt": args.samples_per_prompt,
            "souc": display_path(souc),
            "results_path": display_path(results_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {results_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

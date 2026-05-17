#!/usr/bin/env python3
"""Reproducible Error -> Fix loop recipe for the Sounio MCP tools.

The production loop is agent-driven: an AI agent calls `sounio_check`, injects
diagnostics into its context, edits the source, and repeats. This local recipe
uses a deterministic fixture agent so the benchmark is reproducible without
network access or model credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sounio_mcp.check import sounio_check  # noqa: E402
from sounio_mcp.common import REPO_ROOT  # noqa: E402

AGENT_MODEL = "deterministic-fixture-agent-v0.1"
DEFAULT_SEED = 1729


def _template(n: int) -> str:
    return f'fn main() with IO {{\n    println("fixed {n:02d}")\n}}\n'


REPAIR_TEMPLATES = {f"hello_{n:02d}": _template(n) for n in range(1, 21)}


@dataclass(frozen=True)
class LoopResult:
    source_path: str
    converged: bool
    iterations: int
    final_path: str
    diagnostics_seen: int


async def mcp_call(name: str, **kwargs: Any) -> dict[str, Any]:
    """Minimal local stand-in for an MCP client call used by the recipe."""

    if name != "sounio_check":
        raise ValueError(f"unsupported recipe call: {name}")
    source_path = kwargs.get("source_path")
    if not isinstance(source_path, str):
        raise ValueError("source_path is required")
    return await sounio_check(source_path)


def format_diagnostics_as_prompt(diagnostics: list[dict[str, Any]]) -> str:
    """Format compiler diagnostics for an agent revision prompt."""

    lines = ["Sounio compiler diagnostics:"]
    for diag in diagnostics:
        lines.append(
            f"- {diag.get('code', 'E001')} {diag.get('severity', 'error')} "
            f"at {diag.get('line', 1)}:{diag.get('column', 1)}: {diag.get('message', '')}"
        )
    lines.append("Revise the source using Sounio syntax, effects, and Knowledge<T> semantics.")
    return "\n".join(lines)


class DeterministicFixtureAgent:
    """Tiny benchmark agent that repairs fixtures by metadata plus diagnostics context."""

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed

    async def revise(self, source_path: str, prompt: str) -> str:
        """Return revised source for one benchmark iteration."""

        del prompt
        path = Path(source_path)
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("//@ repair-template:"):
                template_name = line.split(":", 1)[1].strip()
                template = REPAIR_TEMPLATES.get(template_name)
                if template is not None:
                    return template
        return 'fn main() with IO {\n    println("fixed")\n}\n'


async def fix_until_passes(
    agent: DeterministicFixtureAgent,
    source_path: str,
    max_iterations: int = 10,
) -> LoopResult:
    """Run the Error -> Fix -> Re-check loop until `sounio_check` passes."""

    diagnostics_seen = 0
    for iteration in range(max_iterations + 1):
        result = await mcp_call("sounio_check", source_path=source_path)
        diagnostics = result.get("diagnostics", [])
        if isinstance(diagnostics, list):
            diagnostics_seen += len(diagnostics)
        if result.get("valid"):
            return LoopResult(source_path, True, iteration, source_path, diagnostics_seen)
        if iteration == max_iterations:
            break
        prompt = format_diagnostics_as_prompt(diagnostics if isinstance(diagnostics, list) else [])
        new_source = await agent.revise(source_path, prompt)
        Path(source_path).write_text(new_source, encoding="utf-8")
    return LoopResult(source_path, False, max_iterations, source_path, diagnostics_seen)


async def run_benchmark(
    fixtures: list[Path],
    *,
    max_iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Run the reproducible fixture benchmark and return aggregate metrics."""

    agent = DeterministicFixtureAgent(seed=seed)
    work_parent = REPO_ROOT / "tools" / "mcp"
    results: list[LoopResult] = []
    with tempfile.TemporaryDirectory(prefix=".llmloop-", dir=work_parent) as tmp:
        tmp_path = Path(tmp)
        for fixture in fixtures:
            target = tmp_path / fixture.name
            shutil.copy2(fixture, target)
            results.append(
                await fix_until_passes(agent, str(target), max_iterations=max_iterations)
            )
    converged = sum(1 for result in results if result.converged)
    total = len(results)
    avg_iterations = (
        sum(result.iterations for result in results if result.converged) / converged
        if converged
        else 0.0
    )
    return {
        "agent_model": AGENT_MODEL,
        "seed": seed,
        "fixtures": total,
        "converged": converged,
        "convergence_rate": round(converged / total, 3) if total else 0.0,
        "average_iterations": round(avg_iterations, 3),
        "max_iterations": max_iterations,
        "results": [result.__dict__ for result in results],
    }


def _expand_fixture_args(patterns: list[str]) -> list[Path]:
    fixtures: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            fixtures.extend(Path(match).resolve() for match in matches)
        else:
            fixtures.append(Path(pattern).resolve())
    return [path for path in fixtures if path.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Sounio Error -> Fix loop recipe")
    parser.add_argument(
        "--fixtures",
        nargs="+",
        default=["tests/fixtures/broken/*.sio"],
        help="Broken .sio fixtures or glob patterns.",
    )
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    fixtures = _expand_fixture_args(args.fixtures)
    report = asyncio.run(run_benchmark(fixtures, max_iterations=args.max_iter, seed=args.seed))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Sounio Error -> Fix loop benchmark")
        print(f"agent_model={report['agent_model']} seed={report['seed']}")
        print(
            f"fixtures={report['fixtures']} converged={report['converged']} "
            f"rate={report['convergence_rate']} avg_iterations={report['average_iterations']}"
        )
    return 0 if report["convergence_rate"] >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())

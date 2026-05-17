from __future__ import annotations

from pathlib import Path

import pytest

from examples.llmloop_recipe import run_benchmark


@pytest.mark.asyncio
async def test_error_fix_loop_converges_on_fixture_set() -> None:
    fixtures = sorted(Path("tests/fixtures/broken").glob("*.sio"))
    report = await run_benchmark(fixtures, max_iterations=10, seed=1729)
    assert report["fixtures"] == 20
    assert report["convergence_rate"] >= 0.8

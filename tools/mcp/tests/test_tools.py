from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from sounio_mcp.check import sounio_check
from sounio_mcp.compile import sounio_compile
from sounio_mcp.errors_catalog import ERROR_CATALOG, get_error_doc
from sounio_mcp.run import sounio_run
from sounio_mcp.stdlib_docs import get_stdlib_doc
from sounio_mcp.test import sounio_test


@pytest.mark.asyncio
async def test_sounio_check_reports_broken_fixture() -> None:
    result = await sounio_check("tests/fixtures/broken_dose.sio")
    assert result["valid"] is False
    assert result["diagnostics"]
    assert result["diagnostic_schema"] == "sounio.diagnostic.v1"


@pytest.mark.asyncio
async def test_sounio_check_shared_schema_envelope_validates() -> None:
    result = await sounio_check("tests/fixtures/broken_dose.sio")
    with open("tools/shared/diagnostic_schema.json", encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator(schema).validate(result["diagnostic_envelope"])


@pytest.mark.asyncio
async def test_sounio_run_hello() -> None:
    result = await sounio_run("examples/hello.sio")
    assert result["status"] == "ok"
    assert result["exit_code"] == 0
    assert "Compilation successful!" in result["stdout"]


@pytest.mark.asyncio
async def test_sounio_run_rejects_secret_env() -> None:
    result = await sounio_run("examples/hello.sio", env={"API_TOKEN": "nope"})
    assert result["status"] == "runtime_error"
    assert "sensitive environment key" in result["stderr"]


@pytest.mark.asyncio
async def test_sounio_compile_rejects_unknown_target() -> None:
    result = await sounio_compile("examples/hello.sio", target="amiga")
    assert result["status"] == "error"
    assert "unsupported target" in result["stderr"]


@pytest.mark.asyncio
async def test_sounio_test_can_run_filtered_file() -> None:
    result = await sounio_test("examples/hello.sio")
    assert result["total"] == 1
    assert result["passed"] == 1
    assert result["failed"] == 0


def test_stdlib_resource_returns_markdown() -> None:
    doc = get_stdlib_doc("stats")
    assert "stats" in doc.lower()


def test_error_catalog_has_promoted_traceable_entries() -> None:
    assert len(ERROR_CATALOG) >= 20
    doc = get_error_doc("E070")
    assert "E070" in doc
    assert "Provenance" in doc

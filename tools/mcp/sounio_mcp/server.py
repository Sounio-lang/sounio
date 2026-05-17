"""FastMCP server exposing the checked Sounio compiler over local stdio."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import __version__
from .check import sounio_check as check_tool
from .compile import sounio_compile as compile_tool
from .errors_catalog import get_error_doc as error_resource
from .run import sounio_run as run_tool
from .stdlib_docs import get_stdlib_doc as stdlib_resource
from .test import sounio_test as test_tool


def _make_server() -> FastMCP:
    try:
        return FastMCP(
            name="sounio-compiler",
            instructions=(
                "Local-only Sounio compiler server. Use tools to check, compile, run, "
                "and test .sio files inside the workspace; use resources for stdlib "
                "and compiler error context."
            ),
            json_response=True,
        )
    except TypeError:
        return FastMCP("sounio-compiler")


mcp = _make_server()


@mcp.tool()
async def sounio_compile(
    source_path: str,
    target: str = "elf",
    optimization: str = "default",
) -> dict[str, Any]:
    """Compile a `.sio` source file.

    Example:
        `sounio_compile(source_path="examples/hello.sio", target="elf")`
    """

    return await compile_tool(source_path, target, optimization)


@mcp.tool()
async def sounio_check(source_path: str) -> dict[str, Any]:
    """Type-check a `.sio` source file and return structured diagnostics.

    Example:
        `sounio_check(source_path="tests/fixtures/broken_dose.sio")`
    """

    return await check_tool(source_path)


@mcp.tool()
async def sounio_run(
    source_path: str,
    args: list[str] | None = None,
    timeout_sec: int = 30,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compile and run a `.sio` source file, capturing stdout/stderr/exit code.

    Example:
        `sounio_run(source_path="examples/hello.sio", args=[])`
    """

    return await run_tool(source_path, args, timeout_sec, env)


@mcp.tool()
async def sounio_test(
    test_pattern: str = "tests/run-pass",
    test_filter: str | None = None,
) -> dict[str, Any]:
    """Run Sounio tests and return per-test failures.

    Example:
        `sounio_test(test_pattern="tests/fixtures/broken", test_filter="*.sio")`
    """

    return await test_tool(test_pattern, test_filter)


@mcp.resource("sounio://stdlib/{module}")
async def get_stdlib_doc(module: str) -> str:
    """Return Markdown documentation for a Sounio standard-library module."""

    return stdlib_resource(module)


@mcp.resource("sounio://errors/{error_code}")
async def get_error_doc(error_code: str) -> str:
    """Return an explanation, examples, and fix suggestions for a compiler error code."""

    return error_resource(error_code)


def main() -> None:
    """Run the Sounio MCP server."""

    parser = argparse.ArgumentParser(description="Sounio compiler MCP server")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio"],
        help="MCP transport. v1 is local-only and supports stdio only.",
    )
    parser.add_argument("--version", action="version", version=f"sounio-mcp-server {__version__}")
    args = parser.parse_args()
    if args.transport == "stdio":
        _run_jsonrpc_stdio()
    else:  # pragma: no cover - argparse prevents this in v1.
        mcp.run(transport=args.transport)


def _run_jsonrpc_stdio() -> None:
    """Run a line-delimited MCP-compatible JSON-RPC stdio loop."""

    for line in sys.stdin:
        if not line.strip():
            continue
        response = _dispatch_jsonrpc_line(line)
        if response is not None:
            print(json.dumps(response, separators=(",", ":")), flush=True)


def _dispatch_jsonrpc_line(line: str) -> dict[str, Any] | None:
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        return _error_response(None, -32700, f"parse error: {exc.msg}")
    if not isinstance(request, dict):
        return _error_response(None, -32600, "request must be a JSON object")
    legacy_one_shot = "jsonrpc" not in request
    request_id = request.get("id", 1 if legacy_one_shot else None)
    method = request.get("method")
    if not isinstance(method, str):
        return _error_response(request_id, -32600, "missing JSON-RPC method")
    if request_id is None:
        return None
    try:
        result = _handle_method(method, request.get("params", {}))
    except Exception as exc:  # pragma: no cover - adversarial path.
        return _error_response(request_id, -32603, str(exc))
    if request_id is None:
        return None
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _handle_method(method: str, params: object) -> dict[str, Any]:
    if method == "initialize":
        return {
            "protocolVersion": "2025-11-25",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
            "serverInfo": {"name": "sounio-compiler", "version": __version__},
            "instructions": (
                "Use sounio_check, sounio_compile, sounio_run, and sounio_test "
                "for local Sounio compiler loops."
            ),
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": _tool_descriptors()}
    if method == "tools/call":
        return _call_tool(params)
    if method == "resources/list":
        return {"resources": []}
    if method in {"resources/templates/list", "resources/list_templates"}:
        return {
            "resourceTemplates": [
                {
                    "uriTemplate": "sounio://stdlib/{module}",
                    "name": "sounio_stdlib",
                    "description": "Sounio standard-library documentation",
                    "mimeType": "text/markdown",
                },
                {
                    "uriTemplate": "sounio://errors/{error_code}",
                    "name": "sounio_errors",
                    "description": "Sounio compiler error catalogue",
                    "mimeType": "text/markdown",
                },
            ]
        }
    if method == "resources/read":
        return _read_resource(params)
    raise ValueError(f"unknown method: {method}")


def _error_response(request_id: object, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_descriptors() -> list[dict[str, Any]]:
    return [
        {
            "name": "sounio_compile",
            "description": "Compile a Sounio source file (.sio) to a local target artefact.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_path": {"type": "string"},
                    "target": {"type": "string", "default": "elf"},
                    "optimization": {"type": "string", "default": "default"},
                },
                "required": ["source_path"],
            },
        },
        {
            "name": "sounio_check",
            "description": "Type-check a Sounio source file and return structured diagnostics.",
            "inputSchema": {
                "type": "object",
                "properties": {"source_path": {"type": "string"}},
                "required": ["source_path"],
            },
        },
        {
            "name": "sounio_run",
            "description": "Compile and run a Sounio source file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_path": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}},
                    "timeout_sec": {"type": "integer", "default": 30},
                    "env": {"type": "object", "additionalProperties": {"type": "string"}},
                },
                "required": ["source_path"],
            },
        },
        {
            "name": "sounio_test",
            "description": "Run Sounio tests and return aggregate/per-test status.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "test_pattern": {"type": "string", "default": "tests/run-pass"},
                    "test_filter": {"type": "string"},
                },
            },
        },
    ]


def _call_tool(params: object) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("tools/call params must be an object")
    name = params.get("name")
    arguments = params.get("arguments", {})
    if not isinstance(name, str):
        raise ValueError("tools/call params.name must be a string")
    if not isinstance(arguments, dict):
        raise ValueError("tools/call params.arguments must be an object")

    if name == "sounio_compile":
        result = asyncio.run(
            compile_tool(
                str(arguments.get("source_path", "")),
                str(arguments.get("target", "elf")),
                str(arguments.get("optimization", "default")),
            )
        )
    elif name == "sounio_check":
        result = asyncio.run(check_tool(str(arguments.get("source_path", ""))))
    elif name == "sounio_run":
        raw_args = arguments.get("args")
        run_args = [str(item) for item in raw_args] if isinstance(raw_args, list) else None
        raw_env = arguments.get("env")
        run_env = (
            {str(key): str(value) for key, value in raw_env.items()}
            if isinstance(raw_env, dict)
            else None
        )
        timeout = arguments.get("timeout_sec", 30)
        result = asyncio.run(
            run_tool(
                str(arguments.get("source_path", "")),
                run_args,
                int(timeout) if isinstance(timeout, int) else 30,
                run_env,
            )
        )
    elif name == "sounio_test":
        raw_filter = arguments.get("test_filter")
        result = asyncio.run(
            test_tool(
                str(arguments.get("test_pattern", "tests/run-pass")),
                str(raw_filter) if raw_filter is not None else None,
            )
        )
    else:
        raise ValueError(f"unknown tool: {name}")
    return {
        "content": [{"type": "text", "text": json.dumps(result)}],
        "structuredContent": result,
        "isError": False,
    }


def _read_resource(params: object) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("resources/read params must be an object")
    uri = params.get("uri")
    if not isinstance(uri, str):
        raise ValueError("resources/read params.uri must be a string")
    if uri.startswith("sounio://stdlib/"):
        text = stdlib_resource(uri.removeprefix("sounio://stdlib/"))
    elif uri.startswith("sounio://errors/"):
        text = error_resource(uri.removeprefix("sounio://errors/"))
    else:
        raise ValueError(f"unknown resource URI: {uri}")
    return {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": text}]}


if __name__ == "__main__":
    main()

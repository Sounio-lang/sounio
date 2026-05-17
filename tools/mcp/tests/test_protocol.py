from __future__ import annotations

import json
import os
import sys
from datetime import timedelta

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from sounio_mcp.server import _dispatch_jsonrpc_line


def test_one_shot_jsonrpc_returns_structured_content() -> None:
    response = _dispatch_jsonrpc_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "sounio_check",
                    "arguments": {"source_path": "tests/fixtures/broken_dose.sio"},
                },
            }
        )
    )
    assert response is not None
    result = response["result"]
    assert result["structuredContent"]["valid"] is False
    text_payload = json.loads(result["content"][0]["text"])
    assert text_payload["diagnostic_schema"] == "sounio.diagnostic.v1"


def test_jsonrpc_initialized_notification_is_silent() -> None:
    response = _dispatch_jsonrpc_line(
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
    )
    assert response is None


@pytest.mark.asyncio
async def test_mcp_python_client_can_call_tools_and_resources() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in [os.getcwd() + "/tools/mcp", *sys.path] if item
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "sounio_mcp.server", "--transport", "stdio"],
        env=env,
        cwd=os.getcwd(),
    )

    async with stdio_client(params) as streams, ClientSession(*streams) as session:
            init = await session.initialize()
            assert init.serverInfo.name == "sounio-compiler"

            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "sounio_compile",
                "sounio_check",
                "sounio_run",
                "sounio_test",
            }

            check = await session.call_tool(
                "sounio_check",
                {"source_path": "tests/fixtures/broken_dose.sio"},
                read_timeout_seconds=timedelta(seconds=15),
            )
            assert check.structuredContent is not None
            assert check.structuredContent["valid"] is False

            templates = await session.list_resource_templates()
            assert {item.uriTemplate for item in templates.resourceTemplates} == {
                "sounio://stdlib/{module}",
                "sounio://errors/{error_code}",
            }

            resource = await session.read_resource("sounio://errors/E070")
            assert resource.contents
            assert "E070" in resource.contents[0].text

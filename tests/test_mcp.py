"""DeepDive MCP server tests — registry shape + (optional) live MCP roundtrip."""

from __future__ import annotations

import pytest

from deepdive.mcp import build_research_registry


def test_registry_exposes_research_tool():
    reg = build_research_registry()
    names = [t.name for t in reg.list()]
    assert "research" in names


def test_research_tool_has_correct_schema():
    reg = build_research_registry()
    research = next(t for t in reg.list() if t.name == "research")
    schema = research.input_schema
    assert schema["type"] == "object"
    assert "question" in schema["properties"]
    assert "question" in schema["required"]
    # Optional fields exposed
    assert "allow_domains" in schema["properties"]
    assert "block_domains" in schema["properties"]
    assert "max_pages" in schema["properties"]
    assert "export_format" in schema["properties"]


def test_research_tool_description_mentions_cited_report():
    reg = build_research_registry()
    research = next(t for t in reg.list() if t.name == "research")
    desc = research.description.lower()
    assert "research" in desc
    assert "cited" in desc or "report" in desc


# In-process MCP roundtrip — only when the SDK is installed.
pytest.importorskip("mcp")


@pytest.mark.asyncio
async def test_research_tool_callable_via_in_memory_mcp():
    """Wrap the registry in an MCP server, then call from an MCP client.

    We don't actually run the research pipeline (it would hit the network);
    we just verify the tool surface is correctly mounted and reachable.
    """
    from actants.mcp import build_server
    from mcp.shared.memory import create_connected_server_and_client_session

    server = build_server(build_research_registry(), name="deepdive-test")
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        listing = await session.list_tools()
        tools = {t.name for t in listing.tools}
        assert "research" in tools
        # The tool's input schema must round-trip cleanly
        research = next(t for t in listing.tools if t.name == "research")
        assert research.inputSchema is not None
        assert "question" in research.inputSchema.get("properties", {})

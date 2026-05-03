"""DeepDive as an MCP server.

Exposes one tool — ``research(question, allow_domains, block_domains, max_pages)``
— over the Model Context Protocol so Claude Desktop, Cursor, and any MCP-aware
client can invoke DeepDive as a remote research tool. Built on
``actants.mcp.serve``.
"""

from __future__ import annotations

from deepdive.mcp.server import build_research_registry, serve_mcp

__all__ = ["build_research_registry", "serve_mcp"]

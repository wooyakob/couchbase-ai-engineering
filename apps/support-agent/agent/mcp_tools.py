"""Optional: attach Couchbase MCP server tools to the agent (Chapter 12 §12.5).

The MCP server gives the agent the *long tail* of data access — schema discovery and
ad-hoc SQL++ — through a least-privilege database user. Core actions stay as the
hand-written, narrowly-scoped catalog tools in tools/support_tools.py.

Usage:
    tools = await load_mcp_tools()
    agent = create_react_agent(model, tools=[*core_tools, *tools])
"""

import os

from langchain_mcp_adapters.client import MultiServerMCPClient


async def load_mcp_tools() -> list:
    client = MultiServerMCPClient({
        "couchbase": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["couchbase-mcp-server"],
            "env": {
                "CB_CONNECTION_STRING": os.getenv("CB_CONN_STRING", "couchbase://localhost"),
                # a dedicated READ-ONLY user — never the app credentials (Ch. 12 §12.6)
                "CB_USERNAME": os.getenv("MCP_CB_USERNAME", "mcp_readonly"),
                "CB_PASSWORD": os.getenv("MCP_CB_PASSWORD", ""),
                "CB_BUCKET_NAME": os.getenv("CB_BUCKET", "ai"),
            },
        }
    })
    return await client.get_tools()

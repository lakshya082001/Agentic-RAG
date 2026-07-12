import json
import os
import re

from langchain_mcp_adapters.client import MultiServerMCPClient

URL_RE = re.compile(r"https?://\S+")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "mcp_config.json")

_client = None
_tools = None


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        raw = f.read()
    raw = raw.replace("${TAVILY_API_KEY}", os.environ.get("TAVILY_API_KEY", ""))
    return json.loads(raw)


async def get_mcp_tools() -> list:
    """Lazily start the MCP servers and cache their tools."""
    global _client, _tools
    if _tools is None:
        _client = MultiServerMCPClient(_load_config())
        _tools = await _client.get_tools()
    return _tools


def _find_tool(tools: list, *name_parts: str):
    for t in tools:
        if all(p in t.name.lower() for p in name_parts):
            return t
    return None


async def search_and_fetch(query: str, max_urls: int = 4) -> list[str]:
    """Tavily search for `query`, then fetch page content for the top URLs.

    Returns a list of raw text blobs (fed to rag_pipeline as raw_texts).
    """
    tools = await get_mcp_tools()

    search_tool = _find_tool(tools, "search")
    fetch_tool = _find_tool(tools, "fetch")
    if search_tool is None or fetch_tool is None:
        print("[MCP] search/fetch tool not found among:", [t.name for t in tools])
        return []

    print(f"[MCP] Searching: '{query}'")
    search_result = await search_tool.ainvoke({"query": query})
    urls = _extract_urls(search_result)[:max_urls]
    print(f"[MCP] Found {len(urls)} URLs")

    contents = []
    for url in urls:
        try:
            print(f"[MCP] Fetching: {url}")
            page = await fetch_tool.ainvoke({"url": url})
            contents.append(_as_text(page))
        except Exception as e:
            print(f"[MCP] Fetch failed for {url}: {e}")
    return contents


def _as_text(mcp_result) -> str:
    """Flatten an MCP tool result (str, or list of content blocks) into plain text."""
    if isinstance(mcp_result, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in mcp_result
        )
    return str(mcp_result)


def _extract_urls(search_result) -> list[str]:
    """Best-effort URL extraction from a Tavily MCP search response.

    The MCP result can be: a JSON dict with a "results" list, a plain string, or
    (as tavily-mcp actually returns) a list of content blocks like
    [{"type": "text", "text": "Title: ...\\nURL: ...\\nContent: ..."}]. Handle all three.
    """
    data = search_result

    # list of MCP content blocks -> concatenate their text
    if isinstance(data, list):
        return _dedupe(URL_RE.findall(_as_text(data)))

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return _dedupe(URL_RE.findall(data))

    if isinstance(data, dict):
        results = data.get("results", [])
        urls = [r.get("url") for r in results if isinstance(r, dict) and r.get("url")]
        if urls:
            return _dedupe(urls)

    return []


def _dedupe(urls: list[str]) -> list[str]:
    seen, out = set(), []
    for u in urls:
        u = u.rstrip(".,)")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

import asyncio
import pprint

from fastmcp import Client


def pretty_print(tool_name: str, result):
    """
    Pretty-print the result returned by an MCP tool.

    Args:
      tool_name: Name of the tool you called (e.g., "definition_short").
      result: The raw result dict/list returned by the tool.
    """
    print(f"\n=== {tool_name} ===")
    pp = pprint.PrettyPrinter(indent=2, width=100, compact=False)
    pp.pprint(result)
    print("=" * (6 + len(tool_name)))


async def test_tools():
    async with Client("http://127.0.0.1:8000/mcp") as client:
        resp1 = await client.call_tool(
            "definition_short", {"symbol": "MedievalLMTokenizer", "max_chars": 500}
        )
        pretty_print("definition_short", resp1)

        # resp2 = await client.call_tool(
        #     "definition_full", {"symbol": "MyClass", "max_chars": 2000}
        # )
        # pretty_print("definition_full", resp2)

        # resp3 = await client.call_tool(
        #     "references", {"symbol": "MyClass", "max_results": 50}
        # )
        # pretty_print("references", resp3)
        # # repeat for other symbols
        # resp = await client.call_tool(
        #     "outline", {"file_path": "tests/demo.py", "symbol": None, "max_chars": 300}
        # )
        # pretty_print("outline (module)", resp)

        # resp2 = await client.call_tool(
        #     "outline",
        #     {"file_path": "tests/demo.py", "symbol": "MyClass", "max_chars": 300},
        # )
        # pretty_print("outline MyClass", resp2)


asyncio.run(test_tools())

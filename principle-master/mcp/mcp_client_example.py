#!/usr/bin/env python3
"""
Example MCP client to test the principle-master journaling server.

This script demonstrates how to use the MCP server for journal management.
"""

import asyncio
import json
import subprocess
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Test the MCP server functionality."""
    server_params = StdioServerParameters(
        command="python3",
        args=["mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            print("🚀 MCP Server initialized successfully!")
            print("=" * 50)
            
            # List available tools
            print("\n📋 Available tools:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            print("\n" + "=" * 50)
            
            # Test 1: Get current template
            print("\n🔍 Testing: Get journal template")
            try:
                result = await session.call_tool("get_journal_template", {})
                print(f"✅ Template retrieved (length: {len(result.content[0].text)} chars)")
                print(f"First 100 chars: {result.content[0].text[:100]}...")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Test 2: Create a new journal
            print("\n📝 Testing: Create new journal")
            try:
                result = await session.call_tool("create_journal", {"date": "2024-01-15"})
                print(f"✅ {result.content[0].text}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Test 3: List journals
            print("\n📚 Testing: List journals")
            try:
                result = await session.call_tool("list_journals", {})
                print(f"✅ {result.content[0].text}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Test 4: Read a journal
            print("\n👀 Testing: Read journal")
            try:
                result = await session.call_tool("read_journal", {"date": "2024-01-15"})
                print(f"✅ Journal read successfully (length: {len(result.content[0].text)} chars)")
                print(f"First 200 chars: {result.content[0].text[:200]}...")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Test 5: Write journal content
            print("\n✍️  Testing: Write journal content")
            test_content = """# 📅 Daily Journal — 2024-01-15

## Today's Reflection
This is a test journal entry created via MCP server!

## Key Learnings
- MCP server integration works
- Journal management is now accessible via API
- External LLM hosts can create journals

## Tomorrow's Goals
- Continue testing MCP functionality
- Add more features as needed
"""
            try:
                result = await session.call_tool("write_journal_content", {
                    "date": "2024-01-15",
                    "content": test_content
                })
                print(f"✅ {result.content[0].text}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Test 6: List resources
            print("\n🗂️  Testing: List resources")
            try:
                resources = await session.list_resources()
                print(f"✅ Found {len(resources.resources)} resources:")
                for resource in resources.resources:
                    print(f"  - {resource.name}: {resource.description}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("\n" + "=" * 50)
            print("🎉 MCP Server testing completed!")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
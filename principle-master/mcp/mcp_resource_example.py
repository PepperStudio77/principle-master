#!/usr/bin/env python3
"""
Example demonstrating how to use MCP resources from a client/host.

This shows how MCP hosts (like Claude Desktop) can access resources
defined by the principle-master MCP server.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def demonstrate_resources():
    """Demonstrate how to use MCP resources."""
    server_params = StdioServerParameters(
        command="python3",
        args=["mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("🗂️  MCP Resources Demo")
            print("=" * 50)
            
            # 1. List all available resources
            print("\n📋 Listing all available resources:")
            resources = await session.list_resources()
            
            for resource in resources.resources:
                print(f"  📄 {resource.name}")
                print(f"     URI: {resource.uri}")
                print(f"     Description: {resource.description}")
                print(f"     MIME Type: {resource.mimeType}")
                print()
            
            print("=" * 50)
            
            # 2. Read the journal template resource
            print("\n📖 Reading journal template resource:")
            try:
                template_content = await session.read_resource(
                    "principle-master://journal/template"
                )
                print(f"✅ Template content (length: {len(template_content)} chars)")
                print(f"First 200 chars:\n{template_content[:200]}...")
                print()
            except Exception as e:
                print(f"❌ Error reading template: {e}")
            
            # 3. Read the journal list resource
            print("\n📚 Reading journal list resource:")
            try:
                journal_list_json = await session.read_resource(
                    "principle-master://journal/list"
                )
                journal_list = json.loads(journal_list_json)
                print(f"✅ Found {len(journal_list)} journal entries:")
                for journal_date in journal_list:
                    print(f"  📅 {journal_date}")
                print()
            except Exception as e:
                print(f"❌ Error reading journal list: {e}")
            
            print("=" * 50)
            print("📝 Resource Usage Summary:")
            print("• Resources provide READ-ONLY access to data")
            print("• Templates can be fetched without parameters")
            print("• Journal lists return JSON arrays of available dates")
            print("• Resources are cached and efficient for repeated access")
            print("• Use tools for WRITE operations (create, update, delete)")


async def show_resource_vs_tool_usage():
    """Show the difference between resources and tools."""
    print("\n🔄 Resources vs Tools Comparison")
    print("=" * 50)
    
    print("📖 RESOURCES (Read-Only):")
    print("  • principle-master://journal/template")
    print("    → Returns current template content")
    print("    → Fast, cached access")
    print("    → No parameters needed")
    print()
    print("  • principle-master://journal/list") 
    print("    → Returns JSON array of journal dates")
    print("    → Efficient for directory scanning")
    print("    → No parameters needed")
    print()
    
    print("🛠️  TOOLS (Read & Write):")
    print("  • get_journal_template")
    print("    → Same as template resource, but as a tool")
    print("  • list_journals")
    print("    → Same as list resource, but formatted text")
    print("  • create_journal")
    print("    → CREATE new journal entries")
    print("  • read_journal") 
    print("    → READ specific journal content by date")
    print("  • write_journal_content")
    print("    → WRITE/UPDATE journal content")
    print("  • update_journal_template")
    print("    → UPDATE the template")
    print()
    
    print("📋 Usage Guidelines:")
    print("  ✅ Use RESOURCES for:")
    print("     - Quick template access")
    print("     - Directory listings")
    print("     - Metadata queries")
    print("     - Dashboard/overview displays")
    print()
    print("  ✅ Use TOOLS for:")
    print("     - Creating new content")
    print("     - Modifying existing content") 
    print("     - Complex operations with parameters")
    print("     - Interactive workflows")


if __name__ == "__main__":
    asyncio.run(demonstrate_resources())
    asyncio.run(show_resource_vs_tool_usage())
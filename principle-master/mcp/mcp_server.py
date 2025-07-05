#!/usr/bin/env python3
"""
MCP Server for Principle Master Journaling

Exposes journaling functionality from principle-master to allow other LLM hosts
to create and manage journals via the Model Context Protocol.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add the principle-master directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "principle-master"))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
)
from mcp.shared.exceptions import McpError

from core.state import JournalManager, WorkflowState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("principle-master-mcp")

# Initialize the MCP server
app = Server("principle-master-journal")

# Create a global journal manager instance
journal_manager = JournalManager()


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools for journal management."""
    return [
        Tool(
            name="create_journal",
            description="Create a new journal entry for a specific date using the available template",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (defaults to today if not provided)"
                    }
                },
                "additionalProperties": False
            }
        ),
        Tool(
            name="get_journal_template",
            description="Get the current journal template content",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="update_journal_template",
            description="Update the AI journal template with new content",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "New template content to save"
                    }
                },
                "required": ["content"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="list_journals",
            description="List all existing journal files",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="read_journal",
            description="Read the content of a specific journal file",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format"
                    }
                },
                "required": ["date"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="write_journal_content",
            description="Write content to a specific journal file",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the journal"
                    }
                },
                "required": ["date", "content"],
                "additionalProperties": False
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls for journal management."""
    try:
        if name == "create_journal":
            return await create_journal_tool(arguments)
        elif name == "get_journal_template":
            return await get_journal_template_tool(arguments)
        elif name == "update_journal_template":
            return await update_journal_template_tool(arguments)
        elif name == "list_journals":
            return await list_journals_tool(arguments)
        elif name == "read_journal":
            return await read_journal_tool(arguments)
        elif name == "write_journal_content":
            return await write_journal_content_tool(arguments)
        else:
            raise McpError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in tool {name}: {str(e)}")
        raise McpError(f"Tool execution failed: {str(e)}")


async def create_journal_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """Create a new journal entry."""
    date_str = arguments.get("date")
    
    if date_str:
        try:
            # Validate date format
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise McpError("Date must be in YYYY-MM-DD format")
    else:
        date_str = datetime.today().strftime("%Y-%m-%d")
    
    # Create custom journal manager for specific date
    journal_dir = journal_manager.local_journal_dir()
    if not os.path.exists(journal_dir):
        os.makedirs(journal_dir)
    
    journal_file = os.path.join(journal_dir, f"journal-{date_str}.md")
    
    # Check if journal already exists
    if os.path.exists(journal_file):
        return [TextContent(
            type="text",
            text=f"Journal for {date_str} already exists at: {journal_file}"
        )]
    
    # Get template content
    template_content = journal_manager.read_template()
    
    # Create journal with date substitution
    with open(journal_file, "w") as f:
        f.write(template_content.format(DATE=date_str))
    
    return [TextContent(
        type="text",
        text=f"Journal created successfully for {date_str} at: {journal_file}"
    )]


async def get_journal_template_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get the current journal template."""
    template_content = journal_manager.read_template()
    return [TextContent(
        type="text",
        text=template_content
    )]


async def update_journal_template_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """Update the AI journal template."""
    content = arguments.get("content")
    if not content:
        raise McpError("Content is required")
    
    journal_manager.update_template(content)
    return [TextContent(
        type="text",
        text="Journal template updated successfully"
    )]


async def list_journals_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """List all existing journal files."""
    journal_dir = journal_manager.local_journal_dir()
    
    if not os.path.exists(journal_dir):
        return [TextContent(
            type="text",
            text="No journal directory found"
        )]
    
    journal_files = []
    for file in os.listdir(journal_dir):
        if file.startswith("journal-") and file.endswith(".md"):
            # Extract date from filename
            date_part = file.replace("journal-", "").replace(".md", "")
            journal_files.append(date_part)
    
    journal_files.sort()
    
    if not journal_files:
        return [TextContent(
            type="text",
            text="No journal files found"
        )]
    
    return [TextContent(
        type="text",
        text=f"Found {len(journal_files)} journal files:\n" + "\n".join(journal_files)
    )]


async def read_journal_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """Read a specific journal file."""
    date_str = arguments.get("date")
    if not date_str:
        raise McpError("Date is required")
    
    try:
        # Validate date format
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise McpError("Date must be in YYYY-MM-DD format")
    
    journal_dir = journal_manager.local_journal_dir()
    journal_file = os.path.join(journal_dir, f"journal-{date_str}.md")
    
    if not os.path.exists(journal_file):
        raise McpError(f"Journal for {date_str} not found")
    
    with open(journal_file, "r") as f:
        content = f.read()
    
    return [TextContent(
        type="text",
        text=content
    )]


async def write_journal_content_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """Write content to a specific journal file."""
    date_str = arguments.get("date")
    content = arguments.get("content")
    
    if not date_str:
        raise McpError("Date is required")
    if not content:
        raise McpError("Content is required")
    
    try:
        # Validate date format
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise McpError("Date must be in YYYY-MM-DD format")
    
    journal_dir = journal_manager.local_journal_dir()
    if not os.path.exists(journal_dir):
        os.makedirs(journal_dir)
    
    journal_file = os.path.join(journal_dir, f"journal-{date_str}.md")
    
    with open(journal_file, "w") as f:
        f.write(content)
    
    return [TextContent(
        type="text",
        text=f"Journal content written successfully for {date_str}"
    )]


@app.list_resources()
async def list_resources() -> List[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="principle-master://journal/template",
            name="Journal Template",
            description="Current journal template used for creating new entries",
            mimeType="text/markdown"
        ),
        Resource(
            uri="principle-master://journal/list",
            name="Journal List",
            description="List of all available journal entries",
            mimeType="application/json"
        )
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a specific resource."""
    if uri == "principle-master://journal/template":
        return journal_manager.read_template()
    elif uri == "principle-master://journal/list":
        journal_dir = journal_manager.local_journal_dir()
        if not os.path.exists(journal_dir):
            return json.dumps([])
        
        journal_files = []
        for file in os.listdir(journal_dir):
            if file.startswith("journal-") and file.endswith(".md"):
                date_part = file.replace("journal-", "").replace(".md", "")
                journal_files.append(date_part)
        
        journal_files.sort()
        return json.dumps(journal_files)
    else:
        raise McpError(f"Resource not found: {uri}")


async def main():
    """Main entry point for the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `principle-master`, an AI-powered chatbot designed to help users build and refine their own principles through a journaling → reflection → advice cycle. The application uses LlamaIndex for RAG capabilities and supports both OpenAI and Gemini LLMs.

## Key Commands

### Setup Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Configure LLM (one-time setup)
python principle-master/main.py config-llm

# Index content from PDF (one-time setup)
python principle-master/main.py index-content <pdf_path>
```

### Running the Application
```bash
# Run the main application
python principle-master/main.py principle-master --verbose

# Run with dynamic workflow (alternative implementation)
python principle-master/main.py principle-master --verbose --dynamic
```

### Development Commands
```bash
# Install dependencies (using Makefile)
make install
```

## Architecture Overview

### Core Components

- **Main Entry Point**: `principle-master/main.py` - CLI interface using Click
- **Core Workflow**: `principle-master/core/workflow.py` - Main workflow orchestration using LlamaIndex workflows
- **Agent System**: Multi-agent architecture with specialized agents:
  - `IntentionDetectionAgent` - Routes user intent to appropriate functionality
  - `CaseReflectionAgent` - Guides users through structured case reflection
  - `ProfileUpdateAgent` - Manages user profile recording
  - `AdviceAgent` - Provides personalized advice using RAG

### Key Directories

- `core/` - Core business logic, agents, and workflow management
- `utils/` - Utility functions for LLM interaction, PDF processing, tokenization
- `config/` - Configuration files (key.json for API keys)
- `index/` - Vector store and index files for RAG functionality  
- `journal/` - User journal files and templates
- `state/` - Application state management
- `notes/` - User cases, notes, and profile data

### Data Flow

1. User interaction starts through CLI → `IntentionDetectionAgent` routes to appropriate function
2. Available functions: Case Reflection, Profile Recording, Advice, Journal Writing
3. RAG system uses indexed content from PDF books/principles for advice generation
4. State management tracks user profile, cases, and generates personalized templates

### Configuration

- LLM configuration stored in `principle-master/config/key.json`
- Supports both OpenAI and Gemini models for LLM and embeddings
- Index data persisted in `principle-master/index/` directory

### Memory Management

- Uses `ChatMemoryBuffer` with 40,000 token limit for conversation memory
- Fresh memory contexts created for case reflection sessions
- Session-based state management with UUID tracking

## MCP Server Integration

### Running the MCP Server
```bash
# Start the MCP server for journaling functionality
python3 mcp_server.py
```

### Available MCP Tools
- `create_journal` - Create new journal entry for a specific date
- `get_journal_template` - Get current journal template content  
- `update_journal_template` - Update the AI journal template
- `list_journals` - List all existing journal files
- `read_journal` - Read specific journal content
- `write_journal_content` - Write content to a journal file

### MCP Resources
- `principle-master://journal/template` - Current journal template
- `principle-master://journal/list` - List of available journal entries

### Testing MCP Server
```bash
# Run example client to test MCP functionality
python3 mcp_client_example.py
```

## Important Notes

- The application requires indexed content (PDF) to provide advice functionality
- Journal writing opens vim editor for user input
- Two workflow implementations available: static and dynamic agent flows
- MCP server exposes journaling functionality to external LLM hosts
- No formal test suite - manual testing through CLI interface
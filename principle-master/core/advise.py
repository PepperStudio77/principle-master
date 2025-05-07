import asyncio
import os.path
from typing import List

from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, \
    Settings, get_response_synthesizer
from llama_index.core.agent.workflow import AgentOutput, ToolCallResult, ToolCall, FunctionAgent
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.tools import FunctionTool

from core.index import load_persisted_index
from core.state import get_workflow_state
from utils.llm import get_embedding, get_openai_llm


def _get_local_index_store_dir():
    index_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "./index")
    return index_dir


def _load_persisted_index(local_index_store: str):
    storage_context = StorageContext.from_defaults(persist_dir=local_index_store)
    index = load_index_from_storage(storage_context)
    return index


TOP_K = 2


def _create_query_engine_from_index(index: VectorStoreIndex):
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
    )
    # return the original content without using LLM to synthesizer. For later evaluation.
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.NO_TEXT)
    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )
    return query_engine


REWRITE_FACTOR = 2

QUESTION_REWRITE_PROMPT = f"""
You are an assistant that helps reframe user questions into clear, concept-driven statements that match 
the style and topics of Principles by Ray Dalio, and perform look up principle book for relevant content. 

Background:
Principles teaches structured thinking about life and work decisions.
The key ideas are:
* Radical truth and radical transparency
* Decision-making frameworks
* Embracing mistakes as learning
* Diagnosing and solving problems systematically
* Building meritocratic organizations
The content style is practical, structured, focused on systems thinking — not emotional, not vague.

Task:
- Task 1: Rewrite a user’s question into a statement that would match how Ray Dalio frames ideas in Principles. Emphasize problem-solving, systems, truth-seeking, decision-making frameworks.Avoid emotional tone or overly personal phrasing.Keep the rewritten version faithful to the user's original meaning but recast it in Principles language. Use formal, logical, neutral tone.
- Task 2: Look up principle book with given re-wrote statements. You should provide at least {REWRITE_FACTOR} rewrote versions.
- Task 3: Find the most relevant from the book content as your fina answers. 

Example rewrites:
- User Question: "How can I recover after making a mistake?"
  Rewritten Statement: 
  - "The process of learning from mistakes to make consistent progress aligns with the principle that pain + reflection = growth?"
  - "Mistakes are opportunities for learning through pain and reflection, which is essential for personal growth."
  - "Recovering from mistakes involves acknowledging pain, analyzing the cause, and designing improvements."
- User Question: "How do I know if my team is working well together?"
  Rewritten Statement: 
  - "A team's effectiveness is measured by the quality of meaningful work and meaningful relationships achieved through radical truth and transparency."
  - "Evaluating team health requires assessing alignment to radical truth, open communication, and meritocratic collaboration."
  - "Organizational success is determined by how openly and effectively team members pursue shared goals with transparency and trust."

"""


def get_principle_rag_agent():
    index = load_persisted_index()
    query_engine = _create_query_engine_from_index(index)
    query_engine = query_engine

    def look_up_principle_book(original_question: str, rewrote_statement: List[str]) -> List[str]:
        result = []
        for q in rewrote_statement:
            response = query_engine.query(q)
            content = [n.get_content() for n in response.source_nodes]
            result.extend(content)
        return result

    tools = [
        FunctionTool.from_defaults(
            fn=look_up_principle_book,
            name="look_up_principle_book",
            description="Look up principle book with re-wrote queries. Getting the suggestions from the Principle book by Ray Dalio"),
    ]

    agent = FunctionAgent(
        name="principle_reference_loader",
        description="You are a helpful agent will based on user's question and look up the most relevant content in principle book.\n",
        system_prompt=QUESTION_REWRITE_PROMPT,
        tools=tools,
    )
    return agent


ADVISER_PROMPT = """
You are an AI assistant that provides thoughtful, practical, and *deeply personalized* suggestions by combining:
- The user's personal profile and principles
- Insights retrieved from *Principles* by Ray Dalio

---  
User Profile:  

User's Profile:
```
{user_profile}
```

User's principle:
```
{user_principles}
```

Book Content: 
```
{book_content}
```

## ✍️ Style Guidelines:

- Provide the suggestions based on the content of book when relevant. 
- Ground your suggestion in something **specific about the user** (e.g. a strength, weakness, past experiences).
- Provide Top 3 most relevant suggestions. 

---

"""


def get_adviser_agent(user_profile: dict, user_principles: List[str], book_content: str):
    agent = FunctionAgent(
        name="principle_advisor",
        description="You are a helpful advisor which will advise user's question based on the some guidance from Principle book by Ray Dalio and user's own principles.\n",
        system_prompt=ADVISER_PROMPT.format(
            user_principles="\n".join(user_principles),
            user_profile="\n".join([k + ": " + v for (k, v) in user_profile.items()]),
            book_content=book_content,
        ))
    return agent


async def run_agent(agent: FunctionAgent, question, verbose: bool = False):
    chat_history = [
        ChatMessage(
            role="user", content=question,
        ),
    ]
    handler = agent.run(chat_history=chat_history)
    if not verbose:
        output = await handler
        return output.response.content
    result = None
    current_agent = None
    async for event in handler.stream_events():
        if (
                hasattr(event, "current_agent_name")
                and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'=' * 50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'=' * 50}\n")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
                result = event.response.content
            if event.tool_calls:
                print(
                    "🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")
    return result


async def get_advise(session_id: str, question: str, verbose: bool = False):
    state = get_workflow_state(session_id)
    principles = state.load_principle_from_cases()
    profile = state.load_profile()
    rag_agent = get_principle_rag_agent()
    book_content = await run_agent(rag_agent, question=question, verbose=verbose)
    advisor = get_adviser_agent(profile, principles, book_content)
    advise = await run_agent(advisor, question=question, verbose=verbose)
    return advise


if __name__ == '__main__':
    Settings.embed_model = get_embedding()
    Settings.llm = get_openai_llm()
    asyncio.run(get_advise("xxxx", "How to handle stress?", True, True))

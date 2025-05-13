from llama_index.core import Settings
from llama_index.core.agent.workflow import AgentWorkflow, AgentOutput, ToolCallResult, ToolCall, FunctionAgent
from llama_index.core.base.llms.types import ChatMessage

from core.advisor_agents import get_principle_rag_agent, get_interviewer_agent, get_adviser_agent
from core.state import get_workflow_state
from utils.llm import get_openai_llm, get_embedding


def get_advice_dynamic_workflow(session_id: str):
    state = get_workflow_state(session_id)
    interviewer = get_interviewer_agent(True, ["reference_retriever"])
    retriever = get_principle_rag_agent(True, ["principle_advisor"])
    principles = state.load_principle_from_cases()
    profile = state.load_profile()
    advisor = get_adviser_agent(user_principles=principles, user_profile=profile, is_dynamic_agent=True,
                                can_handoff_to=[])
    workflow = AgentWorkflow(
        agents=[interviewer, advisor, retriever],
        root_agent="interviewer",
    )
    return workflow


async def run_dynamic_workflow(session_id: str, question: str, verbose: bool = False):
    workflow = get_advice_dynamic_workflow(session_id)
    user_msg = f"""
    Please give advise me given following question: {question}. 
    You can seek clarification to me further if want to. 
    I would like you to retrieve relevant content from Principle book by Ray Dalio, combine with my profile and existing principle, 
    generate catered suggestions for me. """
    handler = workflow.run(user_msg=user_msg)
    if not verbose:
        output = await handler
        return output.response.content
    # provide verbose output
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


async def run_static_workflow(session_id: str, question: str, verbose: bool = False):
    state = get_workflow_state(session_id)
    principles = state.load_principle_from_cases()
    profile = state.load_profile()
    interviewer = get_interviewer_agent()
    interview_result = await run_agent(interviewer, question=question, verbose=verbose)
    rag_agent = get_principle_rag_agent()
    book_content = await run_agent(rag_agent, question=interview_result, verbose=verbose)
    advisor = get_adviser_agent(profile, principles, book_content)
    advise = await run_agent(advisor, question=interview_result, verbose=verbose)
    return advise


async def run():
    session_id = "test_session"
    question = "How can I improve my time management skills?"
    result = await workflow_run(session_id, question, verbose=True)
    print(result)


if __name__ == "__main__":
    import asyncio

    Settings.llm = get_openai_llm()
    Settings.embed_model = get_embedding()
    asyncio.run(run())

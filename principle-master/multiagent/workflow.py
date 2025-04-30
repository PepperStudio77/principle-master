from llama_index.core.agent.workflow import AgentWorkflow, AgentOutput, ToolCallResult, ToolCall
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer

from multiagent.agents import get_evaluator_agent, get_interviewer_agent, get_planner_agent


def get_agent_workflow(verbose:bool = False):
    evaluator = get_evaluator_agent()
    interviewer = get_interviewer_agent()
    planner = get_planner_agent()
    agent_workflow = AgentWorkflow(
        agents=[evaluator, interviewer, planner],
        root_agent=evaluator.name,
        initial_state={
            "plan_content": "Not written yet"
        },
        verbose=verbose,
    )
    return agent_workflow


async def run(request, verbose=False):
    workflow = get_agent_workflow(verbose=verbose)
    workflow_prompt = (
        f"You need to generate a plan to help users to adopt a good habit, you will evaluate user's request, "
        f"seek for clarification, generate a plan and evaluate the plan.")
    chat_history = [
        ChatMessage(role="system",
                    content=f"User request is: {request}"),
    ]
    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
    memory.put_messages(chat_history)
    handler = workflow.run(
        user_msg=workflow_prompt,
        memory=memory,
    )
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

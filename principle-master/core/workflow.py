import uuid
from typing import Optional

from llama_index.core import Settings
from llama_index.core.agent.workflow import AgentOutput, ToolCallResult, ToolCall
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.workflow import Context, Workflow, step, StartEvent, StopEvent
from rich import print

from core.case_reflection import CaseReflectionAgent
from core.intention import IntentionDetectionAgent
from core.profile import ProfileUpdateAgent
from core.state import CASE_REFLECTION, ROUTING, ENDING, \
    CaseReflectionEvent, RoutingEvent, get_workflow_state, AVAILABLE_FUNCTIONS, RecordProfileEvent, \
    RECORD_PROFILE, Advice, ADVISE
from utils.llm import get_llm, get_embedding


async def print_details(workflow_handler) -> str:
    current_agent = None
    async for event in workflow_handler.stream_events():
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
                return event.response.content
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
    return "Event Streaming Finished without Agent output"


class PrincipleMasterFlow(Workflow):
    GREETING = (
        "Great to see you! Let’s dive into the art of thoughtful decision-making, guided by Ray Dalio’s Principles. "
        "Growth begins with radical honesty. \n"
        "Here is my available functions for you to try, anything do you fancy?\n"
        f"{chr(10).join(list(AVAILABLE_FUNCTIONS))}\n")

    def __init__(self, memory: Optional[BaseMemory] = None, verbose: bool = False):
        self.memory = memory
        self.verbose = verbose
        self.session_id = str(uuid.uuid4())
        self.router_state = get_workflow_state(self.session_id)
        self._greeted = False
        super().__init__(timeout=None, verbose=verbose)

    EVENT_MAP = {
        CASE_REFLECTION: CaseReflectionEvent,
        RECORD_PROFILE: RecordProfileEvent,
        ADVISE: Advice,
        ROUTING: RoutingEvent,
        ENDING: StopEvent,
    }

    @step
    async def start(self, ctx: Context,
                    ev: StartEvent | RoutingEvent) -> CaseReflectionEvent | RecordProfileEvent | Advice | StopEvent:
        agent = IntentionDetectionAgent(session_id=self.session_id, tools=[], memory=self.memory,
                                        verbose=self.verbose)
        if self._greeted is False:
            print(self.GREETING)
            self.memory.put(ChatMessage(
                role="assistant",
                content=self.GREETING,
            ))
            self._greeted = True
        response = agent.start_chat("")
        event_class = self.EVENT_MAP[response]
        return event_class(input=response)

    @step
    async def case_reflection(self, ctx: Context, ev: CaseReflectionEvent) -> RoutingEvent:
        refresh_memory = ChatMemoryBuffer.from_defaults(token_limit=TOKEN_LIMIT)
        agent = CaseReflectionAgent(session_id=self.session_id, memory=refresh_memory,
                                    verbose=self.verbose)
        response = agent.start_chat("Instruct me what should I do")
        return RoutingEvent(input=response)

    @step
    async def record_profile(self, ctx: Context, ev: RecordProfileEvent) -> RoutingEvent:
        agent = ProfileUpdateAgent(session_id=self.session_id, verbose=self.verbose)
        response = agent.start_chat("Instruct me what should I do")
        return RoutingEvent(input=response)

    @step
    async def advice(self, ctx: Context, ev: Advice) -> RoutingEvent:
        pass


TOKEN_LIMIT = 40000


async def run():
    llm = get_llm()
    Settings.llm = llm
    embed_model = get_embedding()
    Settings.embed_model = embed_model
    memory = ChatMemoryBuffer.from_defaults(token_limit=TOKEN_LIMIT)
    workflow = PrincipleMasterFlow(memory=memory, verbose=True)
    result = await workflow.run()
    print(result)


if __name__ == '__main__':
    # asyncio.run(run())
    embed_model = get_embedding()
    # Use the embedding model
    embeddings = embed_model.get_text_embedding("This is a test sentence.")
    print(embeddings)

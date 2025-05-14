import asyncio
import subprocess
import uuid
from typing import Optional

from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.workflow import Context, Workflow, step, StartEvent, StopEvent, Event
from rich import print

from core.advice_agent_flow import get_advice_dynamic_workflow, get_static_workflow
from core.case_reflection import CaseReflectionAgent
from core.intention import IntentionDetectionAgent
from core.profile import ProfileUpdateAgent
from core.state import CASE_REFLECTION, ROUTING, ENDING, get_workflow_state, AVAILABLE_FUNCTIONS, \
    RECORD_PROFILE, ADVISE, JOURNAL
from utils.llm import get_embedding, get_config, get_llm


class CaseReflectionEvent(Event):
    input: str


class RecordProfileEvent(Event):
    input: str


class RoutingEvent(Event):
    input: str


class Advice(Event):
    input: str


class JournalEvent(Event):
    input: str


class PrincipleMasterFlow(Workflow):
    GREETING = (
        "Great to see you! Let’s dive into the art of thoughtful decision-making, guided by Ray Dalio’s Principles. "
        "Growth begins with radical honesty. \n"
        "Here is my available functions for you to try:\n"
        f"{chr(10).join(list(AVAILABLE_FUNCTIONS))}\n")

    def __init__(self, memory: Optional[BaseMemory] = None, verbose: bool = False,
                 is_dynamic_advice_flow: bool = False):
        self.memory = memory
        self.verbose = verbose
        self.session_id = str(uuid.uuid4())
        self.state = get_workflow_state(self.session_id)
        self.conf = get_config()
        self.is_dynamic_advice_flow = is_dynamic_advice_flow
        super().__init__(timeout=None, verbose=verbose)

    EVENT_MAP = {
        CASE_REFLECTION: CaseReflectionEvent,
        RECORD_PROFILE: RecordProfileEvent,
        ADVISE: Advice,
        ROUTING: RoutingEvent,
        ENDING: StopEvent,
        JOURNAL: JournalEvent,
    }

    @step
    async def start(self, ctx: Context,
                    ev: StartEvent) -> CaseReflectionEvent | RecordProfileEvent | Advice | JournalEvent | StopEvent:
        agent = IntentionDetectionAgent(session_id=self.session_id, tools=[], memory=self.memory,
                                        verbose=self.verbose)
        print(self.GREETING)
        self.memory.put(ChatMessage(
            role="assistant",
            content=self.GREETING,
        ))
        response = agent.start_chat()
        event_class = self.EVENT_MAP[response]
        return event_class(input=response)

    @step
    async def case_reflection(self, ctx: Context, ev: CaseReflectionEvent) -> StopEvent:
        refresh_memory = ChatMemoryBuffer.from_defaults(token_limit=TOKEN_LIMIT)
        agent = CaseReflectionAgent(session_id=self.session_id, memory=refresh_memory,
                                    verbose=self.verbose)
        response = agent.start_chat("Instruct me what should I do")
        return StopEvent(input=response)

    @step
    async def record_profile(self, ctx: Context, ev: RecordProfileEvent) -> StopEvent:
        agent = ProfileUpdateAgent(session_id=self.session_id, verbose=self.verbose)
        response = agent.start_chat("Instruct me what should I do")
        return StopEvent(input=response)

    @step
    async def advice(self, ctx: Context, ev: Advice) -> StopEvent:
        uer_question = input("How can I help you today?")
        workflow = get_advice_dynamic_workflow(session_id=self.session_id) if self.is_dynamic_advice_flow else \
            get_static_workflow(session_id=self.session_id, verbose=self.verbose)
        advise = await workflow.run(user_msg=uer_question)
        print(advise)
        return StopEvent(result="Done")

    @step
    async def write_journal(self, ctx: Context, ev: JournalEvent) -> StopEvent:
        print("Write your journal")
        new_journal_file = self.state.new_journal()
        # sleep here because I am lazy to implement a io lock, wait for verbose log to finish first.
        await asyncio.sleep(1)
        subprocess.run(["vim", new_journal_file])
        return StopEvent(result="Done")


TOKEN_LIMIT = 40000


async def run_customise_workflow(verbose: bool = False, is_dynamic_advice_flow: bool = False):
    llm = get_llm()
    Settings.llm = llm
    embed_model = get_embedding()
    Settings.embed_model = embed_model
    memory = ChatMemoryBuffer.from_defaults(token_limit=TOKEN_LIMIT)
    workflow = PrincipleMasterFlow(memory=memory, verbose=verbose, is_dynamic_advice_flow=is_dynamic_advice_flow)
    _ = await workflow.run()

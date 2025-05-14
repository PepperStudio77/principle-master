from typing import List, Optional

from llama_index.core.agent.workflow import AgentWorkflow, AgentOutput, ToolCallResult, ToolCall, FunctionAgent
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.workflow import Workflow, Event, step, Context, StartEvent, StopEvent
from llama_index.core.workflow.handler import WorkflowHandler

from core.advisor_agents import get_principle_rag_agent, get_interviewer_agent, get_adviser_agent, \
    get_template_update_agent
from core.state import get_workflow_state


def get_advice_dynamic_workflow(session_id: str, verbose: bool = False):
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
        verbose=verbose,
    )
    return workflow


def get_static_workflow(session_id: str, verbose: bool = False):
    workflow = AdviceWorkFlow(session_id=session_id, verbose=verbose)
    return workflow


async def _run_agent(agent: FunctionAgent, question, chat_history: Optional[List[ChatMessage]] = None,
                     verbose: bool = False):
    chat_history = [] if chat_history is None else chat_history.copy()
    chat_history.append(
        ChatMessage(
            role="user", content=question,
        ))
    handler = agent.run(chat_history=chat_history)
    if verbose:
        result = await _verbose_print(handler)
        return result
    output = await handler
    return output.response.content


class ReferenceRetrivalEvent(Event):
    question: str


class Advice(Event):
    principles: List[str]
    profile: dict
    question: str
    book_content: str


class UpdateJournalTemplate(Event):
    advice: str
    question: str


class AdviceWorkFlow(Workflow):

    def __init__(self, verbose: bool = False, session_id: str = None):
        self.uuid = session_id
        state = get_workflow_state(session_id)
        self.principles = state.load_principle_from_cases()
        self.profile = state.load_profile()
        self.verbose = verbose
        super().__init__(timeout=None, verbose=verbose)

    @step
    async def interview(self, ctx: Context,
                        ev: StartEvent) -> ReferenceRetrivalEvent:
        # Step 1: Interviewer agent asks questions to the user
        interviewer = get_interviewer_agent()
        question = await _run_agent(interviewer, question=ev.user_msg, verbose=self.verbose)
        return ReferenceRetrivalEvent(question=question)

    @step
    async def retrieve(self, ctx: Context, ev: ReferenceRetrivalEvent) -> Advice:
        # Step 2: RAG agent retrieves relevant content from the book
        rag_agent = get_principle_rag_agent()
        book_content = await _run_agent(rag_agent, question=ev.question, verbose=self.verbose)
        return Advice(principles=self.principles, profile=self.profile,
                      question=ev.question, book_content=book_content)

    @step
    async def advice(self, ctx: Context, ev: Advice) -> UpdateJournalTemplate:
        # Step 3: Adviser agent provides advice based on the user's profile, principles, and book content
        advisor = get_adviser_agent(ev.profile, ev.principles, ev.book_content)
        advice = await _run_agent(advisor, question=ev.question, verbose=self.verbose)
        return UpdateJournalTemplate(advice=advice, question=ev.question)

    @step
    async def update_journal_template(self, ctx: Context, ev: UpdateJournalTemplate) -> StopEvent:
        # Step 4: Update the journal template based on the advice provided
        state = get_workflow_state(self.uuid)
        existing_template = state.read_template()
        template_update_agent = get_template_update_agent(existing_template)
        chat_history = [
            ChatMessage(
                role="assistant",
                content=ev.advice
            )
        ]
        updated_template = await _run_agent(template_update_agent, question=ev.question, chat_history=chat_history,
                                            verbose=self.verbose)
        print("Updated template:\n", updated_template.strip("md").strip("```"))
        print("Do you want to save the updated template? (yes/no)")
        user_input = input(">>")
        if user_input == "y" or user_input == "yes" or user_input == "Yes" or user_input == "YES" or user_input == "Y":
            state.update_template(updated_template)
            print("Updated template saved.")
        return StopEvent(result=updated_template)


async def _verbose_print(handler: WorkflowHandler) -> str:
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
        elif isinstance(event, StopEvent):
            print(f"🔧 Stop Event: {event}")
            result = str(event.result)
    return result

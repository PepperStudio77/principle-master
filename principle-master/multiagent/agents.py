from typing import List

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context, JsonSerializer
from llama_index.llms.deepseek import DeepSeek

from utils.journal_keeper import Response, write_response_to_mark_down
from utils.clarification import save_interview_notes, load_interview_notes
from utils.llm import get_config, get_llm


async def read_interview_notes():
    notes = load_interview_notes()
    if notes is None:
        return "No previous interview notes found"
    notes_str = ""
    for topic, details in notes.items():
        notes_str += f"Question : {topic}"
        for q, a in details.items():
            notes_str += f"{q} {a}\n"
    return f"Here is the previous interview notes:\n{notes_str}"


async def load_feedback():
    pass


def get_evaluator_agent():
    agent = FunctionAgent(
        name="Evaluator",
        description="Evaluate the planning status and decides which Agent should rout to.",
        llm=get_llm(),
        tools=[read_interview_notes],
        system_prompt="You are the master mind and orchestrator of the plan generation to help users adopt good habits which they desire. \n"
                      "You will always attempt to read interview notes at the beginning.\n"
                      "Then you will evaluate whether you have enough data to propose a plan. The areas of evaluations includes:\n"
                      "- Basic information of user including gender, ages, occupation, working duration, prefer time to invest in new habit.\n"
                      "- Whether user is setting a clear of target of good habit he/she want to adopt.\n"
                      "- what is the prefer time in a day for user to adopt something new.\n"
                      "- Whether there is previous attempt of establishing those habit, what was the main obstacles.\n"
                      "- Historical feedback for execution of the plan.\n"
                      "If you think there is clarification required by users, you need to delegate to Interviewer Agent to seek for clarification with existing interview notes and your instruction.\n"
                      "If you think there the plan is clear enough, you can delegate to Planner Agent with existing interview notes.\n "
                      "You do not need to seek for user clarification yourself.",
        can_handoff_to=["Interviewer", "Planner"]
    )
    return agent


async def ask_clarification(ctx: Context, user_request_summary: str, questions_to_ask: List[str]):
    current_state = await ctx.get("state")
    if "interview_records" not in current_state:
        current_state["interview_records"] = {}
    notes = {user_request_summary: {}}
    for q in questions_to_ask:
        notes[user_request_summary][q] = input(q)
    current_state["interview_records"].update(notes)
    await ctx.set("state", current_state)
    # persist state
    save_interview_notes(notes)
    return "Interview feedback collected"


def get_interviewer_agent():
    agent = FunctionAgent(
        name="Interviewer",
        description="Useful for get clarification from users",
        llm=get_llm(),
        tools=[ask_clarification],
        system_prompt="You are an interviewer which seek for further clarification for users' input to facilitate"
                      "user to make up a plan to adopt new habit they want.\n"
                      "You should ask clarification from users with the questions you got. \n "
                      "Once it is done, you pass your clarification result to agent Evaluator for evaluation.\n"
                      "If you don't feel like it is required for ask clarification, you can still handoff to Evaluator\n",
        can_handoff_to=["Evaluator"],
    )
    return agent


async def persist_the_plan(ctx: Context, plan_in_markdown: str):
    current_state = await ctx.get("state")
    current_state["plan_content"] = plan_in_markdown
    await ctx.set("state", current_state)
    response = Response(plan_in_markdown, None)
    write_response_to_mark_down("multi-agent", response, "multi-agent")
    return "Plan drafted"


def get_planner_agent():
    agent = FunctionAgent(
        name="Planner",
        description="Useful for produce a plan to help user establish new good habits.",
        llm=get_llm(),
        tools=[persist_the_plan],
        system_prompt="You are a intelligent, well thought planer which can produce a effective plan to encourage user to establish a new good habit\n"
                      "The criteria of a good plan including:\n"
                      "- The plan consider user's condition including gender, ages, occupation, prefer time.\n"
                      "- You should also take consider in the challenges user face on the previous attempt.\n"
                      "You will then gather all the information and perform produce the plan in markdown format.\n"
                      "You always need to persist the plan asking for permission is not required.",
        can_handoff_to=["Evaluator"]
    )
    return agent

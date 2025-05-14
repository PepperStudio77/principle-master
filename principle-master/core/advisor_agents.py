import asyncio
from typing import List

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.tools import FunctionTool

from core.index import load_persisted_index

DYNAMIC_AGENT_ADJUSTMENT_PROMPT = "You should handover to {next_agent_name} when you are done. "

QUESTION_COUNT = 1

interviewer_prompt = f"""
You are helpful agent which build for raise clarification to users by calling clarification tools. 
Your goal is getting the detail information of user's questions to give throughout guidance. 
You should ask no more than {QUESTION_COUNT} to users.
You should not attempt to answer the question but return summarised content as your output. 
"""


def get_interviewer_agent(is_dynamic_agent: bool = False, can_handoff_to: List[str] = None):
    async def clarification(user_original_message, questions_raised: List[str]) -> str:
        # sleep here because I am lazy to implement a io lock, wait for verbose log to finish first.
        await asyncio.sleep(0.5)
        result = ""
        for q in questions_raised:
            result += q + ":"
            print("Question:" + q)
            response = input("Response:")
            result += response + "\n"
        return result

    tools = [
        FunctionTool.from_defaults(
            fn=clarification,
            name="clarification",
            description="Useful tool take questions for clarification as input, return user's response as output"
        )
    ]
    prompt = interviewer_prompt
    agent = FunctionAgent(
        name="interviewer",
        description="Useful agent to clarify user's questions",
        system_prompt=prompt,
        tools=tools
    )
    if is_dynamic_agent:
        agent.can_handoff_to = can_handoff_to
        agent.system_prompt += DYNAMIC_AGENT_ADJUSTMENT_PROMPT.format(next_agent_name=can_handoff_to[0])
    return agent


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
- Task 1: Clarify the user's question if needed. Ask follow-up questions to ensure you understand the user's intent.
- Task 2: Rewrite a user’s question into a statement that would match how Ray Dalio frames ideas in Principles. Use formal, logical, neutral tone.
- Task 3: Look up principle book with given re-wrote statements. You should provide at least {REWRITE_FACTOR} rewrote versions.
- Task 4: Find the most relevant from the book content as your fina answers. 

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


def get_principle_rag_agent(is_dynamic_agent: bool = False, can_handoff_to: List[str] = None):
    index = load_persisted_index()
    query_engine = _create_query_engine_from_index(index)

    async def look_up_principle_book(original_question: str, rewrote_statement: List[str]) -> List[str]:
        result = []
        for q in rewrote_statement:
            response = query_engine.query(q)
            content = [n.get_content() for n in response.source_nodes]
            result.extend(content)
        return result

    async def clarify_question(original_question: str, your_questions_to_user: List[str]) -> str:
        """
        Clarify the user's question if needed. Ask follow-up questions to ensure you understand the user's intent.
        """
        # sleep here because I am lazy to implement a io lock, wait for verbose log to finish first.
        await asyncio.sleep(0.5)
        response = ""
        for q in your_questions_to_user:
            print(f"Question: {q}")
            r = input("Response:")
            response += f"Question: {q}\nResponse: {r}\n"
        return response

    tools = [
        FunctionTool.from_defaults(
            fn=look_up_principle_book,
            name="look_up_principle_book",
            description="Look up principle book with re-wrote queries. Getting the suggestions from the Principle book by Ray Dalio"),
        FunctionTool.from_defaults(
            fn=clarify_question,
            name="clarify_question",
            description="Clarify the user's question if needed. Ask follow-up questions to ensure you understand the user's intent.",
        )
    ]

    agent = FunctionAgent(
        name="reference_retriever",
        description="You are a helpful agent will based on user's question and look up the most relevant content in principle book.\n",
        system_prompt=QUESTION_REWRITE_PROMPT,
        tools=tools,
    )

    if is_dynamic_agent and can_handoff_to is not None:
        agent.can_handoff_to = can_handoff_to
        agent.system_prompt = agent.system_prompt + DYNAMIC_AGENT_ADJUSTMENT_PROMPT.format(
            next_agent_name=can_handoff_to[0])
    return agent


BOOK_CONTENT_PROMPT = """
Book Content: 
```
{book_content}
```
"""

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
{book_content}

## ✍️ Style Guidelines:

- Provide the suggestions based on the content of book when relevant. 
- Ground your suggestion in something **specific about the user** (e.g. a strength, weakness, past experiences).
- Provide Top 3 most relevant suggestions. 
---
"""


def get_adviser_agent(user_profile: dict, user_principles: List[str], book_content: str = None,
                      is_dynamic_agent: bool = False, can_handoff_to: List[str] = None):
    if not is_dynamic_agent:
        book_content = BOOK_CONTENT_PROMPT.format(book_content=book_content)
    else:
        book_content = ""
    agent = FunctionAgent(
        name="principle_advisor",
        description="You are a helpful advisor which will advise user's question based on the some guidance from Principle book by Ray Dalio and user's own principles.\n",
        system_prompt=ADVISER_PROMPT.format(
            user_principles="\n".join(user_principles),
            user_profile="\n".join([k + ": " + v for (k, v) in user_profile.items()]),
            book_content=book_content,
        ))
    if is_dynamic_agent and can_handoff_to is not None:
        agent.can_handoff_to = can_handoff_to
        agent.system_prompt = agent.system_prompt + DYNAMIC_AGENT_ADJUSTMENT_PROMPT.format(
            next_agent_name=can_handoff_to[0])
    return agent


def get_template_update_agent(existing_template: str, is_dynamic_agent: bool = False, can_handoff_to: List[str] = None):
    template_update_prompt = """
You are an AI assistant that updates a daily journal template based on:
1. Advice provided to the user.
2. User's specific concerns.

Your task:
- Insert the Advice to the **Advice for the Day** section.
- Update the **Daily Goals / Checklist** section to reflect the advice and concerns.
- Update the **Tomorrow's Focus** section to align with the advice and concerns.
- Generate the updated markdown as output, just raw markdown content, do not need to quote it with '```'. Do not modify other parts of the template.

Existing Template:
```
{existing_template}
```
```


"""
    agent = FunctionAgent(
        name="template_updater",
        description="You are a helpful agent which will update the daily journal template based on the advice provided to the user and user's specific concerns.",
        system_prompt=template_update_prompt.format(existing_template=existing_template),
    )
    if is_dynamic_agent and can_handoff_to is not None:
        agent.can_handoff_to = can_handoff_to
        agent.system_prompt = agent.system_prompt + DYNAMIC_AGENT_ADJUSTMENT_PROMPT.format(
            next_agent_name=can_handoff_to[0])
    return agent

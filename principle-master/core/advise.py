import asyncio
import os.path
from typing import List

from llama_index.core import Document, StorageContext, load_index_from_storage, VectorStoreIndex, \
    Settings, get_response_synthesizer, Response
from llama_index.core.agent.workflow import ReActAgent, AgentOutput, ToolCallResult, ToolCall
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.evaluation import RelevancyEvaluator, BaseEvaluator
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.tools import FunctionTool

from core.state import WorkflowState
from utils.llm import get_embedding, get_openai_llm
from utils.pdf_file import load_principle_book_summary_to_string, load_principle_book_full_to_string


def _get_local_index_store_dir():
    index_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "./index")
    return index_dir


def _load_persisted_index(local_index_store: str):
    storage_context = StorageContext.from_defaults(persist_dir=local_index_store)
    index = load_index_from_storage(storage_context)
    return index


def _create_index_for_content(load_full_book: bool, use_cache: bool):
    local_index_store = _get_local_index_store_dir()
    if use_cache and os.path.exists(local_index_store):
        return _load_persisted_index(local_index_store)
    if load_full_book:
        content = load_principle_book_full_to_string()
    else:
        content = load_principle_book_summary_to_string()
    documents = [Document(text=content)]
    vector_index = VectorStoreIndex.from_documents(documents)
    if use_cache:
        os.makedirs(local_index_store, exist_ok=True)
        vector_index.storage_context.persist(persist_dir=local_index_store)
    return vector_index


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
- Task 3: Find the most relevant content as your fina answers. You do not have to re-write or summarise the content. 

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
	
Instruction:
If the user's question is very short or vague, you can seek clarification with users by using tools.  
Always phrase the output as a statement, not a question.
You should look up principle book with your re-wrote when you see fit. 
"""


class PrincipleRagAgent(object):

    def __init__(self, query_engine: BaseQueryEngine, verbose: bool = False):
        self.evaluator = RelevancyEvaluator()
        self.query_engine = query_engine
        self.verbose = verbose

        def look_up_principle_book(original_question, rewrote_queries: List[str]) -> List[str]:
            result = []
            for q in rewrote_queries:
                response = self.query_engine.query(q)
                content = [n.get_content() for n in response.source_nodes]
                result.extend(content)
            return result

        def seek_clarification(question: str) -> str:
            response = input(question + ":")
            return response

        tools = [
            FunctionTool.from_defaults(
                fn=look_up_principle_book,
                name="look_up_principle_book",
                description="Look up principle book with re-wrote queries. Getting the suggestions from the Principle book by Ray Dalio's"),
            FunctionTool.from_defaults(
                fn=seek_clarification,
                name="seek_clarification",
                description="Clarify with users when the user's question is less clear."
            )
        ]

        self.agent = ReActAgent(
            name="principle_reference_loader",
            description="You are a helpful agent will based on user's question and look up the most relevant content in principle book.\n",
            system_prompt=QUESTION_REWRITE_PROMPT,
            tools=tools)

    @classmethod
    def from_default_query_engine(cls, load_full_book: bool = False, use_cache: bool = True, verbose: bool = False):
        index = _create_index_for_content(load_full_book=load_full_book, use_cache=use_cache)
        query_engine = _create_query_engine_from_index(index)
        return cls(query_engine, verbose=verbose)

    async def retrieve_book_content(self, question: str):
        if not self.verbose:
            response = await self.agent.run(user_msg=question)
            return response

        handler = self.agent.run(user_msg=question)
        response = None
        async for event in handler.stream_events():
            if isinstance(event, AgentOutput):
                if event.response.content:
                    print("📤 Output:", event.response.content)
                    response = event.response.content
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
        # return final response
        return response


ADVISER_PROMPT = """
You are an AI assistant that provides thoughtful, practical suggestions by combining:
Use's question:
{user_question}

Book Content retrieved from Principles by Ray Dalio:
```
{book_content}
```

The user's personal principles, beliefs, or context.
User's principle:
```
{user_principles}
```

Principles emphasizes radical truth, radical transparency, decision-making systems, learning from mistakes, believability-weighted evaluation, and systematic problem solving.
Ground your suggestions in the ideas retrieved from Principle, and prioritise to use user's own principle when they are relevant.  
Respect differences if the user's principles differ slightly from Dalio's — adapt, don't override.

Style of Suggestions:
Logical, structured, pragmatic
Neutral and rational tone (avoid emotional bias)
Encourage reflection, learning, and iterative improvement
Where appropriate, recommend structured steps or frameworks (e.g., 5-step process: goals → problems → diagnosis → design → execution)

"""


class PrincipleAdviser(object):

    def __init__(self, question: str, book_content: str, user_principles: List[str], verbose: bool = False):
        self.question = question
        self.book_content = book_content
        self.user_principle = user_principles
        self.verbose = verbose
        self.agent = ReActAgent(
            name="principle_advisor",
            description="You are a helpful advisor which will advise user's question based on the some guidance from Principle book by Ray Dalio and user's own principles.\n",
            system_prompt=ADVISER_PROMPT.format(
                user_question=question,
                book_content=book_content,
                user_principles="\n".join(user_principles)
            ))

    async def advice(self):
        if not self.verbose:
            response = await self.agent.run(user_msg=self.question)
            return response
        handler = self.agent.run(user_msg=self.question)
        response = None
        async for event in handler.stream_events():
            if isinstance(event, AgentOutput):
                if event.response.content:
                    print("📤 Output:", event.response.content)
                    response = event.response.content
        return response


async def run(question: str, principles: List[str], verbose:bool=False):
    rag_agent = PrincipleRagAgent.from_default_query_engine(load_full_book=True, verbose=verbose)
    book_content = await rag_agent.retrieve_book_content(question)
    advise_agent = PrincipleAdviser(question=question, book_content=book_content, user_principles=principles, verbose=verbose)
    return await advise_agent.advice()


if __name__ == '__main__':
    Settings.embed_model = get_embedding()
    Settings.llm = get_openai_llm()
    state = WorkflowState()
    asyncio.run(run("How to handle stress?", state.load_principle_from_cases(), True))

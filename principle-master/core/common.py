from typing import List, Optional

from llama_index.core import Settings
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.tools import BaseTool


TOKEN_LIMIT = 40000
class MyAgentRunner(AgentRunner):
    PRINT_FORMAT = "[green]{message}[/green]"

    @staticmethod
    def get_purpose() -> str:
        raise Exception("Not implemented")

    def print(self, msg):
        return print(self.PRINT_FORMAT.format(message=msg))

    def __init__(self, session_id: str, tools: List[BaseTool],
                 memory: Optional[BaseMemory] = None,
                 verbose: bool = False):
        prefix_message = [
            ChatMessage(role="system",
                        content=self.get_purpose()),
        ]
        worker = FunctionCallingAgentWorker(tools=tools, llm=Settings.llm, prefix_messages=prefix_message,
                                            verbose=verbose)
        if memory is None:
            memory = ChatMemoryBuffer.from_defaults(token_limit=TOKEN_LIMIT)
        memory.put_messages(prefix_message)
        self._greeted = False
        self.session_id = session_id
        super().__init__(worker, memory=memory, verbose=verbose)

    def start_chat(self, msg: str=None):
        return self._my_chat(msg)

    def _my_chat(self, msg: str):
        raise Exception("Not Implemented")
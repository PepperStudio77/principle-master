import hashlib
from typing import Optional

from llama_index.core.memory import BaseMemory
from llama_index.core.tools import FunctionTool

from core.common import MyAgentRunner
from core.state import get_workflow_state, ReflectionCase

TOKEN_LIMIT = 40000


class CaseReflectionAgent(MyAgentRunner):
    END_OUTPUT = "CaseCollected"
    GREETING = "Let's to a case reflections."

    def __init__(self, session_id: str, memory: Optional[BaseMemory] = None,
                 verbose: bool = False) -> None:

        def store_reflection_case(case_summary, case_details, principle_applied, detail_analysis,
                                  new_principle: str) -> str:
            hashed = hashlib.sha256(case_summary.encode()).hexdigest()
            state = get_workflow_state(session_id)
            state.persist_case(self.session_id,
                               ReflectionCase(
                                   case_id=hashed,
                                   summary=case_summary,
                                   detail=case_details,
                                   principle_applied=principle_applied,
                                   detail_analysis=detail_analysis,
                                   new_principle=new_principle,
                                   dialog=self.memory.get_all(),
                               )
                               )
            return f"Case persisted. Case ID : {hashed}. Session ID: {session_id}"

        store_case = FunctionTool.from_defaults(
            fn=store_reflection_case,
            name="store_reflection_case",
            description="Store reflection case when the information of the cases is sufficiently clarified."
        )
        super().__init__(session_id, tools=[store_case], memory=memory, verbose=verbose, max_function_calls=1)

    @staticmethod
    def get_purpose():
        p = (
            "You are an assistant to help user to do a case reflection for practising the instruction in Ray Dalio's Principle.\n"
            "You should ask follow-up questions iteratively until you gather enough clear and structured information about the case.\n"
            "Be sympathetic and patient. Acknowledge and echo the user's feelings based on the information shared.\n"
            "You may also consult relevant content from the principles book to formulate better and more insightful questions.\n\n"
            "A good case reflection should ideally include the following elements:\n"
            "- [Required] Case at hand: Describe what happened.\n"
            "- [Optional] 'One of those': Identify the high-level category this case falls into.\n"
            "- [Optional] Principle Applied: Which principle(s) were applied, and how were they weighed?\n"
            "- [Required] Reflection: What is the user's personal reflection or learning?\n"
            "- [Required] New Principle: You should guide user to come up with a new principle to address this case.\n\n"
            "Continue asking thoughtful questions until the reflection includes at least the required elements.\n"
            "You should able to generate following information based on users' clarification.\n"
            "- Case summary\n"
            "- Case Description\n"
            "- Principle Applied\n"
            "- Detail Analysis\n"
            "- New Principle\n"
            f"**Once you are confident that the reflection is sufficiently clarified, you will trigger the function call"
            f"store_reflection_case to store the case instead of asking for permission from user **\n"
            f"** If you had triggered function call to store the case, you should response: '{CaseReflectionAgent.END_OUTPUT}'. Nothing more.**"
        )
        return p

    def _my_chat(self, msg: str) -> str:
        user_input = msg
        while True:
            if user_input != "":
                response = super().chat(user_input)
                if self.END_OUTPUT in response.response:
                    return self.END_OUTPUT
                self.print(response.response)
            user_input = input(">>")

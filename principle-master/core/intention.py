from core.common import MyAgentRunner
from core.state import AVAILABLE_FUNCTIONS, ROUTING, ENDING, get_workflow_state


class IntentionDetectionAgent(MyAgentRunner):
    ALL_STAGES = set(AVAILABLE_FUNCTIONS).union({ROUTING, ENDING})
    GREETING = "I am a principle practice helper which provide case reflection, make-a-plan, and advises function"

    @staticmethod
    def get_purpose():
        p = (
            "Your task is greeting user what kind of functions you have and identify the user's intention and determine which of the available functions they want to use.\n"
            "The available functions are:\n"
            f"{chr(10).join(list(AVAILABLE_FUNCTIONS))}\n"
            "Once you are confident about the user's intended function, respond with the exact name of the function from the list above.\n"
            "If you are not sure about user's intention, you seek clarification to users."
            "**If yuu are sure about user intention, you should respond with only the function name — no explanation, no extra words, just the function name.**"
            "**If user indicate they wanna finish chatting, output 'Ending'. ")
        return p

    def _my_chat(self, msg: str) -> str:
        while True:
            user_input = input(">>")
            if user_input == "":
                continue
            response = super().chat(user_input)
            stripped = response.response.strip()
            if stripped in self.ALL_STAGES:
                return stripped
            self.print(stripped)

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import FunctionTool

from core.common import MyAgentRunner
from core.state import get_workflow_state, WorkflowState


class ProfileUpdateAgent(MyAgentRunner):
    value_candidate = [
        "To be liked/loved",
        "To be ethically good",
        "To create something new",
        "To help others",
        "To learn/evolve",
        "To impact the world",
        "To achieve your career goals",
        "To live a peaceful life savoring the simple pleasures it has to offer",
        "To attain financial success",
        "To understand the world",
        "To have a life filled with fun and adventure",
        "To have good friends",
        "To have a thriving family"
    ]

    class Question(object):
        def __init__(self, question_key, question, formating, evaluation: str):
            self.question_key = question_key
            self.question = question
            self.formating = formating
            self.evaluation = evaluation

    @staticmethod
    def get_purpose() -> str:
        return (
            f"You are an assistant to help to analysis user's response and see whether it meet the evaluation criteria and record it with proper format. "
        )

    def __init__(self, session_id: str, verbose: bool = False):
        self.question_key = None
        self.questions = [
            ProfileUpdateAgent.Question("mbti", "What is your MBTI?", "Capitalize  strings. e.g. ENTP, INFJ",
                                        "Answer have to be legitimate MBTI type."),
            ProfileUpdateAgent.Question("key_strength", "Can you share what are the three key strength of yourself?",
                                        "Legitimate English sentence without special charactor. Three answers are separated with ';'",
                                        "Answer need to clear answer the three strength of users. Three strength are differentiate from each other."),
            ProfileUpdateAgent.Question("greatest_weakness",
                                        "Can you share what are the three key weakness of yourself?",
                                        "Legitimate English sentence without special charactor. Three answers are separated with ';'",
                                        "Clear answer the three weakness and they are differentiate from each other."),
            ProfileUpdateAgent.Question("one_big_challenge", "What is your 'One Big Challenge'?",
                                        "Legitimate English sentence, no special charactor.",
                                        "Clearly describe the One big challenge user have"),
            ProfileUpdateAgent.Question("most_appreciated_values",
                                        f"What values are most important to you?\n{"\n".join(self.value_candidate)}\nPick up to three, feel free to come up with your own",
                                        "Legitimate English Letter or short sentence. Three answers are separated with ';'",
                                        "words that clearly describe user's value preference."),
            ProfileUpdateAgent.Question("least_appreciated_values",
                                        f"What values are least important to you?\n{"\n".join(self.value_candidate)}\nPick up to three, feel free to come up with your own",
                                        "Legitimate English Letter or short sentence. Three answers are separated with ';'",
                                        "words that clearly describe  values"),
            ProfileUpdateAgent.Question("principles", "Do you have any existing principles you are operating with?",
                                        "Legitimate English sentence", "Answer are clearly express users' principle")
        ]
        state = get_workflow_state(session_id)

        def update_profile(content):
            profile = WorkflowState.Profile()
            profile.update(self.question_key, content)
            state.persist_profile(profile)
            return "Profile saved"

        store_profile = FunctionTool.from_defaults(
            fn=update_profile,
            name="update_profile",
            description="Update the user profile when the answers meet the evaluation criteria. Content parameter is been rewrote to meet the format requirement without losing or adding any information. "
        )

        super().__init__(session_id, [store_profile], verbose=verbose)

    def _my_chat(self, msg: str):
        for question in self.questions:
            self.question_key = question.question_key
            print(question.question)
            self.address_question(question)
        return "Profile updated"

    FINISH_RESPONSE = 'AnswerCollected'

    def get_system_prompt(self, question, answer, formating, evaluation: str):
        return (f"You are helpful assistant to evaluate user's answer to a question whether meet the criteria. \n"
                f"- Question: {question}\n"
                f"- Answer: {answer}\n"
                f"- Formating: {formating}\n"
                f"- Evaluation: {evaluation}\n"
                f"** If you find the answer do not met the criteria, you can response to user how can they answer it properly. **\n"
                f"** If you find the answer met the criteria, you should rewrote it according to formating requirement and update the profile **\n"
                f"** When you update the profile, your should response '{self.FINISH_RESPONSE}'. Nothing more**")

    def address_question(self, question: Question):
        answer = input(">>")
        system_prompt = self.get_system_prompt(question.question, answer, question.formating, question.evaluation)
        system_prompt_chat = ChatMessage(
            role="system",
            content=system_prompt,
        )
        chat_history = [system_prompt_chat]
        while True:
            response = self.chat("Evaluate it.", chat_history=chat_history)
            if response.response == self.FINISH_RESPONSE:
                return "Answer collected", True
            print(response.response)
            user_clarification = input(">>")
            user_chat = ChatMessage(
                role="user",
                content=user_clarification,
            )
            chat_history.append(user_chat)

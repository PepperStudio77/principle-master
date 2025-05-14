import json
import os
from datetime import datetime
from typing import Optional, List

from llama_index.core.base.llms.types import ChatMessage

type Function = str
CASE_REFLECTION: Function = "CaseReflection"
RECORD_PROFILE: Function = "RecordProfile"
ADVISE: Function = "Advice"
ROUTING: Function = "Routing"
ENDING: Function = "Ending"
JOURNAL: Function = "Journal"
_INIT_STATE = ROUTING

AVAILABLE_FUNCTIONS = {
    CASE_REFLECTION,
    RECORD_PROFILE,
    ADVISE,
    JOURNAL,
}

CURRENT_STAGE = "CURRENT_STAGE"
STAGE_META = "STAGE_META"


class Profile(object):
    def __init__(self,
                 mbti: Optional[str] = None,
                 key_strength: Optional[str] = None,
                 greatest_weakness: Optional[str] = None,
                 one_big_challenge: Optional[str] = None,
                 most_appreciated_values: Optional[str] = None,
                 least_appreciated_values: Optional[str] = None,
                 principles: Optional[str] = None):
        self.mbti = mbti
        self.key_strength = key_strength
        self.greatest_weakness = greatest_weakness
        self.one_big_challenge = one_big_challenge
        self.most_appreciated_values = most_appreciated_values
        self.least_appreciated_values = least_appreciated_values
        self.principles = principles

    def to_dict(self):
        r = {
            "mbti": self.mbti,
            "key_strength": self.key_strength,
            "greatest_weakness": self.greatest_weakness,
            "one_big_challenge": self.one_big_challenge,
            "most_appreciated_values": self.most_appreciated_values,
            "least_appreciated_values": self.least_appreciated_values,
            "principles": self.principles,
        }
        cleaned_dict = {k: v for k, v in r.items() if v is not None}
        return cleaned_dict

    def update(self, key, content):
        if not hasattr(self, key):
            raise Exception(f"Profile class do not have filed {key}")
        setattr(self, key, content)


class ReflectionCase(object):

    def __init__(self,
                 case_id: str,
                 summary: str,
                 detail: str,
                 principle_applied: str,
                 detail_analysis: str,
                 new_principle: str,
                 dialog: Optional[List[ChatMessage]] = None):
        self.case_id = case_id
        self.summary = summary
        self.detail = detail
        self.principle_applied = principle_applied
        self.detail_analysis = detail_analysis
        self.new_principle = new_principle
        self.dialog = dialog

    def to_dict(self):
        dialog = []
        if self.dialog is not None:
            for c in self.dialog:
                dialog.append(str(c))
        case = {
            "case_id": self.case_id,
            "summary": self.summary,
            "detail": self.detail,
            "principle_applied": self.principle_applied,
            "detail_analysis": self.detail_analysis,
            "new_principle": self.new_principle,
            "dialog": dialog,
        }
        return case


class JournalManager(object):
    @staticmethod
    def local_journal_dir():
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "journal")

    BASE_TEMPLATE = "template_static.md"
    AI_TEMPLATE = "template.md"

    def new_journal(self) -> str:
        """
        Create a new journal file based on the available template.
        If AI_TEMPLATE exists, use it; otherwise, fallback to BASE_TEMPLATE.
        :return: The path to the created journal file.
        """
        journal_dir = self.local_journal_dir()
        if not os.path.exists(journal_dir):
            os.makedirs(journal_dir)

        today = datetime.today().strftime('%Y-%m-%d')
        journal_file = os.path.join(journal_dir, f"journal-{today}.md")

        template_file = os.path.join(journal_dir, self.AI_TEMPLATE)

        with open(template_file, "r") as template, open(journal_file, "w") as journal:
            journal.write(template.read().format(DATE=today))

        return journal_file

    def read_template(self) -> str:
        """
        Read and return the content of the existing template.
        If AI_TEMPLATE exists, use it; otherwise, fallback to BASE_TEMPLATE.
        :return: The content of the template file.
        """
        journal_dir = self.local_journal_dir()
        template_file = os.path.join(journal_dir, self.AI_TEMPLATE)

        with open(template_file, "r") as template:
            return template.read()

    def update_template(self, content: str):
        """
        Update the AI_TEMPLATE file with the provided content.
        :param content: The content to write to the AI_TEMPLATE file.
        """
        journal_dir = self.local_journal_dir()
        ai_template_file = os.path.join(journal_dir, self.AI_TEMPLATE)

        with open(ai_template_file, "w") as template:
            template.write(content)


class ProfileManager(object):
    PROFILE_FILE = "profile.json"

    @staticmethod
    def local_store_dir():
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes")

    def persist_profile(self, profile: Profile):
        profile_dir = self.local_store_dir()
        profile_file = os.path.join(profile_dir, self.PROFILE_FILE)
        profile_dict = profile.to_dict()
        exist_profile = {}
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir)
        if os.path.exists(profile_file):
            with open(profile_file, "r+") as f:
                exist_profile = json.load(f)
        exist_profile.update(profile_dict)
        with open(profile_file, "w+") as f:
            json.dump(exist_profile, f, indent=4, sort_keys=True)
            f.write('\n')
        return "Profile updated"

    def load_profile(self):
        profile_dir = self.local_store_dir()
        profile_file = os.path.join(profile_dir, self.PROFILE_FILE)
        if not os.path.exists(profile_file):
            raise Exception("Case file do not exist.")
        with open(profile_file, "r+") as f:
            profile = json.load(f)
        return profile


class CaseManager(object):
    CASE_FILE = "cases.json"

    @staticmethod
    def local_store_dir():
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes")

    def persist_case(self, session_id: str, case: ReflectionCase):
        case_dir = self.local_store_dir()
        case_file = os.path.join(case_dir, self.CASE_FILE)
        case = case.to_dict()
        case["session_id"] = session_id
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
        if os.path.exists(case_file):
            with open(case_file, "r+") as f:
                cases = json.load(f)
        else:
            cases = []
        with open(case_file, "w+") as f:
            cases.append(case)
            json.dump(cases, f, indent=4, sort_keys=True)
            f.write('\n')
        return "Case stored"

    def load_cases(self):
        case_dir = self.local_store_dir()
        case_file = os.path.join(case_dir, self.CASE_FILE)
        if not os.path.exists(case_file):
            raise Exception("Case file do not exist.")
        with open(case_file, "r+") as f:
            cases = json.load(f)
        return cases

    def load_principle_from_cases(self):
        cases = self.load_cases()
        principles = []
        for c in cases:
            principles.append(c["new_principle"])
        return principles


class WorkflowState(CaseManager, ProfileManager, JournalManager):
    def __init__(self):
        # todo: reserve for future use
        self.state = {}


_WORKFLOW_STATE = {}


def get_workflow_state(uuid: str) -> WorkflowState:
    global _WORKFLOW_STATE
    if uuid not in _WORKFLOW_STATE:
        _WORKFLOW_STATE[uuid] = WorkflowState()
    return _WORKFLOW_STATE[uuid]

from unittest import TestCase

from core.state import get_workflow_state, WorkflowState


class TestWorkflowState(TestCase):
    def test_persist_profile(self):
        state = get_workflow_state("test")
        profile = WorkflowState.Profile(
            mbti="ENTP",
            key_strength="passionate, innovated, grit",
        )
        state.persist_profile(profile)

    def test_load_persist_profile(self):
        state = get_workflow_state("test")
        profile = state.load_profile()
        print(profile)

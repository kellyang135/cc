# neuroguide/backend/tests/test_agent.py
import pytest
from app.services.agent import NeuroGuideAgent


@pytest.fixture
def agent():
    return NeuroGuideAgent()


def test_agent_initialization(agent):
    """Test agent initializes with correct state."""
    assert agent.conversation_history == []
    assert agent.current_raw is None
    assert agent.tour_state == "idle"


def test_agent_has_system_prompt(agent):
    """Test agent has a system prompt defined."""
    assert agent.system_prompt is not None
    assert "EEG" in agent.system_prompt
    assert "guide" in agent.system_prompt.lower()

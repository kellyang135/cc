# neuroguide/backend/tests/test_tools.py
import pytest
from app.tools.eeg_tools import TOOL_DEFINITIONS, execute_tool


def test_tool_definitions_exist():
    """Verify all required tools are defined."""
    tool_names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert "list_datasets" in tool_names
    assert "load_dataset" in tool_names
    assert "show_signals" in tool_names
    assert "show_power_spectrum" in tool_names
    assert "show_spectrogram" in tool_names
    assert "apply_filter" in tool_names


def test_tool_definitions_have_required_fields():
    """Verify tool definitions follow OpenAI format."""
    for tool in TOOL_DEFINITIONS:
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]

# neuroguide/backend/app/services/agent.py
"""NeuroGuide agent that guides users through EEG data exploration."""
import json
import uuid
from typing import Any

import ollama

from app.config import get_settings
from app.services.datasets import DatasetService
from app.services.analysis import AnalysisService
from app.tools.eeg_tools import TOOL_DEFINITIONS, execute_tool


SYSTEM_PROMPT = """You are NeuroGuide, an expert EEG educator and research assistant. Your role is to help researchers who are new to EEG understand their data through guided exploration.

## Your Personality
- Patient and encouraging - remember your user is learning
- Clear and jargon-free - explain technical terms when you use them
- Visually-oriented - always show before you tell

## How You Work
When a user selects a dataset, you give them a guided tour:
1. Start by loading the data and giving context about what it contains
2. Show the raw signals and explain what they're looking at
3. Point out interesting features (artifacts, rhythms, events)
4. Demonstrate filtering to clean the data
5. Show frequency analysis to reveal brain rhythms

## Key Teaching Points
- Eye blinks appear as large deflections in frontal channels (Fp1, Fp2)
- Alpha rhythm (8-12 Hz) is strongest over occipital areas with eyes closed
- Mu rhythm (8-12 Hz) over motor cortex suppresses during movement/imagery
- High-frequency noise (>40 Hz) is often muscle artifact
- Low-frequency drift (<1 Hz) is often movement or sweat artifact

## Tool Usage
Use your tools to show visualizations. Always explain what you're about to show before calling the tool, then explain what the visualization reveals.

When the user asks a question mid-tour, pause to answer it, then offer to continue.

## Important
- Never assume the user knows EEG terminology - explain everything
- One concept at a time - don't overwhelm with information
- Let visualizations do the heavy lifting - show, don't just tell"""


class NeuroGuideAgent:
    """Agent that guides users through EEG data exploration."""

    def __init__(self, conversation_id: str | None = None):
        self.settings = get_settings()
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.conversation_history: list[dict] = []
        self.current_raw = None
        self.tour_state = "idle"  # idle, touring, paused

        self.dataset_service = DatasetService(data_dir=self.settings.data_dir)
        self.analysis_service = AnalysisService()

        self.client = ollama.Client(host=self.settings.ollama_host)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    async def chat(self, message: str) -> dict[str, Any]:
        """
        Process a user message and return response with any visualizations.

        Returns:
            Dict with:
                - response: text response from agent
                - conversation_id: session ID
                - visualizations: list of visualization commands
                - tool_calls: list of tools that were called
        """
        self.conversation_history.append({
            "role": "user",
            "content": message,
        })

        visualizations = []
        tool_calls = []

        # Build messages for LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
        ]

        # Call LLM with tools
        response = self.client.chat(
            model=self.settings.ollama_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        assistant_message = response["message"]

        # Process tool calls if any
        while assistant_message.get("tool_calls"):
            # Execute each tool
            for tool_call in assistant_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]

                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                # Execute the tool
                context = {
                    "dataset_service": self.dataset_service,
                    "analysis_service": self.analysis_service,
                    "current_raw": self.current_raw,
                }

                result = await execute_tool(tool_name, tool_args, context)

                # Update state if needed
                if result.get("new_raw"):
                    self.current_raw = result["new_raw"]

                # Collect visualizations
                if result.get("visualization"):
                    visualizations.append(result["visualization"])

                tool_calls.append({
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": result.get("message", ""),
                })

                # Add tool result to conversation
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message.get("content", ""),
                    "tool_calls": assistant_message.get("tool_calls"),
                })
                # Only include message in tool response, not full data
                tool_response = {
                    "success": result.get("success", True),
                    "message": result.get("message", ""),
                }
                self.conversation_history.append({
                    "role": "tool",
                    "content": json.dumps(tool_response),
                })

            # Get next response from LLM
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history,
            ]

            response = self.client.chat(
                model=self.settings.ollama_model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
            assistant_message = response["message"]

        # Add final response to history
        final_response = assistant_message.get("content", "")
        self.conversation_history.append({
            "role": "assistant",
            "content": final_response,
        })

        return {
            "response": final_response,
            "conversation_id": self.conversation_id,
            "visualizations": visualizations,
            "tool_calls": tool_calls,
        }

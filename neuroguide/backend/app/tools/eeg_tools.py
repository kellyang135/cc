# neuroguide/backend/app/tools/eeg_tools.py
"""EEG agent tools for the NeuroGuide assistant.

This module defines the tools available to the AI agent for EEG exploration,
following the OpenAI function calling format for Ollama compatibility.
"""
from typing import Any


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_datasets",
            "description": "List all available EEG datasets that can be loaded for exploration.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_dataset",
            "description": "Load an EEG dataset for exploration. This must be called before analyzing data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the dataset to load (e.g., 'sample_eyes', 'eeg_motor_imagery')",
                    },
                    "subject": {
                        "type": "integer",
                        "description": "Subject number to load (default: 1)",
                        "default": 1,
                    },
                    "run": {
                        "type": "integer",
                        "description": "Run number to load (default: 1)",
                        "default": 1,
                    },
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_signals",
            "description": "Display EEG signals in the visualization panel. Shows voltage over time for selected channels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of channel names to display (e.g., ['C3', 'C4', 'Cz'])",
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Start time in seconds (default: 0)",
                        "default": 0,
                    },
                    "end_time": {
                        "type": "number",
                        "description": "End time in seconds (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["channels"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_power_spectrum",
            "description": "Display the power spectrum (frequency content) of a channel. Shows which frequencies are present in the signal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel name to analyze (e.g., 'Cz')",
                    },
                },
                "required": ["channel"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_spectrogram",
            "description": "Display a spectrogram showing how frequency content changes over time. Useful for seeing brain state transitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel name to analyze (e.g., 'Cz')",
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Start time in seconds (default: 0)",
                        "default": 0,
                    },
                    "end_time": {
                        "type": "number",
                        "description": "End time in seconds (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["channel"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_filter",
            "description": "Apply a frequency filter to the data. Useful for removing noise or isolating specific brain rhythms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "low_freq": {
                        "type": "number",
                        "description": "Low cutoff frequency in Hz (removes frequencies below this)",
                    },
                    "high_freq": {
                        "type": "number",
                        "description": "High cutoff frequency in Hz (removes frequencies above this)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "highlight_region",
            "description": "Highlight a time region in the current visualization to draw attention to a specific feature.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "number",
                        "description": "Start time of region in seconds",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "End time of region in seconds",
                    },
                    "label": {
                        "type": "string",
                        "description": "Label for the highlighted region (e.g., 'blink artifact')",
                    },
                },
                "required": ["start_time", "end_time", "label"],
            },
        },
    },
]


async def execute_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute a tool and return the result.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool
        context: Context containing services and current state
            - dataset_service: DatasetService instance
            - analysis_service: AnalysisService instance
            - current_raw: Currently loaded MNE Raw object (if any)

    Returns:
        Dict with:
            - success: bool
            - result: tool output (data for visualization, etc.)
            - visualization: optional visualization command for frontend
            - message: human-readable result description
            - new_raw: optional new Raw object to update context
    """
    dataset_service = context.get("dataset_service")
    analysis_service = context.get("analysis_service")
    current_raw = context.get("current_raw")

    if tool_name == "list_datasets":
        datasets = dataset_service.list_datasets()
        return {
            "success": True,
            "result": datasets,
            "message": f"Found {len(datasets)} available datasets.",
        }

    elif tool_name == "load_dataset":
        dataset_id = tool_args["dataset_id"]
        subject = tool_args.get("subject", 1)
        run = tool_args.get("run", 1)

        raw = await dataset_service.load_dataset(dataset_id, subject, run)
        info = dataset_service.get_dataset_info(dataset_id)

        return {
            "success": True,
            "result": {
                "dataset_id": dataset_id,
                "subject": subject,
                "run": run,
                "channels": raw.info['ch_names'],
                "sample_rate": raw.info['sfreq'],
                "duration": raw.times[-1],
            },
            "new_raw": raw,  # Signal to update context
            "message": f"Loaded {info.name} - Subject {subject}, Run {run}. "
                      f"{len(raw.info['ch_names'])} channels, {raw.times[-1]:.1f} seconds.",
        }

    elif tool_name == "show_signals":
        if current_raw is None:
            return {"success": False, "message": "No dataset loaded. Use load_dataset first."}

        result = analysis_service.get_signal_segment(
            current_raw,
            channels=tool_args["channels"],
            start_time=tool_args.get("start_time", 0),
            end_time=tool_args.get("end_time", 10),
        )

        return {
            "success": True,
            "result": result,
            "visualization": {"type": "time_series", "data": result},
            "message": f"Showing {len(result['channels'])} channels from "
                      f"{tool_args.get('start_time', 0)}s to {tool_args.get('end_time', 10)}s.",
        }

    elif tool_name == "show_power_spectrum":
        if current_raw is None:
            return {"success": False, "message": "No dataset loaded. Use load_dataset first."}

        result = analysis_service.compute_power_spectrum(
            current_raw,
            channel=tool_args["channel"],
        )

        return {
            "success": True,
            "result": result,
            "visualization": {"type": "power_spectrum", "data": result},
            "message": f"Showing power spectrum for channel {tool_args['channel']}.",
        }

    elif tool_name == "show_spectrogram":
        if current_raw is None:
            return {"success": False, "message": "No dataset loaded. Use load_dataset first."}

        result = analysis_service.compute_spectrogram(
            current_raw,
            channel=tool_args["channel"],
            start_time=tool_args.get("start_time", 0),
            end_time=tool_args.get("end_time", 30),
        )

        return {
            "success": True,
            "result": result,
            "visualization": {"type": "spectrogram", "data": result},
            "message": f"Showing spectrogram for channel {tool_args['channel']}.",
        }

    elif tool_name == "apply_filter":
        if current_raw is None:
            return {"success": False, "message": "No dataset loaded. Use load_dataset first."}

        filtered_raw = analysis_service.apply_filter(
            current_raw,
            low_freq=tool_args.get("low_freq"),
            high_freq=tool_args.get("high_freq"),
        )

        filter_desc = []
        if tool_args.get("low_freq"):
            filter_desc.append(f">{tool_args['low_freq']}Hz")
        if tool_args.get("high_freq"):
            filter_desc.append(f"<{tool_args['high_freq']}Hz")

        return {
            "success": True,
            "result": {"filtered": True},
            "new_raw": filtered_raw,
            "message": f"Applied filter ({', '.join(filter_desc)}). Data is now filtered.",
        }

    elif tool_name == "highlight_region":
        return {
            "success": True,
            "result": tool_args,
            "visualization": {"type": "highlight", "data": tool_args},
            "message": f"Highlighted region: {tool_args['label']} "
                      f"({tool_args['start_time']}s - {tool_args['end_time']}s).",
        }

    else:
        return {"success": False, "message": f"Unknown tool: {tool_name}"}

# NeuroGuide: EEG Exploration Assistant

**Date:** 2026-01-16
**Status:** Design Complete

## Overview

NeuroGuide is a visual workbench that helps researchers new to EEG understand their data through AI-narrated exploration. It combines a visualization panel with a conversational AI that guides users through EEG concepts, artifacts, and analysis patterns.

### Target User

Researchers from adjacent fields (CS, engineering, materials science, etc.) who need to work with EEG data but aren't signal processing experts.

### What Makes It Different

- **Not a tool that expects expertise** - Unlike MNE or EEGLAB, you don't need to know what functions to call
- **Not just a chatbot** - Visualizations are first-class, not afterthoughts
- **Explainability is the product** - Every visualization comes with "here's what you're seeing and why it matters"

### HCI Focus

- Explainable AI: Making EEG analysis understandable
- Human-AI collaboration: How do narration and visualization work together?
- Pacing and interruption: When should the AI pause vs. continue?

## User Experience Flow

### Starting a Session

User arrives at a split interface: visualization panel (left/top) and chat panel (right/bottom). Welcome message offers dataset choices:

- BCI Competition IV - Motor Imagery (4 classes, 9 subjects)
- PhysioNet EEG Motor Movement (64 channels, 109 subjects)
- Sample: Eyes Open vs Closed

### The Guided Tour

Once a dataset is selected, the AI takes over:

1. **Context setting** - Explains what the data is and where it came from
2. **Raw signal overview** - Shows multi-channel time series, explains the noise
3. **Pointing out features** - Highlights artifacts, events with visual overlays
4. **Building understanding** - Demonstrates filtering, shows before/after

### User Interruption

At any point, user can type a question. AI pauses, answers, and offers to continue or go deeper.

## Architecture

### Frontend (React + TypeScript)

- **Visualization Panel** - Plotly.js for time series, spectrograms, power spectra
- **Chat Panel** - Markdown-supported narration display, user input
- **State sync** - Agent commands update visualizations in real-time

### Backend (FastAPI + Python)

- **Agent service** - Orchestrates narration flow, responds to questions
- **Analysis service** - Wraps MNE-Python for EEG operations
- **Dataset service** - Manages public datasets, handles downloads/caching

### Agent (Ollama / Local LLM)

Tools available to the agent:

| Tool | Description |
|------|-------------|
| `load_dataset(name)` | Load a specific dataset |
| `show_signals(channels, time_range)` | Update visualization with raw/filtered signals |
| `highlight_region(start, end, label)` | Draw attention to a specific time window |
| `show_spectrogram(channel)` | Display frequency content over time |
| `show_power_spectrum(channel)` | Display frequency content |
| `apply_filter(low, high)` | Apply bandpass filter and show result |
| `explain(topic)` | Retrieve explanation for EEG concepts |

Agent maintains "tour state" - knows where it is in the narrative and what the user has seen.

## Visualizations

### MVP Visualization Types

1. **Multi-channel time series** - Voltage over time for multiple electrodes, with zoom/scroll and highlight overlays
2. **Single-channel detail view** - Zoomed-in view of one electrode
3. **Power spectrum** - Frequency content (amplitude vs. Hz)
4. **Spectrogram** - Frequency content over time (heatmap)

### Design Principle

One plot at a time by default. Clean, clear labels. The AI tells you what to look at rather than showing everything at once.

### Deferred

- Topographic maps (scalp heatmaps)
- 3D head models
- Interactive channel selection on head diagram

## Datasets

### MVP Datasets

1. **PhysioNet EEG Motor Movement/Imagery Dataset**
   - Free, well-documented, widely used
   - 109 subjects, 64 channels
   - Tasks: motor execution and motor imagery
   - Format: EDF files (MNE loads directly)

2. **BCI Competition IV Dataset 2a**
   - Classic benchmark, 9 subjects, 22 channels, 4 classes
   - Good for explaining motor imagery classification

3. **Eyes Open vs Closed sample**
   - Simple, clear alpha rhythm difference
   - Great for first-time users

### Data Management

- Pre-download and cache on first use
- Store in `data/` directory
- Agent knows available datasets and their contents

## MVP Scope

### In Scope

- Visual workbench with chat + visualization panels
- 2-3 public datasets pre-loaded
- AI-narrated tour covering:
  - Data context and origin
  - Raw signal exploration
  - Artifact identification (blinks, muscle, noise)
  - Basic filtering (before/after)
  - Frequency content (alpha, beta, mu)
- User interruption and questions anytime
- 4 visualization types

### Out of Scope (Future)

- File upload for custom data
- Full preprocessing pipeline builder
- Classification/ML workflows
- Topographic maps
- Real-time streaming
- User accounts or saved sessions
- Multiple LLM backends

## Success Criteria

A researcher with no EEG background can load the motor imagery dataset and, after a 10-minute guided tour, correctly identify:

- What an eye blink artifact looks like
- Where to look for motor-related activity
- What the alpha band is and how to see it

## Technical Stack

| Layer | Technology |
|-------|------------|
| Frontend | React, TypeScript, Plotly.js, Vite |
| Backend | FastAPI, Python, MNE-Python |
| AI | Ollama (local LLM) |
| Data | PhysioNet, BCI Competition datasets |

## Resume Positioning

This project demonstrates:

- **AI/ML Engineering** - Building agentic workflows with tool use
- **HCI** - Designing human-AI collaboration and explainable interfaces
- **Full-stack development** - React + FastAPI + data pipelines
- **Domain application** - Biomedical signal processing

Differentiates from MatExplorer by focusing on AI-driven guidance rather than search/visualization tools.

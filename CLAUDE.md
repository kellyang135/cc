# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This monorepo contains two AI-powered research applications:
- **MatAssistant** - Semiconductor materials research agent using Materials Project API
- **NeuroGuide** - EEG exploration assistant with AI-narrated tours

Both projects follow the same architecture: FastAPI backend + React/TypeScript frontend.

## Common Commands

### Backend (Python)

```bash
# Setup (from either matassistant/backend or neuroguide/backend)
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run server
uvicorn app.main:app --reload

# Run tests
pytest -v

# Run single test file
pytest tests/test_datasets.py -v

# Run single test
pytest tests/test_datasets.py::test_get_available_datasets -v

# Lint
ruff check app/

# Format
ruff format app/
```

### Frontend (React/TypeScript)

```bash
# Setup (from either matassistant/frontend or neuroguide/frontend)
npm install

# Run dev server
npm run dev

# Build
npm run build

# Lint
npm run lint
```

## Architecture

### Backend Structure (both projects)

```
backend/app/
├── main.py          # FastAPI app, CORS config, router includes
├── config.py        # Settings via pydantic-settings, loads .env
├── routers/         # API endpoint definitions
├── services/        # Business logic layer
└── tools/           # Agent tool definitions (Ollama-compatible format)
```

**Patterns:**
- Config uses `@lru_cache` singleton pattern
- Services are classes instantiated per-request or as singletons
- Tools define function schemas for LLM function calling
- CORS allows `http://localhost:5173` (Vite dev server)

### Frontend Structure

Both frontends use Vite + React 19 + TypeScript. MatAssistant uses Three.js for 3D crystal visualization; NeuroGuide uses Plotly.js for EEG signal plots.

### Key Services

**MatAssistant:**
- `MaterialsAgent` - Orchestrates LLM with materials search tools
- `MaterialsDatabase` - Materials Project API client

**NeuroGuide:**
- `DatasetService` - Lists/loads PhysioNet EEG datasets via MNE
- `AnalysisService` - Signal filtering, power spectrum, spectrogram

## Environment Variables

**MatAssistant** requires `MP_API_KEY` (Materials Project API key).

**Both projects** can configure:
- `OLLAMA_HOST` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `llama3.1:8b`)

Copy `.env.example` to `.env` in each backend directory.

## Prerequisites

- Python 3.11+
- Node.js/npm
- Ollama running locally with a model pulled (e.g., `ollama pull llama3.1:8b`)

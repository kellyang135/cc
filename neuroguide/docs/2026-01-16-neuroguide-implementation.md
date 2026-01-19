# NeuroGuide Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an EEG exploration assistant that helps researchers new to EEG understand their data through AI-narrated tours with synchronized visualizations.

**Architecture:** Split-panel web app (React + Plotly.js frontend, FastAPI + MNE-Python backend) with an Ollama-powered agent that orchestrates guided tours through public EEG datasets. Agent tools update visualizations in real-time while explaining concepts.

**Tech Stack:** FastAPI, MNE-Python, Ollama, React, TypeScript, Plotly.js, Vite

---

## Task 1: Project Structure Setup

**Files:**
- Create: `neuroguide/backend/app/__init__.py`
- Create: `neuroguide/backend/app/main.py`
- Create: `neuroguide/backend/app/config.py`
- Create: `neuroguide/backend/pyproject.toml`
- Create: `neuroguide/backend/.env.example`

**Step 1: Create directory structure**

```bash
mkdir -p neuroguide/backend/app/routers
mkdir -p neuroguide/backend/app/services
mkdir -p neuroguide/backend/app/tools
mkdir -p neuroguide/backend/tests
mkdir -p neuroguide/frontend/src/components
mkdir -p neuroguide/data
touch neuroguide/backend/app/__init__.py
touch neuroguide/backend/app/routers/__init__.py
touch neuroguide/backend/app/services/__init__.py
touch neuroguide/backend/app/tools/__init__.py
```

**Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "neuroguide"
version = "0.1.0"
description = "EEG exploration assistant with AI-narrated tours"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "httpx>=0.26.0",
    "ollama>=0.4.0",
    "python-dotenv>=1.0.0",
    "mne>=1.6.0",
    "numpy>=1.26.0",
    "scipy>=1.12.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.2.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
```

**Step 3: Create config.py**

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Data settings
    data_dir: str = "../data"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Step 4: Create main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="NeuroGuide",
    description="EEG exploration assistant with AI-narrated tours",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Welcome to NeuroGuide"}
```

**Step 5: Create .env.example**

```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
DATA_DIR=../data
```

**Step 6: Verify backend runs**

```bash
cd neuroguide/backend
pip install -e ".[dev]"
uvicorn app.main:app --reload
# Expected: Server starts on http://localhost:8000
# Visit http://localhost:8000/health - should return {"status": "healthy"}
```

**Step 7: Commit**

```bash
git add neuroguide/
git commit -m "feat: initialize neuroguide project structure with FastAPI backend"
```

---

## Task 2: Frontend Setup

**Files:**
- Create: `neuroguide/frontend/package.json`
- Create: `neuroguide/frontend/vite.config.ts`
- Create: `neuroguide/frontend/tsconfig.json`
- Create: `neuroguide/frontend/index.html`
- Create: `neuroguide/frontend/src/main.tsx`
- Create: `neuroguide/frontend/src/App.tsx`

**Step 1: Create package.json**

```json
{
  "name": "neuroguide-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "lint": "eslint ."
  },
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "plotly.js": "^2.35.0",
    "react-plotly.js": "^2.6.0"
  },
  "devDependencies": {
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@types/react-plotly.js": "^2.6.3",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "~5.6.0",
    "vite": "^6.0.0",
    "eslint": "^9.0.0"
  }
}
```

**Step 2: Create vite.config.ts**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})
```

**Step 3: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

**Step 4: Create index.html**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>NeuroGuide - EEG Exploration Assistant</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

**Step 5: Create main.tsx**

```typescript
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

**Step 6: Create App.tsx (placeholder)**

```typescript
function App() {
  return (
    <div style={{ padding: '20px' }}>
      <h1>NeuroGuide</h1>
      <p>EEG Exploration Assistant - Coming Soon</p>
    </div>
  )
}

export default App
```

**Step 7: Create index.css**

```css
:root {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.5;
  font-weight: 400;
  color: #213547;
  background-color: #f8f9fa;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  min-height: 100vh;
}
```

**Step 8: Install and verify**

```bash
cd neuroguide/frontend
npm install
npm run dev
# Expected: Vite dev server starts on http://localhost:5173
# Page shows "NeuroGuide" header
```

**Step 9: Commit**

```bash
git add neuroguide/frontend/
git commit -m "feat: add React frontend with Vite and Plotly.js"
```

---

## Task 3: Dataset Service

**Files:**
- Create: `neuroguide/backend/app/services/datasets.py`
- Create: `neuroguide/backend/tests/test_datasets.py`

**Step 1: Write the failing test**

```python
# neuroguide/backend/tests/test_datasets.py
import pytest
from app.services.datasets import DatasetService


@pytest.fixture
def dataset_service():
    return DatasetService(data_dir="./test_data")


def test_list_available_datasets(dataset_service):
    datasets = dataset_service.list_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    assert "eeg_motor_imagery" in [d["id"] for d in datasets]


def test_dataset_has_required_fields(dataset_service):
    datasets = dataset_service.list_datasets()
    for dataset in datasets:
        assert "id" in dataset
        assert "name" in dataset
        assert "description" in dataset
        assert "subjects" in dataset
        assert "channels" in dataset
```

**Step 2: Run test to verify it fails**

```bash
cd neuroguide/backend
pytest tests/test_datasets.py -v
# Expected: FAIL - ModuleNotFoundError: No module named 'app.services.datasets'
```

**Step 3: Write the implementation**

```python
# neuroguide/backend/app/services/datasets.py
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    id: str
    name: str
    description: str
    subjects: int
    channels: int
    sample_rate: float
    tasks: list[str]
    source: str
    download_url: str | None = None


AVAILABLE_DATASETS = [
    DatasetInfo(
        id="eeg_motor_imagery",
        name="PhysioNet EEG Motor Movement/Imagery",
        description="EEG recordings of subjects performing motor execution and motor imagery tasks. "
                    "Subjects opened/closed fists or imagined opening/closing fists.",
        subjects=109,
        channels=64,
        sample_rate=160.0,
        tasks=["rest", "left_fist", "right_fist", "both_fists", "both_feet"],
        source="PhysioNet",
        download_url="https://physionet.org/content/eegmmidb/1.0.0/",
    ),
    DatasetInfo(
        id="sample_eyes",
        name="Eyes Open vs Closed Sample",
        description="Sample dataset showing clear alpha rhythm differences between eyes open "
                    "and eyes closed states. Great for learning about frequency analysis.",
        subjects=1,
        channels=64,
        sample_rate=160.0,
        tasks=["eyes_open", "eyes_closed"],
        source="MNE Sample",
    ),
]


class DatasetService:
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> list[dict]:
        """Return list of available datasets with metadata."""
        return [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "subjects": d.subjects,
                "channels": d.channels,
                "sample_rate": d.sample_rate,
                "tasks": d.tasks,
                "source": d.source,
            }
            for d in AVAILABLE_DATASETS
        ]

    def get_dataset_info(self, dataset_id: str) -> DatasetInfo | None:
        """Get info for a specific dataset."""
        for d in AVAILABLE_DATASETS:
            if d.id == dataset_id:
                return d
        return None

    def is_downloaded(self, dataset_id: str) -> bool:
        """Check if dataset is already downloaded."""
        dataset_path = self.data_dir / dataset_id
        return dataset_path.exists() and any(dataset_path.iterdir())
```

**Step 4: Run test to verify it passes**

```bash
cd neuroguide/backend
pytest tests/test_datasets.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add neuroguide/backend/app/services/datasets.py neuroguide/backend/tests/test_datasets.py
git commit -m "feat: add dataset service with available dataset listing"
```

---

## Task 4: Dataset Download & Loading

**Files:**
- Modify: `neuroguide/backend/app/services/datasets.py`
- Create: `neuroguide/backend/tests/test_dataset_loading.py`

**Step 1: Write the failing test**

```python
# neuroguide/backend/tests/test_dataset_loading.py
import pytest
from app.services.datasets import DatasetService


@pytest.fixture
def dataset_service():
    return DatasetService(data_dir="./test_data")


@pytest.mark.asyncio
async def test_load_sample_dataset(dataset_service):
    """Test loading the built-in sample dataset."""
    raw = await dataset_service.load_dataset("sample_eyes", subject=1, run=1)
    assert raw is not None
    assert hasattr(raw, 'info')
    assert raw.info['nchan'] > 0


@pytest.mark.asyncio
async def test_get_raw_data_array(dataset_service):
    """Test extracting numpy array from raw data."""
    raw = await dataset_service.load_dataset("sample_eyes", subject=1, run=1)
    data, times = raw.get_data(return_times=True)
    assert data.shape[0] > 0  # channels
    assert data.shape[1] > 0  # samples
    assert len(times) == data.shape[1]
```

**Step 2: Run test to verify it fails**

```bash
cd neuroguide/backend
pytest tests/test_dataset_loading.py -v
# Expected: FAIL - AttributeError: 'DatasetService' object has no attribute 'load_dataset'
```

**Step 3: Add load_dataset method**

```python
# Add to neuroguide/backend/app/services/datasets.py

import asyncio
import mne


class DatasetService:
    # ... existing code ...

    async def load_dataset(
        self,
        dataset_id: str,
        subject: int = 1,
        run: int = 1
    ) -> mne.io.Raw:
        """Load a dataset and return MNE Raw object."""
        if dataset_id == "sample_eyes":
            return await self._load_sample_eyes(subject, run)
        elif dataset_id == "eeg_motor_imagery":
            return await self._load_motor_imagery(subject, run)
        else:
            raise ValueError(f"Unknown dataset: {dataset_id}")

    async def _load_sample_eyes(self, subject: int, run: int) -> mne.io.Raw:
        """Load eyes open/closed sample using MNE's eegbci dataset."""
        # MNE has built-in access to PhysioNet data
        def _load():
            # Run 1 = eyes open, Run 2 = eyes closed
            raw_fnames = mne.datasets.eegbci.load_data(subject, runs=[run])
            raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)
            mne.datasets.eegbci.standardize(raw)
            return raw

        return await asyncio.to_thread(_load)

    async def _load_motor_imagery(self, subject: int, run: int) -> mne.io.Raw:
        """Load motor imagery dataset from PhysioNet."""
        def _load():
            # Runs 4, 8, 12 = motor imagery (left vs right fist)
            imagery_runs = [4, 8, 12]
            run_idx = min(run - 1, len(imagery_runs) - 1)
            raw_fnames = mne.datasets.eegbci.load_data(subject, runs=[imagery_runs[run_idx]])
            raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)
            mne.datasets.eegbci.standardize(raw)
            return raw

        return await asyncio.to_thread(_load)

    def get_channel_names(self, raw: mne.io.Raw) -> list[str]:
        """Get list of channel names from Raw object."""
        return raw.info['ch_names']

    def get_sample_rate(self, raw: mne.io.Raw) -> float:
        """Get sampling frequency from Raw object."""
        return raw.info['sfreq']
```

**Step 4: Run test to verify it passes**

```bash
cd neuroguide/backend
pytest tests/test_dataset_loading.py -v
# Expected: PASS (may take a moment to download sample data on first run)
```

**Step 5: Commit**

```bash
git add neuroguide/backend/app/services/datasets.py neuroguide/backend/tests/test_dataset_loading.py
git commit -m "feat: add dataset loading with MNE for PhysioNet EEG data"
```

---

## Task 5: Analysis Service - Signal Processing

**Files:**
- Create: `neuroguide/backend/app/services/analysis.py`
- Create: `neuroguide/backend/tests/test_analysis.py`

**Step 1: Write the failing test**

```python
# neuroguide/backend/tests/test_analysis.py
import pytest
import numpy as np
from app.services.datasets import DatasetService
from app.services.analysis import AnalysisService


@pytest.fixture
def dataset_service():
    return DatasetService(data_dir="./test_data")


@pytest.fixture
def analysis_service():
    return AnalysisService()


@pytest.mark.asyncio
async def test_get_signal_segment(dataset_service, analysis_service):
    """Test extracting a time segment from raw data."""
    raw = await dataset_service.load_dataset("sample_eyes", subject=1, run=1)

    result = analysis_service.get_signal_segment(
        raw,
        channels=["C3", "C4", "Cz"],
        start_time=0.0,
        end_time=5.0
    )

    assert "data" in result
    assert "times" in result
    assert "channels" in result
    assert len(result["channels"]) == 3
    assert result["data"].shape[0] == 3  # 3 channels
    assert result["times"][-1] <= 5.0


@pytest.mark.asyncio
async def test_apply_bandpass_filter(dataset_service, analysis_service):
    """Test applying bandpass filter."""
    raw = await dataset_service.load_dataset("sample_eyes", subject=1, run=1)

    filtered_raw = analysis_service.apply_filter(raw, low_freq=1.0, high_freq=40.0)

    assert filtered_raw is not None
    assert filtered_raw.info['nchan'] == raw.info['nchan']
```

**Step 2: Run test to verify it fails**

```bash
cd neuroguide/backend
pytest tests/test_analysis.py -v
# Expected: FAIL - ModuleNotFoundError: No module named 'app.services.analysis'
```

**Step 3: Write the implementation**

```python
# neuroguide/backend/app/services/analysis.py
import numpy as np
import mne
from scipy import signal


class AnalysisService:
    """Service for EEG signal analysis operations."""

    def get_signal_segment(
        self,
        raw: mne.io.Raw,
        channels: list[str] | None = None,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> dict:
        """
        Extract a time segment of signal data.

        Returns dict with:
            - data: numpy array (channels x samples)
            - times: numpy array of time points
            - channels: list of channel names
            - sample_rate: sampling frequency
        """
        if channels is None:
            channels = raw.info['ch_names'][:10]  # Default to first 10

        # Ensure channels exist
        available = set(raw.info['ch_names'])
        channels = [ch for ch in channels if ch in available]

        if not channels:
            raise ValueError("No valid channels specified")

        if end_time is None:
            end_time = raw.times[-1]

        # Get data for specified channels and time range
        picks = mne.pick_channels(raw.info['ch_names'], include=channels)
        start_sample = int(start_time * raw.info['sfreq'])
        end_sample = int(end_time * raw.info['sfreq'])

        data = raw.get_data(picks=picks, start=start_sample, stop=end_sample)
        times = raw.times[start_sample:end_sample]

        return {
            "data": data.tolist(),  # Convert to list for JSON serialization
            "times": times.tolist(),
            "channels": channels,
            "sample_rate": raw.info['sfreq'],
        }

    def apply_filter(
        self,
        raw: mne.io.Raw,
        low_freq: float | None = None,
        high_freq: float | None = None,
    ) -> mne.io.Raw:
        """Apply bandpass filter to raw data."""
        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=low_freq, h_freq=high_freq, verbose=False)
        return raw_filtered

    def compute_power_spectrum(
        self,
        raw: mne.io.Raw,
        channel: str,
        fmin: float = 0.5,
        fmax: float = 50.0,
    ) -> dict:
        """
        Compute power spectral density for a channel.

        Returns dict with:
            - frequencies: array of frequency values
            - power: array of power values
            - channel: channel name
        """
        picks = mne.pick_channels(raw.info['ch_names'], include=[channel])
        if len(picks) == 0:
            raise ValueError(f"Channel {channel} not found")

        spectrum = raw.compute_psd(
            method='welch',
            picks=picks,
            fmin=fmin,
            fmax=fmax,
            verbose=False
        )

        freqs = spectrum.freqs
        power = spectrum.get_data()[0]  # First (and only) channel

        return {
            "frequencies": freqs.tolist(),
            "power": power.tolist(),
            "channel": channel,
        }

    def compute_spectrogram(
        self,
        raw: mne.io.Raw,
        channel: str,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> dict:
        """
        Compute spectrogram (time-frequency representation) for a channel.

        Returns dict with:
            - times: array of time values
            - frequencies: array of frequency values
            - power: 2D array (frequencies x times)
            - channel: channel name
        """
        picks = mne.pick_channels(raw.info['ch_names'], include=[channel])
        if len(picks) == 0:
            raise ValueError(f"Channel {channel} not found")

        if end_time is None:
            end_time = raw.times[-1]

        # Get data segment
        start_sample = int(start_time * raw.info['sfreq'])
        end_sample = int(end_time * raw.info['sfreq'])
        data = raw.get_data(picks=picks, start=start_sample, stop=end_sample)[0]

        # Compute spectrogram using scipy
        fs = raw.info['sfreq']
        nperseg = int(fs * 2)  # 2 second windows
        noverlap = int(nperseg * 0.75)  # 75% overlap

        f, t, Sxx = signal.spectrogram(
            data,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density'
        )

        # Limit frequency range to 0-50 Hz
        freq_mask = f <= 50

        return {
            "times": (t + start_time).tolist(),
            "frequencies": f[freq_mask].tolist(),
            "power": Sxx[freq_mask, :].tolist(),
            "channel": channel,
        }
```

**Step 4: Run test to verify it passes**

```bash
cd neuroguide/backend
pytest tests/test_analysis.py -v
# Expected: PASS
```

**Step 5: Add tests for power spectrum and spectrogram**

```python
# Add to neuroguide/backend/tests/test_analysis.py

@pytest.mark.asyncio
async def test_compute_power_spectrum(dataset_service, analysis_service):
    """Test computing power spectral density."""
    raw = await dataset_service.load_dataset("sample_eyes", subject=1, run=1)

    result = analysis_service.compute_power_spectrum(raw, channel="Cz")

    assert "frequencies" in result
    assert "power" in result
    assert len(result["frequencies"]) == len(result["power"])
    assert all(f >= 0 for f in result["frequencies"])


@pytest.mark.asyncio
async def test_compute_spectrogram(dataset_service, analysis_service):
    """Test computing spectrogram."""
    raw = await dataset_service.load_dataset("sample_eyes", subject=1, run=1)

    result = analysis_service.compute_spectrogram(
        raw,
        channel="Cz",
        start_time=0.0,
        end_time=10.0
    )

    assert "times" in result
    assert "frequencies" in result
    assert "power" in result
    assert len(result["power"]) == len(result["frequencies"])
```

**Step 6: Run all tests**

```bash
cd neuroguide/backend
pytest tests/test_analysis.py -v
# Expected: All PASS
```

**Step 7: Commit**

```bash
git add neuroguide/backend/app/services/analysis.py neuroguide/backend/tests/test_analysis.py
git commit -m "feat: add analysis service with filtering, power spectrum, and spectrogram"
```

---

## Task 6: Agent Tools Definition

**Files:**
- Create: `neuroguide/backend/app/tools/eeg_tools.py`
- Create: `neuroguide/backend/tests/test_tools.py`

**Step 1: Write the failing test**

```python
# neuroguide/backend/tests/test_tools.py
import pytest
from app.tools.eeg_tools import TOOL_DEFINITIONS, execute_tool
from app.services.datasets import DatasetService
from app.services.analysis import AnalysisService


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
```

**Step 2: Run test to verify it fails**

```bash
cd neuroguide/backend
pytest tests/test_tools.py -v
# Expected: FAIL - ModuleNotFoundError
```

**Step 3: Write tool definitions**

```python
# neuroguide/backend/app/tools/eeg_tools.py
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
```

**Step 4: Run test to verify it passes**

```bash
cd neuroguide/backend
pytest tests/test_tools.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add neuroguide/backend/app/tools/eeg_tools.py neuroguide/backend/tests/test_tools.py
git commit -m "feat: add EEG agent tools with execute_tool function"
```

---

## Task 7: Agent Service

**Files:**
- Create: `neuroguide/backend/app/services/agent.py`
- Create: `neuroguide/backend/tests/test_agent.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

```bash
cd neuroguide/backend
pytest tests/test_agent.py -v
# Expected: FAIL - ModuleNotFoundError
```

**Step 3: Write the agent implementation**

```python
# neuroguide/backend/app/services/agent.py
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
                self.conversation_history.append({
                    "role": "tool",
                    "content": json.dumps(result),
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
```

**Step 4: Run test to verify it passes**

```bash
cd neuroguide/backend
pytest tests/test_agent.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add neuroguide/backend/app/services/agent.py neuroguide/backend/tests/test_agent.py
git commit -m "feat: add NeuroGuide agent with tour system prompt and tool execution"
```

---

## Task 8: Chat API Router

**Files:**
- Create: `neuroguide/backend/app/routers/chat.py`
- Modify: `neuroguide/backend/app/main.py`

**Step 1: Create the chat router**

```python
# neuroguide/backend/app/routers/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.agent import NeuroGuideAgent

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Store active agents by conversation_id
active_agents: dict[str, NeuroGuideAgent] = {}


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class VisualizationCommand(BaseModel):
    type: str
    data: dict


class ToolCallInfo(BaseModel):
    name: str
    arguments: dict
    result: str


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    visualizations: list[VisualizationCommand]
    tool_calls: list[ToolCallInfo]


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the NeuroGuide agent."""
    try:
        # Get or create agent for this conversation
        if request.conversation_id and request.conversation_id in active_agents:
            agent = active_agents[request.conversation_id]
        else:
            agent = NeuroGuideAgent(conversation_id=request.conversation_id)
            active_agents[agent.conversation_id] = agent

        # Process message
        result = await agent.chat(request.message)

        return ChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            visualizations=[
                VisualizationCommand(type=v["type"], data=v["data"])
                for v in result["visualizations"]
            ],
            tool_calls=[
                ToolCallInfo(name=tc["name"], arguments=tc["arguments"], result=tc["result"])
                for tc in result["tool_calls"]
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def list_datasets():
    """List available datasets."""
    from app.services.datasets import DatasetService
    service = DatasetService()
    return service.list_datasets()
```

**Step 2: Update main.py to include router**

```python
# neuroguide/backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat

app = FastAPI(
    title="NeuroGuide",
    description="EEG exploration assistant with AI-narrated tours",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Welcome to NeuroGuide"}
```

**Step 3: Verify API works**

```bash
cd neuroguide/backend
uvicorn app.main:app --reload
# In another terminal:
curl http://localhost:8000/api/chat/datasets
# Expected: JSON array of datasets
```

**Step 4: Commit**

```bash
git add neuroguide/backend/app/routers/chat.py neuroguide/backend/app/main.py
git commit -m "feat: add chat API router with conversation management"
```

---

## Task 9: Frontend Layout Component

**Files:**
- Create: `neuroguide/frontend/src/components/Layout.tsx`
- Create: `neuroguide/frontend/src/components/Layout.css`
- Modify: `neuroguide/frontend/src/App.tsx`

**Step 1: Create Layout component**

```typescript
// neuroguide/frontend/src/components/Layout.tsx
import './Layout.css'

interface LayoutProps {
  visualization: React.ReactNode
  chat: React.ReactNode
}

export function Layout({ visualization, chat }: LayoutProps) {
  return (
    <div className="layout">
      <header className="header">
        <h1>NeuroGuide</h1>
        <p>EEG Exploration Assistant</p>
      </header>
      <main className="main">
        <div className="viz-panel">
          {visualization}
        </div>
        <div className="chat-panel">
          {chat}
        </div>
      </main>
    </div>
  )
}
```

**Step 2: Create Layout CSS**

```css
/* neuroguide/frontend/src/components/Layout.css */
.layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #f8f9fa;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem 2rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.header h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
}

.header p {
  margin: 0.25rem 0 0;
  font-size: 0.875rem;
  opacity: 0.9;
}

.main {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.viz-panel {
  flex: 1;
  padding: 1rem;
  overflow: auto;
  background: white;
  border-right: 1px solid #e9ecef;
}

.chat-panel {
  width: 400px;
  min-width: 350px;
  max-width: 500px;
  display: flex;
  flex-direction: column;
  background: #f8f9fa;
}

@media (max-width: 900px) {
  .main {
    flex-direction: column;
  }

  .viz-panel {
    flex: none;
    height: 50%;
    border-right: none;
    border-bottom: 1px solid #e9ecef;
  }

  .chat-panel {
    width: 100%;
    max-width: none;
    flex: 1;
  }
}
```

**Step 3: Update App.tsx to use Layout**

```typescript
// neuroguide/frontend/src/App.tsx
import { Layout } from './components/Layout'
import './index.css'

function App() {
  return (
    <Layout
      visualization={
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          color: '#6c757d'
        }}>
          Select a dataset to begin exploring
        </div>
      }
      chat={
        <div style={{ padding: '1rem' }}>
          Chat panel coming soon...
        </div>
      }
    />
  )
}

export default App
```

**Step 4: Verify layout works**

```bash
cd neuroguide/frontend
npm run dev
# Expected: Split panel layout with header, viz panel on left, chat on right
```

**Step 5: Commit**

```bash
git add neuroguide/frontend/src/components/ neuroguide/frontend/src/App.tsx
git commit -m "feat: add split-panel layout component"
```

---

## Task 10: Chat Panel Component

**Files:**
- Create: `neuroguide/frontend/src/components/ChatPanel.tsx`
- Create: `neuroguide/frontend/src/components/ChatPanel.css`

**Step 1: Create ChatPanel component**

```typescript
// neuroguide/frontend/src/components/ChatPanel.tsx
import { useState, useRef, useEffect } from 'react'
import './ChatPanel.css'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}

interface ToolCall {
  name: string
  arguments: Record<string, unknown>
  result: string
}

interface ChatPanelProps {
  onVisualization?: (viz: { type: string; data: unknown }) => void
}

export function ChatPanel({ onVisualization }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Welcome to NeuroGuide! I'm here to help you explore and understand EEG data.\n\nWould you like me to guide you through a dataset? Just say something like \"Let's explore the motor imagery dataset\" or \"Show me the eyes open/closed data\".",
    },
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/api/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          conversation_id: conversationId,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()

      setConversationId(data.conversation_id)

      // Handle visualizations
      if (data.visualizations && onVisualization) {
        for (const viz of data.visualizations) {
          onVisualization(viz)
        }
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error('Chat error:', error)
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please make sure the backend server is running.',
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
        {isLoading && (
          <div className="message assistant">
            <div className="message-content loading">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about EEG data..."
          disabled={isLoading}
          rows={2}
        />
        <button onClick={sendMessage} disabled={isLoading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  )
}
```

**Step 2: Create ChatPanel CSS**

```css
/* neuroguide/frontend/src/components/ChatPanel.css */
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.message {
  max-width: 90%;
  padding: 0.75rem 1rem;
  border-radius: 1rem;
  line-height: 1.5;
}

.message.user {
  align-self: flex-end;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-bottom-right-radius: 0.25rem;
}

.message.assistant {
  align-self: flex-start;
  background: white;
  color: #213547;
  border-bottom-left-radius: 0.25rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.message-content {
  white-space: pre-wrap;
  word-wrap: break-word;
}

.message-content.loading {
  display: flex;
  gap: 0.25rem;
  padding: 0.25rem 0;
}

.dot {
  width: 8px;
  height: 8px;
  background: #667eea;
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out both;
}

.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }
.dot:nth-child(3) { animation-delay: 0s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

.input-area {
  padding: 1rem;
  background: white;
  border-top: 1px solid #e9ecef;
  display: flex;
  gap: 0.5rem;
}

.input-area textarea {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #dee2e6;
  border-radius: 0.5rem;
  resize: none;
  font-family: inherit;
  font-size: 0.875rem;
}

.input-area textarea:focus {
  outline: none;
  border-color: #667eea;
}

.input-area button {
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 0.5rem;
  font-weight: 500;
  cursor: pointer;
  transition: opacity 0.2s;
}

.input-area button:hover:not(:disabled) {
  opacity: 0.9;
}

.input-area button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

**Step 3: Commit**

```bash
git add neuroguide/frontend/src/components/ChatPanel.tsx neuroguide/frontend/src/components/ChatPanel.css
git commit -m "feat: add chat panel component with message handling"
```

---

## Task 11: Time Series Visualization Component

**Files:**
- Create: `neuroguide/frontend/src/components/TimeSeriesPlot.tsx`

**Step 1: Create TimeSeriesPlot component**

```typescript
// neuroguide/frontend/src/components/TimeSeriesPlot.tsx
import Plot from 'react-plotly.js'
import { Data, Layout } from 'plotly.js'

interface TimeSeriesData {
  data: number[][]  // channels x samples
  times: number[]
  channels: string[]
  sample_rate: number
}

interface Highlight {
  start_time: number
  end_time: number
  label: string
}

interface TimeSeriesPlotProps {
  data: TimeSeriesData | null
  highlights?: Highlight[]
}

export function TimeSeriesPlot({ data, highlights = [] }: TimeSeriesPlotProps) {
  if (!data) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        color: '#6c757d'
      }}>
        No signal data to display
      </div>
    )
  }

  // Create traces for each channel with vertical offset
  const traces: Data[] = data.channels.map((channel, i) => {
    // Normalize and offset each channel for display
    const channelData = data.data[i]
    const mean = channelData.reduce((a, b) => a + b, 0) / channelData.length
    const std = Math.sqrt(
      channelData.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / channelData.length
    )
    const normalized = channelData.map((v) => (v - mean) / (std || 1))
    const offset = (data.channels.length - 1 - i) * 3  // Offset each channel

    return {
      x: data.times,
      y: normalized.map((v) => v + offset),
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: channel,
      line: { width: 1 },
    }
  })

  // Add highlight shapes
  const shapes = highlights.map((h) => ({
    type: 'rect' as const,
    xref: 'x' as const,
    yref: 'paper' as const,
    x0: h.start_time,
    x1: h.end_time,
    y0: 0,
    y1: 1,
    fillcolor: 'rgba(255, 193, 7, 0.3)',
    line: { width: 0 },
  }))

  // Add highlight annotations
  const annotations = highlights.map((h) => ({
    x: (h.start_time + h.end_time) / 2,
    y: 1,
    xref: 'x' as const,
    yref: 'paper' as const,
    text: h.label,
    showarrow: false,
    font: { size: 12, color: '#856404' },
    bgcolor: 'rgba(255, 243, 205, 0.9)',
    borderpad: 4,
  }))

  const layout: Partial<Layout> = {
    title: 'EEG Signals',
    xaxis: {
      title: 'Time (s)',
      showgrid: true,
      gridcolor: '#e9ecef',
    },
    yaxis: {
      title: '',
      showticklabels: false,
      showgrid: false,
    },
    showlegend: true,
    legend: {
      orientation: 'h',
      y: -0.15,
    },
    margin: { t: 50, r: 20, b: 80, l: 50 },
    shapes,
    annotations,
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
  }

  return (
    <Plot
      data={traces}
      layout={layout}
      style={{ width: '100%', height: '100%' }}
      config={{ responsive: true }}
    />
  )
}
```

**Step 2: Commit**

```bash
git add neuroguide/frontend/src/components/TimeSeriesPlot.tsx
git commit -m "feat: add time series visualization component with highlight support"
```

---

## Task 12: Power Spectrum and Spectrogram Components

**Files:**
- Create: `neuroguide/frontend/src/components/PowerSpectrumPlot.tsx`
- Create: `neuroguide/frontend/src/components/SpectrogramPlot.tsx`

**Step 1: Create PowerSpectrumPlot component**

```typescript
// neuroguide/frontend/src/components/PowerSpectrumPlot.tsx
import Plot from 'react-plotly.js'
import { Data, Layout } from 'plotly.js'

interface PowerSpectrumData {
  frequencies: number[]
  power: number[]
  channel: string
}

interface PowerSpectrumPlotProps {
  data: PowerSpectrumData | null
}

export function PowerSpectrumPlot({ data }: PowerSpectrumPlotProps) {
  if (!data) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        color: '#6c757d'
      }}>
        No spectrum data to display
      </div>
    )
  }

  // Convert power to dB scale for better visualization
  const powerDb = data.power.map((p) => 10 * Math.log10(p + 1e-10))

  const traces: Data[] = [
    {
      x: data.frequencies,
      y: powerDb,
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      line: { color: '#667eea', width: 2 },
      fillcolor: 'rgba(102, 126, 234, 0.2)',
    },
  ]

  // Add frequency band annotations
  const bandAnnotations = [
    { range: [1, 4], label: 'Delta', color: 'rgba(108, 117, 125, 0.1)' },
    { range: [4, 8], label: 'Theta', color: 'rgba(40, 167, 69, 0.1)' },
    { range: [8, 12], label: 'Alpha', color: 'rgba(255, 193, 7, 0.15)' },
    { range: [12, 30], label: 'Beta', color: 'rgba(0, 123, 255, 0.1)' },
    { range: [30, 50], label: 'Gamma', color: 'rgba(220, 53, 69, 0.1)' },
  ]

  const shapes = bandAnnotations.map((band) => ({
    type: 'rect' as const,
    xref: 'x' as const,
    yref: 'paper' as const,
    x0: band.range[0],
    x1: band.range[1],
    y0: 0,
    y1: 1,
    fillcolor: band.color,
    line: { width: 0 },
  }))

  const annotations = bandAnnotations.map((band) => ({
    x: (band.range[0] + band.range[1]) / 2,
    y: 1.05,
    xref: 'x' as const,
    yref: 'paper' as const,
    text: band.label,
    showarrow: false,
    font: { size: 10, color: '#6c757d' },
  }))

  const layout: Partial<Layout> = {
    title: `Power Spectrum - ${data.channel}`,
    xaxis: {
      title: 'Frequency (Hz)',
      showgrid: true,
      gridcolor: '#e9ecef',
      range: [0, 50],
    },
    yaxis: {
      title: 'Power (dB)',
      showgrid: true,
      gridcolor: '#e9ecef',
    },
    showlegend: false,
    margin: { t: 60, r: 20, b: 60, l: 60 },
    shapes,
    annotations,
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
  }

  return (
    <Plot
      data={traces}
      layout={layout}
      style={{ width: '100%', height: '100%' }}
      config={{ responsive: true }}
    />
  )
}
```

**Step 2: Create SpectrogramPlot component**

```typescript
// neuroguide/frontend/src/components/SpectrogramPlot.tsx
import Plot from 'react-plotly.js'
import { Data, Layout } from 'plotly.js'

interface SpectrogramData {
  times: number[]
  frequencies: number[]
  power: number[][]  // frequencies x times
  channel: string
}

interface SpectrogramPlotProps {
  data: SpectrogramData | null
}

export function SpectrogramPlot({ data }: SpectrogramPlotProps) {
  if (!data) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        color: '#6c757d'
      }}>
        No spectrogram data to display
      </div>
    )
  }

  // Convert power to dB scale
  const powerDb = data.power.map((row) =>
    row.map((p) => 10 * Math.log10(p + 1e-10))
  )

  const traces: Data[] = [
    {
      x: data.times,
      y: data.frequencies,
      z: powerDb,
      type: 'heatmap',
      colorscale: 'Viridis',
      colorbar: {
        title: 'Power (dB)',
        titleside: 'right',
      },
    },
  ]

  const layout: Partial<Layout> = {
    title: `Spectrogram - ${data.channel}`,
    xaxis: {
      title: 'Time (s)',
      showgrid: false,
    },
    yaxis: {
      title: 'Frequency (Hz)',
      showgrid: false,
    },
    margin: { t: 50, r: 80, b: 60, l: 60 },
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
  }

  return (
    <Plot
      data={traces}
      layout={layout}
      style={{ width: '100%', height: '100%' }}
      config={{ responsive: true }}
    />
  )
}
```

**Step 3: Commit**

```bash
git add neuroguide/frontend/src/components/PowerSpectrumPlot.tsx neuroguide/frontend/src/components/SpectrogramPlot.tsx
git commit -m "feat: add power spectrum and spectrogram visualization components"
```

---

## Task 13: Integrate Visualization Components

**Files:**
- Modify: `neuroguide/frontend/src/App.tsx`

**Step 1: Update App.tsx with full integration**

```typescript
// neuroguide/frontend/src/App.tsx
import { useState } from 'react'
import { Layout } from './components/Layout'
import { ChatPanel } from './components/ChatPanel'
import { TimeSeriesPlot } from './components/TimeSeriesPlot'
import { PowerSpectrumPlot } from './components/PowerSpectrumPlot'
import { SpectrogramPlot } from './components/SpectrogramPlot'
import './index.css'

type VisualizationType = 'none' | 'time_series' | 'power_spectrum' | 'spectrogram'

interface Visualization {
  type: VisualizationType
  data: unknown
}

interface Highlight {
  start_time: number
  end_time: number
  label: string
}

function App() {
  const [visualization, setVisualization] = useState<Visualization>({ type: 'none', data: null })
  const [highlights, setHighlights] = useState<Highlight[]>([])

  const handleVisualization = (viz: { type: string; data: unknown }) => {
    if (viz.type === 'highlight') {
      // Add highlight to existing visualization
      setHighlights((prev) => [...prev, viz.data as Highlight])
    } else {
      // Update main visualization and clear highlights
      setVisualization({ type: viz.type as VisualizationType, data: viz.data })
      setHighlights([])
    }
  }

  const renderVisualization = () => {
    switch (visualization.type) {
      case 'time_series':
        return (
          <TimeSeriesPlot
            data={visualization.data as Parameters<typeof TimeSeriesPlot>[0]['data']}
            highlights={highlights}
          />
        )
      case 'power_spectrum':
        return (
          <PowerSpectrumPlot
            data={visualization.data as Parameters<typeof PowerSpectrumPlot>[0]['data']}
          />
        )
      case 'spectrogram':
        return (
          <SpectrogramPlot
            data={visualization.data as Parameters<typeof SpectrogramPlot>[0]['data']}
          />
        )
      default:
        return (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: '#6c757d',
            textAlign: 'center',
            padding: '2rem',
          }}>
            <h2 style={{ marginBottom: '1rem', color: '#495057' }}>
              Welcome to NeuroGuide
            </h2>
            <p style={{ maxWidth: '400px', lineHeight: 1.6 }}>
              Start a conversation in the chat panel to explore EEG data.
              I'll guide you through understanding brain signals step by step.
            </p>
          </div>
        )
    }
  }

  return (
    <Layout
      visualization={renderVisualization()}
      chat={<ChatPanel onVisualization={handleVisualization} />}
    />
  )
}

export default App
```

**Step 2: Verify integration works**

```bash
cd neuroguide/frontend
npm run dev
# Expected: Full app with chat panel and visualization area
```

**Step 3: Commit**

```bash
git add neuroguide/frontend/src/App.tsx
git commit -m "feat: integrate all visualization components with chat-driven updates"
```

---

## Task 14: End-to-End Testing

**Files:**
- Manual testing steps

**Step 1: Start backend**

```bash
cd neuroguide/backend
pip install -e ".[dev]"
uvicorn app.main:app --reload
# Should start on http://localhost:8000
```

**Step 2: Start frontend**

```bash
cd neuroguide/frontend
npm install
npm run dev
# Should start on http://localhost:5173
```

**Step 3: Test the flow**

1. Open http://localhost:5173
2. Type: "Show me the available datasets"
3. Type: "Let's explore the eyes open/closed sample"
4. Type: "Show me 5 seconds of the raw signals"
5. Type: "What's the alpha band?"
6. Type: "Show me the power spectrum for channel Oz"

**Expected behavior:**
- Agent responds with explanations
- Visualizations update in the left panel
- No console errors

**Step 4: Commit final state**

```bash
git add -A
git commit -m "feat: complete NeuroGuide MVP - EEG exploration assistant"
```

---

## Summary

This plan creates a complete MVP of NeuroGuide with:

1. **Backend (FastAPI + MNE)**
   - Dataset service for loading PhysioNet EEG data
   - Analysis service for signal processing
   - Agent with tool-calling for guided exploration
   - RESTful API for chat and visualizations

2. **Frontend (React + Plotly)**
   - Split-panel layout (viz + chat)
   - Time series, power spectrum, spectrogram visualizations
   - Chat interface with conversation management

3. **Agent Capabilities**
   - Load and explain datasets
   - Show signals with artifact highlighting
   - Apply filters and show before/after
   - Display frequency analysis

**Total tasks:** 14
**Estimated implementation time with Claude Code:** Iterative, task by task

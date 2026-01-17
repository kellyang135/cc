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
        raw, channels=["C3", "C4", "Cz"], start_time=0.0, end_time=5.0
    )
    assert "data" in result
    assert "times" in result
    assert "channels" in result
    assert len(result["channels"]) == 3
    assert result["times"][-1] <= 5.0


@pytest.mark.asyncio
async def test_apply_bandpass_filter(dataset_service, analysis_service):
    """Test applying bandpass filter."""
    raw = await dataset_service.load_dataset("sample_eyes", subject=1, run=1)
    filtered_raw = analysis_service.apply_filter(raw, low_freq=1.0, high_freq=40.0)
    assert filtered_raw is not None
    assert filtered_raw.info['nchan'] == raw.info['nchan']


@pytest.mark.asyncio
async def test_compute_power_spectrum(dataset_service, analysis_service):
    """Test computing power spectral density."""
    raw = await dataset_service.load_dataset("sample_eyes", subject=1, run=1)
    result = analysis_service.compute_power_spectrum(raw, channel="C3")
    assert "frequencies" in result
    assert "power" in result
    assert len(result["frequencies"]) == len(result["power"])


@pytest.mark.asyncio
async def test_compute_spectrogram(dataset_service, analysis_service):
    """Test computing spectrogram."""
    raw = await dataset_service.load_dataset("sample_eyes", subject=1, run=1)
    result = analysis_service.compute_spectrogram(raw, channel="C3", start_time=0.0, end_time=10.0)
    assert "times" in result
    assert "frequencies" in result
    assert "power" in result

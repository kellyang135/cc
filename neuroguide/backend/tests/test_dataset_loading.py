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

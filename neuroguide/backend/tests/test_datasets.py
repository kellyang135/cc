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

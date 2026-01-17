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
        assert "sample_rate" in dataset
        assert "tasks" in dataset
        assert "source" in dataset


def test_get_dataset_info_valid_id(dataset_service):
    """Test get_dataset_info returns DatasetInfo for valid dataset ID."""
    from app.services.datasets import DatasetInfo

    result = dataset_service.get_dataset_info("eeg_motor_imagery")
    assert result is not None
    assert isinstance(result, DatasetInfo)
    assert result.id == "eeg_motor_imagery"
    assert result.name == "PhysioNet EEG Motor Movement/Imagery"


def test_get_dataset_info_invalid_id(dataset_service):
    """Test get_dataset_info returns None for invalid dataset ID."""
    result = dataset_service.get_dataset_info("nonexistent_dataset")
    assert result is None


def test_is_downloaded_true(tmp_path):
    """Test is_downloaded returns True when dataset directory exists with files."""
    from app.services.datasets import DatasetService

    # Create a dataset service with tmp_path as data_dir
    service = DatasetService(data_dir=str(tmp_path))

    # Create dataset directory with a file inside
    dataset_dir = tmp_path / "eeg_motor_imagery"
    dataset_dir.mkdir()
    (dataset_dir / "data.edf").touch()

    assert service.is_downloaded("eeg_motor_imagery") is True


def test_is_downloaded_false(tmp_path):
    """Test is_downloaded returns False when dataset is not downloaded."""
    from app.services.datasets import DatasetService

    # Create a dataset service with tmp_path as data_dir
    service = DatasetService(data_dir=str(tmp_path))

    # Don't create any dataset directory
    assert service.is_downloaded("eeg_motor_imagery") is False

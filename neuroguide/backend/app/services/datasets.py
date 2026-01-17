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

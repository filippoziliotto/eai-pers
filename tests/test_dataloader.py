import pytest
from dataset.dataloader import get_dataloader
from torch.utils.data import DataLoader

def test_dataloader_initialization(tmp_path):
    """
    Test if the DataLoader initializes properly and returns the expected number of samples.
    """
    # Mock data directory setup
    data_dir = tmp_path / "mock_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create a mock dataset (e.g., 100 samples in JSON format or .npz files if needed)
    for i in range(100):
        sample_file = data_dir / f"sample_{i}.json"
        sample_file.write_text(f'{{"id": {i}, "value": "data_{i}"}}')

    # Initialize DataLoader
    batch_size = 10
    dataloader = get_dataloader(
        data_dir=str(data_dir),
        data_split="val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Use 0 workers for simplicity in tests
    )

    # Assert that the dataloader is an instance of DataLoader
    assert isinstance(dataloader, DataLoader)

    # Collect all batches and count the number of samples
    total_samples = 0
    for batch in dataloader:
        total_samples += len(batch)

    # Check if the total number of samples matches the mock dataset
    assert total_samples == 100

    # Check the number of batches (100 samples / 10 batch size = 10 batches)
    assert len(list(dataloader)) == 10

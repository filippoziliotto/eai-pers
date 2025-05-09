import torch
import pytest
from dataset.transform import MapTransform


@pytest.fixture
def sample_data():
    B, W, H, C = 10, 50, 50, 256
    return {
        "feature_map": torch.randn(B, W, H, C),
        "xy_coords": [25, 25],
        "description": "This is a test. No augmentation. Just a test. Really."
    }


def test_no_augmentation(sample_data):
    transform = MapTransform(use_aug=False)
    out_feature_map, out_xy_coords, out_description = transform(
        sample_data["feature_map"],
        sample_data["xy_coords"],
        sample_data["description"]
    )

    assert torch.equal(out_feature_map, sample_data["feature_map"])
    assert out_xy_coords == sample_data["xy_coords"]
    assert out_description == sample_data["description"]


def test_horizontal_flip(sample_data, monkeypatch):
    transform = MapTransform(use_aug=True, use_horizontal_flip=True)
    transform.prob = 0.5

    with monkeypatch.context() as m:
        m.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.9]))
        out_feature_map, out_xy_coords, _ = transform(
            sample_data["feature_map"],
            sample_data["xy_coords"],
            sample_data["description"]
        )

    expected = torch.flip(sample_data["feature_map"], dims=(2,))
    assert torch.equal(out_feature_map, expected)
    assert out_xy_coords[0] == sample_data["feature_map"].shape[2] - sample_data["xy_coords"][0]


def test_vertical_flip(sample_data, monkeypatch):
    transform = MapTransform(use_aug=True, use_vertical_flip=True)
    transform.prob = 0.5

    with monkeypatch.context() as m:
        m.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.9]))
        out_feature_map, out_xy_coords, _ = transform(
            sample_data["feature_map"],
            sample_data["xy_coords"],
            sample_data["description"]
        )

    expected = torch.flip(sample_data["feature_map"], dims=(1,))
    assert torch.equal(out_feature_map, expected)
    assert out_xy_coords[1] == sample_data["feature_map"].shape[1] - sample_data["xy_coords"][1]


def test_random_crop(sample_data, monkeypatch):
    def fake_crop(feature_map, xy_coords):
        cropped = feature_map[:, :-1, :-1]
        new_xy = [xy_coords[0] - 1, xy_coords[1] - 1]
        return cropped, new_xy

    monkeypatch.setattr("dataset.utils.random_crop_preserving_target", fake_crop)

    transform = MapTransform(use_aug=True, use_random_crop=True)
    transform.prob = 0.5

    with monkeypatch.context() as m:
        m.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.9]))
        out_feature_map, out_xy_coords, _ = transform(
            sample_data["feature_map"],
            sample_data["xy_coords"],
            sample_data["description"]
        )

    cropped, new_xy = fake_crop(sample_data["feature_map"], sample_data["xy_coords"])
    assert torch.equal(out_feature_map, cropped)
    assert out_xy_coords == new_xy


def test_description_augmentation(sample_data, monkeypatch):
    transform = MapTransform(use_aug=True, use_desc_aug=True)
    transform.prob = 0.5

    with monkeypatch.context() as m:
        m.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.9]))
        _, _, out_description = transform(
            sample_data["feature_map"],
            sample_data["xy_coords"],
            sample_data["description"]
        )

    assert out_description != sample_data["description"]


def test_all_augmentations_combined(sample_data, monkeypatch):
    transform = MapTransform(
        use_aug=True,
        use_horizontal_flip=True,
        use_vertical_flip=True,
        use_random_crop=True,
        use_desc_aug=True
    )
    transform.prob = 0.5

    def fake_crop(feature_map, xy_coords):
        cropped = feature_map[:, :-1, :-1]
        new_xy = [xy_coords[0] - 1, xy_coords[1] - 1]
        return cropped, new_xy

    monkeypatch.setattr("dataset.utils.random_crop_preserving_target", fake_crop)

    with monkeypatch.context() as m:
        m.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.9]))
        out_feature_map, out_xy_coords, out_description = transform(
            sample_data["feature_map"],
            sample_data["xy_coords"],
            sample_data["description"]
        )

    assert not torch.equal(out_feature_map, sample_data["feature_map"])
    assert out_xy_coords != sample_data["xy_coords"]
    assert out_description != sample_data["description"]

import torch
import pytest

# Import the class to test
from dataset.transform import MapTransform
from dataset.utils import random_crop_preserving_target


B, W, H, C = 10, 50, 50, 256
XY_COORDS = [25, 25]
DESCRIPTION = "This is a test. No augmentation. Just a test. Really."
FEAT_MAP = torch.randn(B, W, H, C)

# Test when augmentation is turned off
def test_no_augmentation():
    
    transform = MapTransform(use_aug=False)
    feature_map = FEAT_MAP  
    xy_coords = XY_COORDS
    description = DESCRIPTION
    
    out_feature_map, out_xy_coords, out_description = transform(feature_map, xy_coords, description)
    
    # When augmentation is off, output should match input exactly.
    assert torch.equal(out_feature_map, feature_map)
    assert out_xy_coords == xy_coords
    assert out_description == description

# Test horizontal flip augmentation
def test_horizontal_flip(monkeypatch):
    
    transform = MapTransform(use_aug=True, use_horizontal_flip=True)
    # Manually set the probability attribute (since it’s referenced in __call__)
    transform.prob = 0.5
    
    # Create a simple tensor for which a horizontal flip is easy to verify.
    feature_map =  FEAT_MAP
    xy_coords = XY_COORDS
    description = DESCRIPTION
    
    # Force the flip by making torch.rand return a value that ensures the condition is met.
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.9]))
    
    out_feature_map, out_xy_coords, _ = transform(feature_map, xy_coords, description)
    
    # Expected behavior: feature_map should be flipped horizontally (flip on dimension 1)
    expected_feature_map = torch.flip(feature_map, dims=(1,))
    assert torch.equal(out_feature_map, expected_feature_map)
    
    # The x-coordinate should be updated accordingly.
    assert out_xy_coords[0] == feature_map.shape[2] - xy_coords[0]

# Test vertical flip augmentation
def test_vertical_flip(monkeypatch):
    
    transform = MapTransform(use_aug=True, use_vertical_flip=True)
    transform.prob = 0.5
    feature_map =  FEAT_MAP
    xy_coords = XY_COORDS
    description = DESCRIPTION
    
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.9]))
    
    out_feature_map, out_xy_coords, _ = transform(feature_map, xy_coords, description)
    
    expected_feature_map = torch.flip(feature_map, dims=(0,))
    assert torch.equal(out_feature_map, expected_feature_map)
    assert out_xy_coords[1] == feature_map.shape[1] - xy_coords[1]

# Test random crop augmentation by faking the crop function
def test_random_crop(monkeypatch):
    
    # Define a fake crop function so we can control the output.
    def fake_crop(feature_map, xy_coords):
        # For example, remove the last row and column:
        cropped = feature_map[:, :-1, :-1]
        new_xy = [xy_coords[0] - 1, xy_coords[1] - 1]
        return cropped, new_xy

    # Replace the utility function with our fake version.
    monkeypatch.setattr("dataset.utils.random_crop_preserving_target", fake_crop)
    
    transform = MapTransform(use_aug=True, use_random_crop=True)
    transform.prob = 0.5
    feature_map =  FEAT_MAP
    xy_coords = XY_COORDS
    description = DESCRIPTION
    
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.9]))
    
    out_feature_map, out_xy_coords, _ = transform(feature_map, xy_coords, description)
    
    cropped, new_xy = fake_crop(feature_map, xy_coords)
    assert torch.equal(out_feature_map, cropped)
    assert out_xy_coords == new_xy

# Test description augmentation
def test_description_augmentation(monkeypatch):
    transform = MapTransform(use_aug=True, use_desc_aug=True)
    transform.prob = 0.5
    feature_map =  FEAT_MAP
    xy_coords = XY_COORDS
    description = DESCRIPTION
    
    # Force the description augmentation by patching torch.rand.
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.9]))
    
    _, _, out_description = transform(feature_map, xy_coords, description)
    
    # Verify that the description is changed (since the condition is forced).
    # Here, we simply check that the output is not the same as the input.
    print(out_description)
    assert out_description != description
    
def test_all_augmentation(monkeypatch):
    transform = MapTransform(use_aug=True, use_horizontal_flip=True, use_vertical_flip=True, use_random_crop=True, use_desc_aug=True)
    transform.prob = 0.5
    feature_map =  FEAT_MAP
    xy_coords = XY_COORDS
    description = DESCRIPTION
    
    # Force all augmentations by patching torch.rand.
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.9]))
    
    out_feature_map, out_xy_coords, out_description = transform(feature_map, xy_coords, description)
    
    # Verify that all augmentations are applied.
    # We can reuse the previous tests to verify the individual augmentations.
    test_horizontal_flip(monkeypatch)
    test_vertical_flip(monkeypatch)
    test_random_crop(monkeypatch)
    test_description_augmentation(monkeypatch)
    
    # We just need to check that all augmentations are applied together.
    # The output should be different from the input for all augmentations.
    assert out_feature_map != feature_map
    assert out_xy_coords != xy_coords
    assert out_description != description

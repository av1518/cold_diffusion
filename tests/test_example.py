from src.strat_funcs import single_alternating_zoom


import torch
from torchvision.transforms import InterpolationMode

img = torch.rand(3, 28, 28)  # 3-channel 28x28 image


def test_no_zoom():
    result = single_alternating_zoom(img, 0)
    assert torch.equal(img, result), "No zoom should return the original image"


def test_max_zoom():
    result = single_alternating_zoom(img, 27)
    assert result.shape == img.shape, "Maximum zoom should still return a 28x28 image"


def test_mid_range_zoom():
    mid_step = 14
    result = single_alternating_zoom(img, mid_step)
    assert (
        result.shape == img.shape
    ), f"Zoom at step {mid_step} should still result in a 28x28 image"


def test_output_shape_consistency():
    step = 5
    result = single_alternating_zoom(img, step)
    assert (
        result.shape == img.shape
    ), "Output image should have the same dimensions as the input image"

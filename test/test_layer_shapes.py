import torch

from swin import layers


def test_patch_partition_output_shape():
    pp = layers.PatchPartition(4)
    input = torch.rand(1, 3, 32, 32)
    result = pp(input)
    assert list(result.shape) == [1, 48, 8, 8]

def test_patch_partition_output_shape_with_padding():
    pp = layers.PatchPartition(4)
    input = torch.rand(1, 3, 19, 19)
    result = pp(input)
    assert list(result.shape) == [1, 48, 5, 5]


def test_patch_merge_output_shape():
    pm = layers.PatchMerge(96)
    input = torch.rand(1, 96, 8, 8)
    result = pm(input)
    assert list(result.shape) == [1, 192, 4, 4]


def test_shuffle_layer_norm():
    ln = layers.ShuffleLayerNorm2D(96)
    input = torch.rand(1, 96, 8, 8)
    result = ln(input)
    assert result.shape == input.shape


def test_wmsa_output_shape():
    wmsa = layers.WindowedMultiheadSelfAttention(96)
    input = torch.rand(1, 96, 8, 8)
    result = wmsa(input)
    assert result.shape == input.shape


def test_wmsa_output_shape_with_padding():
    wmsa = layers.WindowedMultiheadSelfAttention(96)
    input = torch.rand(1, 96, 7, 7)
    result = wmsa(input)
    assert result.shape == input.shape


def test_swmsa_output_shape():
    swmsa = layers.ShiftedWindowedMultiheadSelfAttention(2, 96)
    input = torch.rand(1, 96, 8, 8)
    result = swmsa(input)
    assert result.shape == input.shape


def test_swmsa_output_shape_no_pad():
    swmsa = layers.ShiftedWindowedMultiheadSelfAttention(2, 96)
    input = torch.rand(1, 96, 6, 6)
    result = swmsa(input)
    assert result.shape == input.shape
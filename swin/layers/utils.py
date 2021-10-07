import torch
from torch import nn


def pad_if_necessary(
    inputs: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    if inputs.size(-1) % window_size != 0 or inputs.size(-2) % window_size != 0:
        h = (inputs.size(-2) + window_size - 1) // window_size * window_size
        w = (inputs.size(-1) + window_size - 1) // window_size * window_size
        padded = torch.zeros((inputs.size(0), inputs.size(1), h, w), dtype=inputs.dtype, device=inputs.device)
        padded[..., :inputs.size(-2), :inputs.size(-1)] = inputs
        inputs = padded
    return inputs


class ShuffleLayerNorm2D(nn.LayerNorm):
    """
    Shuffles channels to perform layernorm over dimension 1 rather than -1
    """
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        inputs = inputs.permute(0, 2, 3, 1)
        result = super(ShuffleLayerNorm2D, self).forward(inputs)
        return result.permute(0, 3, 1, 2)

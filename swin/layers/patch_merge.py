import torch
from torch import nn


class PatchMerge(nn.Module):
    """
    Implements the Patch Merge operator from Swin Transformer
    """
    def __init__(
        self,
        channels: int,
        window_size: int = 2,
    ):
        super(PatchMerge, self).__init__()
        self.merger = nn.Conv2d(
            in_channels = channels,
            out_channels = window_size * channels,
            kernel_size = window_size,
            stride = window_size,
            padding = window_size // 2
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.merger(inputs)

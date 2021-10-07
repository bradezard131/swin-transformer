from typing import Sequence
import torch
from torch import nn

from swin import layers


class SwinBlock(nn.Module):
    """
    Implements the basic Swin Transformer Block
    """
    def __init__(
        self,
        channels: int,
        heads: int = 3,
        window_size: int = 4,
        shift_offset: int = 0,
        expansion: int = 4,
        dropout: float = 0.1,
    ):
        super(SwinBlock, self).__init__()
        self.layernorm1 = layers.ShuffleLayerNorm2D(channels)
        if shift_offset == 0:
            self.wmsa = layers.WindowedMultiheadSelfAttention(channels, heads, window_size)
        else:
            self.wmsa = layers.ShiftedWindowedMultiheadSelfAttention(shift_offset, channels, heads, window_size)
        
        self.layernorm2 = layers.ShuffleLayerNorm2D(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, expansion * channels, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(expansion * channels, channels, 1)
        )
        self.final_activation = nn.GELU()
    
    def forward(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        # Attention Block
        residual = inputs
        inputs = self.layernorm1(inputs)
        features = self.wmsa(inputs)
        features = residual + features
        
        # MLP Block
        residual = features
        features = self.layernorm2(features)
        features = self.mlp(features)
        result = self.final_activation(residual + features)
        
        return result


def build_swin_model(
    out_classes: int,
    layer_configuration: Sequence[int] = [2, 2, 6, 2],
    initial_channels: int = 96,
    patch_size: int = 4,
    heads: int = 3,
    window_size: int = 4,
    offset_size: int = 2,
    expansion: int = 4,
    dropout: float = 0.1,
) -> nn.Sequential:
    modules = [layers.PatchPartition(patch_size),
               nn.Conv2d(patch_size * patch_size * 3, initial_channels, 1)]  # Linear patch embedding
    
    channels = initial_channels
    shift = False
    for config in layer_configuration:
        # Add the Swin Blocks
        for _ in range(config):
            modules.append(SwinBlock(
                channels,
                heads,
                window_size,
                shift_offset=offset_size if shift else 0,
                expansion=expansion,
                dropout=dropout,
            ))
            shift = not shift
        
        # Add the PatchMerge
        modules.append(layers.PatchMerge(channels))
        channels = channels * 2
    
    return nn.Sequential(
        *modules,
        nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
        nn.Flatten(1),
        nn.Linear(channels, out_classes)
    )
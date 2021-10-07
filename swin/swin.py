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

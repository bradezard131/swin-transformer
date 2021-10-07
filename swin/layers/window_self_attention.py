import numpy as np
import torch
from torch import nn

from swin.layers.utils import pad_if_necessary


def _generate_relative_idxs(
    window_size: int,
) -> torch.Tensor:
    """
    Generates relative positional indexes for relative positional attention
    """
    a = torch.arange(window_size - 1, 2 * window_size - 1)
    y, x = torch.meshgrid(a, a)
    x = x.flatten().numpy()
    y = y.flatten().numpy()
    idxs = []
    for i in range(window_size):
        for j in range(window_size):
            idxs.append(np.ravel_multi_index((x-j, y-i), (2 * window_size - 1, 2 * window_size - 1)))
    return torch.from_numpy(np.stack(idxs, 0))


class WindowedMultiheadSelfAttention(nn.Module):
    """
    Implements the Windowed MHSA from Swin Transformers
    """
    def __init__(
        self,
        channels: int,
        heads: int = 3,
        window_size: int = 4,
    ):
        super(WindowedMultiheadSelfAttention, self).__init__()
        self.channels = channels
        self.heads = heads
        self.scale = (channels / heads) ** 0.5
        self.window_size = window_size
        self.qkv_project = nn.Conv2d(channels, 3 * channels, 1)
        self.out_project = nn.Conv2d(channels, channels, 1)
        
        self.relative_bias = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2))  # type: ignore
        self.relative_idxs = _generate_relative_idxs(window_size)
        
        self.out_project = nn.Conv2d(channels, channels, 1, bias=False)
    
    def _reshape_qkv(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        inputs = inputs.reshape(inputs.size(0), 
                                inputs.size(1) // self.heads, self.heads,
                                inputs.size(2) // self.window_size, self.window_size,
                                inputs.size(3) // self.window_size, self.window_size)
        inputs = inputs.permute(0, 2, 3, 5, 4, 6, 1)  # Batch, Heads, Height, Width, WinSize, WinSize, Channels
        inputs = inputs.flatten(start_dim=4, end_dim=5)  # Batch, Heads, Height, Width, SeqLen, Channels
        return inputs
    
    def _reshape_output(
        self,
        output: torch.Tensor,
    ) -> torch.Tensor:
        output = output.reshape(output.size(0), output.size(1), output.size(2), output.size(3),
                                self.window_size, self.window_size, output.size(-1))
        output = output.permute(0, 6, 1, 2, 4, 3, 5)  # Batch, Channels, Heads, Height, WinSize, Width, WinSize
        return output.reshape(output.size(0),
                              output.size(1) * self.heads,
                              output.size(3) * self.window_size,
                              output.size(5) * self.window_size)
    
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        original_height, original_width = inputs.shape[-2:]
        inputs = pad_if_necessary(inputs, self.window_size)
        
        q, k, v = self.qkv_project(inputs).split([self.channels, self.channels, self.channels], dim=1)
        q = self._reshape_qkv(q)
        k = self._reshape_qkv(k)
        
        attention = ((q @ k.transpose(-2, -1)) / self.scale).softmax(-1)
        output = attention @ self._reshape_qkv(v)
        output = self._reshape_output(output)
        return self.out_project(output[..., :original_height, :original_width])


class ShiftedWindowedMultiheadSelfAttention(WindowedMultiheadSelfAttention):
    """
    Applies Windowed MHSA on shifted windows
    """
    def __init__(
        self,
        shift_offset: int,
        *args, **kwargs,  # allows passing through of other arguments to the superclass constructor
    ):
        super(ShiftedWindowedMultiheadSelfAttention, self).__init__(*args, **kwargs)
        self.shift_offset = shift_offset
    
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        original_height, original_width = inputs.shape[-2:]
        
        # Calculate padded new image size
        new_width = (inputs.size(-1) + self.shift_offset + self.window_size - 1) // self.window_size * self.window_size
        new_height = (inputs.size(-2) + self.shift_offset + self.window_size -1) // self.window_size * self.window_size
        
        # Pad the inputs
        padded = torch.zeros((inputs.size(0), inputs.size(1), new_height, new_width),
                             dtype=inputs.dtype, device=inputs.device)
        padded[..., 
               self.shift_offset:self.shift_offset+original_height, 
               self.shift_offset:self.shift_offset+new_width] = inputs
        
        # Do the forward pass
        outputs = super(ShiftedWindowedMultiheadSelfAttention, self).forward(padded)
        
        # Unpad the result
        return outputs[..., 
               self.shift_offset:self.shift_offset+original_height, 
               self.shift_offset:self.shift_offset+new_width]
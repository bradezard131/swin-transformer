import torch
from torch import nn

from swin.layers.utils import pad_if_necessary


class PatchPartition(nn.Module):
    """
    Implements the Patch Partition module from Swin-Transformers
    """
    def __init__(
        self,
        patch_size: int = 4
    ):
        super(PatchPartition, self).__init__()
        self.patch_size = patch_size
    
    def forward(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        images = pad_if_necessary(images, self.patch_size)
        patches = images.reshape(images.size(0), images.size(1),
                                 images.size(2) // self.patch_size, self.patch_size,
                                 images.size(3) // self.patch_size, self.patch_size)  # break into windows
        features = patches.permute(0, 1, 3, 5, 2, 4).flatten(start_dim=1, end_dim=3)  # flatten patches
        return features

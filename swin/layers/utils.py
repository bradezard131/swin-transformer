import torch


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

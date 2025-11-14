import torch
import torch.nn.functional as F
from typing import List


def empirical_est(
    x: torch.Tensor, y: torch.Tensor, order: int, beta: float = 1
) -> List[torch.Tensor]:
    """
    Computes an empirical estimation vector for each possible index based on the input tensor `x` and target `y`.

    Args:
        x (torch.Tensor): Input tensor of shape (1, N). It is squeezed to shape (N,).
        y (torch.Tensor): Target tensor of shape (1, N). It is squeezed to shape (N,).
        order (int): The order determines the number of binary divisions (2^order indices).
        beta (float, optional): Smoothing parameter for cumulative sum adjustment. Default is 1.

    Returns:
        List[torch.Tensor]: A list where each element corresponds to an empirical estimation vector
        for each of the 2^order possible indices.
    """

    # Ensure the input tensor x has the correct first dimension
    assert x.size(0) == 1, "Input tensor x should have a batch size of 1."

    device = x.device  # Get the device (e.g., 'cpu' or 'cuda') for tensor operations

    # Convert x and y to float, and remove the first dimension by squeezing
    x = x.float().squeeze()
    y = y.float().squeeze()

    # Create a powers tensor for 2^i for i in [0, ..., order-1], reversed order for indexing
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(device)

    # Compute index for each entry in x by performing a 1D convolution
    idx = F.conv1d(x.view(1, -1), powers.view(1, 1, -1)).squeeze()

    est_vec = []  # List to store estimation vectors

    # Loop over all possible 2^order indices
    for i in range(2**order):
        mask = (
            idx == i
        )  # Create a mask for selecting elements where the index matches i

        # Select corresponding values in y and adjust the start of the sequence
        s = y[order - 1 :][mask]
        s = torch.cat(
            (torch.Tensor([0]).to(device), s[:-1])
        )  # Prepend 0 and remove the last element

        # Cumulative sum with smoothing based on the `beta` parameter
        s = (s.cumsum(0) + beta) / (torch.arange(len(s), device=device) + 2 * beta)

        est_vec.append(s)  # Append the resulting vector to the list

    return est_vec

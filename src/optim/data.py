import torch
from torch import Tensor
from typing import Any, Tuple, Optional


def get_random_P(
    order: int,
    batch_size_per_chain: int,
    num_chains: int,
    generator: Optional[torch.Generator],
    dist: Optional[torch.distributions.Distribution],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Generate a random tensor P with either a uniform random distribution or by sampling
    from a provided distribution, then shuffle the tensor before returning.

    Args:
        order (int): Determines the number of dimensions for the random tensor P (2^order).
        batch_size_per_chain (int): Batch size for each chain.
        num_chains (int): Number of chains to generate.
        generator (Optional[torch.Generator]): Torch generator for random number generation.
        dist (Optional[Distribution]): Distribution to sample from. If None, uses random uniform distribution.
        device (torch.device): The device to create the tensors on (e.g., 'cpu' or 'cuda').
        dtype (torch.dtype): The data type for the tensors (e.g., torch.float32).

    Returns:
        Tensor: A shuffled tensor of shape (num_chains * batch_size_per_chain, 2^order, 2).
    """

    if dist is None:
        # Generate random probabilities (pk) for binary outcomes (2^order rows)
        pk = torch.rand(
            (num_chains, 2**order, 1), generator=generator, dtype=dtype, device=device
        )
        # Create P matrix with probabilities for both binary outcomes (0 and 1)
        P = torch.cat([1 - pk, pk], dim=2)
    else:
        # Sample P from the provided distribution and move to the correct device
        P = dist.sample().to(device)

    # Expand P to match the required batch size for each chain and reshape
    P = (
        P.unsqueeze(1)
        .expand(-1, batch_size_per_chain, *P.shape[1:])
        .reshape(-1, *P.shape[1:])
    )

    # Shuffle the rows of the tensor P
    shuffled_P = P[torch.randperm(P.size(0))]

    return shuffled_P


def get_batch(
    P: Optional[torch.Tensor],
    order: int,
    seq_length: int,
    batch_size_per_chain: int,
    num_chains: int,
    generator: torch.Generator,
    dist: torch.distributions.Distribution,
    extra_args: Any,
    return_P: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Generates a batch of sequences for a Markov process, where either a provided transition matrix `P` or
    randomly generated transition probabilities are used to generate the data.

    Args:
        P (Optional[torch.Tensor]): Transition matrix. If None, random P will be generated.
        order (int): Order of the Markov process determining the number of past states.
        seq_length (int): Length of the sequence to generate.
        batch_size_per_chain (int): Number of sequences to generate per chain.
        num_chains (int): Number of chains in the batch.
        generator (torch.Generator): Torch random generator for reproducibility.
        dist (torch.distributions.Distribution): Distribution used in case `P` is None.
        extra_args (Any): Additional arguments such as device, dtype, and initial distribution method.
        return_P (bool): If True, returns the transition matrix `P` along with the generated data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - x: Tensor of generated sequences of shape (batch_size, seq_length).
            - y: Tensor of next-step sequences of shape (batch_size, seq_length).
            - P: Transition matrix if `return_P` is True, otherwise None.
    """
    # Calculate total batch size
    batch_size = batch_size_per_chain * num_chains

    # Initialize a tensor to store the generated data
    data = torch.zeros(batch_size, seq_length + 1, device=extra_args.device)

    # Precompute powers of 2 for computing indices from the past states
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(
        extra_args.device
    )

    if P is None:
        # If P is not provided, generate the first `order` bits randomly
        alpha = 0.5  # Assuming equal probability for initial states
        data[:, :order] = torch.bernoulli(
            alpha * torch.ones((batch_size, order), device=extra_args.device),
            generator=generator,
        )

        # Generate the random transition matrix P
        P = get_random_P(
            order,
            batch_size_per_chain,
            num_chains,
            generator,
            dist,
            extra_args.device,
            extra_args.dtype,
        )
        batch_indices = torch.arange(batch_size)

        # Generate the following bits based on the Markov process
        for i in range(order, seq_length + 1):
            prev_symbols = data[:, i - order : i]  # Previous `order` symbols
            idx = (prev_symbols @ powers).int()  # Compute indices using dot product
            next_symbols = torch.multinomial(
                P[batch_indices, idx], 1, generator=generator
            ).squeeze(1)
            data[:, i] = next_symbols
    else:
        if P.dim() == 2:
            # Case where the same fixed P is used for all sequences
            if extra_args.initial == "steady":
                if P.size(0) == 2:
                    alpha = P[1, 0] / (
                        P[0, 1] + P[1, 0]
                    )  # Steady-state initial distribution
                else:
                    alpha = 0.5
            else:
                alpha = 0.5

            # Generate the first `order` bits
            data[:, :order] = torch.bernoulli(
                alpha * torch.ones((batch_size, order), device=extra_args.device),
                generator=generator,
            )

            # Generate the following bits
            for i in range(order, seq_length + 1):
                prev_symbols = data[:, i - order : i]
                idx = (prev_symbols @ powers).int()  # Compute indices
                next_symbols = torch.multinomial(
                    P[idx], 1, generator=generator
                ).squeeze(1)
                data[:, i] = next_symbols
        else:
            # Case where a random P is used for each chain
            alpha = 0.5
            data[:, :order] = torch.bernoulli(
                alpha * torch.ones((batch_size, order), device=extra_args.device),
                generator=generator,
            )
            batch_indices = torch.arange(batch_size)

            # Generate the following bits
            for i in range(order, seq_length + 1):
                prev_symbols = data[:, i - order : i]
                idx = (prev_symbols @ powers).int()
                next_symbols = torch.multinomial(
                    P[batch_indices, idx], 1, generator=generator
                ).squeeze(1)
                data[:, i] = next_symbols

    # Extract sequences and their next-step counterparts
    x = data[:, :seq_length].to(int)
    y = data[:, 1:].to(int)

    if return_P:
        return x, y, P

    return x, y

import numpy as np
import torch
import os
from typing import IO, Any, BinaryIO


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
):
    L = len(x)

    start_index = np.random.randint(0, L - context_length, batch_size)

    input_tokens = np.stack([x[i : i + context_length] for i in start_index])
    target_tokens = np.stack([x[i + 1 : i + context_length + 1] for i in start_index])

    input_tokens = torch.tensor(input_tokens, dtype=torch.int64, device=device)
    target_tokens = torch.tensor(target_tokens, dtype=torch.int64, device=device)

    return (input_tokens, target_tokens)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    obj = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "iteration": iteration
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    obj = torch.load(src)
    model.load_state_dict(obj["model_state"])
    optimizer.load_state_dict(obj["optimizer_state"])
    return obj["iteration"]
import numpy as np
import torch
import numpy.typing as npt
import typing
import os

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    n = len(dataset) - context_length
    idx = np.random.randint(0, n, size=batch_size)
    inputs = np.array([dataset[i:i+context_length] for i in idx], dtype=np.int64)
    targets = np.array([dataset[i+1:i+context_length+1] for i in idx], dtype=np.int64)

    return (torch.tensor(inputs, device=device), torch.tensor(targets, device=device))

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out:  str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = {
        'model_state' : model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(obj, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']

import os
import torch
import random
import numpy as np

"""
Helpers for runtime and device management on different platforms.

This module centralises common checks for Kaggle environments, GPU availability,
and distributed training metadata. It also exposes simple wrappers to
initialise distributed process groups and seed all relevant random number
generators.  Keeping this logic in one place makes it easier to reuse across
training scripts and prevents conditional code from cluttering the main
application.
"""

def is_kaggle() -> bool:
    """Return True if running inside a Kaggle notebook environment."""
    # Kaggle mounts the working directory at /kaggle/working and also uses /kaggle
    return os.path.exists("/kaggle") or os.path.exists("/kaggle/working")

def gpu_count() -> int:
    """Return the number of available CUDA devices."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def rank() -> int:
    """Process rank in a distributed setup (0 if not distributed)."""
    return int(os.environ.get("RANK", "0"))

def world_size() -> int:
    """Total number of processes in a distributed run (1 if not distributed)."""
    return int(os.environ.get("WORLD_SIZE", "1"))

def local_rank() -> int:
    """The index of the GPU this process should use in a multi-GPU run."""
    return int(os.environ.get("LOCAL_RANK", "0"))

def init_distributed():
    """
    Initialise the default process group for distributed training.

    This helper checks the world size and only initialises the process group
    if more than one process is participating.  It uses the `nccl` backend
    which is optimised for multi-GPU training.  If the process group has
    already been initialised by an external launcher (e.g. torchrun) then
    calling this function is a no-op.
    """
    if world_size() > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            # If using torchrun the necessary env variables (MASTER_ADDR, MASTER_PORT,
            # RANK, WORLD_SIZE, LOCAL_RANK) are already populated.
            dist.init_process_group(backend="nccl")

def set_seed(seed: int = 42):
    """
    Seed Python, NumPy and PyTorch for reproducibility.

    When using multi-GPU training each worker should call this function
    independently to ensure deterministic behaviour.  CUDA's random state
    is also seeded if GPUs are available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
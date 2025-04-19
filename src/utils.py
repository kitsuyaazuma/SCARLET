import os
import subprocess
import random
import numpy as np
import torch
import torch.nn.functional as F
import shutil
from pathlib import Path
from typing import Callable, Union


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_random_state() -> dict:
    return {
        "random": random.getstate(),
        "environ": os.environ["PYTHONHASHSEED"],
        "numpy": np.random.get_state(),
        "torch": {
            "manual_seed": torch.initial_seed(),
            "cuda_manual_seed": torch.cuda.initial_seed(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
    }


def set_random_state(state: dict) -> None:
    random.setstate(state["random"])
    os.environ["PYTHONHASHSEED"] = state["environ"]
    np.random.set_state(state["numpy"])
    torch.manual_seed(state["torch"]["manual_seed"])
    torch.cuda.manual_seed(state["torch"]["cuda_manual_seed"])
    torch.backends.cudnn.deterministic = state["torch"]["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = state["torch"]["cudnn_benchmark"]


def get_optimizer(name: str):
    if name == "SGD":
        return torch.optim.SGD
    else:
        raise ValueError(f"Optimizer {name} not found")


def get_criterion(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    match name:
        case "cross_entropy":
            return lambda inp, tgt: F.cross_entropy(inp, tgt)
        case "kl_div":
            return lambda inp, tgt: F.kl_div(inp, tgt, reduction="batchmean")
        case "mse_loss":
            return lambda inp, tgt: F.mse_loss(inp, tgt)
        case _:
            raise ValueError(f"Criterion {name} not found")


def clean_up(dirs: Union[list[Path], Path]):
    if isinstance(dirs, Path):
        dirs = [dirs]
    for dir in dirs:
        shutil.rmtree(dir)


def get_git_commit_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode("utf-8")
    )


def get_cuda_info() -> str:
    if torch.cuda.is_available():
        return (
            f"version={torch.version.cuda}, "  # type: ignore
            f"device_count={torch.cuda.device_count()}, "
            f"device_name={torch.cuda.get_device_name()}"
        )
    return "CUDA is not available"

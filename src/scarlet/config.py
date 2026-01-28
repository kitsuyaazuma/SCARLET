from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import tyro

from scarlet.dataset import CommonPartitionStrategy, CommonPrivateTask, CommonPublicTask
from scarlet.models import CommonModelName


@dataclass
class DSFLConfig:
    """DS-FL algorithm configuration."""

    era_temperature: float = 0.1
    """Temperature for ERA (T)."""


@dataclass
class SCARLETConfig:
    """SCARLET algorithm configuration."""

    enhanced_era_exponent: float = 2.0
    """Exponent for Enhanced ERA (beta)."""

    cache_duration: int = 50
    """Cache duration for soft-labels (D)."""


@dataclass
class WandBConfig:
    mode: Literal["online", "offline", "disabled", "shared"] = "online"


@dataclass
class CommonConfig:
    dataset_root_dir: Path = Path("/tmp/scarlet/dataset")
    """Root directory for the dataset."""

    state_root_dir: Path = Path("/tmp/scarlet/state")
    """Root directory to save intermediate states."""

    seed: int = 42
    """Seed for reproducibility."""

    num_clients: int = 100
    """Total number of clients in the federation."""

    private_task: CommonPrivateTask = CommonPrivateTask.CIFAR10
    """Task name for private dataset."""

    public_task: CommonPublicTask = CommonPublicTask.CIFAR100
    """Task name for public dataset."""

    partition: CommonPartitionStrategy = CommonPartitionStrategy.DIRICHLET
    """Dataset partition strategy."""

    dir_alpha: float = 0.05
    """Alpha for Dirichlet distribution based partitioning."""

    public_size: int = 10000
    """Total size of the public dataset."""

    model_name: CommonModelName = CommonModelName.RESNET20
    """Name of the model to be used."""

    global_round: int = 5
    """Total number of federated learning rounds."""

    epochs: int = 5
    """Number of local training epochs per client."""

    batch_size: int = 50
    """Batch size for local training."""

    lr: float = 0.1
    """Learning rate for the client optimizer."""

    kd_epochs: int = 5
    """Number of epochs for knowledge distillation."""

    kd_batch_size: int = 50
    """Batch size for knowledge distillation."""

    kd_lr: float = 0.1
    """Learning rate for knowledge distillation."""

    public_size_per_round: int = 1000
    """Number of public samples used per round."""

    sample_ratio: float = 1.0
    """Fraction of clients to sample in each round."""

    num_parallels: int = 10
    """Number of parallel processes for training."""


@dataclass
class Config:
    algorithm: tyro.conf.OmitSubcommandPrefixes[
        (
            Annotated[
                DSFLConfig,
                tyro.conf.subcommand("dsfl"),
            ]
            | Annotated[
                SCARLETConfig,
                tyro.conf.subcommand("scarlet"),
            ]
        )
    ]
    """Select the FL algorithm to run."""

    common: tyro.conf.OmitArgPrefixes[CommonConfig]
    """Common configuration for FL experiments."""

    wandb: WandBConfig
    """Weights & Biases configuration."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import torch
import torch.multiprocessing as mp
import typer
import wandb

from algorithm import (
    AlgorithmName,
    CommonClientArgs,
    CommonPipeline,
    CommonServerArgs,
    DSFLClientTrainer,
    DSFLServerHandler,
    SCARLETClientTrainer,
    SCARLETServerHandler,
)
from core import setup_reproducibility
from dataset import (
    CommonPartitionedDataset,
    CommonPartitionStrategy,
    CommonPrivateTask,
    CommonPublicTask,
)
from models import CommonModelName, CommonModelSelector


def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main(
    dataset_root_dir: Annotated[
        Path, typer.Option(help="Root directory for the dataset.")
    ] = Path("/tmp/scarlet/dataset"),
    state_root_dir: Annotated[
        Path, typer.Option(help="Directory to save intermediate states.")
    ] = Path("/tmp/scarlet/state"),
    seed: Annotated[int, typer.Option(help="Seed for reproducibility.")] = 42,
    num_clients: Annotated[
        int, typer.Option(help="Total number of clients in the federation.")
    ] = 100,
    private_task: Annotated[
        CommonPrivateTask, typer.Option(help="Task name for private dataset.")
    ] = CommonPrivateTask.CIFAR10,
    public_task: Annotated[
        CommonPublicTask, typer.Option(help="Task name for public dataset.")
    ] = CommonPublicTask.CIFAR100,
    partition: Annotated[
        CommonPartitionStrategy,
        typer.Option(help="Dataset partition strategy."),
    ] = CommonPartitionStrategy.DIRICHLET,
    dir_alpha: Annotated[
        float,
        typer.Option(help="Alpha for Dirichlet distribution based partitioning."),
    ] = 0.05,
    public_size: Annotated[
        int, typer.Option(help="Total size of the public dataset.")
    ] = 10000,
    model_name: Annotated[
        CommonModelName, typer.Option(help="Name of the model to be used.")
    ] = CommonModelName.RESNET20,
    global_round: Annotated[
        int, typer.Option(help="Total number of federated learning rounds.")
    ] = 5,
    kd_epochs: Annotated[
        int, typer.Option(help="Number of epochs for knowledge distillation.")
    ] = 5,
    kd_batch_size: Annotated[
        int, typer.Option(help="Batch size for knowledge distillation.")
    ] = 50,
    kd_lr: Annotated[
        float, typer.Option(help="Learning rate for knowledge distillation.")
    ] = 0.1,
    public_size_per_round: Annotated[
        int, typer.Option(help="Number of public samples used per round.")
    ] = 1000,
    private_val_ratio: Annotated[
        float, typer.Option(help="Validation ratio for private datasets.")
    ] = 0.0,
    public_val_ratio: Annotated[
        float, typer.Option(help="Validation ratio for public dataset.")
    ] = 0.0,
    sample_ratio: Annotated[
        float, typer.Option(help="Fraction of clients to sample in each round.")
    ] = 1.0,
    epochs: Annotated[
        int, typer.Option(help="Number of local training epochs per client.")
    ] = 5,
    batch_size: Annotated[
        int, typer.Option(help="Batch size for local training.")
    ] = 50,
    lr: Annotated[
        float, typer.Option(help="Learning rate for the client optimizer.")
    ] = 0.1,
    num_parallels: Annotated[
        int,
        typer.Option(help="Number of parallel threads for training."),
    ] = 10,
    algorithm_name: Annotated[
        AlgorithmName, typer.Option(help="Algorithm to use.")
    ] = AlgorithmName.SCARLET,
    # Algorithm-specific args
    era_temperature: Annotated[
        float, typer.Option(help="Temperature for ERA (DSFL).")
    ] = 0.1,
    enhanced_era_exponent: Annotated[
        float, typer.Option(help="Exponent for Enhanced ERA (SCARLET).")
    ] = 2.0,
    cache_duration: Annotated[
        int, typer.Option(help="Cache duration for soft-labels (SCARLET).")
    ] = 50,
) -> None:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    config = locals()
    run = wandb.init(config=config)

    setup_logging()
    logging.info("\n" + "\n".join([f"  {k}: {v}" for k, v in config.items()]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_split_dir = dataset_root_dir / timestamp
    dataset_split_dir.mkdir(parents=True, exist_ok=True)

    state_dir = state_root_dir / timestamp
    state_dir.mkdir(parents=True, exist_ok=True)

    setup_reproducibility(seed)

    dataset = CommonPartitionedDataset(
        root=dataset_root_dir,
        path=dataset_split_dir,
        num_clients=num_clients,
        seed=seed,
        private_task=private_task,
        public_task=public_task,
        partition=partition,
        dir_alpha=dir_alpha,
        public_size=public_size,
        private_val_ratio=private_val_ratio,
        public_val_ratio=public_val_ratio,
    )
    model_selector = CommonModelSelector(num_classes=dataset.num_classes, seed=seed)

    handler: DSFLServerHandler | SCARLETServerHandler | None = None
    trainer: DSFLClientTrainer | SCARLETClientTrainer | None = None

    common_server_args = CommonServerArgs(
        model_selector=model_selector,
        model_name=model_name,
        dataset=dataset,
        global_round=global_round,
        num_clients=num_clients,
        device=device,
        sample_ratio=sample_ratio,
        kd_epochs=kd_epochs,
        kd_lr=kd_lr,
        kd_batch_size=kd_batch_size,
        seed=seed,
        public_size_per_round=public_size_per_round,
    )
    common_client_args = CommonClientArgs(
        model_selector=model_selector,
        model_name=model_name,
        dataset=dataset,
        seed=seed,
        device=device,
        num_clients=num_clients,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        num_parallels=num_parallels,
        kd_epochs=kd_epochs,
        kd_lr=kd_lr,
        kd_batch_size=kd_batch_size,
        public_size_per_round=public_size_per_round,
        state_dir=state_dir,
    )

    match algorithm_name:
        case AlgorithmName.DSFL:
            handler = DSFLServerHandler(
                common_args=common_server_args,
                era_temperature=era_temperature,
            )
            trainer = DSFLClientTrainer(common_args=common_client_args)
        case AlgorithmName.SCARLET:
            handler = SCARLETServerHandler(
                common_args=common_server_args,
                enhanced_era_exponent=enhanced_era_exponent,
                cache_duration=cache_duration,
            )
            trainer = SCARLETClientTrainer(
                common_args=common_client_args,
            )

    try:
        pipeline = CommonPipeline(
            handler=handler,
            trainer=trainer,
            run=run,
        )
        pipeline.main()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    typer.run(main)

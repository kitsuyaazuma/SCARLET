import logging
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from blazefl.reproducibility import setup_reproducibility
from hydra.core import hydra_config
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter

from algorithm import (
    DSFLClientTrainer,
    DSFLServerHandler,
    SCARLETClientTrainer,
    SCARLETServerHandler,
)
from dataset import CommonPartitionedDataset
from models.selector import CommonModelSelector
from pipeline import CommonPipeline


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    log_dir = hydra_config.HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(log_dir=log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root_dir = Path(cfg.dataset_root_dir)
    dataset_split_dir = dataset_root_dir.joinpath(timestamp)
    share_dir = Path(cfg.share_dir).joinpath(timestamp)
    state_dir = Path(cfg.state_dir).joinpath(timestamp)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    logging.info(f"device: {device}")

    setup_reproducibility(cfg.seed)

    dataset = CommonPartitionedDataset(
        root=dataset_root_dir,
        path=dataset_split_dir,
        num_clients=cfg.num_clients,
        seed=cfg.seed,
        private_task=cfg.private_task,
        public_task=cfg.public_task,
        partition=cfg.partition,
        dir_alpha=cfg.dir_alpha,
        public_size=cfg.public_size,
    )
    model_selector = CommonModelSelector(num_classes=dataset.num_classes, seed=cfg.seed)

    handler_args = {
        "model_selector": model_selector,
        "model_name": cfg.model_name,
        "dataset": dataset,
        "global_round": cfg.global_round,
        "num_clients": cfg.num_clients,
        "kd_epochs": cfg.kd_epochs,
        "kd_batch_size": cfg.kd_batch_size,
        "kd_lr": cfg.kd_lr,
        "public_size_per_round": cfg.public_size_per_round,
        "device": device,
        "sample_ratio": cfg.sample_ratio,
        "seed": cfg.seed,
    }
    trainer_args = {
        "model_selector": model_selector,
        "model_name": cfg.model_name,
        "share_dir": share_dir,
        "state_dir": state_dir,
        "seed": cfg.seed,
        "dataset": dataset,
        "device": device,
        "num_clients": cfg.num_clients,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "kd_epochs": cfg.kd_epochs,
        "kd_batch_size": cfg.kd_batch_size,
        "kd_lr": cfg.kd_lr,
        "num_parallels": cfg.num_parallels,
        "public_size_per_round": cfg.public_size_per_round,
    }

    handler: DSFLServerHandler | SCARLETServerHandler | None = None
    trainer: DSFLClientTrainer | SCARLETClientTrainer | None = None
    match cfg.algorithm.name:
        case "dsfl":
            handler = DSFLServerHandler(
                **handler_args,
                era_temperature=cfg.algorithm.era_temperature,
            )
            trainer = DSFLClientTrainer(
                **trainer_args,
            )
        case "scarlet":
            handler = SCARLETServerHandler(
                **handler_args,
                enhanced_era_exponent=cfg.algorithm.enhanced_era_exponent,
                cache_duration=cfg.algorithm.cache_duration,
            )
            trainer = SCARLETClientTrainer(
                **trainer_args,
            )
        case _:
            raise ValueError(f"Invalid algorithm: {cfg.algorithm.name}")

    try:
        pipeline = CommonPipeline(
            handler=handler,
            trainer=trainer,
            writer=writer,
        )
        pipeline.main()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")


if __name__ == "__main__":
    # NOTE: To use CUDA with multiprocessing, you must use the 'spawn' start method
    mp.set_start_method("spawn")

    main()

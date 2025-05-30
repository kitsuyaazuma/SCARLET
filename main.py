from datetime import datetime
import logging
from pathlib import Path

from blazefl.utils import seed_everything
import hydra
from hydra.core import hydra_config
from omegaconf import DictConfig, OmegaConf
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard.writer import SummaryWriter

from algorithm import (
    DSFLServerHandler,
    DSFLClientTrainer,
    SCARLETServerHandler,
    SCARLETClientTrainer,
)
from pipeline import DSFLPipeline, SCARLETPipeline
from dataset import CommonPartitionedDataset
from models.selector import CommonModelSelector


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

    seed_everything(cfg.seed, device=device)

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
    model_selector = CommonModelSelector(num_classes=dataset.num_classes)

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
    }

    match cfg.algorithm.name:
        case "dsfl":
            handler = DSFLServerHandler(
                **handler_args,
                era_temperature=cfg.algorithm.era_temperature,
            )
            trainer = DSFLClientTrainer(
                **trainer_args,
            )
            pipeline = DSFLPipeline(
                handler=handler,
                trainer=trainer,
                writer=writer,
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
            pipeline = SCARLETPipeline(
                handler=handler,
                trainer=trainer,
                writer=writer,
            )
        case _:
            raise ValueError(f"Invalid algorithm: {cfg.algorithm.name}")

    try:
        pipeline.main()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt: Stopping the pipeline.")
    except Exception as e:
        logging.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    # NOTE: To use CUDA with multiprocessing, you must use the 'spawn' start method
    mp.set_start_method("spawn")

    main()

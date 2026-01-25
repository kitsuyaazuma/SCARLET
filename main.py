import logging
import pprint
import sys
from dataclasses import asdict
from datetime import datetime

import torch
import torch.multiprocessing as mp
import tyro
import wandb

from algorithm import (
    CommonClientArgs,
    CommonPipeline,
    CommonServerArgs,
    DSFLClientTrainer,
    DSFLServerHandler,
    SCARLETClientTrainer,
    SCARLETServerHandler,
)
from config import Config, DSFLConfig, SCARLETConfig
from core import setup_reproducibility
from dataset import CommonPartitionedDataset
from models import CommonModelSelector


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


def main(cfg: Config) -> None:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    run = wandb.init(mode=cfg.wandb.mode, config=asdict(cfg))

    setup_logging()
    logging.info(pprint.pformat(asdict(cfg)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_split_dir = cfg.common.dataset_root_dir / timestamp
    dataset_split_dir.mkdir(parents=True, exist_ok=True)

    state_dir = cfg.common.state_root_dir / timestamp
    state_dir.mkdir(parents=True, exist_ok=True)

    setup_reproducibility(cfg.common.seed)

    dataset = CommonPartitionedDataset(
        root=cfg.common.dataset_root_dir,
        path=dataset_split_dir,
        num_clients=cfg.common.num_clients,
        seed=cfg.common.seed,
        private_task=cfg.common.private_task,
        public_task=cfg.common.public_task,
        partition=cfg.common.partition,
        dir_alpha=cfg.common.dir_alpha,
        public_size=cfg.common.public_size,
    )
    model_selector = CommonModelSelector(
        num_classes=dataset.num_classes, seed=cfg.common.seed
    )

    handler: DSFLServerHandler | SCARLETServerHandler | None = None
    trainer: DSFLClientTrainer | SCARLETClientTrainer | None = None

    common_server_args = CommonServerArgs(
        model_selector=model_selector,
        model_name=cfg.common.model_name,
        dataset=dataset,
        global_round=cfg.common.global_round,
        num_clients=cfg.common.num_clients,
        device=device,
        sample_ratio=cfg.common.sample_ratio,
        kd_epochs=cfg.common.kd_epochs,
        kd_lr=cfg.common.kd_lr,
        kd_batch_size=cfg.common.kd_batch_size,
        seed=cfg.common.seed,
        public_size_per_round=cfg.common.public_size_per_round,
    )
    common_client_args = CommonClientArgs(
        model_selector=model_selector,
        model_name=cfg.common.model_name,
        dataset=dataset,
        seed=cfg.common.seed,
        device=device,
        num_clients=cfg.common.num_clients,
        epochs=cfg.common.epochs,
        lr=cfg.common.lr,
        batch_size=cfg.common.batch_size,
        num_parallels=cfg.common.num_parallels,
        kd_epochs=cfg.common.kd_epochs,
        kd_lr=cfg.common.kd_lr,
        kd_batch_size=cfg.common.kd_batch_size,
        public_size_per_round=cfg.common.public_size_per_round,
        state_dir=state_dir,
    )

    match cfg.algorithm:
        case DSFLConfig():
            handler = DSFLServerHandler(
                common_args=common_server_args,
                era_temperature=cfg.algorithm.era_temperature,
            )
            trainer = DSFLClientTrainer(common_args=common_client_args)
        case SCARLETConfig():
            handler = SCARLETServerHandler(
                common_args=common_server_args,
                enhanced_era_exponent=cfg.algorithm.enhanced_era_exponent,
                cache_duration=cfg.algorithm.cache_duration,
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
    cfg = tyro.cli(Config)
    main(cfg)

import logging
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
import hydra
import shutil

from algorithm import (
    DSFLParallelClientTrainer,
    DSFLServerHandler,
    SCARLETParallelClientTrainer,
    SCARLETServerHandler,
    COMETServerHandler,
    CentralizedServerHandler,
)
from algorithm.cfd import CFDParallelClientTrainer, CFDServerHandler
from algorithm.comet import COMETParallelClientTrainer
from algorithm.individual import IndividualParallelClientTrainer
from algorithm.selectivefd import (
    SelectiveFDParallelClientTrainer,
    SelectiveFDServerHandler,
)
from dataset import PartitionedDataset
from pipeline import (
    CFDPipeline,
    COMETPipeline,
    IndividualPipeline,
    SCARLETPipeline,
    DSFLPipeline,
    SelectiveFDPipeline,
    CentralizedPipeline,
)
from utils import get_cuda_info, seed_everything, get_git_commit_hash

data_dir = None


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:  # noqa: C901
    global data_dir
    print(OmegaConf.to_yaml(cfg))

    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore
    writer = SummaryWriter(log_dir=log_dir)

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if cfg.data_dir == "":
        data_dir = (
            Path(__file__).resolve().parents[0].joinpath("data").joinpath(date_time)
        )
    else:
        data_dir = Path(cfg.data_dir).joinpath(date_time)
    data_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"git commit hash: {get_git_commit_hash()}")
    logging.info(f"cuda: {get_cuda_info()}")

    seed_everything(cfg.seed)

    # data
    partitioned_dataset = PartitionedDataset(
        data_dir=data_dir,
        num_clients=cfg.num_clients,
        partition=cfg.partition,
        dir_alpha=cfg.dir_alpha,
        private_task=cfg.private_task,
        public_task=cfg.public_task,
        public_size=cfg.public_size,
        train_batch_size=cfg.batch_size,
        test_batch_size=cfg.test_batch_size,
        validation_ratio=cfg.validation_ratio,
    )
    partitioned_dataset.save_distribution_stats(dir=Path(log_dir))

    state_dict_dir = data_dir.joinpath("state_dict")
    state_dict_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path(log_dir).joinpath("analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    match cfg.algorithm.name:
        case "dsfl":
            handler = DSFLServerHandler(
                model_name=cfg.server_model,
                global_round=cfg.global_round,
                sample_ratio=cfg.sample_ratio,
                cuda=cuda,
                public_size_per_round=cfg.public_size_per_round,
                era_temperature=cfg.algorithm.era_temperature,
                dataset=partitioned_dataset,
            )
            trainer = DSFLParallelClientTrainer(
                model_name=cfg.client_model,
                num_clients=cfg.num_clients,
                cuda=cuda,
                state_dict_dir=state_dict_dir,
                seed=cfg.seed,
                num_parallels=cfg.num_parallels,
            )
            pipeline = DSFLPipeline(
                handler=handler,
                trainer=trainer,
                writer=writer,
            )
        case "scarlet":
            handler = SCARLETServerHandler(
                model_name=cfg.server_model,
                global_round=cfg.global_round,
                sample_ratio=cfg.sample_ratio,
                cuda=cuda,
                public_size_per_round=cfg.public_size_per_round,
                dataset=partitioned_dataset,
                era_exponent=cfg.algorithm.era_exponent,
                cache_ratio=cfg.algorithm.cache_ratio,
                cache_duration=cfg.algorithm.cache_duration,
                analysis_dir=analysis_dir,
            )
            trainer = SCARLETParallelClientTrainer(
                model_name=cfg.client_model,
                num_clients=cfg.num_clients,
                cuda=cuda,
                state_dict_dir=state_dict_dir,
                seed=cfg.seed,
                num_parallels=cfg.num_parallels,
            )
            pipeline = SCARLETPipeline(
                handler=handler,
                trainer=trainer,
                writer=writer,
            )
        case "comet":
            handler = COMETServerHandler(
                model_name=cfg.server_model,
                global_round=cfg.global_round,
                sample_ratio=cfg.sample_ratio,
                cuda=cuda,
                public_size_per_round=cfg.public_size_per_round,
                dataset=partitioned_dataset,
                num_clusters=cfg.algorithm.num_clusters,
                kmeans_device=cfg.algorithm.kmeans_device,
                enable_cache=cfg.algorithm.enable_cache,
                cache_duration=cfg.algorithm.cache_duration,
            )
            trainer = COMETParallelClientTrainer(
                model_name=cfg.client_model,
                num_clients=cfg.num_clients,
                cuda=cuda,
                state_dict_dir=state_dict_dir,
                seed=cfg.seed,
                num_parallels=cfg.num_parallels,
                regularization_weight=cfg.algorithm.regularization_weight,
                enable_cache=cfg.algorithm.enable_cache,
            )
            pipeline = COMETPipeline(
                handler=handler,
                trainer=trainer,
                writer=writer,
            )
        case "selectivefd":
            handler = SelectiveFDServerHandler(
                model_name=cfg.server_model,
                global_round=cfg.global_round,
                sample_ratio=cfg.sample_ratio,
                cuda=cuda,
                public_size_per_round=cfg.public_size_per_round,
                dataset=partitioned_dataset,
                l1_distance_threshold=cfg.algorithm.l1_distance_threshold,
                enable_cache=cfg.algorithm.enable_cache,
                cache_duration=cfg.algorithm.cache_duration,
            )
            trainer = SelectiveFDParallelClientTrainer(
                model_name=cfg.client_model,
                num_clients=cfg.num_clients,
                cuda=cuda,
                state_dict_dir=state_dict_dir,
                seed=cfg.seed,
                num_parallels=cfg.num_parallels,
                gaussian_kernel_width=cfg.algorithm.gaussian_kernel_width,
                lambda_exponent=cfg.algorithm.lambda_exponent,
                kulsif_test_batch_size=cfg.algorithm.kulsif_test_batch_size,
                dre_threshold=cfg.algorithm.dre_threshold,
                enable_cache=cfg.algorithm.enable_cache,
            )
            pipeline = SelectiveFDPipeline(
                handler=handler,
                trainer=trainer,
                writer=writer,
            )
        case "cfd":
            handler = CFDServerHandler(
                model_name=cfg.server_model,
                global_round=cfg.global_round,
                sample_ratio=cfg.sample_ratio,
                cuda=cuda,
                public_size_per_round=cfg.public_size_per_round,
                dataset=partitioned_dataset,
                enable_cache=cfg.algorithm.enable_cache,
                cache_duration=cfg.algorithm.cache_duration,
            )
            trainer = CFDParallelClientTrainer(
                model_name=cfg.client_model,
                num_clients=cfg.num_clients,
                cuda=cuda,
                state_dict_dir=state_dict_dir,
                seed=cfg.seed,
                num_parallels=cfg.num_parallels,
                enable_cache=cfg.algorithm.enable_cache,
            )
            pipeline = CFDPipeline(
                handler=handler,
                trainer=trainer,
                writer=writer,
            )
        case "individual":
            handler = None  # type: ignore
            trainer = IndividualParallelClientTrainer(
                model_name=cfg.client_model,
                num_clients=cfg.num_clients,
                cuda=cuda,
                state_dict_dir=state_dict_dir,
                seed=cfg.seed,
                num_parallels=cfg.num_parallels,
            )
            pipeline = IndividualPipeline(  # type: ignore
                trainer=trainer,
                writer=writer,
                global_round=cfg.global_round,
            )
        case "centralized":
            handler = CentralizedServerHandler(
                model_name=cfg.server_model,
                global_round=cfg.global_round,
                cuda=cuda,
                dataset=partitioned_dataset,
                epochs=cfg.epochs,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
            )
            trainer = None  # type: ignore
            pipeline = CentralizedPipeline(  # type: ignore
                handler=handler,
                writer=writer,
            )
        case _:
            raise ValueError(f"Invalid algorithm name: {cfg.algorithm}")
    if trainer is not None:
        trainer.setup_datasets(partitioned_dataset)
    if handler is not None:
        handler.test_batch_size = cfg.test_batch_size
        handler.setup_kd_optim(cfg.kd_epochs, cfg.kd_batch_size, cfg.kd_lr)
    if trainer is not None:
        trainer.setup_worker(
            cfg.epochs,
            cfg.batch_size,
            cfg.lr,
            cfg.kd_epochs,
            cfg.kd_batch_size,
            cfg.kd_lr,
            analysis_dir,
        )
    pipeline.main()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    try:
        main()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")
    except Exception as e:
        logging.exception(e)
    finally:
        if data_dir is not None and data_dir.exists():
            shutil.rmtree(data_dir)

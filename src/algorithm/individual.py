from pathlib import Path
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from dataclasses import dataclass

from algorithm.base import (
    BaseSerialClientTrainer,
    BaseClientWorkerProcess,
)
from dataset import PartitionedDataset


@dataclass
class IndividualClientWorkerProcess(BaseClientWorkerProcess):
    pass


def dsfl_client_worker(
    device: str,
    client_id: int,
    process: IndividualClientWorkerProcess,
    dataset: PartitionedDataset,
) -> list[torch.Tensor]:
    process.prepare(device, client_id, dataset)
    process.train()
    process.evaluate()
    process.save()
    return []


class IndividualParallelClientTrainer(BaseSerialClientTrainer):
    def __init__(
        self,
        model_name: str,
        num_clients: int,
        cuda: bool,
        state_dict_dir: Path,
        seed: int,
        num_parallels: int,
    ) -> None:
        super().__init__(
            model_name, num_clients, cuda, state_dict_dir, seed, num_parallels
        )
        self.round = 0

    def setup_worker(
        self,
        epochs: int,
        batch_size: int,
        lr: float,
        kd_epochs: int,
        kd_lr: float,
        kd_batch_size: int,
        analysis_dir: Path,
    ):
        _ = (kd_epochs, kd_lr, kd_batch_size)
        self.process = IndividualClientWorkerProcess(
            model_name=self.model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer_name="SGD",
            criterion_name="cross_entropy",
            state_dict_dir=self.state_dict_dir,
            seed=self.seed,
            analysis_dir=analysis_dir,
        )

    def local_process(self):
        id_list = list(range(self.num_clients))
        pool = mp.Pool(processes=self.num_parallels)
        jobs = [
            pool.apply_async(
                dsfl_client_worker,
                (
                    f"cuda:{client_id % self.device_count}" if self.cuda else "cpu",
                    client_id,
                    self.process,
                    self.dataset,
                ),
            )
            for client_id in id_list
        ]
        for job in tqdm(jobs, desc=f"Round {self.round}"):
            self.cache.append(job.get())
        self.round += 1

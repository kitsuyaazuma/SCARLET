from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
from tqdm import tqdm
from dataclasses import dataclass

from algorithm.base import (
    BaseSerialClientTrainer,
    BaseServerHandler,
    BaseClientWorkerProcess,
)
from dataset import NonLabelDataset, PartitionedDataset
from utils import get_criterion, get_optimizer


@dataclass
class DSFLClientWorkerProcess(BaseClientWorkerProcess):
    kd_epochs: int
    kd_lr: float
    kd_batch_size: int
    kd_optimizer_name: str
    kd_criterion_name: str

    def prepare(self, device: str, client_id: int, dataset: PartitionedDataset):
        super().prepare(device, client_id, dataset)
        self.kd_optimizer = get_optimizer(self.kd_optimizer_name)(
            self.model.parameters(), lr=self.kd_lr
        )
        self.kd_criterion = get_criterion(self.kd_criterion_name)
        if self.state_dict_path.exists():
            self.kd_optimizer.load_state_dict(
                torch.load(self.state_dict_path)["kd_optimizer"]
            )

    def distill(
        self, public_probs: torch.Tensor, public_indices: torch.Tensor
    ) -> float:
        self.model.train()
        epoch_loss, epoch_samples = 0.0, 0
        if public_probs.numel() != 0 and public_indices.numel() != 0:
            public_data_loader = DataLoader(
                Subset(
                    self.dataset.get_public_train_dataset(), public_indices.tolist()
                ),
                batch_size=self.kd_batch_size,
            )
            public_probs_loader = DataLoader(
                NonLabelDataset(data=list(torch.unbind(public_probs, dim=0))),
                batch_size=self.kd_batch_size,
            )
            for kd_epoch in range(self.kd_epochs):
                for (data, _), probs in zip(public_data_loader, public_probs_loader):
                    data, probs = data.to(self.device), probs.to(self.device)

                    output = F.log_softmax(self.model(data), dim=1)
                    kd_loss = self.kd_criterion(output, probs.squeeze(1))

                    self.kd_optimizer.zero_grad()
                    kd_loss.backward()
                    self.kd_optimizer.step()
                    if kd_epoch == self.kd_epochs - 1:
                        epoch_loss += kd_loss.item() * data.size(0)
                        epoch_samples += data.size(0)

        self.save_dict["kd_optimizer"] = self.kd_optimizer.state_dict()
        return epoch_loss / epoch_samples if epoch_samples > 0 else 0.0

    def predict(self, next_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        predict_data_loader = DataLoader(
            Subset(self.dataset.get_public_train_dataset(), next_indices.tolist()),
            batch_size=self.batch_size,
        )

        local_probs: list[torch.Tensor] = []
        with torch.no_grad():
            for data, _ in predict_data_loader:
                data = data.to(self.device)

                output = self.model(data)
                probs = F.softmax(output, dim=1)
                local_probs.extend([prob.detach().cpu() for prob in probs])

        return torch.stack(local_probs), next_indices


def dsfl_client_worker(
    device: str,
    client_id: int,
    process: DSFLClientWorkerProcess,
    dataset: PartitionedDataset,
    public_probs: torch.Tensor,
    public_indices: torch.Tensor,
    next_indices: torch.Tensor,
) -> list[torch.Tensor]:
    process.prepare(device, client_id, dataset)
    process.distill(public_probs, public_indices)
    process.train()
    probs, indices = process.predict(next_indices)
    process.evaluate()
    process.save()
    return [probs, indices]


class DSFLParallelClientTrainer(BaseSerialClientTrainer):
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
        kd_batch_size: int,
        kd_lr: float,
        analysis_dir: Path,
    ):
        self.process = DSFLClientWorkerProcess(
            model_name=self.model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer_name="SGD",
            criterion_name="cross_entropy",
            state_dict_dir=self.state_dict_dir,
            seed=self.seed,
            kd_epochs=kd_epochs,
            kd_batch_size=kd_batch_size,
            kd_lr=kd_lr,
            kd_optimizer_name="SGD",
            kd_criterion_name="kl_div",
            analysis_dir=analysis_dir,
        )

    def local_process(self, payload: list, id_list: list[int]):
        public_probs, public_indices, next_indices = payload

        public_probs.share_memory_()
        public_indices.share_memory_()
        next_indices.share_memory_()

        pool = mp.Pool(processes=self.num_parallels)
        jobs = [
            pool.apply_async(
                dsfl_client_worker,
                (
                    f"cuda:{client_id % self.device_count}" if self.cuda else "cpu",
                    client_id,
                    self.process,
                    self.dataset,
                    public_probs,
                    public_indices,
                    next_indices,
                ),
            )
            for client_id in id_list
        ]
        for job in tqdm(jobs, desc=f"Round {self.round}"):
            self.cache.append(job.get())
        self.round += 1


class DSFLServerHandler(BaseServerHandler):
    def __init__(
        self,
        model_name: str,
        global_round: int,
        sample_ratio: float,
        cuda: bool,
        public_size_per_round: int,
        era_temperature: float,
        dataset: PartitionedDataset,
    ):
        super().__init__(
            model_name,
            global_round,
            sample_ratio,
            cuda,
            public_size_per_round,
            dataset,
        )
        self.public_probs = torch.empty(0)
        self.public_indices = torch.empty(0)
        self.era_temperature = era_temperature
        self.next_public_indices = torch.empty(0)
        self.set_next_public_indices()

    def setup_kd_optim(self, kd_epochs: int, kd_batch_size: int, kd_lr: float):
        self.kd_epochs = kd_epochs
        self.kd_batch_size = kd_batch_size
        self.kd_lr = kd_lr
        self.kd_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.kd_lr)
        self.kd_criterion = F.kl_div

    def set_next_public_indices(self) -> None:
        shuffled_indices = torch.randperm(self.dataset.public_train_size)
        self.next_public_indices = shuffled_indices[: self.public_size_per_round]

    def global_update(self, buffer: list) -> None:
        probs_list = [ele[0] for ele in buffer]
        indices_list = [ele[1] for ele in buffer]

        public_probs_stack = defaultdict(list)
        for probs, indices in zip(probs_list, indices_list):
            for prob, indice in zip(probs, indices):
                public_probs_stack[indice.item()].append(prob)

        public_probs: list[torch.Tensor] = []
        public_indices: list[int] = []
        for indice, probs in public_probs_stack.items():
            public_indices.append(indice)
            mean_prob = torch.stack(probs).mean(dim=0).cpu()
            # Entropy Reduction Aggregation
            era_prob = F.softmax(mean_prob / self.era_temperature, dim=0)
            public_probs.append(era_prob)

        self.model.train()
        public_subset = Subset(self.dataset.get_public_train_dataset(), public_indices)
        public_loader = DataLoader(public_subset, batch_size=self.kd_batch_size)
        public_probs_loader = DataLoader(
            NonLabelDataset(data=public_probs),
            batch_size=self.kd_batch_size,
        )
        for _ in range(self.kd_epochs):
            for (data, target), prob in zip(public_loader, public_probs_loader):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                    prob = prob.cuda(self.device)

                output = F.log_softmax(self.model(data), dim=1)
                prob = prob.squeeze(1)
                kd_loss = self.kd_criterion(output, prob, reduction="batchmean")

                self.kd_optimizer.zero_grad()
                kd_loss.backward()
                self.kd_optimizer.step()

        self.public_indices = torch.tensor(public_indices)
        self.public_probs = torch.stack(public_probs)

        self.set_next_public_indices()

    @property
    def downlink_package(self) -> list[torch.Tensor]:
        return [self.public_probs, self.public_indices, self.next_public_indices]

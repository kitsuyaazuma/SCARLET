import csv
import os
from collections import defaultdict, deque
import enum
from typing import NamedTuple, override
import torch
from dataclasses import dataclass
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from algorithm.dsfl import (
    DSFLClientWorkerProcess,
    DSFLParallelClientTrainer,
    DSFLServerHandler,
)
from dataset import NonLabelDataset, PartitionedDataset


class CacheType(enum.IntEnum):
    NOT_HIT = 0
    ALREADY_HIT = 1
    NEWLY_HIT = 2
    EXPIRED = 3


@dataclass
class SCARLETClientWorkerProcess(DSFLClientWorkerProcess):
    def prepare(self, device: str, client_id: int, dataset: PartitionedDataset):
        super().prepare(device, client_id, dataset)
        self.cache: list[torch.Tensor | None] = [
            None for _ in range(self.dataset.public_size)
        ]
        if self.state_dict_path.exists():
            self.cache = torch.load(self.state_dict_path)["cache"]
            self.kd_optimizer.load_state_dict(
                torch.load(self.state_dict_path)["kd_optimizer"]
            )
        self.metrics_dir = self.analysis_dir.joinpath("metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.metrics_dir.joinpath(f"{self.client_id}.csv")
        self.header = [
            "private_train_loss",
            "private_train_acc",
            "private_val_loss",
            "private_val_acc",
            "public_train_loss",
            "public_val_loss",
        ]
        if not self.csv_path.exists():
            with open(self.csv_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)
        self.metrics: dict[str, float] = {}

    def set_cache(self, new_cache: torch.Tensor):
        self.new_cache_list = new_cache.tolist()

    def update_cache(
        self, indices: torch.Tensor, probs: torch.Tensor, stale_indices: torch.Tensor
    ):
        if stale_indices.numel() != 0:
            for index in stale_indices:
                self.cache[index.item()] = None
        if indices.numel() != 0 and probs.numel() != 0:
            for index, prob in zip(indices, probs):
                self.cache[index.item()] = prob
        self.save_dict["cache"] = self.cache

    def train(self) -> tuple[float, float]:
        loss, acc = super().train()
        self.metrics["private_train_loss"] = loss
        self.metrics["private_train_acc"] = acc
        return loss, acc

    def distill(self, public_probs: torch.Tensor, public_indices: torch.Tensor):
        if public_probs.numel() != 0 and public_indices.numel() != 0:
            public_probs_queue = deque(torch.unbind(public_probs, dim=0))
            public_probs_with_cache = []
            for index, cache in zip(public_indices, self.new_cache_list):
                match cache:
                    case CacheType.NOT_HIT.value:
                        public_probs_with_cache.append(public_probs_queue.popleft())
                    case CacheType.ALREADY_HIT.value:
                        public_probs_with_cache.append(self.cache[index])
                    case CacheType.NEWLY_HIT.value:
                        self.cache[index] = public_probs_queue.popleft()
                        public_probs_with_cache.append(self.cache[index])
                    case CacheType.EXPIRED.value:
                        self.cache[index] = None
                        public_probs_with_cache.append(public_probs_queue.popleft())
            public_probs = torch.stack(public_probs_with_cache)
            assert public_probs.shape[0] == public_indices.shape[0]

        loss = super().distill(public_probs, public_indices)
        self.metrics["public_train_loss"] = loss
        self.save_dict["cache"] = self.cache

    def validate(
        self,
        public_probs: torch.Tensor,
        public_indices: torch.Tensor,
        next_public_indices: torch.Tensor,
    ):
        self.model.eval()
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

                    if kd_epoch == self.kd_epochs - 1:
                        epoch_loss += kd_loss.item() * data.size(0)
                        epoch_samples += data.size(0)
        self.metrics["public_val_loss"] = (
            epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        )

        loss_sum = 0.0
        acc_sum = 0.0
        total = 0
        valset = self.dataset.get_private_validation_dataset(self.client_id)
        val_loader = DataLoader(valset, batch_size=100, shuffle=False)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss_ = F.cross_entropy(outputs, labels)

                _, pred = outputs.topk(5, 1, largest=True, sorted=True)

                labels = labels.view(labels.size(0), -1).expand_as(pred)
                correct = pred.eq(labels).float()

                acc_sum += correct[:, :1].sum().item()

                loss_sum += loss_.item()
                total += labels.size(0)

        loss = loss_sum / len(val_loader)
        acc = acc_sum / total
        self.metrics["private_val_loss"] = loss
        self.metrics["private_val_acc"] = acc

        if next_public_indices.numel() != 0:
            predict_data_loader = DataLoader(
                Subset(
                    self.dataset.get_public_validation_dataset(),
                    next_public_indices.tolist(),
                ),
                batch_size=self.batch_size,
            )

            local_probs: list[torch.Tensor] = []
            with torch.no_grad():
                for data, _ in predict_data_loader:
                    data = data.to(self.device)

                    output = self.model(data)
                    probs = F.softmax(output, dim=1)
                    local_probs.extend([prob.detach().cpu() for prob in probs])

            return torch.stack(local_probs), next_public_indices
        return torch.empty(0), torch.empty(0)

    def save(self):
        super().save()
        with open(self.csv_path, "a") as f:
            writer = csv.writer(f)
            row = [self.metrics[name] for name in self.header]
            writer.writerow(row)


def scarlet_client_worker(
    device: str,
    client_id: int,
    process: SCARLETClientWorkerProcess,
    dataset: PartitionedDataset,
    public_probs: torch.Tensor,
    public_indices: torch.Tensor,
    next_indices: torch.Tensor,
    new_cache: torch.Tensor,
    cache_update: list[torch.Tensor] | None,
    val_public_probs: torch.Tensor,
    val_public_indices: torch.Tensor,
    next_val_public_indices: torch.Tensor,
    round: int | None = None,
) -> list[torch.Tensor]:
    process.prepare(device, client_id, dataset)
    if cache_update is not None:
        process.update_cache(cache_update[0], cache_update[1], cache_update[2])
    process.set_cache(new_cache)
    process.distill(public_probs, public_indices)
    process.train()
    if next_indices.numel() == 0:
        probs, indices = torch.empty(0), torch.empty(0)
    else:
        probs, indices = process.predict(next_indices)
    process.evaluate(round)
    package = [probs, indices]
    if dataset.validation_ratio > 0:
        val_probs, val_indices = process.validate(
            val_public_probs, val_public_indices, next_val_public_indices
        )
        package.extend([val_probs, val_indices])
    process.save()
    return package


class SCARLETParallelClientTrainer(DSFLParallelClientTrainer):
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
        self.process = SCARLETClientWorkerProcess(
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

    def local_process(  # type: ignore  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        payload: tuple[list[torch.Tensor], dict[int, list[torch.Tensor]]],
        id_list: list[int],
    ):
        if self.dataset.validation_ratio == 0:
            public_probs, public_indices, next_indices, new_cache = payload[0]
            val_public_probs, val_public_indices, next_val_public_indices = (
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
            )
        else:
            (
                public_probs,
                public_indices,
                next_indices,
                new_cache,
                val_public_probs,
                val_public_indices,
                next_val_public_indices,
            ) = payload[0]
        cache_update_by_client = payload[1]

        public_probs.share_memory_()
        public_indices.share_memory_()
        next_indices.share_memory_()
        new_cache.share_memory_()
        val_public_probs.share_memory_()
        val_public_indices.share_memory_()
        next_val_public_indices.share_memory_()

        pool = mp.Pool(processes=self.num_parallels)
        jobs = [
            pool.apply_async(
                scarlet_client_worker,
                (
                    f"cuda:{client_id % self.device_count}" if self.cuda else "cpu",
                    client_id,
                    self.process,
                    self.dataset,
                    public_probs,
                    public_indices,
                    next_indices,
                    new_cache,
                    cache_update_by_client.get(client_id, None),
                    val_public_probs,
                    val_public_indices,
                    next_val_public_indices,
                    self.round,
                ),
            )
            for client_id in id_list
        ]
        for job in tqdm(jobs, desc=f"Round {self.round}"):
            self.cache.append(job.get())
        self.round += 1


class ServerCache(NamedTuple):
    prob: torch.Tensor | None
    round: int


class SCARLETServerHandler(DSFLServerHandler):
    def __init__(
        self,
        model_name: str,
        global_round: int,
        sample_ratio: float,
        cuda: bool,
        public_size_per_round: int,
        dataset: PartitionedDataset,
        era_exponent: float,
        cache_ratio: float,
        cache_duration: int,
        analysis_dir: Path,
    ):
        super(DSFLServerHandler, self).__init__(
            model_name, global_round, sample_ratio, cuda, public_size_per_round, dataset
        )
        self.public_probs = torch.empty(0)
        self.public_indices = torch.empty(0)
        self.next_public_indices = torch.empty(0)
        self.new_cache = torch.empty(0)
        self.era_exponent = era_exponent
        self.cache_ratio = cache_ratio
        self.cache_duration = cache_duration
        self.cache: list[ServerCache] = [
            ServerCache(prob=None, round=0) for _ in range(self.dataset.public_size)
        ]
        self.client_mock_caches: list[list[torch.Tensor | None]] = [
            [None for _ in range(self.dataset.public_size)]
            for _ in range(self.dataset.num_clients)
        ]
        self.cache_update_by_client: dict[int, list[torch.Tensor]] = {}
        self.sampled_clients: list[int] = []

        self.metrics_dir = analysis_dir.joinpath("metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.header = [
            "public_train_loss",
            "public_val_loss",
        ]
        self.csv_path = self.metrics_dir.joinpath("server.csv")
        if not self.csv_path.exists():
            with open(self.csv_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)
        self.metrics: dict[str, float] = {}
        self.public_val_probs = torch.empty(0)
        self.public_val_indices = torch.empty(0)
        self.next_public_val_indices = torch.empty(0)

        self.set_next_public_indices()

        self.enable_instrumentation = (
            os.environ.get("ENABLE_SCARLET_INSTRUMENTATION", "false").lower() == "true"
        )
        if self.enable_instrumentation:
            print("[SCARLET] Instrumentation for soft-label disagreement is ENABLED.")
            self.avg_variance: float = 0.0
            self.avg_entropy_of_mean: float = 0.0
            self.avg_client_entropy_mean: float = 0.0
            self.avg_client_entropy_var: float = 0.0

    @override
    def sample_clients(self):
        self.sampled_clients = super().sample_clients()
        return self.sampled_clients

    def set_next_public_indices(self) -> None:
        super().set_next_public_indices()
        next_request_indices = []
        self.next_cached_indices = []
        next_public_indices = self.next_public_indices.tolist()
        for i in next_public_indices:
            if (
                self.cache[i].prob is not None
                and self.cache[i].round + self.cache_duration > self.round
            ):
                self.next_cached_indices.append(i)
            else:
                next_request_indices.append(i)
        self.next_public_indices = torch.tensor(next_request_indices)

        if self.dataset.validation_ratio > 0:
            self.next_public_val_indices = torch.randperm(
                self.dataset.public_validation_size
            )

    def global_update(self, buffer: list[list[torch.Tensor]]) -> None:  # noqa: C901
        probs_list = [ele[0] for ele in buffer]
        indices_list = [ele[1] for ele in buffer]

        public_probs_stack = defaultdict(list)
        for probs, indices in zip(probs_list, indices_list):
            if probs.numel() == 0 and indices.numel() == 0:
                continue
            for prob, indice in zip(probs, indices):
                public_probs_stack[indice.item()].append(prob)

        if self.enable_instrumentation:
            round_variances = []
            round_avg_soft_label_entropies = []
            round_client_entropy_means = []
            round_client_entropy_vars = []

            for _, client_probs_list in public_probs_stack.items():
                if not client_probs_list:
                    continue
                client_probs_tensor = torch.stack(
                    client_probs_list
                )  # (num_clients, num_classes)

                if client_probs_tensor.shape[0] <= 1:
                    continue

                variance_per_class = torch.var(client_probs_tensor, dim=0)
                round_variances.append(variance_per_class.mean().item())

                mean_prob = client_probs_tensor.mean(dim=0)
                avg_soft_label_entropy = -torch.sum(
                    mean_prob * torch.log(mean_prob + 1e-9)
                )
                round_avg_soft_label_entropies.append(avg_soft_label_entropy.item())

                entropies = -torch.sum(
                    client_probs_tensor * torch.log(client_probs_tensor + 1e-9), dim=1
                )
                round_client_entropy_means.append(entropies.mean().item())
                round_client_entropy_vars.append(entropies.var().item())

            self.avg_variance = (
                float(np.mean(round_variances)) if round_variances else 0.0
            )
            self.avg_entropy_of_mean = (
                float(np.mean(round_avg_soft_label_entropies))
                if round_avg_soft_label_entropies
                else 0.0
            )
            self.avg_client_entropy_mean = (
                float(np.mean(round_client_entropy_means))
                if round_client_entropy_means
                else 0
            )
            self.avg_client_entropy_var = (
                float(np.mean(round_client_entropy_vars))
                if round_client_entropy_vars
                else 0
            )

        public_probs: list[torch.Tensor] = []
        public_indices: list[int] = []
        for index, probs_by_index in public_probs_stack.items():
            public_indices.append(index)
            mean_prob = torch.stack(probs_by_index).mean(dim=0).cpu()
            # Enhanced Entropy Reduction Aggregation
            era_prob = mean_prob**self.era_exponent / torch.sum(
                mean_prob**self.era_exponent
            )
            public_probs.append(era_prob)

        # add cached data
        for i in self.next_cached_indices:
            public_indices.append(i)
            public_probs.append(self.cache[i].prob)

        self.calculate_cache_diff(public_indices)
        # update cache
        new_cache = self.update_cache(public_probs, public_indices)

        # update global model
        self.model.train()
        public_subset = Subset(self.dataset.get_public_train_dataset(), public_indices)
        public_loader = DataLoader(public_subset, batch_size=self.kd_batch_size)
        public_probs_loader = DataLoader(
            NonLabelDataset(data=public_probs),
            batch_size=self.kd_batch_size,
        )
        epoch_loss, epoch_samples = 0.0, 0
        for kd_epoch in range(self.kd_epochs):
            for (data, target), prob in zip(public_loader, public_probs_loader):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                    prob = prob.cuda(self.device)

                output = F.log_softmax(self.model(data), dim=1)
                prob = prob.squeeze(1)
                kd_loss = self.kd_criterion(output, prob, reduction="batchmean")
                if kd_epoch == self.kd_epochs - 1:
                    epoch_loss += kd_loss.item() * data.size(0)
                    epoch_samples += data.size(0)

                self.kd_optimizer.zero_grad()
                kd_loss.backward()
                self.kd_optimizer.step()
        self.metrics["public_train_loss"] = epoch_loss / epoch_samples

        if self.dataset.validation_ratio > 0:
            val_probs_list = [ele[2] for ele in buffer]
            val_indices_list = [ele[3] for ele in buffer]
            val_public_probs_stack = defaultdict(list)
            for probs, indices in zip(val_probs_list, val_indices_list):
                if probs.numel() == 0 and indices.numel() == 0:
                    continue
                for prob, indice in zip(probs, indices):
                    val_public_probs_stack[indice.item()].append(prob)
            val_public_probs: list[torch.Tensor] = []
            val_public_indices: list[int] = []
            for index, probs_by_index in val_public_probs_stack.items():
                val_public_indices.append(index)
                mean_prob = torch.stack(probs_by_index).mean(dim=0).cpu()
                # Enhanced Entropy Reduction Aggregation
                era_prob = mean_prob**self.era_exponent / torch.sum(
                    mean_prob**self.era_exponent
                )
                val_public_probs.append(era_prob)
            self.public_val_indices = torch.tensor(val_public_indices)
            self.public_val_probs = torch.stack(val_public_probs)

            self.model.eval()
            public_val_subset = Subset(
                self.dataset.get_public_validation_dataset(), val_public_indices
            )
            public_val_loader = DataLoader(
                public_val_subset, batch_size=self.kd_batch_size
            )
            public_val_probs_loader = DataLoader(
                NonLabelDataset(data=val_public_probs),
                batch_size=self.kd_batch_size,
            )
            val_loss, val_samples = 0.0, 0
            with torch.no_grad():
                for (data, target), prob in zip(
                    public_val_loader, public_val_probs_loader
                ):
                    if self.cuda:
                        data = data.cuda(self.device)
                        target = target.cuda(self.device)
                        prob = prob.cuda(self.device)

                    output = F.log_softmax(self.model(data), dim=1)
                    prob = prob.squeeze(1)
                    kd_loss = self.kd_criterion(output, prob, reduction="batchmean")
                    if kd_epoch == self.kd_epochs - 1:
                        val_loss += kd_loss.item() * data.size(0)
                        val_samples += data.size(0)
            self.metrics["public_val_loss"] = (
                val_loss / val_samples if val_samples > 0 else 0.0
            )

        self.public_indices = torch.tensor(public_indices)
        not_already_cached_probs = [
            prob
            for i, prob in enumerate(public_probs)
            if new_cache[i] != CacheType.ALREADY_HIT
        ]
        if len(not_already_cached_probs) == 0:
            self.public_probs = torch.empty(0)
        else:
            self.public_probs = torch.stack(not_already_cached_probs)
        self.new_cache = torch.tensor([cache.value for cache in new_cache])

        with open(self.csv_path, "a") as f:
            writer = csv.writer(f)
            row = [self.metrics[name] for name in self.header]
            writer.writerow(row)
        self.metrics = {}

        self.set_next_public_indices()

    def calculate_cache_diff(self, public_indices: list[int]) -> None:
        # calculate cache difference for each selected client
        self.cache_update_by_client = {}
        for client_id in range(self.dataset.num_clients):
            update_indices, stale_indices = [], []
            update_probs: list[torch.Tensor] = []

            mock_cache: list[torch.Tensor | None] = self.client_mock_caches[client_id]
            for i in range(len(mock_cache)):
                if i not in public_indices:
                    continue
                client_cache_prob = mock_cache[i]
                server_cache_prob = self.cache[i].prob
                if client_cache_prob is None and server_cache_prob is not None:
                    update_indices.append(i)
                    update_probs.append(server_cache_prob)
                elif client_cache_prob is not None and server_cache_prob is None:
                    stale_indices.append(i)
                elif client_cache_prob is not None and server_cache_prob is not None:
                    if not torch.allclose(client_cache_prob, server_cache_prob):
                        update_indices.append(i)
                        update_probs.append(server_cache_prob)

            self.cache_update_by_client[client_id] = [
                torch.tensor(update_indices)
                if len(update_indices) > 0
                else torch.empty(0),
                torch.stack(update_probs) if len(update_probs) > 0 else torch.empty(0),
                torch.tensor(stale_indices)
                if len(stale_indices) > 0
                else torch.empty(0),
            ]

    def update_cache(
        self, probs: list[torch.Tensor], indices: list[int]
    ) -> list[CacheType]:
        candidate_indices = []
        for i in indices:
            if self.cache[i].prob is None:
                candidate_indices.append(i)
        selected_indices = np.random.choice(
            candidate_indices,
            int(self.cache_ratio * len(candidate_indices)),
            replace=False,
        )
        new_cache = []
        for i, prob in zip(indices, probs):
            if self.cache[i].prob is None:
                if i in selected_indices:
                    self.cache[i] = ServerCache(prob=prob, round=self.round)
                    new_cache.append(CacheType.NEWLY_HIT)
                else:
                    new_cache.append(CacheType.NOT_HIT)
            else:
                if self.round - self.cache[i].round <= self.cache_duration:
                    new_cache.append(CacheType.ALREADY_HIT)
                else:
                    self.cache[i] = ServerCache(prob=None, round=self.round)
                    new_cache.append(CacheType.EXPIRED)
        return new_cache

    @property
    def downlink_package(  # type: ignore  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> tuple[list[torch.Tensor], dict[int, list[torch.Tensor]]]:
        downlink_package = super().downlink_package
        downlink_package.append(self.new_cache)

        if self.dataset.validation_ratio > 0:
            downlink_package.append(self.public_val_probs)
            downlink_package.append(self.public_val_indices)
            downlink_package.append(self.next_public_val_indices)

        # keep mock cache up-to-date for each selected client
        public_indices = downlink_package[1]
        for client_id in self.sampled_clients:
            for i in public_indices:
                self.client_mock_caches[client_id][i] = self.cache[i].prob

        return downlink_package, self.cache_update_by_client

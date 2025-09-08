from collections import defaultdict, deque
import enum
from typing import NamedTuple, Optional
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
        self.cache: list[Optional[torch.Tensor]] = [
            None for _ in range(self.dataset.public_size)
        ]
        if self.state_dict_path.exists():
            self.cache = torch.load(self.state_dict_path)["cache"]
            self.kd_optimizer.load_state_dict(
                torch.load(self.state_dict_path)["kd_optimizer"]
            )

    def set_cache(self, new_cache: torch.Tensor):
        self.new_cache_list = new_cache.tolist()

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

        super().distill(public_probs, public_indices)
        self.save_dict["cache"] = self.cache


def scarlet_client_worker(
    device: str,
    client_id: int,
    process: SCARLETClientWorkerProcess,
    dataset: PartitionedDataset,
    public_probs: torch.Tensor,
    public_indices: torch.Tensor,
    next_indices: torch.Tensor,
    new_cache: torch.Tensor,
) -> list[torch.Tensor]:
    process.prepare(device, client_id, dataset)
    process.set_cache(new_cache)
    process.distill(public_probs, public_indices)
    process.train()
    if next_indices.numel() == 0:
        probs, indices = torch.empty(0), torch.empty(0)
    else:
        probs, indices = process.predict(next_indices)
    process.evaluate()
    process.save()
    return [probs, indices]


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

    def local_process(self, payload: list, id_list: list[int]):
        public_probs, public_indices, next_indices, new_cache = payload

        public_probs.share_memory_()
        public_indices.share_memory_()
        next_indices.share_memory_()
        new_cache.share_memory_()

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
                ),
            )
            for client_id in id_list
        ]
        for job in tqdm(jobs, desc=f"Round {self.round}"):
            self.cache.append(job.get())
        self.round += 1


class ServerCache(NamedTuple):
    prob: Optional[torch.Tensor]
    round: int


def exponential_moving_average(
    tensors: list[torch.Tensor], alpha: Optional[float] = None
) -> torch.Tensor:
    """Exponential Moving Average

    Args:
        tensors (list[torch.Tensor]): list of tensors (latest to oldest)
        alpha (float, optional): smoothing factor. Defaults to 1 / (len(tensors) + 1).
    """
    if alpha is None:
        alpha = 1 / (len(tensors) + 1)
    ema_tensor = tensors[0].detach()
    for i in range(1, len(tensors)):
        ema_tensor = ema_tensor * (1 - alpha) + tensors[i].detach() * alpha
    return ema_tensor


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
        self.set_next_public_indices()

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

    def global_update(self, buffer: list) -> None:  # noqa: C901
        probs_list = [ele[0] for ele in buffer]
        indices_list = [ele[1] for ele in buffer]

        public_probs_stack = defaultdict(list)
        for probs, indices in zip(probs_list, indices_list):
            if probs.numel() == 0 and indices.numel() == 0:
                continue
            for prob, indice in zip(probs, indices):
                public_probs_stack[indice.item()].append(prob)

        public_probs: list[torch.Tensor] = []
        public_indices: list[int] = []
        for indice, probs in public_probs_stack.items():
            public_indices.append(indice)
            mean_prob = torch.stack(probs).mean(dim=0).cpu()
            # Entropy Reduction Aggregation
            era_prob = mean_prob**self.era_exponent / torch.sum(
                mean_prob**self.era_exponent
            )
            public_probs.append(era_prob)

        # add cached data
        for i in self.next_cached_indices:
            public_indices.append(i)
            public_probs.append(self.cache[i].prob)

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

        self.set_next_public_indices()

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
    def downlink_package(self) -> list[torch.Tensor]:
        downlink_package = super().downlink_package
        downlink_package.append(self.new_cache)
        return downlink_package

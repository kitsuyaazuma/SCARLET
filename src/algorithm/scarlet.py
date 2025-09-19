from collections import defaultdict, deque
import enum
from typing import NamedTuple, Optional, override
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
    cache_update: list[torch.Tensor] | None,
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

    def local_process(  # type: ignore  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        payload: tuple[list[torch.Tensor], dict[int, list[torch.Tensor]]],
        id_list: list[int],
    ):
        public_probs, public_indices, next_indices, new_cache = payload[0]
        cache_update_by_client = payload[1]

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
                    cache_update_by_client.get(client_id, None),
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
        self.client_mock_caches = [
            [None for _ in range(self.dataset.public_size)]
            for _ in range(self.dataset.num_clients)
        ]
        self.cache_update_by_client: dict[int, list[torch.Tensor]] = {}
        self.sampled_clients: list[int] = []
        self.set_next_public_indices()

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

    def global_update(self, buffer: list[list[torch.Tensor]]) -> None:  # noqa: C901
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

        # calculate cache difference for each selected client
        self.cache_update_by_client = {}
        for client_id in self.sampled_clients:
            update_indices, update_probs, stale_indices = [], [], []
            mock_cache = self.client_mock_caches[client_id]
            for i in public_indices:
                if mock_cache[i] is None and self.cache[i].prob is not None:
                    update_indices.append(i)
                    update_probs.append(self.cache[i].prob)
                elif mock_cache[i] is not None and self.cache[i].prob is None:
                    stale_indices.append(i)
                elif mock_cache[i] is not None and self.cache[i].prob is not None:
                    assert isinstance(mock_cache[i], torch.Tensor) and isinstance(
                        self.cache[i].prob, torch.Tensor
                    )
                    if not torch.allclose(mock_cache[i], self.cache[i].prob):
                        update_indices.append(i)
                        update_probs.append(self.cache[i].prob)

            self.cache_update_by_client[client_id] = [
                torch.tensor(update_indices)
                if len(update_indices) > 0
                else torch.empty(0),
                torch.stack(update_probs) if len(update_probs) > 0 else torch.empty(0),
                torch.tensor(stale_indices)
                if len(stale_indices) > 0
                else torch.empty(0),
            ]

        # update cache
        new_cache = self.update_cache(public_probs, public_indices)

        # keep cache up-to-date for each selected client
        for client_id in self.sampled_clients:
            for i in public_indices:
                self.client_mock_caches[client_id][i] = self.cache[i].prob

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
    def downlink_package(  # type: ignore  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> tuple[list[torch.Tensor], dict[int, list[torch.Tensor]]]:
        downlink_package = super().downlink_package
        downlink_package.append(self.new_cache)
        return downlink_package, self.cache_update_by_client

from collections import defaultdict, deque
from typing import Optional
import torch
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from pathlib import Path

from algorithm.dsfl import (
    DSFLClientWorkerProcess,
    DSFLParallelClientTrainer,
    DSFLServerHandler,
)
from dataset import NonLabelDataset, PartitionedDataset
from algorithm.scarlet import ServerCache, CacheType


@dataclass
class CFDClientWorkerProcess(DSFLClientWorkerProcess):
    cache_enabled: bool = False

    def prepare(self, device: str, client_id: int, dataset: PartitionedDataset):
        super().prepare(device, client_id, dataset)
        if self.cache_enabled:
            self.cache: list[Optional[torch.Tensor]] = [
                None for _ in range(self.dataset.public_size)
            ]
            if self.state_dict_path.exists():
                self.cache = torch.load(self.state_dict_path)["cache"]

    def set_cache(self, new_cache: torch.Tensor):
        self.new_cache_list = new_cache.tolist()

    def predict(self, next_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs, indices = super().predict(next_indices)
        labels = torch.tensor([prob.argmax() for prob in probs])
        return labels, indices

    def distill(self, public_probs: torch.Tensor, public_indices: torch.Tensor):
        if not self.cache_enabled:
            super().distill(public_probs, public_indices)
            return

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


def cfd_client_worker(
    device: str,
    client_id: int,
    process: CFDClientWorkerProcess,
    dataset: PartitionedDataset,
    public_probs: torch.Tensor,
    public_indices: torch.Tensor,
    next_indices: torch.Tensor,
    new_cache: Optional[torch.Tensor],
) -> list[torch.Tensor]:
    process.prepare(device, client_id, dataset)
    if new_cache is not None:
        process.set_cache(new_cache)
    process.distill(public_probs, public_indices)
    process.train()
    probs, indices = process.predict(next_indices)
    process.evaluate()
    process.save()
    return [probs, indices]


class CFDParallelClientTrainer(DSFLParallelClientTrainer):
    def __init__(
        self,
        model_name: str,
        num_clients: int,
        cuda: bool,
        state_dict_dir: Path,
        seed: int,
        num_parallels: int,
        enable_cache: bool = False,
    ):
        super().__init__(
            model_name, num_clients, cuda, state_dict_dir, seed, num_parallels
        )
        self.cache_enabled = enable_cache

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
        self.process = CFDClientWorkerProcess(
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
            cache_enabled=self.cache_enabled,
        )

    def local_process(self, payload: list, id_list: list[int]):
        public_probs, public_indices, next_indices, *_ = payload

        public_probs.share_memory_()
        public_indices.share_memory_()
        next_indices.share_memory_()
        if self.cache_enabled:
            new_cache = _[0]
            new_cache.share_memory_()

        pool = mp.Pool(processes=self.num_parallels)
        jobs = [
            pool.apply_async(
                cfd_client_worker,
                (
                    f"cuda:{client_id % self.device_count}" if self.cuda else "cpu",
                    client_id,
                    self.process,
                    self.dataset,
                    public_probs,
                    public_indices,
                    next_indices,
                    new_cache if self.cache_enabled else None,
                ),
            )
            for client_id in id_list
        ]
        for job in tqdm(jobs, desc=f"Round {self.round}"):
            self.cache.append(job.get())
        self.round += 1


class CFDServerHandler(DSFLServerHandler):
    def __init__(
        self,
        model_name: str,
        global_round: int,
        sample_ratio: float,
        cuda: bool,
        public_size_per_round: int,
        dataset: PartitionedDataset,
        enable_cache: bool = False,
        cache_duration: int = 0,
    ):
        super(DSFLServerHandler, self).__init__(
            model_name, global_round, sample_ratio, cuda, public_size_per_round, dataset
        )
        self.public_probs = torch.empty(0)
        self.public_indices = torch.empty(0)
        self.next_public_indices = torch.empty(0)
        if enable_cache:
            self.new_cache = torch.empty(0)
            self.cache_enabled = True
            self.cache_duration = cache_duration
            self.cache: list[ServerCache] = [
                ServerCache(prob=None, round=0) for _ in range(self.dataset.public_size)
            ]
        else:
            self.cache_enabled = False
        self.set_next_public_indices()

    def set_next_public_indices(self) -> None:
        super().set_next_public_indices()
        if not self.cache_enabled:
            return
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

    def update_cache(
        self, probs: list[torch.Tensor], indices: list[int]
    ) -> list[CacheType]:
        candidate_indices = []
        for i in indices:
            if self.cache[i].prob is None:
                candidate_indices.append(i)
        selected_indices = np.random.choice(
            candidate_indices,
            len(candidate_indices),
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

    def global_update(self, buffer: list) -> None:  # noqa: C901
        labels_list = [ele[0] for ele in buffer]
        indices_list = [ele[1] for ele in buffer]

        public_probs_stack = defaultdict(list)
        num_classes = self.dataset.num_classes
        for labels, indices in zip(labels_list, indices_list):
            if labels.numel() == 0 and indices.numel() == 0:
                continue
            for label, indice in zip(labels, indices):
                prob = torch.zeros(num_classes)
                prob[label.item()] = 1.0
                public_probs_stack[indice.item()].append(prob)

        public_probs: list[torch.Tensor] = []
        public_indices: list[int] = []
        for indice, probs in public_probs_stack.items():
            public_indices.append(indice)
            mean_prob = torch.stack(probs).mean(dim=0).cpu()
            public_probs.append(mean_prob)

        if self.cache_enabled:
            for i in self.next_cached_indices:
                public_indices.append(i)
                public_probs.append(self.cache[i].prob)
            new_cache = self.update_cache(public_probs, public_indices)

        # update global model
        self.model.train()
        public_subset = Subset(self.dataset.get_public_train_dataset(), public_indices)
        public_loader = DataLoader(public_subset, batch_size=self.kd_batch_size)
        public_probs_loader = DataLoader(
            NonLabelDataset(data=public_probs),
            batch_size=self.kd_batch_size,
        )
        public_model_probs: list[torch.Tensor] = []
        for kd_epoch in range(self.kd_epochs):
            for (data, target), prob in zip(public_loader, public_probs_loader):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                    prob = prob.cuda(self.device)

                output = self.model(data)
                if kd_epoch == self.kd_epochs - 1:
                    with torch.no_grad():
                        model_probs = F.softmax(output, dim=1)
                        public_model_probs.extend(
                            [model_prob.detach().cpu() for model_prob in model_probs]
                        )
                log_softmax_output = F.log_softmax(output, dim=1)
                prob = prob.squeeze(1)

                kd_loss = self.kd_criterion(
                    log_softmax_output, prob, reduction="batchmean"
                )

                self.kd_optimizer.zero_grad()
                kd_loss.backward()
                self.kd_optimizer.step()

        # prepare package
        self.public_indices = torch.tensor(public_indices)

        if self.cache_enabled:
            not_already_cached_probs = [
                prob
                for i, prob in enumerate(public_model_probs)
                if new_cache[i] != CacheType.ALREADY_HIT
            ]
            if len(not_already_cached_probs) == 0:
                self.public_probs = torch.empty(0)
            else:
                self.public_probs = torch.stack(not_already_cached_probs)
            self.new_cache = torch.tensor([cache.value for cache in new_cache])
        else:
            self.public_probs = torch.stack(public_model_probs)

        self.set_next_public_indices()

    @property
    def downlink_package(self) -> list[torch.Tensor]:
        downlink_package = super().downlink_package
        if self.cache_enabled:
            downlink_package.append(self.new_cache)
        return downlink_package

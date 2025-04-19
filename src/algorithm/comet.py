from collections import defaultdict, deque
from typing import Optional
import torch
from sklearn.cluster import KMeans
from dataclasses import dataclass
from fast_pytorch_kmeans import KMeans as FastKMeans
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from pathlib import Path
import numpy as np

from algorithm.dsfl import (
    DSFLClientWorkerProcess,
    DSFLParallelClientTrainer,
    DSFLServerHandler,
)
from dataset import NonLabelDataset, PartitionedDataset
from algorithm.scarlet import ServerCache, CacheType


@dataclass
class COMETClientWorkerProcess(DSFLClientWorkerProcess):
    regularization_weight: float
    cache_enabled: bool

    def prepare(self, device: str, client_id: int, dataset: PartitionedDataset):
        super().prepare(device, client_id, dataset)
        if self.cache_enabled:
            self.cache: list[Optional[torch.Tensor]] = [
                None for _ in range(self.dataset.public_size)
            ]
        if self.state_dict_path.exists():
            if self.cache_enabled:
                self.cache = torch.load(self.state_dict_path)["cache"]

    def get_best_centroids(
        self, public_centroids: torch.Tensor, public_indices: torch.Tensor
    ) -> torch.Tensor:
        if public_centroids.numel() == 0 or public_indices.numel() == 0:
            return torch.empty(0)
        best_centroids = []
        probs, _ = self.predict(public_indices)
        for centroids, prob in zip(public_centroids, probs):
            distances = []
            for centroid in centroids:
                distances.append(torch.linalg.vector_norm(prob - centroid).item())
            min_idx = np.argmin(distances)
            best_centroids.append(centroids[min_idx])
        return torch.stack(best_centroids)

    def set_cache(self, new_cache: torch.Tensor):
        self.new_cache_list = new_cache.tolist()

    def distill(self, public_probs: torch.Tensor, public_indices: torch.Tensor):  # noqa: C901
        if self.cache_enabled:
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

        self.model.train()
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
            for _ in range(self.kd_epochs):
                for (data, _), probs in zip(public_data_loader, public_probs_loader):
                    data, probs = data.to(self.device), probs.to(self.device)

                    output = F.log_softmax(self.model(data), dim=1)
                    kd_loss = (
                        self.kd_criterion(output, probs.squeeze(1))
                        * self.regularization_weight
                    )

                    self.kd_optimizer.zero_grad()
                    kd_loss.backward()
                    self.kd_optimizer.step()

        self.save_dict["kd_optimizer"] = self.kd_optimizer.state_dict()
        if self.cache_enabled:
            self.save_dict["cache"] = self.cache


def comet_client_worker(
    device: str,
    client_id: int,
    process: COMETClientWorkerProcess,
    dataset: PartitionedDataset,
    public_centroids: torch.Tensor,
    public_indices: torch.Tensor,
    next_indices: torch.Tensor,
    new_cache: Optional[torch.Tensor],
) -> list[torch.Tensor]:
    process.prepare(device, client_id, dataset)
    if new_cache is not None:
        process.set_cache(new_cache)
    public_probs = process.get_best_centroids(public_centroids, public_indices)
    process.distill(public_probs, public_indices)
    process.train()
    probs, indices = process.predict(next_indices)
    process.evaluate()
    process.save()
    return [probs, indices]


class COMETParallelClientTrainer(DSFLParallelClientTrainer):
    def __init__(
        self,
        model_name: str,
        num_clients: int,
        cuda: bool,
        state_dict_dir: Path,
        seed: int,
        num_parallels: int,
        regularization_weight: float,
        enable_cache: bool = False,
    ) -> None:
        super().__init__(
            model_name, num_clients, cuda, state_dict_dir, seed, num_parallels
        )
        self.regularization_weight = regularization_weight
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
        self.process = COMETClientWorkerProcess(
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
            regularization_weight=self.regularization_weight,
            cache_enabled=self.cache_enabled,
        )

    def local_process(self, payload: list, id_list: list[int]):
        public_centroids, public_indices, next_indices, *_ = payload

        public_centroids.share_memory_()
        public_indices.share_memory_()
        next_indices.share_memory_()
        if self.cache_enabled:
            new_cache = _[0]
            new_cache.share_memory_()

        pool = mp.Pool(processes=self.num_parallels)
        jobs = [
            pool.apply_async(
                comet_client_worker,
                (
                    f"cuda:{client_id % self.device_count}" if self.cuda else "cpu",
                    client_id,
                    self.process,
                    self.dataset,
                    public_centroids,
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


class COMETServerHandler(DSFLServerHandler):
    def __init__(
        self,
        model_name: str,
        global_round: int,
        sample_ratio: float,
        cuda: bool,
        public_size_per_round: int,
        dataset: PartitionedDataset,
        num_clusters: int,
        kmeans_device: str,
        enable_cache: bool = False,
        cache_duration: int = 0,
    ):
        super(DSFLServerHandler, self).__init__(
            model_name, global_round, sample_ratio, cuda, public_size_per_round, dataset
        )
        self.public_centroids = torch.empty(0)
        self.public_indices = torch.empty(0)
        self.next_public_indices = torch.empty(0)
        self.new_cache = torch.empty(0)
        self.num_clusters = num_clusters
        self.kmeans_device = kmeans_device
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
        public_centroids: list[torch.Tensor] = []
        for indice, probs in public_probs_stack.items():
            public_indices.append(indice)
            candidate_centroids = self.get_candidate_centroids(probs)
            public_centroids.append(torch.stack(candidate_centroids))
            mean_prob = torch.stack(probs).mean(dim=0).cpu()
            public_probs.append(mean_prob)

        if self.cache_enabled:
            for i in self.next_cached_indices:
                public_indices.append(i)
                public_probs.append(self.cache[i].prob)
            new_cache = self.update_cache(public_probs, public_indices)

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

        if self.cache_enabled:
            not_already_cached_centroids = [
                centroid
                for i, centroid in enumerate(public_centroids)
                if new_cache[i] != CacheType.ALREADY_HIT
            ]
            if len(not_already_cached_centroids) == 0:
                self.public_centroids = torch.empty(0)
            else:
                self.public_centroids = torch.stack(not_already_cached_centroids)
            self.new_cache = torch.tensor([cache.value for cache in new_cache])
        else:
            self.public_centroids = torch.stack(public_centroids)

        self.set_next_public_indices()

    def get_candidate_centroids(self, probs: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.kmeans_device == "cpu":
            centroids = []
            X = [prob.numpy() for prob in probs]
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(X)
            for c in kmeans.cluster_centers_:
                centroids.append(torch.tensor(c, dtype=torch.float32))
            return centroids
        elif self.kmeans_device == "cuda":
            x = torch.stack(probs).to(self.device)
            kmeans = FastKMeans(n_clusters=self.num_clusters)
            kmeans.fit(x)
            centroids = kmeans.centroids
            assert isinstance(centroids, torch.Tensor)
            return list(torch.unbind(centroids.to("cpu"), dim=0))
        else:
            raise ValueError("Invalid device")

    @property
    def downlink_package(self) -> list[torch.Tensor]:
        downlink_package = [
            self.public_centroids,
            self.public_indices,
            self.next_public_indices,
        ]
        if self.cache_enabled:
            downlink_package.append(self.new_cache)
        return downlink_package

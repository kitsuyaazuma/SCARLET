from collections import defaultdict, deque
from typing import DefaultDict, Optional
import torch
import random
from dataclasses import dataclass
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from filelock import FileLock
from fedlab.contrib.dataset import Subset as FedlabSubset

from algorithm.dsfl import (
    DSFLClientWorkerProcess,
    DSFLParallelClientTrainer,
    DSFLServerHandler,
)
from dataset import NonLabelDataset, PartitionedDataset
from torchvision import transforms
from algorithm.scarlet import ServerCache, CacheType

calc_dtype = {
    "cifar10": torch.float64,
    "cifar100": torch.float64,
    "tiny-imagenet-200": torch.float64,
    "caltech256": torch.float64,
}


class GrayscaleToRGB(object):
    def __call__(self, img):
        if img.mode == "L":
            img = img.convert("RGB")
        return img


@dataclass
class SelectiveFDClientWorkerProcess(DSFLClientWorkerProcess):
    gaussian_kernel_width: float
    lambda_exponent: float
    dre_threshold: float
    kulsif_test_batch_size: int
    cache_enabled: bool = False

    def prepare(self, device: str, client_id: int, dataset: PartitionedDataset):
        super().prepare(device, client_id, dataset)
        self.outlier_public_indices: DefaultDict[int, bool] = defaultdict(bool)
        if self.cache_enabled:
            self.cache: list[Optional[torch.Tensor]] = [
                None for _ in range(self.dataset.public_size)
            ]
        if self.state_dict_path.exists():
            self.outlier_public_indices = torch.load(self.state_dict_path)[
                "outlier_public_indices"
            ]
            self.save_dict["outlier_public_indices"] = self.outlier_public_indices
            if self.cache_enabled:
                self.cache = torch.load(self.state_dict_path)["cache"]
        else:
            if self.device.startswith("cuda"):
                # avoid CUDA out of memory
                with FileLock(f"/tmp/{str(self.device)}.lock"):
                    self.estimate_density_ratio()
            else:
                self.estimate_density_ratio()

    def set_cache(self, new_cache: torch.Tensor):
        self.new_cache_list = new_cache.tolist()

    def estimate_density_ratio(self):  # noqa: C901
        trainset = self.dataset.get_private_train_dataset(self.client_id)
        assert isinstance(trainset, FedlabSubset)
        public_trainset = self.dataset.get_public_train_dataset()
        assert isinstance(public_trainset, FedlabSubset)
        _dtype = calc_dtype[self.dataset.private_task]
        with torch.no_grad():
            shape = trainset.data[0].shape
            shape = (32, 32, 3)
            x = torch.from_numpy(
                np.random.uniform(
                    low=-1.0,
                    high=1.0,
                    size=(len(trainset) // 2, shape[0] * shape[1] * shape[2]),
                )
            ).to(_dtype)
            shuffled_indices = random.sample(range(len(trainset)), len(trainset) // 2)
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((shape[0], shape[1])),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            train_data = []
            for j in shuffled_indices[: int(len(shuffled_indices) * 0.9)]:
                train_data.append(transform(trainset.data[j]))
            y = torch.stack([td.view(-1) for td in train_data]).to(_dtype)
            kulsif = KuLSIF(
                x=x,
                y=y,
                gaussian_kernel_width=self.gaussian_kernel_width,
                lambda_=len(trainset) ** self.lambda_exponent,
                device=torch.device(self.device),
            )
            eval_data = []
            for j in shuffled_indices[int(len(shuffled_indices) * 0.9) :]:
                eval_data.append(transform(trainset.data[j]))
            z = [ed.view(-1) for ed in eval_data]
            eval_ratio = kulsif.estimate(torch.stack(z).to(_dtype))
            threshold = np.quantile(eval_ratio, q=self.dre_threshold)

            z = []
            test_data = []
            if self.dataset.public_task == "caltech256":
                transform = transforms.Compose(
                    [
                        GrayscaleToRGB(),
                        transforms.ToTensor(),
                        transforms.Resize((shape[0], shape[1])),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )

            for j in range(len(public_trainset)):
                test_data.append(transform(public_trainset.data[j]))
            for start_idx in range(0, len(test_data), self.kulsif_test_batch_size):
                end_idx = min(start_idx + self.kulsif_test_batch_size, len(test_data))
                z_batch = [td.view(-1) for td in test_data[start_idx:end_idx]]
                test_ratio_batch = kulsif.estimate(torch.stack(z_batch).to(_dtype))
                for j, r in enumerate(test_ratio_batch):
                    if r < threshold:
                        self.outlier_public_indices[start_idx + j] = True

            del kulsif
            torch.cuda.empty_cache()

        self.save_dict["outlier_public_indices"] = self.outlier_public_indices

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

    def predict(self, next_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inlier_next_indices = [
            idx for idx in next_indices.tolist() if not self.outlier_public_indices[idx]
        ]
        if len(inlier_next_indices) == 0:
            return torch.empty(0), torch.empty(0)
        next_indices = torch.tensor(inlier_next_indices)
        return super().predict(next_indices)


class KuLSIF:
    """
    Statistical Analysis of Kernel-Based Least-Squares Density-Ratio Estimation
    https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=cf3cf21510af20118a6e38509dbac4ca359e4320
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        gaussian_kernel_width: float,
        lambda_: float,
        device: torch.device,
    ):
        self.device = device
        self.x = x.to(self.device)
        self.n = x.shape[0]
        self.y = y.to(self.device)
        self.m = y.shape[0]
        self.gaussian_kernel_width = gaussian_kernel_width
        self.lambda_ = lambda_
        self.k = self._calculate_gaussian_kernel

        self.k_11 = self.k(self.x.unsqueeze(0), self.x.unsqueeze(1))
        self.k_12 = self.k(self.x.unsqueeze(1), self.y.unsqueeze(0))
        self.alpha = self._calculate_alpha()
        del self.k_11, self.k_12
        torch.cuda.empty_cache()

    def _calculate_gaussian_kernel(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            # NOTE: High GPU memory consumption
            distance_matrix = torch.linalg.norm(a - b, ord=2, dim=2)
            k = torch.exp(-(distance_matrix**2) / (2 * self.gaussian_kernel_width**2))
            del distance_matrix
            torch.cuda.empty_cache()
        return k

    def _calculate_alpha(self) -> torch.Tensor:
        with torch.no_grad():
            lhs_matrix = self.k_11 / self.n + self.lambda_ * torch.eye(
                self.n, device=self.device
            )
            try:
                inverse_lhs_matrix = torch.linalg.inv(lhs_matrix)
            except torch.linalg.LinAlgError:
                inverse_lhs_matrix = torch.linalg.pinv(lhs_matrix)

            rhs_matrix = -torch.mm(
                self.k_12,
                torch.ones((self.m, 1), device=self.device).to(self.k_12.dtype),
            ) / (self.n * self.m * self.lambda_)

            assert lhs_matrix.dtype == inverse_lhs_matrix.dtype
            alpha = torch.mm(inverse_lhs_matrix, rhs_matrix)

            del lhs_matrix, inverse_lhs_matrix, rhs_matrix
            torch.cuda.empty_cache()
        return alpha

    def estimate(self, z: torch.Tensor) -> np.ndarray:
        z = z.to(self.device)
        with torch.no_grad():
            alpha_column = self.alpha.view(self.n, 1)
            k_zx = self.k(z.unsqueeze(1), self.x.unsqueeze(0))
            k_zy = self.k(z.unsqueeze(1), self.y.unsqueeze(0))

            assert k_zx.dtype == alpha_column.dtype
            left = torch.mm(k_zx, alpha_column).view(-1)
            right = torch.mean(k_zy, dim=1) / self.lambda_

            wz = left + right

        wz_numpy = wz.cpu().numpy()
        assert isinstance(wz_numpy, np.ndarray)
        return wz_numpy


def selectivefd_client_worker(
    device: str,
    client_id: int,
    process: SelectiveFDClientWorkerProcess,
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


class SelectiveFDParallelClientTrainer(DSFLParallelClientTrainer):
    def __init__(
        self,
        model_name: str,
        num_clients: int,
        cuda: bool,
        state_dict_dir: Path,
        seed: int,
        num_parallels: int,
        gaussian_kernel_width: float,
        lambda_exponent: float,
        kulsif_test_batch_size: int,
        dre_threshold: float,
        enable_cache: bool = False,
    ) -> None:
        super().__init__(
            model_name, num_clients, cuda, state_dict_dir, seed, num_parallels
        )
        self.gaussian_kernel_width = gaussian_kernel_width
        self.lambda_exponent = lambda_exponent
        self.kulsif_test_batch_size = kulsif_test_batch_size
        self.dre_threshold = dre_threshold
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
        self.process = SelectiveFDClientWorkerProcess(
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
            gaussian_kernel_width=self.gaussian_kernel_width,
            lambda_exponent=self.lambda_exponent,
            dre_threshold=self.dre_threshold,
            kulsif_test_batch_size=self.kulsif_test_batch_size,
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
                selectivefd_client_worker,
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


class SelectiveFDServerHandler(DSFLServerHandler):
    def __init__(
        self,
        model_name: str,
        global_round: int,
        sample_ratio: float,
        cuda: bool,
        public_size_per_round: int,
        dataset: PartitionedDataset,
        l1_distance_threshold: float,
        enable_cache: bool = False,
        cache_duration: int = 0,
    ):
        super(DSFLServerHandler, self).__init__(
            model_name, global_round, sample_ratio, cuda, public_size_per_round, dataset
        )
        self.public_probs = torch.empty(0)
        self.public_indices = torch.empty(0)
        self.next_public_indices = torch.empty(0)
        if enable_cache:  # NOTE: Cache module added from SCARLET
            self.new_cache = torch.empty(0)
            self.cache_enabled = True
            self.cache_duration = cache_duration
            self.cache: list[ServerCache] = [
                ServerCache(prob=None, round=0) for _ in range(self.dataset.public_size)
            ]
        else:
            self.cache_enabled = False
        self.l1_distance_threshold = l1_distance_threshold
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
        for indice, probs in public_probs_stack.items():
            mean_prob = torch.stack(probs).mean(dim=0).cpu()
            hard_prob = mean_prob.clone()
            hard_prob[mean_prob == mean_prob.max()] = 1.0
            hard_prob[mean_prob != mean_prob.max()] = 0.0
            l1_distance = torch.sum(torch.abs(hard_prob - mean_prob)).item()
            if l1_distance < self.l1_distance_threshold:
                public_indices.append(indice)
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
        else:
            self.public_probs = torch.stack(public_probs)

        self.set_next_public_indices()

    @property
    def downlink_package(self) -> list[torch.Tensor]:
        downlink_package = super().downlink_package
        if self.cache_enabled:
            downlink_package.append(self.new_cache)
        return downlink_package

import threading
from collections import defaultdict, deque
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from multiprocessing.pool import ApplyResult
from pathlib import Path
from typing import NamedTuple

import torch
from blazefl.core import (
    BaseServerHandler,
    ProcessPoolClientTrainer,
    SHMHandle,
)
from blazefl.reproducibility import (
    RNGSuite,
    create_rng_suite,
    setup_reproducibility,
)
from torch import multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from algorithm.dsfl import DSFLClientTrainer, DSFLServerHandler
from dataset import CommonPartitionedDataset
from dataset.dataset import CommonPartitionType
from models import CommonModelSelector
from models.selector import CommonModelName


@dataclass
class SCARLETUplinkPackage:
    cid: int
    soft_labels: torch.Tensor
    indices: torch.Tensor
    metadata: dict


class SCARLETProcessPoolUplinkPackage(SCARLETUplinkPackage):
    soft_labels: torch.Tensor | SHMHandle  # type: ignore
    indices: torch.Tensor | SHMHandle  # type: ignore


@dataclass
class SCARLETDownlinkPackage:
    soft_labels: torch.Tensor | None
    indices: torch.Tensor | None
    next_indices: torch.Tensor
    cache_signals: torch.Tensor | None


class CacheSignal(IntEnum):
    CACHED = 0
    NEWLY_CACHED = 1
    EXPIRED = 2


class GlobalCacheEntry(NamedTuple):
    soft_label: torch.Tensor | None
    round: int


class SCARLETServerHandler(
    BaseServerHandler[SCARLETUplinkPackage, SCARLETDownlinkPackage]
):
    def __init__(
        self,
        model_selector: CommonModelSelector,
        model_name: str,
        dataset: CommonPartitionedDataset,
        global_round: int,
        num_clients: int,
        sample_ratio: float,
        device: str,
        kd_epochs: int,
        kd_batch_size: int,
        kd_lr: float,
        public_size_per_round: int,
        enhanced_era_exponent: float,
        cache_duration: int,
        seed: int,
    ) -> None:
        self.model = model_selector.select_model(CommonModelName(model_name))
        self.dataset = dataset
        self.global_round = global_round
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.device = device
        self.kd_epochs = kd_epochs
        self.kd_batch_size = kd_batch_size
        self.kd_lr = kd_lr
        self.public_size_per_round = public_size_per_round
        self.enhanced_era_exponent = enhanced_era_exponent
        self.cache_duration = cache_duration
        self.seed = seed

        self.kd_optimizer = torch.optim.SGD(self.model.parameters(), lr=kd_lr)
        self.client_buffer_cache: list[SCARLETUplinkPackage] = []
        self.global_soft_labels: torch.Tensor | None = None
        self.global_indices: torch.Tensor | None = None
        self.num_clients_per_round = int(self.num_clients * self.sample_ratio)
        self.round = 0

        self.global_cache: list[GlobalCacheEntry] = [
            GlobalCacheEntry(soft_label=None, round=0)
            for _ in range(self.dataset.public_size)
        ]
        self.cache_signals: torch.Tensor | None = None
        self.rng_suite = create_rng_suite(seed)

    def sample_clients(self) -> list[int]:
        sampled_clients = self.rng_suite.python.sample(
            range(self.num_clients), self.num_clients_per_round
        )

        return sorted(sampled_clients)

    def get_next_indices(self) -> torch.Tensor:
        shuffled_indices = torch.randperm(
            self.dataset.public_size, generator=self.rng_suite.torch_cpu
        )
        next_indices = shuffled_indices[: self.public_size_per_round]
        self.cached_indices = []
        request_indices = []
        for i in next_indices.tolist():
            cache_entry = self.global_cache[i]
            if (
                cache_entry.soft_label is not None
                and cache_entry.round + self.cache_duration > self.round
            ):
                # Record cached indices to restore it from global cache later
                self.cached_indices.append(i)
            else:
                request_indices.append(i)
        return torch.tensor(request_indices)

    def if_stop(self) -> bool:
        return self.round >= self.global_round

    def load(self, payload: SCARLETUplinkPackage) -> bool:
        self.client_buffer_cache.append(payload)

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1
            self.client_buffer_cache = []
            return True
        else:
            return False

    def global_update(self, buffer) -> None:
        buffer.sort(key=lambda x: x.cid)
        soft_labels_list = [ele.soft_labels for ele in buffer]
        indices_list = [ele.indices for ele in buffer]
        self.metadata_list = [ele.metadata for ele in buffer]

        soft_labels_stack: defaultdict[int, list[torch.Tensor]] = defaultdict(
            list[torch.Tensor]
        )
        for i, (soft_labels, indices) in enumerate(
            zip(soft_labels_list, indices_list, strict=True)
        ):
            num_samples = self.metadata_list[i]["num_samples"]
            for soft_label, index in zip(
                soft_labels[:num_samples], indices[:num_samples], strict=True
            ):
                soft_labels_stack[int(index.item())].append(soft_label)

        global_soft_labels: list[torch.Tensor] = []
        global_indices: list[int] = []
        for indices, soft_labels in sorted(
            soft_labels_stack.items(), key=lambda x: x[0]
        ):
            global_indices.append(indices)
            mean_soft_labels = torch.mean(torch.stack(soft_labels), dim=0)
            # Enhanced ERA
            enhanced_era_soft_labels = (
                mean_soft_labels**self.enhanced_era_exponent
            ) / torch.sum(mean_soft_labels**self.enhanced_era_exponent)
            global_soft_labels.append(enhanced_era_soft_labels)

        # Restore cached indices and soft-labels
        for i in self.cached_indices:
            global_indices.append(i)
            assert self.global_cache[i].soft_label is not None
            global_soft_labels.append(self.global_cache[i].soft_label)  # type: ignore

        # Update global cache
        cache_signals = self.update_global_cache(global_indices, global_soft_labels)
        self.cache_signals = torch.tensor([signal.value for signal in cache_signals])

        public_dataset = self.dataset.get_dataset(
            type_=CommonPartitionType.PUBLIC, cid=None
        )
        public_loader = DataLoader(
            Subset(public_dataset, global_indices),
            batch_size=self.kd_batch_size,
        )
        DSFLServerHandler.distill(
            self.model,
            self.kd_optimizer,
            public_loader,
            global_soft_labels,
            self.kd_epochs,
            self.kd_batch_size,
            self.device,
            stop_event=None,
        )

        updated_global_soft_labels = [
            soft_label
            for i, soft_label in enumerate(global_soft_labels)
            if cache_signals[i] != CacheSignal.CACHED
        ]
        if len(updated_global_soft_labels) == 0:
            self.global_soft_labels = torch.empty(0)
        else:
            self.global_soft_labels = torch.stack(updated_global_soft_labels)

        self.global_indices = torch.tensor(global_indices)

    def update_global_cache(
        self, indices: list[int], soft_labels: list[torch.Tensor]
    ) -> list[CacheSignal]:
        cache_signals = []
        for i, soft_label in zip(indices, soft_labels, strict=True):
            if self.global_cache[i].soft_label is None:
                self.global_cache[i] = GlobalCacheEntry(
                    soft_label=soft_label, round=self.round
                )
                cache_signals.append(CacheSignal.NEWLY_CACHED)
            else:
                if self.round - self.global_cache[i].round <= self.cache_duration:
                    cache_signals.append(CacheSignal.CACHED)
                else:
                    self.global_cache[i] = GlobalCacheEntry(
                        soft_label=None, round=self.round
                    )
                    cache_signals.append(CacheSignal.EXPIRED)
        return cache_signals

    def get_summary(self) -> dict[str, float]:
        server_loss, server_acc = DSFLServerHandler.evaulate(
            self.model,
            self.dataset.get_dataloader(
                type_=CommonPartitionType.TEST,
                cid=None,
                batch_size=self.kd_batch_size,
            ),
            self.device,
        )
        client_loss = sum(m["loss"] for m in self.metadata_list) / len(
            self.metadata_list
        )
        client_acc = sum(m["acc"] for m in self.metadata_list) / len(self.metadata_list)
        return {
            "server_acc": server_acc,
            "server_loss": server_loss,
            "client_acc": client_acc,
            "client_loss": client_loss,
        }

    def downlink_package(self) -> SCARLETDownlinkPackage:
        next_indices = self.get_next_indices()
        return SCARLETDownlinkPackage(
            self.global_soft_labels,
            self.global_indices,
            next_indices,
            self.cache_signals,
        )


class LocalCacheEntry(NamedTuple):
    soft_label: torch.Tensor | None


@dataclass
class SCARLETClientConfig:
    model_selector: CommonModelSelector
    model_name: CommonModelName
    dataset: CommonPartitionedDataset
    epochs: int
    batch_size: int
    lr: float
    kd_epochs: int
    kd_batch_size: int
    kd_lr: float
    cid: int
    seed: int
    state_path: Path


@dataclass
class SCARLETClientState:
    random: RNGSuite
    model: dict[str, torch.Tensor]
    optimizer: dict[str, torch.Tensor]
    kd_optimizer: dict[str, torch.Tensor] | None
    local_cache: list[LocalCacheEntry]


class SCARLETClientTrainer(
    ProcessPoolClientTrainer[
        SCARLETProcessPoolUplinkPackage, SCARLETDownlinkPackage, SCARLETClientConfig
    ]
):
    def __init__(
        self,
        model_selector: CommonModelSelector,
        model_name: str,
        share_dir: Path,
        state_dir: Path,
        dataset: CommonPartitionedDataset,
        device: str,
        num_clients: int,
        epochs: int,
        batch_size: int,
        lr: float,
        kd_epochs: int,
        kd_batch_size: int,
        kd_lr: float,
        seed: int,
        num_parallels: int,
    ) -> None:
        self.model_selector = model_selector
        self.model_name = CommonModelName(model_name)
        self.share_dir = share_dir
        self.share_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.device = device
        if self.device == "cuda":
            self.device_count = torch.cuda.device_count()
        self.num_clients = num_clients
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.kd_epochs = kd_epochs
        self.kd_batch_size = kd_batch_size
        self.kd_lr = kd_lr
        self.seed = seed
        self.num_parallels = num_parallels

        self.ipc_mode = "shared_memory"
        self.manager = mp.Manager()
        self.stop_event = self.manager.Event()
        self.cache: list[SCARLETProcessPoolUplinkPackage] = []

        self.soft_labels_buffer = torch.zeros(
            (1000, self.dataset.num_classes),
            dtype=torch.float32,
        )
        self.indices_buffer = torch.zeros(1000, dtype=torch.int64)

    def progress_fn(
        self,
        it: list[ApplyResult],
    ) -> Iterable[ApplyResult]:
        return tqdm(it, desc="Client", leave=False)

    def prepare_uplink_package_buffer(self) -> SCARLETProcessPoolUplinkPackage:
        return SCARLETProcessPoolUplinkPackage(
            cid=-1,
            soft_labels=self.soft_labels_buffer.clone(),
            indices=self.indices_buffer.clone(),
            metadata={"acc": 0.0, "loss": 0.0, "num_samples": 0},
        )

    @staticmethod
    def worker(
        config: SCARLETClientConfig | Path,
        payload: SCARLETDownlinkPackage | Path,
        device: str,
        stop_event: threading.Event,
        *,
        shm_buffer: SCARLETProcessPoolUplinkPackage | None = None,
    ) -> SCARLETProcessPoolUplinkPackage:
        assert isinstance(config, SCARLETClientConfig) and isinstance(
            payload, SCARLETDownlinkPackage
        )
        setup_reproducibility(config.seed)

        model = config.model_selector.select_model(config.model_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        kd_optimizer: torch.optim.SGD | None = None

        if config.state_path.exists():
            state = torch.load(config.state_path, weights_only=False)
            assert isinstance(state, SCARLETClientState)
            rng_suite = state.random
            local_cache = state.local_cache
            model.load_state_dict(state.model)
            optimizer.load_state_dict(state.optimizer)
            if state.kd_optimizer is not None:
                kd_optimizer = torch.optim.SGD(model.parameters(), lr=config.kd_lr)
                kd_optimizer.load_state_dict(state.kd_optimizer)
        else:
            rng_suite = create_rng_suite(config.seed)
            local_cache = [
                LocalCacheEntry(soft_label=None)
                for _ in range(config.dataset.public_size)
            ]

        # Distill
        public_dataset = config.dataset.get_dataset(
            type_=CommonPartitionType.PUBLIC, cid=None
        )
        if (
            payload.indices is not None
            and payload.soft_labels is not None
            and payload.cache_signals is not None
        ):
            local_cache, global_soft_labels = SCARLETClientTrainer.update_local_cache(
                global_soft_labels=payload.soft_labels,
                global_indices=payload.indices,
                local_cache=local_cache,
                cache_signals=payload.cache_signals,
            )
            global_indices = payload.indices.tolist()
            if kd_optimizer is None:
                kd_optimizer = torch.optim.SGD(model.parameters(), lr=config.kd_lr)

            open_loader = DataLoader(
                Subset(public_dataset, global_indices),
                batch_size=config.kd_batch_size,
            )
            DSFLServerHandler.distill(
                model=model,
                optimizer=kd_optimizer,
                open_loader=open_loader,
                global_soft_labels=global_soft_labels,
                kd_epochs=config.kd_epochs,
                kd_batch_size=config.kd_batch_size,
                device=device,
                stop_event=stop_event,
            )

        # Train
        private_loader = config.dataset.get_dataloader(
            type_=CommonPartitionType.PRIVATE,
            cid=config.cid,
            batch_size=config.batch_size,
            generator=rng_suite.torch_cpu,
        )
        DSFLClientTrainer.train(
            model=model,
            optimizer=optimizer,
            data_loader=private_loader,
            device=device,
            epochs=config.epochs,
            stop_event=stop_event,
        )

        # Predict
        public_loader = DataLoader(
            Subset(public_dataset, payload.next_indices.tolist()),
            batch_size=config.batch_size,
        )
        soft_labels = DSFLClientTrainer.predict(
            model=model,
            data_loader=public_loader,
            device=device,
        )

        # Evaluate
        test_loader = config.dataset.get_dataloader(
            type_=CommonPartitionType.TEST,
            cid=config.cid,
            batch_size=config.batch_size,
        )
        loss, acc = DSFLServerHandler.evaulate(
            model=model,
            test_loader=test_loader,
            device=device,
        )

        num_samples = soft_labels.shape[0]
        package = SCARLETProcessPoolUplinkPackage(
            cid=config.cid,
            soft_labels=soft_labels,
            indices=payload.next_indices,
            metadata={"loss": loss, "acc": acc, "num_samples": num_samples},
        )
        assert (
            shm_buffer is not None
            and isinstance(shm_buffer.soft_labels, torch.Tensor)
            and isinstance(shm_buffer.indices, torch.Tensor)
            and isinstance(package.soft_labels, torch.Tensor)
            and isinstance(package.indices, torch.Tensor)
        )
        shm_buffer.soft_labels[:num_samples].copy_(package.soft_labels)
        shm_buffer.indices[:num_samples].copy_(package.indices)
        package.soft_labels = SHMHandle()
        package.indices = SHMHandle()

        state = SCARLETClientState(
            random=rng_suite,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            kd_optimizer=kd_optimizer.state_dict() if kd_optimizer else None,
            local_cache=local_cache,
        )
        torch.save(state, config.state_path)
        return package

    @staticmethod
    def update_local_cache(
        global_soft_labels: torch.Tensor,
        global_indices: torch.Tensor,
        local_cache: list[LocalCacheEntry],
        cache_signals: torch.Tensor,
    ) -> tuple[list[LocalCacheEntry], list[torch.Tensor]]:
        global_soft_labels_queue = deque(torch.unbind(global_soft_labels, dim=0))
        restored_global_soft_labels = []
        for i, cache_signal in zip(
            global_indices.tolist(), cache_signals.tolist(), strict=True
        ):
            match cache_signal:
                case CacheSignal.CACHED.value:
                    assert local_cache[i].soft_label is not None
                    restored_global_soft_labels.append(local_cache[i].soft_label)  # type: ignore
                case CacheSignal.NEWLY_CACHED.value:
                    soft_label = global_soft_labels_queue.popleft()
                    local_cache[i] = LocalCacheEntry(soft_label=soft_label)
                    restored_global_soft_labels.append(soft_label)
                case CacheSignal.EXPIRED.value:
                    local_cache[i] = LocalCacheEntry(soft_label=None)
                    restored_global_soft_labels.append(
                        global_soft_labels_queue.popleft()
                    )
        return local_cache, restored_global_soft_labels

    def get_client_config(self, cid: int) -> SCARLETClientConfig:
        config = SCARLETClientConfig(
            model_selector=self.model_selector,
            model_name=self.model_name,
            dataset=self.dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            kd_epochs=self.kd_epochs,
            kd_batch_size=self.kd_batch_size,
            kd_lr=self.kd_lr,
            cid=cid,
            seed=self.seed,
            state_path=self.state_dir.joinpath(f"{cid}.pt"),
        )
        return config

    def uplink_package(self) -> list[SCARLETProcessPoolUplinkPackage]:
        package = deepcopy(self.cache)
        self.cache = []
        return package

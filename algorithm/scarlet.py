import threading
from collections import defaultdict, deque
from collections.abc import Iterable
from concurrent.futures import Future, as_completed
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import torch
from blazefl.core import (
    BaseServerHandler,
    ThreadPoolClientTrainer,
)
from blazefl.reproducibility import create_rng_suite
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from algorithm.dsfl import DSFLClientTrainer, DSFLServerHandler
from dataset import CommonPartitionedDataset
from dataset.dataset import CommonPartitionType
from models import CommonModelName, CommonModelSelector


@dataclass
class SCARLETUplinkPackage:
    cid: int
    soft_labels: torch.Tensor
    indices: torch.Tensor
    metadata: dict
    # Validation
    val_soft_labels: torch.Tensor
    val_indices: torch.Tensor


@dataclass
class SCARLETDownlinkPackage:
    soft_labels: torch.Tensor | None
    indices: torch.Tensor | None
    next_indices: torch.Tensor
    cache_signals: torch.Tensor | None
    # Validation
    val_soft_labels: torch.Tensor | None
    val_indices: torch.Tensor | None
    val_next_indices: torch.Tensor | None = None


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

        # Start Validation ###
        self.global_val_soft_labels: torch.Tensor | None = None
        self.global_val_indices: torch.Tensor | None = None
        self.global_val_next_indices: torch.Tensor | None = None
        self.public_train_loss = 0.0
        self.public_val_dataset = self.dataset.get_dataset(
            type_=CommonPartitionType.VALIDATION_PUBLIC, cid=None
        )
        self.public_val_loss = 0.0
        # End Validation ###

    def sample_clients(self) -> list[int]:
        sampled_clients = self.rng_suite.python.sample(
            range(self.num_clients), self.num_clients_per_round
        )

        return sorted(sampled_clients)

    def get_next_indices(self) -> torch.Tensor:
        public_train_size = int(
            self.dataset.public_size * (1 - self.dataset.public_val_ratio)
        )
        shuffled_indices = torch.randperm(
            public_train_size, generator=self.rng_suite.torch_cpu
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

        # Start Validation ###
        public_val_size = self.dataset.public_size - public_train_size
        if public_val_size > 0:
            self.global_val_next_indices = torch.randperm(
                public_val_size, generator=self.rng_suite.torch_cpu
            )
        # End Validation ###

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
        for soft_labels, indices in zip(soft_labels_list, indices_list, strict=True):
            for soft_label, index in zip(soft_labels, indices, strict=True):
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
            type_=CommonPartitionType.TRAIN_PUBLIC, cid=None
        )
        public_loader = DataLoader(
            Subset(public_dataset, global_indices),
            batch_size=self.kd_batch_size,
        )
        self.public_train_loss = DSFLServerHandler.distill(
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

        # Start Validation ###
        if self.dataset.public_val_ratio > 0:
            val_soft_labels_list = [ele.val_soft_labels for ele in buffer]
            val_indices_list = [ele.val_indices for ele in buffer]

            val_soft_labels_stack: defaultdict[int, list[torch.Tensor]] = defaultdict(
                list[torch.Tensor]
            )
            for soft_labels, indices in zip(
                val_soft_labels_list, val_indices_list, strict=True
            ):
                for soft_label, index in zip(soft_labels, indices, strict=True):
                    val_soft_labels_stack[int(index.item())].append(soft_label)

            val_global_soft_labels: list[torch.Tensor] = []
            val_global_indices: list[int] = []
            for indices, soft_labels in sorted(
                val_soft_labels_stack.items(), key=lambda x: x[0]
            ):
                val_global_indices.append(indices)
                mean_soft_labels = torch.mean(torch.stack(soft_labels), dim=0)
                # Enhanced ERA
                enhanced_era_soft_labels = (
                    mean_soft_labels**self.enhanced_era_exponent
                ) / torch.sum(mean_soft_labels**self.enhanced_era_exponent)
                val_global_soft_labels.append(enhanced_era_soft_labels)
            self.global_val_soft_labels = torch.stack(val_global_soft_labels)
            self.global_val_indices = torch.tensor(val_global_indices)

            public_val_loader = DataLoader(
                Subset(self.public_val_dataset, val_global_indices),
                batch_size=self.kd_batch_size,
            )
            self.public_val_loss = DSFLServerHandler.distill(
                self.model,
                self.kd_optimizer,
                public_val_loader,
                global_soft_labels,
                self.kd_epochs,
                self.kd_batch_size,
                self.device,
                stop_event=None,
                update_weights=False,
            )
        # End Validation ###

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
        server_test_loss, server_test_acc = DSFLServerHandler.evaulate(
            self.model,
            self.dataset.get_dataloader(
                type_=CommonPartitionType.TEST,
                cid=None,
                batch_size=self.kd_batch_size,
            ),
            self.device,
        )
        client_test_loss = sum(m["test_loss"] for m in self.metadata_list) / len(
            self.metadata_list
        )
        client_test_acc = sum(m["test_acc"] for m in self.metadata_list) / len(
            self.metadata_list
        )
        client_private_train_loss = sum(
            m["private_train_loss"] for m in self.metadata_list
        ) / len(self.metadata_list)
        client_private_train_acc = sum(
            m["private_train_acc"] for m in self.metadata_list
        ) / len(self.metadata_list)
        summary = {
            "server_test_acc": server_test_acc,
            "server_test_loss": server_test_loss,
            "client_private_train_acc": client_private_train_acc,
            "client_private_train_loss": client_private_train_loss,
            "server_public_train_loss": self.public_train_loss,
            "server_public_val_loss": self.public_val_loss,
            "client_test_acc": client_test_acc,
            "client_test_loss": client_test_loss,
        }
        if self.dataset.private_val_ratio > 0:
            summary["client_private_val_loss"] = sum(
                m["private_val_loss"] for m in self.metadata_list
            ) / len(self.metadata_list)
            summary["client_private_val_acc"] = sum(
                m["private_val_acc"] for m in self.metadata_list
            ) / len(self.metadata_list)
        if self.dataset.public_val_ratio > 0 and any(
            m["public_val_loss"] is not None for m in self.metadata_list
        ):
            summary["client_public_val_loss"] = sum(
                m["public_val_loss"]
                for m in self.metadata_list
                if m["public_val_loss"] is not None
            ) / len([m for m in self.metadata_list if m["public_val_loss"] is not None])
        return summary

    def downlink_package(self) -> SCARLETDownlinkPackage:
        next_indices = self.get_next_indices()
        return SCARLETDownlinkPackage(
            self.global_soft_labels,
            self.global_indices,
            next_indices,
            self.cache_signals,
            self.global_val_soft_labels,
            self.global_val_indices,
            self.global_val_next_indices,
        )


class LocalCacheEntry(NamedTuple):
    soft_label: torch.Tensor | None


class SCARLETClientTrainer(
    ThreadPoolClientTrainer[SCARLETUplinkPackage, SCARLETDownlinkPackage]
):
    def __init__(
        self,
        model_selector: CommonModelSelector,
        model_name: CommonModelName,
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
        public_size_per_round: int,
    ) -> None:
        self.models = [
            model_selector.select_model(model_name) for _ in range(num_clients)
        ]
        self.optimizers = [
            torch.optim.SGD(model.parameters(), lr=lr) for model in self.models
        ]
        self.kd_optimizers = [
            torch.optim.SGD(model.parameters(), lr=kd_lr) for model in self.models
        ]
        self.rng_suites = [create_rng_suite(seed + cid) for cid in range(num_clients)]
        self.local_caches = [
            [LocalCacheEntry(soft_label=None) for _ in range(dataset.public_size)]
            for _ in range(num_clients)
        ]
        self.dataset = dataset
        self.device = device
        if self.device == "cuda":
            self.device_count = torch.cuda.device_count()
        self.num_clients = num_clients
        self.epochs = epochs
        self.batch_size = batch_size
        self.kd_epochs = kd_epochs
        self.kd_batch_size = kd_batch_size
        self.seed = seed
        self.num_parallels = num_parallels

        self.stop_event = threading.Event()
        self.cache: list[SCARLETUplinkPackage] = []

        self.soft_labels_buffer = torch.zeros(
            (public_size_per_round, self.dataset.num_classes),
            dtype=torch.float32,
        )
        self.indices_buffer = torch.zeros(public_size_per_round, dtype=torch.int64)

    def progress_fn(
        self,
        it: list[Future],
    ) -> Iterable[Future]:
        return tqdm(as_completed(it), total=len(it), desc="Client", leave=False)

    def worker(
        self,
        cid: int,
        device: str,
        payload: SCARLETDownlinkPackage,
        stop_event: threading.Event,
    ) -> SCARLETUplinkPackage:
        model = self.models[cid]
        optimizer = self.optimizers[cid]
        kd_optimizer = self.kd_optimizers[cid]
        rng_suite = self.rng_suites[cid]
        local_cache = self.local_caches[cid]

        # Distill
        public_dataset = self.dataset.get_dataset(
            type_=CommonPartitionType.TRAIN_PUBLIC, cid=None
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
            public_loader = DataLoader(
                Subset(public_dataset, global_indices),
                batch_size=self.kd_batch_size,
            )
            DSFLServerHandler.distill(
                model=model,
                optimizer=kd_optimizer,
                public_loader=public_loader,
                global_soft_labels=global_soft_labels,
                kd_epochs=self.kd_epochs,
                kd_batch_size=self.kd_batch_size,
                device=device,
                stop_event=stop_event,
            )

        # Train
        private_loader = self.dataset.get_dataloader(
            type_=CommonPartitionType.TRAIN_PRIVATE,
            cid=cid,
            batch_size=self.batch_size,
            generator=rng_suite.torch_cpu,
        )
        train_loss, train_acc = DSFLClientTrainer.train(
            model=model,
            optimizer=optimizer,
            data_loader=private_loader,
            device=device,
            epochs=self.epochs,
            stop_event=stop_event,
        )

        # Predict
        public_loader = DataLoader(
            Subset(public_dataset, payload.next_indices.tolist()),
            batch_size=self.batch_size,
        )
        soft_labels = DSFLClientTrainer.predict(
            model=model,
            data_loader=public_loader,
            device=device,
        )

        # Evaluate
        test_loader = self.dataset.get_dataloader(
            type_=CommonPartitionType.TEST,
            cid=cid,
            batch_size=self.batch_size,
        )
        test_loss, test_acc = DSFLServerHandler.evaulate(
            model=model,
            test_loader=test_loader,
            device=device,
        )

        # Start Validation ###
        val_loss, val_acc = None, None
        if self.dataset.private_val_ratio > 0:
            private_val_dataloader = self.dataset.get_dataloader(
                type_=CommonPartitionType.VALIDATION_PRIVATE,
                cid=cid,
                batch_size=self.batch_size,
            )
            val_loss, val_acc = DSFLServerHandler.evaulate(
                model=model,
                test_loader=private_val_dataloader,
                device=device,
            )

        public_val_loss = None
        val_soft_labels = torch.empty(0)
        val_indices = torch.empty(0)
        if self.dataset.public_val_ratio > 0:
            public_val_dataset = self.dataset.get_dataset(
                type_=CommonPartitionType.VALIDATION_PUBLIC, cid=None
            )

            if payload.val_soft_labels is not None and payload.val_indices is not None:
                public_val_loader = DataLoader(
                    Subset(public_val_dataset, payload.val_indices.tolist()),
                    batch_size=self.kd_batch_size,
                )
                global_val_soft_labels = list(
                    torch.unbind(payload.val_soft_labels, dim=0)
                )
                public_val_loss = DSFLServerHandler.distill(
                    model=model,
                    optimizer=kd_optimizer,
                    public_loader=public_val_loader,
                    global_soft_labels=global_val_soft_labels,
                    kd_epochs=self.kd_epochs,
                    kd_batch_size=self.kd_batch_size,
                    device=device,
                    stop_event=stop_event,
                    update_weights=False,
                )

            if payload.val_next_indices is not None:
                public_val_loader = DataLoader(
                    Subset(public_val_dataset, payload.val_next_indices.tolist()),
                    batch_size=self.batch_size,
                )
                val_soft_labels = DSFLClientTrainer.predict(
                    model=model,
                    data_loader=public_val_loader,
                    device=device,
                )
                val_indices = payload.val_next_indices
        # End Validation ###

        package = SCARLETUplinkPackage(
            cid=cid,
            soft_labels=soft_labels,
            indices=payload.next_indices,
            metadata={
                "private_train_loss": train_loss,
                "private_train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "private_val_loss": val_loss,
                "private_val_acc": val_acc,
                "public_val_loss": public_val_loss,
            },
            val_soft_labels=val_soft_labels,
            val_indices=val_indices,
        )

        self.models[cid] = model
        self.optimizers[cid] = optimizer
        self.kd_optimizers[cid] = kd_optimizer
        self.rng_suites[cid] = rng_suite
        self.local_caches[cid] = local_cache
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

    def uplink_package(self) -> list[SCARLETUplinkPackage]:
        package = deepcopy(self.cache)
        self.cache = []
        return package

import random
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

from blazefl.core import (
    ParallelClientTrainer,
    ServerHandler,
)
from blazefl.utils import (
    RandomState,
    seed_everything,
)
import torch
from torch.utils.data import DataLoader, Subset

from algorithm.dsfl import (
    DSFLClientState,
    DSFLServerHandler,
    DSFLClientTrainer,
    DiskSharedData,
)
from dataset import CommonPartitionedDataset
from models import CommonModelSelector


@dataclass
class SCARLETUplinkPackage:
    soft_labels: torch.Tensor
    indices: torch.Tensor
    metadata: dict


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


class SCARLETServerHandler(ServerHandler[SCARLETUplinkPackage, SCARLETDownlinkPackage]):
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
    ) -> None:
        self.model = model_selector.select_model(model_name)
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

    def sample_clients(self) -> list[int]:
        sampled_clients = random.sample(
            range(self.num_clients), self.num_clients_per_round
        )

        return sorted(sampled_clients)

    def get_next_indices(self) -> torch.Tensor:
        shuffled_indices = torch.randperm(self.dataset.public_size)
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
        for indices, soft_labels in soft_labels_stack.items():
            global_indices.append(indices)
            mean_soft_labels = torch.mean(torch.stack(soft_labels), dim=0)
            # Enhanced ERA
            enhanced_era_soft_labels = (
                mean_soft_labels** self.enhanced_era_exponent
                / torch.sum(mean_soft_labels**self.enhanced_era_exponent)
            )
            global_soft_labels.append(enhanced_era_soft_labels)

        # Restore cached indices and soft-labels
        for i in self.cached_indices:
            global_indices.append(i)
            global_soft_labels.append(self.global_cache[i].soft_label)

        # Update global cache
        cache_signals = self.update_global_cache(global_indices, global_soft_labels)
        self.cache_signals = torch.tensor([signal.value for signal in cache_signals])

        DSFLServerHandler.distill(
            self.model,
            self.kd_optimizer,
            self.dataset,
            global_soft_labels,
            global_indices,
            self.kd_epochs,
            self.kd_batch_size,
            self.device,
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
        for i, soft_label in zip(indices, soft_labels):
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
                type_="test",
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


@dataclass
class SCARLETDiskSharedData(DiskSharedData):
    payload: SCARLETDownlinkPackage


class LocalCacheEntry(NamedTuple):
    soft_label: torch.Tensor | None


@dataclass
class SCARLETClientState(DSFLClientState):
    local_cache: list[LocalCacheEntry]


class SCARLETClientTrainer(
    ParallelClientTrainer[
        SCARLETUplinkPackage, SCARLETDownlinkPackage, SCARLETDiskSharedData
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
        super().__init__(num_parallels, share_dir, device)
        self.model_selector = model_selector
        self.model_name = model_name
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.kd_epochs = kd_epochs
        self.kd_batch_size = kd_batch_size
        self.kd_lr = kd_lr
        self.device = device
        self.num_clients = num_clients
        self.seed = seed

        if self.device == "cuda":
            self.device_count = torch.cuda.device_count()

    @staticmethod
    def process_client(path: Path, device: str) -> Path:
        data = torch.load(path, weights_only=False)
        assert isinstance(data, SCARLETDiskSharedData)

        model = data.model_selector.select_model(data.model_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=data.lr)
        kd_optimizer: torch.optim.SGD | None = None

        state: SCARLETClientState | None = None
        if data.state_path.exists():
            state = torch.load(data.state_path, weights_only=False)
            assert isinstance(state, SCARLETClientState)
            RandomState.set_random_state(state.random)
            local_cache = state.local_cache
            model.load_state_dict(state.model)
            optimizer.load_state_dict(state.optimizer)
            if state.kd_optimizer is not None:
                kd_optimizer = torch.optim.SGD(model.parameters(), lr=data.kd_lr)
                kd_optimizer.load_state_dict(state.kd_optimizer)
        else:
            seed_everything(data.seed, device=device)
            local_cache = [
                LocalCacheEntry(soft_label=None)
                for _ in range(data.dataset.public_size)
            ]

        # Distill
        public_dataset = data.dataset.get_dataset(type_="public", cid=None)
        if (
            data.payload.indices is not None
            and data.payload.soft_labels is not None
            and data.payload.cache_signals is not None
        ):
            local_cache, global_soft_labels = SCARLETClientTrainer.update_local_cache(
                global_soft_labels=data.payload.soft_labels,
                global_indices=data.payload.indices,
                local_cache=local_cache,
                cache_signals=data.payload.cache_signals,
            )
            global_indices = data.payload.indices.tolist()
            if kd_optimizer is None:
                kd_optimizer = torch.optim.SGD(model.parameters(), lr=data.kd_lr)
            DSFLServerHandler.distill(
                model=model,
                optimizer=kd_optimizer,
                dataset=data.dataset,
                global_soft_labels=global_soft_labels,
                global_indices=global_indices,
                kd_epochs=data.kd_epochs,
                kd_batch_size=data.kd_batch_size,
                device=device,
            )

        # Train
        private_loader = data.dataset.get_dataloader(
            type_="private",
            cid=data.cid,
            batch_size=data.batch_size,
        )
        DSFLClientTrainer.train(
            model=model,
            optimizer=optimizer,
            data_loader=private_loader,
            device=device,
            epochs=data.epochs,
            lr=data.lr,
        )

        # Predict
        public_loader = DataLoader(
            Subset(public_dataset, data.payload.next_indices.tolist()),
            batch_size=data.batch_size,
        )
        soft_labels = DSFLClientTrainer.predict(
            model=model,
            data_loader=public_loader,
            device=device,
        )

        # Evaluate
        test_loader = data.dataset.get_dataloader(
            type_="test",
            cid=data.cid,
            batch_size=data.batch_size,
        )
        loss, acc = DSFLServerHandler.evaulate(
            model=model,
            test_loader=test_loader,
            device=device,
        )

        package = SCARLETUplinkPackage(
            soft_labels=torch.stack(soft_labels),
            indices=data.payload.next_indices,
            metadata={"loss": loss, "acc": acc},
        )

        torch.save(package, path)
        state = SCARLETClientState(
            random=RandomState.get_random_state(device=device),
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            kd_optimizer=kd_optimizer.state_dict() if kd_optimizer else None,
            local_cache=local_cache,
        )
        torch.save(state, data.state_path)
        return path

    @staticmethod
    def update_local_cache(
        global_soft_labels: torch.Tensor,
        global_indices: torch.Tensor,
        local_cache: list[LocalCacheEntry],
        cache_signals: torch.Tensor,
    ) -> tuple[list[LocalCacheEntry], list[torch.Tensor]]:
        global_soft_labels_queue = deque(torch.unbind(global_soft_labels, dim=0))
        restored_global_soft_labels = []
        for i, cache_signal in zip(global_indices, cache_signals.tolist()):
            match cache_signal:
                case CacheSignal.CACHED.value:
                    restored_global_soft_labels.append(local_cache[i].soft_label)
                case CacheSignal.NEWLY_CACHED.value:
                    local_cache[i] = LocalCacheEntry(
                        soft_label=global_soft_labels_queue.popleft()
                    )
                    restored_global_soft_labels.append(local_cache[i].soft_label)
                case CacheSignal.EXPIRED.value:
                    local_cache[i] = LocalCacheEntry(soft_label=None)
                    restored_global_soft_labels.append(
                        global_soft_labels_queue.popleft()
                    )
        return local_cache, restored_global_soft_labels

    def get_shared_data(
        self, cid: int, payload: SCARLETDownlinkPackage
    ) -> SCARLETDiskSharedData:
        data = SCARLETDiskSharedData(
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
            payload=payload,
            state_path=self.state_dir.joinpath(f"{cid}.pt"),
        )
        return data

    def uplink_package(self) -> list[SCARLETUplinkPackage]:
        package = deepcopy(self.cache)
        self.cache: list[SCARLETUplinkPackage] = []
        return package

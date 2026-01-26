import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, Subset

from algorithm import (
    CommonClientArgs,
    CommonClientTrainer,
    CommonMetricType,
    CommonServerArgs,
    CommonServerHandler,
    distill,
    evaulate,
    predict,
    train,
)
from algorithm.dsfl import (
    DSFLClientConfig,
    DSFLClientState,
)
from core import (
    SHMHandle,
    create_rng_suite,
    setup_reproducibility,
)
from dataset.dataset import CommonPartitionType


@dataclass
class SCARLETUplinkPackage:
    cid: int
    soft_labels: torch.Tensor | SHMHandle
    indices: torch.Tensor | SHMHandle
    metrics: dict[str, float]


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
    CommonServerHandler[SCARLETUplinkPackage, SCARLETDownlinkPackage]
):
    def __init__(
        self,
        common_args: CommonServerArgs,
        model: torch.nn.Module,
        enhanced_era_exponent: float,
        cache_duration: int,
    ) -> None:
        super().__init__(common_args, model)

        self.enhanced_era_exponent = enhanced_era_exponent
        self.cache_duration = cache_duration

        self.global_cache: list[GlobalCacheEntry] = [
            GlobalCacheEntry(soft_label=None, round=0)
            for _ in range(self.dataset.public_train_size)
        ]
        self.cache_signals: torch.Tensor | None = None

    def get_next_indices(self) -> torch.Tensor:
        next_indices = super().get_next_indices()
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

    def global_update(self, buffer) -> None:
        buffer.sort(key=lambda x: x.cid)
        soft_labels_list = [ele.soft_labels for ele in buffer]
        indices_list = [ele.indices for ele in buffer]
        self.metrics_list = [ele.metrics for ele in buffer]

        soft_labels_stack: defaultdict[int, list[torch.Tensor]] = defaultdict(
            list[torch.Tensor]
        )
        for soft_labels, indices in zip(soft_labels_list, indices_list, strict=True):
            assert type(soft_labels) is torch.Tensor and type(indices) is torch.Tensor
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
            num_workers=0,
        )
        self.public_train_loss = distill(
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
class SCARLETClientConfig(DSFLClientConfig): ...


@dataclass
class SCARLETClientState(DSFLClientState):
    local_cache: list[LocalCacheEntry]


class SCARLETClientTrainer(
    CommonClientTrainer[
        SCARLETUplinkPackage,
        SCARLETDownlinkPackage,
        SCARLETClientConfig,
    ]
):
    def __init__(
        self,
        common_args: CommonClientArgs,
    ) -> None:
        super().__init__(common_args)

        self.request_size = self.public_size_per_round
        self.soft_labels_buffer = torch.zeros(
            (self.public_size_per_round, self.dataset.num_classes),
            dtype=torch.float32,
        )
        self.indices_buffer = torch.zeros(self.public_size_per_round, dtype=torch.int64)

    def prepare_uplink_package_buffer(self) -> SCARLETUplinkPackage:
        return SCARLETUplinkPackage(
            cid=-1,
            soft_labels=self.soft_labels_buffer[: self.request_size].clone(),
            indices=self.indices_buffer[: self.request_size].clone(),
            metrics={
                CommonMetricType.CLIENT_TRAIN_LOSS: 0.0,
                CommonMetricType.CLIENT_TRAIN_ACC: 0.0,
                CommonMetricType.CLIENT_TEST_LOSS: 0.0,
                CommonMetricType.CLIENT_TEST_ACC: 0.0,
            },
        )

    def local_process(
        self, payload: SCARLETDownlinkPackage, cid_list: list[int]
    ) -> None:
        self.request_size = len(payload.next_indices)
        return super().local_process(payload, cid_list)

    @staticmethod
    def worker(
        config: SCARLETClientConfig,
        payload: SCARLETDownlinkPackage,
        device: str,
        stop_event: threading.Event,
        *,
        shm_buffer: SCARLETUplinkPackage | None = None,
    ) -> SCARLETUplinkPackage:
        setup_reproducibility(config.seed)
        model = config.model_selector.select_model(config.model_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        kd_optimizer: torch.optim.SGD | None = None
        if config.state_path.exists():
            state = torch.load(config.state_path, weights_only=False)
            assert isinstance(state, SCARLETClientState)
            model.load_state_dict(state.model)
            optimizer.load_state_dict(state.optimizer)
            if state.kd_optimizer is not None:
                kd_optimizer = torch.optim.SGD(model.parameters(), lr=config.kd_lr)
                kd_optimizer.load_state_dict(state.kd_optimizer)
            rng_suite = state.random
            local_cache = state.local_cache
        else:
            rng_suite = create_rng_suite(config.seed)
            local_cache = [
                LocalCacheEntry(soft_label=None)
                for _ in range(config.dataset.public_train_size)
            ]
        model.to(device)

        # Distill
        public_dataset = config.dataset.get_dataset(
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
            if kd_optimizer is None:
                kd_optimizer = torch.optim.SGD(model.parameters(), lr=config.kd_lr)
            public_loader = DataLoader(
                Subset(public_dataset, global_indices),
                batch_size=config.kd_batch_size,
            )
            distill(
                model=model,
                optimizer=kd_optimizer,
                public_loader=public_loader,
                global_soft_labels=global_soft_labels,
                kd_epochs=config.kd_epochs,
                kd_batch_size=config.kd_batch_size,
                device=device,
                stop_event=stop_event,
            )

        # Train
        private_loader = config.dataset.get_dataloader(
            type_=CommonPartitionType.TRAIN_PRIVATE,
            cid=config.cid,
            batch_size=config.batch_size,
            generator=rng_suite.torch_cpu,
        )
        train_loss, train_acc = train(
            model=model,
            optimizer=optimizer,
            data_loader=private_loader,
            device=device,
            epochs=config.epochs,
            stop_event=stop_event,
        )

        # Predict
        soft_labels = torch.empty(0)
        if len(payload.next_indices) > 0:
            public_loader = DataLoader(
                Subset(public_dataset, payload.next_indices.tolist()),
                batch_size=config.batch_size,
            )
            soft_labels = predict(
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
        test_loss, test_acc = evaulate(
            model=model,
            test_loader=test_loader,
            device=device,
        )

        package = SCARLETUplinkPackage(
            cid=config.cid,
            soft_labels=soft_labels,
            indices=payload.next_indices,
            metrics={
                CommonMetricType.CLIENT_TRAIN_LOSS: train_loss,
                CommonMetricType.CLIENT_TRAIN_ACC: train_acc,
                CommonMetricType.CLIENT_TEST_LOSS: test_loss,
                CommonMetricType.CLIENT_TEST_ACC: test_acc,
            },
        )
        state = SCARLETClientState(
            random=rng_suite,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            kd_optimizer=kd_optimizer.state_dict()
            if kd_optimizer is not None
            else None,
            local_cache=local_cache,
        )
        torch.save(state, config.state_path)

        # config.dataset.set_dataset(
        #     type_=CommonPartitionType.TRAIN_PRIVATE,
        #     cid=config.cid,
        #     dataset=private_loader.dataset,
        # )

        assert shm_buffer is not None
        assert isinstance(shm_buffer.soft_labels, torch.Tensor) and isinstance(
            package.soft_labels, torch.Tensor
        )
        shm_buffer.soft_labels.copy_(package.soft_labels)
        package.soft_labels = SHMHandle()
        assert isinstance(shm_buffer.indices, torch.Tensor) and isinstance(
            package.indices, torch.Tensor
        )
        shm_buffer.indices.copy_(package.indices)
        package.indices = SHMHandle()
        return package

    def get_client_config(self, cid: int) -> SCARLETClientConfig:
        return SCARLETClientConfig(
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
            seed=self.seed + cid,
            state_path=self.state_dir.joinpath(f"{cid}.pt"),
        )

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
                    restored_global_soft_labels.append(local_cache[i].soft_label)
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

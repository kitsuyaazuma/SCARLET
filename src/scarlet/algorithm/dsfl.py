import threading
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from scarlet.core import (
    ModelSelector,
    RNGSuite,
    SHMHandle,
    create_rng_suite,
    setup_reproducibility,
)
from scarlet.dataset import CommonPartitionType, DatasetProvider
from scarlet.models import CommonModelName

from .common import (
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


@dataclass
class DSFLUplinkPackage:
    cid: int
    soft_labels: torch.Tensor | SHMHandle
    indices: torch.Tensor | SHMHandle
    metrics: dict[str, float]


@dataclass
class DSFLDownlinkPackage:
    soft_labels: torch.Tensor | None
    indices: torch.Tensor | None
    next_indices: torch.Tensor


class DSFLServerHandler(CommonServerHandler[DSFLUplinkPackage, DSFLDownlinkPackage]):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: DatasetProvider,
        global_round: int,
        num_clients: int,
        sample_ratio: float,
        device: str,
        kd_epochs: int,
        kd_batch_size: int,
        kd_lr: float,
        public_size_per_round: int,
        seed: int,
        era_temperature: float,
    ) -> None:
        super().__init__(
            model=model,
            dataset=dataset,
            global_round=global_round,
            num_clients=num_clients,
            sample_ratio=sample_ratio,
            device=device,
            kd_epochs=kd_epochs,
            kd_batch_size=kd_batch_size,
            kd_lr=kd_lr,
            public_size_per_round=public_size_per_round,
            seed=seed,
        )
        self.era_temperature = era_temperature

    @classmethod
    def from_args(  # type: ignore[override]
        cls: type[Self],
        args: CommonServerArgs,
        model: torch.nn.Module,
        *,
        era_temperature: float,
        **kwargs,
    ) -> Self:
        return cls(
            model=model,
            dataset=args.dataset,
            global_round=args.global_round,
            num_clients=args.num_clients,
            sample_ratio=args.sample_ratio,
            device=args.device,
            kd_epochs=args.kd_epochs,
            kd_batch_size=args.kd_batch_size,
            kd_lr=args.kd_lr,
            public_size_per_round=args.public_size_per_round,
            seed=args.seed,
            era_temperature=era_temperature,
        )

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
            # Entropy Reduction Aggregation (ERA)
            era_soft_labels = F.softmax(mean_soft_labels / self.era_temperature, dim=0)
            global_soft_labels.append(era_soft_labels)

        public_dataset = self.dataset.get_dataset(
            type_=CommonPartitionType.TRAIN_PUBLIC, cid=None
        )
        public_loader = DataLoader(
            Subset(public_dataset, global_indices),
            batch_size=self.kd_batch_size,
        )
        distill(
            self.model,
            self.kd_optimizer,
            public_loader,
            global_soft_labels,
            self.kd_epochs,
            self.kd_batch_size,
            self.device,
            stop_event=None,
        )

        self.global_soft_labels = torch.stack(global_soft_labels)
        self.global_indices = torch.tensor(global_indices)

    def downlink_package(self) -> DSFLDownlinkPackage:
        next_indices = self.get_next_indices()
        return DSFLDownlinkPackage(
            self.global_soft_labels, self.global_indices, next_indices
        )


@dataclass
class DSFLClientConfig:
    model_selector: ModelSelector
    model_name: CommonModelName
    dataset: DatasetProvider
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
class DSFLClientState:
    random: RNGSuite
    model: dict[str, torch.Tensor]
    optimizer: dict[str, torch.Tensor]
    kd_optimizer: dict[str, torch.Tensor] | None


class DSFLClientTrainer(
    CommonClientTrainer[DSFLUplinkPackage, DSFLDownlinkPackage, DSFLClientConfig]
):
    def __init__(
        self,
        model_selector: ModelSelector,
        model_name: CommonModelName,
        dataset: DatasetProvider,
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
        state_dir: Path,
        manager: SyncManager | None,
    ) -> None:
        super().__init__(
            model_selector=model_selector,
            model_name=model_name,
            dataset=dataset,
            device=device,
            num_clients=num_clients,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            kd_epochs=kd_epochs,
            kd_batch_size=kd_batch_size,
            kd_lr=kd_lr,
            seed=seed,
            num_parallels=num_parallels,
            public_size_per_round=public_size_per_round,
            state_dir=state_dir,
            manager=manager,
        )

        self.soft_labels_buffer = torch.zeros(
            (self.public_size_per_round, self.dataset.num_classes),
            dtype=torch.float32,
        )
        self.indices_buffer = torch.zeros(self.public_size_per_round, dtype=torch.int64)

    @classmethod
    def from_args(  # type: ignore[override]
        cls: type[Self],
        args: CommonClientArgs,
        model_selector: ModelSelector,
        model_name: CommonModelName,
        manager: SyncManager | None = None,
        **kwargs,
    ) -> Self:
        return cls(
            model_selector=model_selector,
            model_name=model_name,
            dataset=args.dataset,
            device=args.device,
            num_clients=args.num_clients,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            kd_epochs=args.kd_epochs,
            kd_batch_size=args.kd_batch_size,
            kd_lr=args.kd_lr,
            seed=args.seed,
            num_parallels=args.num_parallels,
            public_size_per_round=args.public_size_per_round,
            state_dir=args.state_dir,
            manager=manager,
        )

    def prepare_uplink_package_buffer(self) -> DSFLUplinkPackage:
        return DSFLUplinkPackage(
            cid=-1,
            soft_labels=self.soft_labels_buffer.clone(),
            indices=self.indices_buffer.clone(),
            metrics={
                CommonMetricType.CLIENT_TRAIN_LOSS: 0.0,
                CommonMetricType.CLIENT_TRAIN_ACC: 0.0,
                CommonMetricType.CLIENT_TEST_LOSS: 0.0,
                CommonMetricType.CLIENT_TEST_ACC: 0.0,
            },
        )

    @staticmethod
    def worker(
        config: DSFLClientConfig,
        payload: DSFLDownlinkPackage,
        device: str,
        stop_event: threading.Event,
        *,
        shm_buffer: DSFLUplinkPackage | None = None,
    ) -> DSFLUplinkPackage:
        setup_reproducibility(config.seed)
        model = config.model_selector.select_model(config.model_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        kd_optimizer: torch.optim.SGD | None = None
        if config.state_path.exists():
            state = torch.load(config.state_path, weights_only=False)
            assert isinstance(state, DSFLClientState)
            model.load_state_dict(state.model)
            optimizer.load_state_dict(state.optimizer)
            if state.kd_optimizer is not None:
                kd_optimizer = torch.optim.SGD(model.parameters(), lr=config.kd_lr)
                kd_optimizer.load_state_dict(state.kd_optimizer)
            rng_suite = state.random
        else:
            rng_suite = create_rng_suite(config.seed)
        model.to(device)

        # Distill
        public_dataset = config.dataset.get_dataset(
            type_=CommonPartitionType.TRAIN_PUBLIC, cid=None
        )
        if payload.indices is not None and payload.soft_labels is not None:
            global_soft_labels = list(torch.unbind(payload.soft_labels, dim=0))
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

        package = DSFLUplinkPackage(
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

        state = DSFLClientState(
            random=rng_suite,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            kd_optimizer=kd_optimizer.state_dict()
            if kd_optimizer is not None
            else None,
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

    def get_client_config(self, cid: int) -> DSFLClientConfig:
        return DSFLClientConfig(
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

import logging
import threading
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Protocol, Self, TypeVar

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core import (
    BaseClientTrainer,
    BaseServerHandler,
    FilteredDataset,
    ModelSelector,
    ProcessPoolClientTrainer,
    create_rng_suite,
)
from dataset import CommonPartitionType
from dataset.interface import DatasetProvider
from models import CommonModelName

UplinkPackage = TypeVar("UplinkPackage")
DownlinkPackage = TypeVar("DownlinkPackage")
ClientConfig = TypeVar("ClientConfig")


class CommonMetricType(StrEnum):
    SERVER_TEST_LOSS = "server_test_loss"
    SERVER_TEST_ACC = "server_test_acc"
    CLIENT_TRAIN_LOSS = "client_train_loss"
    CLIENT_TRAIN_ACC = "client_train_acc"
    CLIENT_TEST_LOSS = "client_test_loss"
    CLIENT_TEST_ACC = "client_test_acc"


@dataclass(frozen=True)
class CommonServerArgs:
    dataset: DatasetProvider
    global_round: int
    num_clients: int
    sample_ratio: float
    device: str
    kd_epochs: int
    kd_batch_size: int
    kd_lr: float
    public_size_per_round: int
    seed: int


class CommonServerHandler(BaseServerHandler[UplinkPackage, DownlinkPackage], ABC):
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
    ) -> None:
        self.dataset = dataset
        self.global_round = global_round
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.device = device
        self.kd_epochs = kd_epochs
        self.kd_batch_size = kd_batch_size
        self.kd_lr = kd_lr
        self.public_size_per_round = public_size_per_round
        self.seed = seed

        self.model = model
        self.model.to(self.device)
        self.kd_optimizer = torch.optim.SGD(self.model.parameters(), lr=kd_lr)
        self.client_buffer_cache: list[UplinkPackage] = []
        self.global_soft_labels: torch.Tensor | None = None
        self.global_indices: torch.Tensor | None = None
        self.num_clients_per_round = int(self.num_clients * self.sample_ratio)
        self.round = 0

        self.rng_suite = create_rng_suite(seed)
        self.metrics_list: list[dict[str, float]] = []

    @classmethod
    def from_args(
        cls: type[Self], args: CommonServerArgs, model: torch.nn.Module, **kwargs
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
            **kwargs,
        )

    def get_round(self) -> int:
        return self.round

    def if_stop(self) -> bool:
        return self.round >= self.global_round

    def get_next_indices(self) -> torch.Tensor:
        shuffled_indices = torch.randperm(
            self.dataset.public_train_size, generator=self.rng_suite.torch_cpu
        )
        return shuffled_indices[: self.public_size_per_round]

    def sample_clients(self) -> list[int]:
        sampled_clients = self.rng_suite.python.sample(
            range(self.num_clients), self.num_clients_per_round
        )
        return sorted(sampled_clients)

    def load(self, payload: UplinkPackage) -> bool:
        self.client_buffer_cache.append(payload)

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1
            self.client_buffer_cache = []
            return True
        else:
            return False

    def get_summary(self) -> dict[str, float]:
        server_test_loss, server_test_acc = evaulate(
            self.model,
            self.dataset.get_dataloader(
                type_=CommonPartitionType.TEST,
                cid=None,
                batch_size=self.kd_batch_size,
            ),
            self.device,
        )
        client_test_loss = sum(
            m[CommonMetricType.CLIENT_TEST_LOSS] for m in self.metrics_list
        ) / len(self.metrics_list)
        client_test_acc = sum(
            m[CommonMetricType.CLIENT_TEST_LOSS] for m in self.metrics_list
        ) / len(self.metrics_list)
        return {
            CommonMetricType.SERVER_TEST_ACC: server_test_acc,
            CommonMetricType.SERVER_TEST_LOSS: server_test_loss,
            CommonMetricType.CLIENT_TEST_ACC: client_test_acc,
            CommonMetricType.CLIENT_TEST_LOSS: client_test_loss,
        }


@dataclass(frozen=True)
class CommonClientArgs:
    dataset: DatasetProvider
    device: str
    num_clients: int
    epochs: int
    batch_size: int
    lr: float
    kd_epochs: int
    kd_batch_size: int
    kd_lr: float
    seed: int
    num_parallels: int
    public_size_per_round: int
    state_dir: Path


class CommonClientTrainer(
    ProcessPoolClientTrainer[UplinkPackage, DownlinkPackage, ClientConfig], ABC
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
        self.model_selector = model_selector
        self.model_name = model_name
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
        self.public_size_per_round = public_size_per_round
        self.state_dir = state_dir

        self.manager = manager
        self.stop_event = self.manager.Event() if self.manager else threading.Event()
        self.cache: list[UplinkPackage] = []

    @classmethod
    def from_args(
        cls: type[Self],
        args: CommonClientArgs,
        model_selector: ModelSelector,
        model_name: CommonModelName,
        manager: SyncManager | None,
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
            **kwargs,
        )

    def uplink_package(self) -> list[UplinkPackage]:
        package = deepcopy(self.cache)
        self.cache = []
        return package


SummarizableUplinkPackage = TypeVar("SummarizableUplinkPackage")
SummarizableDownlinkPackage = TypeVar("SummarizableDownlinkPackage", covariant=True)


class SummarizableBaseServerHandler(
    BaseServerHandler[SummarizableUplinkPackage, SummarizableDownlinkPackage], Protocol
):
    def get_round(self) -> int: ...

    def get_summary(self) -> dict[str, float]: ...


class Logger(Protocol):
    def log(self, data: dict[str, float], step: int | None = None) -> None: ...


class CommonPipeline:
    def __init__(
        self,
        handler: SummarizableBaseServerHandler,
        trainer: BaseClientTrainer,
        logger: Logger,
    ) -> None:
        self.handler = handler
        self.trainer = trainer
        self.logger = logger

    def main(self) -> None:
        while not self.handler.if_stop():
            round_ = self.handler.get_round()
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package()

            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package()

            # server side
            for pack in uploads:
                self.handler.load(pack)

            summary = self.handler.get_summary()
            self.logger.log(summary, step=round_)
            formatted_summary = ", ".join(f"{k}: {v:.3f}" for k, v in summary.items())
            logging.info(f"round: {round_}, {formatted_summary}")

        logging.info("done!")


def evaulate(
    model: torch.nn.Module, test_loader: DataLoader, device: str
) -> tuple[float, float]:
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction="mean")
            total_loss += loss.item() * labels.size(0)

            predicted = outputs.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def distill(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    public_loader: DataLoader,
    global_soft_labels: list[torch.Tensor],
    kd_epochs: int,
    kd_batch_size: int,
    device: str,
    stop_event: threading.Event | None,
    update_weights: bool = True,
) -> float:
    model.to(device)
    model.train(update_weights)
    global_soft_label_loader = DataLoader(
        FilteredDataset(
            indices=list(range(len(global_soft_labels))),
            original_data=global_soft_labels,
        ),
        batch_size=kd_batch_size,
    )
    epoch_loss, epoch_samples = 0.0, 0
    for kd_epoch in range(kd_epochs):
        if stop_event is not None and stop_event.is_set():
            break
        for data, soft_label in zip(
            public_loader, global_soft_label_loader, strict=True
        ):
            data = data.to(device)
            soft_label = soft_label.to(device).squeeze(1)

            with torch.set_grad_enabled(update_weights):
                output = model(data)
                loss = F.kl_div(
                    F.log_softmax(output, dim=1), soft_label, reduction="batchmean"
                )
            if kd_epoch == kd_epochs - 1:
                epoch_loss += loss.item() * soft_label.size(0)
                epoch_samples += soft_label.size(0)

            if update_weights:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    avg_loss = epoch_loss / epoch_samples
    return avg_loss


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: str,
    epochs: int,
    stop_event: threading.Event,
) -> tuple[float, float]:
    model.to(device)
    model.train()

    epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
    for epoch in range(epochs):
        if stop_event.is_set():
            break
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = F.cross_entropy(output, target, reduction="mean")
            if epoch == epochs - 1:
                epoch_loss += loss.item() * target.size(0)
                predicted = output.argmax(dim=1)
                epoch_correct += (predicted == target).sum().item()
                epoch_samples += target.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    avg_loss = epoch_loss / epoch_samples
    avg_acc = epoch_correct / epoch_samples
    return avg_loss, avg_acc


def predict(
    model: torch.nn.Module, data_loader: DataLoader, device: str
) -> torch.Tensor:
    model.to(device)
    model.eval()

    soft_labels_list = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            output = model(data)
            soft_label = F.softmax(output, dim=1)

            soft_labels_list.append(soft_label.detach())

    soft_labels = torch.cat(soft_labels_list, dim=0)
    return soft_labels.cpu()

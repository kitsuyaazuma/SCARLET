import threading
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import Future, as_completed
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from blazefl.core import (
    BaseServerHandler,
    FilteredDataset,
    ThreadPoolClientTrainer,
)
from blazefl.reproducibility import (
    RNGSuite,
    create_rng_suite,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import CommonPartitionedDataset
from dataset.dataset import CommonPartitionType
from models import CommonModelSelector
from models.selector import CommonModelName


@dataclass
class DSFLUplinkPackage:
    cid: int
    soft_labels: torch.Tensor
    indices: torch.Tensor
    metadata: dict


@dataclass
class DSFLDownlinkPackage:
    soft_labels: torch.Tensor | None
    indices: torch.Tensor | None
    next_indices: torch.Tensor


class DSFLServerHandler(BaseServerHandler[DSFLUplinkPackage, DSFLDownlinkPackage]):
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
        era_temperature: float,
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
        self.era_temperature = era_temperature
        self.seed = seed

        self.kd_optimizer = torch.optim.SGD(self.model.parameters(), lr=kd_lr)
        self.client_buffer_cache: list[DSFLUplinkPackage] = []
        self.global_soft_labels: torch.Tensor | None = None
        self.global_indices: torch.Tensor | None = None
        self.num_clients_per_round = int(self.num_clients * self.sample_ratio)
        self.round = 0

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
        return shuffled_indices[: self.public_size_per_round]

    def if_stop(self) -> bool:
        return self.round >= self.global_round

    def load(self, payload: DSFLUplinkPackage) -> bool:
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
            # Entropy Reduction Aggregation (ERA)
            era_soft_labels = F.softmax(mean_soft_labels / self.era_temperature, dim=0)
            global_soft_labels.append(era_soft_labels)

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

        self.global_soft_labels = torch.stack(global_soft_labels)
        self.global_indices = torch.tensor(global_indices)

    @staticmethod
    def distill(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        open_loader: DataLoader,
        global_soft_labels: list[torch.Tensor],
        kd_epochs: int,
        kd_batch_size: int,
        device: str,
        stop_event: threading.Event | None,
    ) -> None:
        model.to(device)
        model.train()
        global_soft_label_loader = DataLoader(
            FilteredDataset(
                indices=list(range(len(global_soft_labels))),
                original_data=global_soft_labels,
            ),
            batch_size=kd_batch_size,
        )
        for _ in range(kd_epochs):
            if stop_event is not None and stop_event.is_set():
                break
            for data, soft_label in zip(
                open_loader, global_soft_label_loader, strict=True
            ):
                data = data.to(device)
                soft_label = soft_label.to(device).squeeze(1)

                output = model(data)
                loss = F.kl_div(
                    F.log_softmax(output, dim=1), soft_label, reduction="batchmean"
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return

    @staticmethod
    def evaulate(
        model: torch.nn.Module, test_loader: DataLoader, device: str
    ) -> tuple[float, float]:
        model.to(device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                correct = torch.sum(predicted.eq(labels)).item()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += int(correct)
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

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

    def downlink_package(self) -> DSFLDownlinkPackage:
        next_indices = self.get_next_indices()
        return DSFLDownlinkPackage(
            self.global_soft_labels, self.global_indices, next_indices
        )


@dataclass
class DSFLClientState:
    random: RNGSuite
    model: dict[str, torch.Tensor]
    optimizer: dict[str, torch.Tensor]
    kd_optimizer: dict[str, torch.Tensor] | None


class DSFLClientTrainer(
    ThreadPoolClientTrainer[DSFLUplinkPackage, DSFLDownlinkPackage]
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
        self.cache: list[DSFLUplinkPackage] = []

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
        payload: DSFLDownlinkPackage,
        stop_event: threading.Event,
    ) -> DSFLUplinkPackage:
        model = self.models[cid]
        optimizer = self.optimizers[cid]
        kd_optimizer = self.kd_optimizers[cid]
        rng_suite = self.rng_suites[cid]

        # Distill
        public_dataset = self.dataset.get_dataset(
            type_=CommonPartitionType.PUBLIC, cid=None
        )
        if payload.indices is not None and payload.soft_labels is not None:
            global_soft_labels = list(torch.unbind(payload.soft_labels, dim=0))
            global_indices = payload.indices.tolist()
            open_loader = DataLoader(
                Subset(public_dataset, global_indices),
                batch_size=self.kd_batch_size,
            )
            DSFLServerHandler.distill(
                model=model,
                optimizer=kd_optimizer,
                open_loader=open_loader,
                global_soft_labels=global_soft_labels,
                kd_epochs=self.kd_epochs,
                kd_batch_size=self.kd_batch_size,
                device=device,
                stop_event=stop_event,
            )

        # Train
        private_loader = self.dataset.get_dataloader(
            type_=CommonPartitionType.PRIVATE,
            cid=cid,
            batch_size=self.batch_size,
            generator=rng_suite.torch_cpu,
        )
        DSFLClientTrainer.train(
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
        loss, acc = DSFLServerHandler.evaulate(
            model=model,
            test_loader=test_loader,
            device=device,
        )

        package = DSFLUplinkPackage(
            cid=cid,
            soft_labels=soft_labels,
            indices=payload.next_indices,
            metadata={"loss": loss, "acc": acc},
        )

        self.models[cid] = model
        self.optimizers[cid] = optimizer
        self.kd_optimizers[cid] = kd_optimizer
        self.rng_suites[cid] = rng_suite
        return package

    @staticmethod
    def train(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        device: str,
        epochs: int,
        stop_event: threading.Event,
    ) -> None:
        model.to(device)
        model.train()
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            if stop_event.is_set():
                break
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return

    @staticmethod
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

    def uplink_package(self) -> list[DSFLUplinkPackage]:
        package = deepcopy(self.cache)
        self.cache = []
        return package

from dataclasses import dataclass
import csv
import torch
from fedlab.contrib.algorithm import SGDSerialClientTrainer, SyncServerHandler
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F

from dataset import PartitionedDataset
from model import get_model
from utils import (
    get_criterion,
    get_optimizer,
    seed_everything,
    get_random_state,
    set_random_state,
)


@dataclass
class BaseClientWorkerProcess:
    model_name: str
    epochs: int
    lr: float
    batch_size: int
    optimizer_name: str
    criterion_name: str
    state_dict_dir: Path
    seed: int
    analysis_dir: Path

    def prepare(self, device: str, client_id: int, dataset: PartitionedDataset):
        self.state_dict_path = self.state_dict_dir.joinpath(f"{client_id}.pt")
        if self.state_dict_path.exists():
            set_random_state(torch.load(self.state_dict_path)["random_state"])
        else:
            seed_everything(self.seed)
        self.device = device
        self.dataset = dataset
        self.client_id = client_id
        self.analysis_dir = self.analysis_dir
        self.model = get_model(self.model_name, num_classes=dataset.num_classes).to(
            self.device
        )
        self.optimizer = get_optimizer(self.optimizer_name)(
            self.model.parameters(), lr=self.lr
        )
        self.criterion = get_criterion(self.criterion_name)
        if self.state_dict_path.exists():
            self.model.load_state_dict(torch.load(self.state_dict_path)["model"])
            self.optimizer.load_state_dict(
                torch.load(self.state_dict_path)["optimizer"]
            )
        self.save_dict: dict = {}

    def train(self):
        self.model.train()

        trainset = self.dataset.get_private_train_dataset(self.client_id)
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.save_dict["model"] = self.model.state_dict()
        self.save_dict["optimizer"] = self.optimizer.state_dict()

    def evaluate(self):
        self.model.eval()
        loss_sum = 0.0
        top1_acc_sum = 0.0
        top5_acc_sum = 0.0
        total = 0
        testset = self.dataset.get_test_dataset(self.client_id)
        test_loader = DataLoader(testset, batch_size=100, shuffle=False)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss_ = F.cross_entropy(outputs, labels)

                _, pred = outputs.topk(5, 1, largest=True, sorted=True)

                labels = labels.view(labels.size(0), -1).expand_as(pred)
                correct = pred.eq(labels).float()

                top5_acc_sum += correct[:, :5].sum().item()
                top1_acc_sum += correct[:, :1].sum().item()

                loss_sum += loss_.item()
                total += labels.size(0)

        loss = loss_sum / len(test_loader)
        top1_acc = top1_acc_sum / total
        top5_acc = top5_acc_sum / total
        with open(self.analysis_dir.joinpath(f"{self.client_id}.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow([loss, top1_acc, top5_acc])

    def save(self):
        self.save_dict["random_state"] = get_random_state()
        torch.save(
            self.save_dict,
            self.state_dict_path,
        )


class BaseSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(
        self,
        model_name: str,
        num_clients: int,
        cuda: bool,
        state_dict_dir: Path,
        seed: int,
        num_parallels: int,
    ):
        super().__init__(
            model=get_model(model_name), num_clients=num_clients, cuda=cuda
        )
        self.seed = seed
        self.model_name = model_name
        self.state_dict_dir = state_dict_dir
        self.device_count = torch.cuda.device_count()
        self.num_parallels = num_parallels

    def setup_datasets(self, dataset: PartitionedDataset):
        self.dataset = dataset


class BaseServerHandler(SyncServerHandler):
    def __init__(
        self,
        model_name: str,
        global_round: int,
        sample_ratio: float,
        cuda: bool,
        public_size_per_round: int,
        dataset: PartitionedDataset,
    ):
        super().__init__(
            model=get_model(model_name, num_classes=dataset.num_classes),
            global_round=global_round,
            sample_ratio=sample_ratio,
            cuda=cuda,
        )
        self.model_name = model_name
        self.public_size_per_round = public_size_per_round
        self.test_batch_size = 1000
        self.dataset = dataset

    def evaluate(self) -> tuple[float, float, float]:
        self.model.eval()
        loss_sum = 0.0
        top1_acc_sum = 0.0
        top5_acc_sum = 0.0
        total = 0
        testset = self.dataset.get_test_dataset()
        test_loader = DataLoader(
            testset, batch_size=self.test_batch_size, shuffle=False
        )
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss_ = F.cross_entropy(outputs, labels)

                _, pred = outputs.topk(5, 1, largest=True, sorted=True)

                labels = labels.view(labels.size(0), -1).expand_as(pred)
                correct = pred.eq(labels).float()

                top5_acc_sum += correct[:, :5].sum().item()
                top1_acc_sum += correct[:, :1].sum().item()

                loss_sum += loss_.item()
                total += labels.size(0)

        loss = loss_sum / len(test_loader)
        top1_acc = top1_acc_sum / total
        top5_acc = top5_acc_sum / total
        return loss, top1_acc, top5_acc

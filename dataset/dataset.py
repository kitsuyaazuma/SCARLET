from collections.abc import Sized
from enum import StrEnum
from pathlib import Path

from blazefl.core import PartitionedDataset
from blazefl.utils import FilteredDataset
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from dataset.functional import balance_split, client_inner_dirichlet_partition_faster


class PrivateTask(StrEnum):
    CIFAR10 = "cifar10"


class PublicTask(StrEnum):
    CIFAR100 = "cifar100"


class CommonPartitionedDataset(PartitionedDataset):
    def __init__(
        self,
        root: Path,
        path: Path,
        num_clients: int,
        seed: int,
        private_task: str,
        public_task: str,
        partition: str,
        dir_alpha: float,
        public_size: int,
    ) -> None:
        self.root = root
        self.path = path
        self.num_clients = num_clients
        self.seed = seed
        self.private_task = private_task
        self.public_task = public_task
        self.partition = partition
        self.dir_alpha = dir_alpha
        self.public_size = public_size
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.target_transform = None

        self._preprocess()

    def _preprocess(self):
        self.root.mkdir(parents=True, exist_ok=True)
        match self.private_task:
            case PrivateTask.CIFAR10:
                private_dataset = torchvision.datasets.CIFAR10(
                    root=self.root,
                    train=True,
                    download=True,
                )
                test_dataset = torchvision.datasets.CIFAR10(
                    root=self.root,
                    train=False,
                    download=True,
                )
            case _:
                raise ValueError(f"Invalid private task: {self.private_task}")
        assert self.private_task != self.public_task
        match self.public_task:
            case PublicTask.CIFAR100:
                public_dataset = torchvision.datasets.CIFAR100(
                    root=self.root,
                    train=True,
                    download=True,
                )
            case _:
                raise ValueError(f"Invalid public task: {self.public_task}")

        for type_ in ["private", "public", "test"]:
            self.path.joinpath(type_).mkdir(parents=True)

        match self.partition:
            case "dirichlet":
                assert self.dir_alpha is not None
                self.num_classes = len(private_dataset.classes)
                private_client_dict, class_priors = (
                    client_inner_dirichlet_partition_faster(
                        targets=private_dataset.targets,
                        num_clients=self.num_clients,
                        num_classes=self.num_classes,
                        dir_alpha=self.dir_alpha,
                        client_sample_nums=balance_split(
                            num_clients=self.num_clients,
                            num_samples=len(private_dataset.targets),
                        ),
                        verbose=False,
                    )
                )
                test_client_dict, _ = client_inner_dirichlet_partition_faster(
                    targets=test_dataset.targets,
                    num_clients=self.num_clients,
                    num_classes=self.num_classes,
                    dir_alpha=self.dir_alpha,
                    client_sample_nums=balance_split(
                        num_clients=self.num_clients,
                        num_samples=len(test_dataset.targets),
                    ),
                    class_priors=class_priors,
                    verbose=False,
                )
            case _:
                raise ValueError(f"Invalid partition: {self.partition}")

        for cid, indices in private_client_dict.items():
            client_private_dataset = FilteredDataset(
                indices.tolist(),
                private_dataset.data,
                private_dataset.targets,
                transform=self.train_transform,
                target_transform=self.target_transform,
            )
            torch.save(
                client_private_dataset, self.path.joinpath("private", f"{cid}.pkl")
            )

        for cid, indices in test_client_dict.items():
            client_test_dataset = FilteredDataset(
                indices.tolist(),
                test_dataset.data,
                test_dataset.targets,
                transform=self.test_transform,
                target_transform=self.target_transform,
            )
            torch.save(client_test_dataset, self.path.joinpath("test", f"{cid}.pkl"))

        public_indices, _ = train_test_split(
            range(len(public_dataset)),
            test_size=1 - self.public_size / len(public_dataset),
        )
        torch.save(
            FilteredDataset(
                public_indices,
                public_dataset.data,
                original_targets=None,
                transform=self.train_transform,
                target_transform=self.target_transform,
            ),
            self.path.joinpath("public", "public.pkl"),
        )

        torch.save(
            FilteredDataset(
                list(range(len(test_dataset))),
                test_dataset.data,
                test_dataset.targets,
                transform=self.test_transform,
                target_transform=self.target_transform,
            ),
            self.path.joinpath("test", "test.pkl"),
        )

    def get_dataset(self, type_: str, cid: int | None) -> Dataset:
        match type_:
            case "private":
                dataset = torch.load(
                    self.path.joinpath(type_, f"{cid}.pkl"),
                    weights_only=False,
                )
            case "public":
                dataset = torch.load(
                    self.path.joinpath(type_, "public.pkl"),
                    weights_only=False,
                )
            case "test":
                if cid is not None:
                    dataset = torch.load(
                        self.path.joinpath(type_, f"{cid}.pkl"), weights_only=False
                    )
                else:
                    dataset = torch.load(
                        self.path.joinpath(type_, "test.pkl"), weights_only=False
                    )
            case _:
                raise ValueError(f"Invalid dataset type: {type_}")
        assert isinstance(dataset, Dataset)
        return dataset

    def get_dataloader(
        self, type_: str, cid: int | None, batch_size: int | None = None
    ) -> DataLoader:
        dataset = self.get_dataset(type_, cid)
        assert isinstance(dataset, Sized)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader

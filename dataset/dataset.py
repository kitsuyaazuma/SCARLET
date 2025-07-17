from collections.abc import Sized
from enum import StrEnum
from pathlib import Path

import torch
import torchvision
from blazefl.core import FilteredDataset, PartitionedDataset
from blazefl.reproducibility import create_rng_suite
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset.functional import balance_split, client_inner_dirichlet_partition_faster
from dataset.transforms import GeneratorRandomCrop, GeneratorRandomHorizontalFlip


class PrivateTask(StrEnum):
    CIFAR10 = "cifar10"


class PublicTask(StrEnum):
    CIFAR100 = "cifar100"


class CommonPartitionType(StrEnum):
    PRIVATE = "private"
    PUBLIC = "public"
    TEST = "test"


class CommonPartitionedDataset(PartitionedDataset[CommonPartitionType]):
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

        self.rng_suite = create_rng_suite(seed)

        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                GeneratorRandomHorizontalFlip(
                    p=0.5, generator=self.rng_suite.torch_cpu
                ),
                GeneratorRandomCrop(32, padding=4, generator=self.rng_suite.torch_cpu),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self._preprocess()

    def _preprocess(self):
        self.root.mkdir(parents=True, exist_ok=True)
        assert self.private_task != self.public_task
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
        match self.public_task:
            case PublicTask.CIFAR100:
                public_dataset = torchvision.datasets.CIFAR100(
                    root=self.root,
                    train=True,
                    download=True,
                )
            case _:
                raise ValueError(f"Invalid public task: {self.public_task}")

        for type_ in CommonPartitionType:
            self.path.joinpath(type_.value).mkdir(parents=True)

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
                        numpy_rng=self.rng_suite.numpy,
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
                    numpy_rng=self.rng_suite.numpy,
                )
            case _:
                raise ValueError(f"Invalid partition: {self.partition}")

        for cid, indices in private_client_dict.items():
            client_private_dataset = FilteredDataset(
                indices.tolist(),
                private_dataset.data,
                private_dataset.targets,
                transform=self.train_transform,
            )
            torch.save(
                client_private_dataset,
                self.path.joinpath(CommonPartitionType.PRIVATE, f"{cid}.pkl"),
            )

        for cid, indices in test_client_dict.items():
            client_test_dataset = FilteredDataset(
                indices.tolist(),
                test_dataset.data,
                test_dataset.targets,
                transform=self.test_transform,
            )
            torch.save(
                client_test_dataset,
                self.path.joinpath(CommonPartitionType.TEST, f"{cid}.pkl"),
            )

        public_indices = self.rng_suite.numpy.choice(
            len(public_dataset),
            size=self.public_size,
        )
        torch.save(
            FilteredDataset(
                public_indices.tolist(),
                public_dataset.data,
                original_targets=None,
                transform=self.train_transform,
            ),
            self.path.joinpath(CommonPartitionType.PUBLIC, "public.pkl"),
        )

        torch.save(
            FilteredDataset(
                list(range(len(test_dataset))),
                test_dataset.data,
                test_dataset.targets,
                transform=self.test_transform,
            ),
            self.path.joinpath(CommonPartitionType.TEST, "test.pkl"),
        )

    def get_dataset(self, type_: CommonPartitionType, cid: int | None) -> Dataset:
        match type_:
            case CommonPartitionType.PRIVATE:
                dataset = torch.load(
                    self.path.joinpath(type_, f"{cid}.pkl"),
                    weights_only=False,
                )
            case CommonPartitionType.PUBLIC:
                dataset = torch.load(
                    self.path.joinpath(type_, "public.pkl"),
                    weights_only=False,
                )
            case CommonPartitionType.TEST:
                if cid is not None:
                    dataset = torch.load(
                        self.path.joinpath(type_, f"{cid}.pkl"), weights_only=False
                    )
                else:
                    dataset = torch.load(
                        self.path.joinpath(type_, "test.pkl"), weights_only=False
                    )
        assert isinstance(dataset, Dataset)
        return dataset

    def set_dataset(
        self, type_: CommonPartitionType, cid: int | None, dataset: Dataset
    ) -> None:
        match type_:
            case CommonPartitionType.PRIVATE:
                torch.save(dataset, self.path.joinpath(type_, f"{cid}.pkl"))
            case CommonPartitionType.PUBLIC:
                torch.save(dataset, self.path.joinpath(f"{type_}.pkl"))
            case CommonPartitionType.TEST:
                if cid is not None:
                    torch.save(dataset, self.path.joinpath(type_, f"{cid}.pkl"))
                else:
                    torch.save(dataset, self.path.joinpath(type_, "default.pkl"))

    def get_dataloader(
        self,
        type_: CommonPartitionType,
        cid: int | None,
        batch_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DataLoader:
        dataset = self.get_dataset(type_, cid)
        assert isinstance(dataset, Sized)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=type_ == CommonPartitionType.PRIVATE,
            generator=generator,
        )
        return data_loader

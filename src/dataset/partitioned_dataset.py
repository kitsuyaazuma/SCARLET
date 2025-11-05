from typing import Optional
import random
import numpy as np
import pandas as pd
import torch
import torchvision
from fedlab.utils.dataset.functional import balance_split
from fedlab.contrib.dataset import Subset
from torch.utils.data import Dataset
from torchvision import transforms
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from dataset.tiny_imagenet_200 import TinyImageNet

ROOT_DIR = Path(__file__).resolve().parents[0]

CLASS_NUM = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny-imagenet-200": 200,
    "caltech256": 257,
}

ORIGINAL_IMAGE_SIZE = {
    "cifar10": 32,
    "cifar100": 32,
    "tiny-imagenet-200": 64,
    "caltech256": None,
}


class GrayscaleToRGB(object):
    def __call__(self, img):
        if img.mode == "L":
            img = img.convert("RGB")
        return img


@dataclass
class PartitionedDataset:
    data_dir: Path
    num_clients: int
    partition: str
    dir_alpha: float
    private_task: str
    public_task: str
    public_size: int
    validation_ratio: float
    train_batch_size: int
    test_batch_size: int

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.joinpath("validation").mkdir(parents=True, exist_ok=True)
        self.num_classes = CLASS_NUM[self.private_task]
        self.public_train_size = self.public_size - int(
            self.public_size * self.validation_ratio
        )
        self.public_validation_size = int(self.public_size * self.validation_ratio)
        self._prepare()

    def _get_transform(self, task: str, train: bool):
        image_size = ORIGINAL_IMAGE_SIZE[self.private_task]
        transform_list = [
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        if train:
            padding = 8 if image_size == 64 else 4
            transform_list[2:2] = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(image_size, padding=padding),
            ]

            if task == "caltech256":
                transform_list = [GrayscaleToRGB()] + transform_list

        transform = transforms.Compose(transform_list)

        return transform

    def _get_dataset(self, task: str, split: str = "train"):
        root = f"{ROOT_DIR}/{task}/"
        match task:
            case "cifar10":
                return torchvision.datasets.CIFAR10(
                    root=root, train=split == "train", download=True
                )
            case "cifar100":
                return torchvision.datasets.CIFAR100(
                    root=root, train=split == "train", download=True
                )
            case "tiny-imagenet-200":
                return TinyImageNet(
                    root=root,
                    split="train" if split == "train" else "val",
                    download=True,
                )
            case "caltech256":
                return torchvision.datasets.Caltech256(
                    root=root, download=True, transform=None
                )
            case _:
                raise ValueError(f"Invalid task: {task}")

    def save_distribution_stats(self, dir: Path) -> None:
        stats_dict = {}
        assert isinstance(self.train_targets, np.ndarray) or isinstance(
            self.train_targets, list
        )
        for cid, indices in self.client_to_indices.items():
            class_count = [0] * self.num_classes
            for index in indices:
                class_count[self.train_targets[index]] += 1
            stats_dict[cid] = class_count
        stats_df = pd.DataFrame.from_dict(
            stats_dict,
            orient="index",
            columns=list(map(str, range(self.num_classes))),  # type: ignore
        )
        stats_df.to_csv(dir.joinpath("distribution.csv"))
        del self.train_targets, self.client_to_indices

    def _prepare(self) -> None:  # noqa: C901
        private_trainset = self._get_dataset(task=self.private_task, split="train")
        private_train_transform = self._get_transform(
            task=self.private_task, train=True
        )
        public_split = "train"
        public_trainset = self._get_dataset(task=self.public_task, split=public_split)
        public_train_transform = self._get_transform(task=self.public_task, train=True)
        public_val_transform = self._get_transform(task=self.public_task, train=False)
        testset = self._get_dataset(task=self.private_task, split="test")
        test_transform = self._get_transform(task=self.private_task, train=False)
        match self.partition:
            case "client_inner_dirichlet":
                if isinstance(private_trainset, torchvision.datasets.Caltech256):
                    self.train_targets = private_trainset.y
                else:
                    self.train_targets = private_trainset.targets
                assert self.train_targets is not None
                self.client_to_indices, class_priors = (
                    client_inner_dirichlet_partition_faster(
                        targets=self.train_targets,
                        num_clients=self.num_clients,
                        num_classes=self.num_classes,
                        dir_alpha=self.dir_alpha,
                        client_sample_nums=balance_split(
                            self.num_clients, len(self.train_targets)
                        ),
                        verbose=False,
                    )
                )
                assert not isinstance(testset, torchvision.datasets.Caltech256)
                test_targets = testset.targets
                assert test_targets is not None
                client_to_test_indices, _ = client_inner_dirichlet_partition_faster(
                    targets=test_targets,
                    num_clients=self.num_clients,
                    num_classes=self.num_classes,
                    dir_alpha=self.dir_alpha,
                    client_sample_nums=balance_split(
                        self.num_clients, len(test_targets)
                    ),
                    verbose=False,
                    class_priors=class_priors,
                )
            case _:
                raise ValueError(f"Invalid partition method: {self.partition}")

        train_dir = self.data_dir.joinpath("private_train")
        train_dir.mkdir(parents=True, exist_ok=True)
        for client_id, indices in tqdm(
            self.client_to_indices.items(), desc="Saving train data"
        ):
            train_indices = np.random.choice(
                indices,
                size=int(len(indices) * (1 - self.validation_ratio)),
                replace=False,
            )
            val_indices = list(set(indices) - set(train_indices))

            torch.save(
                Subset(
                    dataset=private_trainset,
                    indices=list(indices),
                    transform=private_train_transform,
                ),
                train_dir.joinpath(f"{client_id}.pkl"),
            )
            torch.save(
                Subset(
                    dataset=private_trainset,
                    indices=val_indices,
                    transform=test_transform,
                ),
                self.data_dir.joinpath("validation").joinpath(f"{client_id}.pkl"),
            )

        test_dir = self.data_dir.joinpath("test")
        test_dir.mkdir(parents=True, exist_ok=True)
        for client_id, indices in tqdm(
            client_to_test_indices.items(), desc="Saving test data"
        ):
            torch.save(
                Subset(
                    dataset=testset, indices=list(indices), transform=test_transform
                ),
                test_dir.joinpath(f"{client_id}.pkl"),
            )

        torch.save(
            Subset(
                dataset=testset,
                indices=range(len(testset)),
                transform=test_transform,
            ),
            self.data_dir.joinpath("test.pkl"),
        )

        public_indices = random.sample(range(len(public_trainset)), self.public_size)
        public_train_indices = np.random.choice(
            public_indices,
            size=int(len(public_indices) * (1 - self.validation_ratio)),
            replace=False,
        )
        public_val_indices = list(set(public_indices) - set(public_train_indices))
        if isinstance(public_trainset, torchvision.datasets.Caltech256):
            public_subset = Caltech256Subset(
                dataset=public_trainset,
                indices=public_train_indices,
                transform=public_train_transform,
            )
            public_val_subset = Caltech256Subset(
                dataset=public_trainset,
                indices=public_val_indices,
                transform=public_val_transform,
            )
        else:
            public_subset = Subset(
                dataset=public_trainset,
                indices=public_train_indices,
                transform=public_train_transform,
            )
            public_val_subset = Subset(
                dataset=public_trainset,
                indices=public_val_indices,
                transform=public_val_transform,
            )
        torch.save(
            public_subset,
            self.data_dir.joinpath("public_train.pkl"),
        )
        torch.save(
            public_val_subset,
            self.data_dir.joinpath("public_validation.pkl"),
        )

    def get_private_train_dataset(self, client_id: int) -> Dataset:
        dataset = torch.load(
            self.data_dir.joinpath("private_train").joinpath(f"{client_id}.pkl")
        )
        assert isinstance(dataset, Dataset)
        return dataset

    def get_private_validation_dataset(self, client_id: int) -> Dataset:
        dataset = torch.load(
            self.data_dir.joinpath("validation").joinpath(f"{client_id}.pkl")
        )
        assert isinstance(dataset, Dataset)
        return dataset

    def get_public_train_dataset(self) -> Dataset:
        dataset = torch.load(self.data_dir.joinpath("public_train.pkl"))
        assert isinstance(dataset, Dataset)
        return dataset

    def get_public_validation_dataset(self) -> Dataset:
        dataset = torch.load(self.data_dir.joinpath("public_validation.pkl"))
        assert isinstance(dataset, Dataset)
        return dataset

    def get_whole_train_dataset(self) -> Dataset:
        dataset = torch.load(self.data_dir.joinpath("centralized.pkl"))
        assert isinstance(dataset, Dataset)
        return dataset

    def get_test_dataset(self, client_id: Optional[int] = None) -> Dataset:
        if client_id is None:
            dataset = torch.load(self.data_dir.joinpath("test.pkl"))
        else:
            dataset = torch.load(
                self.data_dir.joinpath("test").joinpath(f"{client_id}.pkl")
            )
        assert isinstance(dataset, Dataset)
        return dataset


class Caltech256Subset(Subset):
    def __init__(
        self,
        dataset: torchvision.datasets.Caltech256,
        indices,
        transform=None,
        target_transform=None,
        save_image_size=64,
    ):
        self.data = []
        pre_transform = transforms.Compose(
            [transforms.Resize(save_image_size)]
        )  # for saving memory
        for idx in indices:
            self.data.append(pre_transform(dataset[idx][0]))
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.data[index]
        label = self.indices[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.indices)


class NonLabelDataset(Dataset):
    def __init__(self, data: list, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def client_inner_dirichlet_partition_faster(
    targets,
    num_clients,
    num_classes,
    dir_alpha,
    client_sample_nums,
    verbose=True,
    class_priors: Optional[np.ndarray] = None,
):
    """Non-iid Dirichlet partition.

    The method is from The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.
    This function can be used by given specific sample number for all clients ``client_sample_nums``.
    It's different from :func:`hetero_dir_partition`.

    Args:
        targets (list or numpy.ndarray): Sample targets.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        client_sample_nums (numpy.ndarray): A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.

    Returns:
        dict: ``{ client_id: indices}``.

    Note:
        Adapted from https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
    """  # noqa: E501
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    if class_priors is None:  # CHANGED: use given class_priors if provided
        class_priors = np.random.dirichlet(
            alpha=[dir_alpha] * num_classes, size=num_clients
        )
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [
        np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in range(num_clients)
    ]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        if verbose:
            print("Remaining Data: %d" % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                # Exception handling: If the current class has no samples left, randomly select a non-zero class # noqa: E501
                while True:
                    new_class = np.random.randint(num_classes)
                    if class_amount[new_class] > 0:
                        curr_class = new_class
                        break
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = idx_list[
                curr_class
            ][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    return client_dict, class_priors

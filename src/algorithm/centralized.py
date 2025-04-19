import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from algorithm.base import (
    BaseServerHandler,
)
from dataset import PartitionedDataset


class CentralizedServerHandler(BaseServerHandler):
    def __init__(
        self,
        model_name: str,
        global_round: int,
        cuda: bool,
        dataset: PartitionedDataset,
        epochs: int,
        lr: float,
        batch_size: int,
    ):
        super().__init__(
            model_name,
            global_round,
            sample_ratio=0.0,
            cuda=cuda,
            public_size_per_round=0,
            dataset=dataset,
        )
        self.epochs = epochs
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = F.cross_entropy
        self.batch_size = batch_size

    def global_update(self, buffer: list) -> None:
        trainset = self.dataset.get_whole_train_dataset()
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for _ in tqdm(
            range(self.epochs), desc=f"Global Round {self.round}", leave=False
        ):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def setup_kd_optim(self, kd_epochs: int, kd_batch_size: int, kd_lr: float):
        pass

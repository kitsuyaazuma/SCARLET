import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch

from algorithm.dsfl import DSFLClientTrainer, DSFLServerHandler
from dataset.interface import DatasetProvider
from models import CommonModelName


class TestRefactoringSuccess(unittest.TestCase):
    def setUp(self):
        self.mock_model = torch.nn.Linear(10, 2)

        self.mock_dataset = MagicMock(spec=DatasetProvider)
        self.mock_dataset.num_classes = 2
        self.mock_dataset.public_train_size = 100

        self.mock_selector = MagicMock()
        self.mock_selector.select_model.return_value = self.mock_model

    def test_server_instantiation(self):
        handler = DSFLServerHandler(
            model=self.mock_model,
            dataset=self.mock_dataset,
            global_round=10,
            num_clients=5,
            sample_ratio=1.0,
            device="cpu",
            kd_epochs=1,
            kd_batch_size=32,
            kd_lr=0.01,
            public_size_per_round=100,
            seed=42,
            era_temperature=0.1,
        )
        self.assertIsNotNone(handler)
        print("ServerHandler instantiated successfully without heavy deps!")

    def test_client_instantiation(self):
        trainer = DSFLClientTrainer(
            model_selector=self.mock_selector,
            model_name=CommonModelName.RESNET20,
            dataset=self.mock_dataset,
            manager=None,
            device="cpu",
            num_clients=5,
            epochs=1,
            batch_size=32,
            lr=0.01,
            kd_epochs=1,
            kd_batch_size=32,
            kd_lr=0.01,
            seed=42,
            num_parallels=1,
            public_size_per_round=100,
            state_dir=Path("/tmp"),
        )
        self.assertIsNotNone(trainer)
        print("ClientTrainer instantiated successfully without multiprocessing!")

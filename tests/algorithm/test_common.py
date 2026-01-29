import threading
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from scarlet.algorithm import (
    CommonClientTrainer,
    CommonServerHandler,
    distill,
    evaulate,
    predict,
    train,
)
from scarlet.dataset import CommonPartitionType


class RawTensorDataset(Dataset):
    """Dataset that returns a single tensor instead of a tuple."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class MockDatasetProvider:
    def __init__(self) -> None:
        self.num_classes = 2
        self.public_train_size = 10
        self.data = torch.randn(20, 2)
        self.targets = torch.randint(0, 2, (20,))

    def get_dataset(self, type_: Any, cid: int | None) -> Dataset:
        _ = cid
        if type_ in [CommonPartitionType.TRAIN_PRIVATE, CommonPartitionType.TEST]:
            return TensorDataset(self.data, self.targets)
        return RawTensorDataset(self.data)

    def get_dataloader(
        self,
        type_: Any,
        cid: int | None,
        batch_size: int | None = None,
        generator: Any = None,
    ) -> DataLoader:
        _ = generator
        dataset = self.get_dataset(type_, cid)
        return DataLoader(dataset, batch_size=batch_size or 4)


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DummyCommonServerHandler(CommonServerHandler):
    def global_update(self, buffer: Any) -> None:
        _ = buffer
        pass

    def downlink_package(self) -> Any:
        return "dummy_downlink"


class DummyCommonClientTrainer(CommonClientTrainer):
    def prepare_uplink_package_buffer(self) -> Any:
        return "dummy_buffer"

    @staticmethod
    def worker(
        config: Any,
        payload: Any,
        device: str,
        stop_event: threading.Event,
        *,
        shm_buffer: Any | None = None,
    ) -> str:
        _ = (config, payload, device, stop_event, shm_buffer)
        return "dummy_uplink"

    def get_client_config(self, cid: int) -> Any:
        return {"cid": cid}


class MockLogger:
    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, float], int | None]] = []

    def log(self, data: dict[str, float], step: int | None = None) -> None:
        self.logs.append((data, step))


def test_evaluate() -> None:
    model = SimpleModel()
    dataset = TensorDataset(torch.randn(10, 2), torch.randint(0, 2, (10,)))
    loader = DataLoader(dataset, batch_size=2)
    device = "cpu"

    loss, acc = evaulate(model, loader, device)
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0 <= acc <= 1


def test_train() -> None:
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = TensorDataset(torch.randn(10, 2), torch.randint(0, 2, (10,)))
    loader = DataLoader(dataset, batch_size=2)
    device = "cpu"
    stop_event = threading.Event()

    loss, acc = train(model, optimizer, loader, device, epochs=1, stop_event=stop_event)
    assert isinstance(loss, float)
    assert isinstance(acc, float)


def test_predict() -> None:
    model = SimpleModel()
    dataset = RawTensorDataset(torch.randn(5, 2))
    loader = DataLoader(dataset, batch_size=2)
    device = "cpu"

    soft_labels = predict(model, loader, device)
    assert soft_labels.shape == (5, 2)
    assert torch.allclose(soft_labels.sum(dim=1), torch.ones(5))


def test_distill() -> None:
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = RawTensorDataset(torch.randn(10, 2))
    loader = DataLoader(dataset, batch_size=2)
    global_soft_labels = [torch.rand(2) for _ in range(10)]
    device = "cpu"

    loss = distill(
        model,
        optimizer,
        loader,
        global_soft_labels,
        kd_epochs=1,
        kd_batch_size=2,
        device=device,
        stop_event=None,
    )
    assert isinstance(loss, float)


@pytest.fixture
def server_handler() -> DummyCommonServerHandler:
    model = SimpleModel()
    dataset_provider = MockDatasetProvider()
    handler = DummyCommonServerHandler(
        model=model,
        dataset=dataset_provider,
        global_round=2,
        num_clients=10,
        sample_ratio=0.5,
        device="cpu",
        kd_epochs=1,
        kd_batch_size=2,
        kd_lr=0.01,
        public_size_per_round=4,
        seed=42,
    )
    return handler


def test_common_server_handler(server_handler: DummyCommonServerHandler) -> None:
    handler = server_handler

    assert handler.get_round() == 0
    assert not handler.if_stop()

    # Test sampling
    clients = handler.sample_clients()
    assert len(clients) == 5

    # Test indices generation
    indices = handler.get_next_indices()
    assert len(indices) == 4

    # --- Round 0 ---
    # Partial load (4/5 clients): should NOT trigger update
    for _ in range(4):
        assert not handler.load("uplink_pkg")

    # 5th client: triggers update
    assert handler.load("uplink_pkg") is True
    assert handler.get_round() == 1

    # --- Round 1 ---
    # Verify automatic buffer reset
    for _ in range(4):
        assert handler.load("pkg_round_2") is False

    assert handler.load("pkg_round_2") is True
    assert handler.get_round() == 2

    assert handler.if_stop()


def test_common_server_handler_get_summary(
    server_handler: DummyCommonServerHandler,
) -> None:
    handler = server_handler

    # Inject dummy metrics data
    handler.metrics_list = [
        {"client_test_loss": 0.5, "client_test_acc": 0.8},
        {"client_test_loss": 0.3, "client_test_acc": 0.9},
    ]

    summary = handler.get_summary()

    # Verify server-side evaluation keys
    assert "server_test_loss" in summary
    assert "server_test_acc" in summary

    # Verify client-side aggregation (average)
    # loss: (0.5 + 0.3) / 2 = 0.4
    # acc: (0.8 + 0.9) / 2 = 0.85
    # Use pytest.approx for floating point comparison
    assert summary["client_test_loss"] == pytest.approx(0.4)
    assert summary["client_test_acc"] == pytest.approx(0.85)


def test_common_client_trainer(tmp_path: Path) -> None:
    model = SimpleModel()

    class MockModelSelector:
        def get_model(self, name: Any, device: Any) -> Any:
            _ = (name, device)
            return model

    dataset_provider = MockDatasetProvider()

    trainer = DummyCommonClientTrainer(
        model_selector=MockModelSelector(),  # type: ignore[arg-type]
        model_name="dummy_model",  # type: ignore[arg-type]
        dataset=dataset_provider,
        device="cpu",
        num_clients=10,
        epochs=1,
        batch_size=2,
        lr=0.01,
        kd_epochs=1,
        kd_batch_size=2,
        kd_lr=0.01,
        seed=42,
        num_parallels=1,
        public_size_per_round=4,
        state_dir=tmp_path,
        manager=None,
    )

    assert len(trainer.cache) == 0

    # Simulate cache update
    dummy_payload = "test_payload"
    trainer.cache.append(dummy_payload)
    assert len(trainer.cache) == 1

    package = trainer.uplink_package()

    # Verify package content and cache clearing
    assert len(package) == 1
    assert package[0] == dummy_payload
    assert len(trainer.cache) == 0

    # Verify deep copy
    assert package is not trainer.cache


def test_common_pipeline() -> None:
    """Test the main loop of CommonPipeline."""
    from scarlet.algorithm.common import CommonPipeline

    # 1. Setup
    model = SimpleModel()
    dataset_provider = MockDatasetProvider()

    # Configure handler to stop after 2 rounds
    handler = DummyCommonServerHandler(
        model=model,
        dataset=dataset_provider,
        global_round=2,
        num_clients=4,
        sample_ratio=1.0,  # All clients participate
        device="cpu",
        kd_epochs=1,
        kd_batch_size=2,
        kd_lr=0.01,
        public_size_per_round=2,
        seed=42,
    )
    # Initialize with dummy metrics to prevent get_summary errors (division by zero)
    handler.metrics_list = [{"client_test_loss": 0.0, "client_test_acc": 0.0}]

    # Trainer: Mock implementation
    class MockClientTrainer(DummyCommonClientTrainer):
        def local_process(self, payload: Any, cid_list: list[int]) -> None:
            _ = payload
            # Simulate client processing
            for _ in cid_list:
                self.cache.append("dummy_uplink_pkg")

        def uplink_package(self) -> list[Any]:
            pkg = self.cache[:]
            self.cache = []
            return pkg

    trainer = MockClientTrainer(
        model_selector=None,  # type: ignore[arg-type]
        model_name="dummy",  # type: ignore[arg-type]
        dataset=dataset_provider,
        device="cpu",
        num_clients=4,
        epochs=1,
        batch_size=2,
        lr=0.01,
        kd_epochs=1,
        kd_batch_size=2,
        kd_lr=0.01,
        seed=42,
        num_parallels=1,
        public_size_per_round=2,
        state_dir=Path("."),
        manager=None,
    )

    logger = MockLogger()

    # 2. Pipeline Execution
    pipeline = CommonPipeline(handler=handler, trainer=trainer, logger=logger)
    pipeline.main()

    # 3. Assertions
    # Ensure the pipeline ran for 2 rounds
    assert handler.get_round() == 2
    assert handler.if_stop() is True

    # Verify logging occurred for each round (Round 0 and Round 1)
    assert len(logger.logs) == 2
    assert logger.logs[0][1] == 0  # step=0
    assert logger.logs[1][1] == 1  # step=1

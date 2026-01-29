import threading
from pathlib import Path
from typing import Any

import pytest
import torch

from scarlet.algorithm.common import CommonServerArgs
from scarlet.algorithm.dsfl import (
    DSFLClientConfig,
    DSFLClientTrainer,
    DSFLDownlinkPackage,
    DSFLServerHandler,
    DSFLUplinkPackage,
)

from .test_common import MockDatasetProvider, SimpleModel


@pytest.fixture
def dsfl_server_handler() -> DSFLServerHandler:
    model = SimpleModel()
    dataset = MockDatasetProvider()
    args = CommonServerArgs(
        dataset=dataset,
        global_round=2,
        num_clients=2,
        sample_ratio=1.0,
        device="cpu",
        kd_epochs=1,
        kd_batch_size=2,
        kd_lr=0.01,
        public_size_per_round=4,
        seed=42,
    )
    return DSFLServerHandler.from_args(args, model, era_temperature=0.5)


def test_dsfl_server_handler_global_update(
    dsfl_server_handler: DSFLServerHandler,
) -> None:
    handler = dsfl_server_handler
    # Prepare mock uplink packages
    buffer = [
        DSFLUplinkPackage(
            cid=0,
            soft_labels=torch.randn(4, 2),
            indices=torch.tensor([0, 1, 2, 3]),
            metrics={"client_test_loss": 0.5, "client_test_acc": 0.8},
        ),
        DSFLUplinkPackage(
            cid=1,
            soft_labels=torch.randn(4, 2),
            indices=torch.tensor([0, 1, 2, 3]),
            metrics={"client_test_loss": 0.3, "client_test_acc": 0.9},
        ),
    ]

    handler.global_update(buffer)
    assert handler.round == 0
    assert (
        handler.global_soft_labels is not None
        and handler.global_soft_labels.shape == (4, 2)
    )
    assert len(handler.metrics_list) == 2


def test_dsfl_client_trainer_worker(tmp_path: Path) -> None:
    model_name = "resnet18"
    dataset = MockDatasetProvider()

    class MockModelSelector:
        def select_model(self, name: Any) -> torch.nn.Module:
            _ = name
            return SimpleModel()

    config = DSFLClientConfig(
        model_selector=MockModelSelector(),  # type: ignore[arg-type]
        model_name=model_name,  # type: ignore[arg-type]
        dataset=dataset,
        epochs=1,
        batch_size=2,
        lr=0.01,
        kd_epochs=1,
        kd_batch_size=2,
        kd_lr=0.01,
        cid=0,
        seed=42,
        state_path=tmp_path / "0.pt",
    )

    payload = DSFLDownlinkPackage(
        soft_labels=torch.randn(2, 2),
        indices=torch.tensor([0, 1]),
        next_indices=torch.tensor([2, 3]),
    )

    # Mock SHM buffer
    shm_buffer = DSFLUplinkPackage(
        cid=-1,
        soft_labels=torch.zeros(2, 2),
        indices=torch.zeros(2, dtype=torch.long),
        metrics={},
    )

    stop_event = threading.Event()
    package = DSFLClientTrainer.worker(
        config, payload, "cpu", stop_event, shm_buffer=shm_buffer
    )

    assert package.cid == 0
    assert "client_train_loss" in package.metrics
    assert config.state_path.exists()

import threading
from dataclasses import dataclass
from typing import TypeVar

import pytest
import torch
import torch.multiprocessing as mp

from scarlet.core.client_trainer import ProcessPoolClientTrainer
from scarlet.core.utils import SHMHandle

UplinkPackage = TypeVar("UplinkPackage")
DownlinkPackage = TypeVar("DownlinkPackage")
ClientConfig = TypeVar("ClientConfig")


@dataclass
class DummyUplinkPackage:
    cid: int
    message: str
    tensor: torch.Tensor | SHMHandle


@dataclass
class DummyDownlinkPackage:
    message: str


@dataclass
class DummyClientConfig:
    cid: int


class DummyProcessPoolClientTrainer(
    ProcessPoolClientTrainer[
        DummyUplinkPackage, DummyDownlinkPackage, DummyClientConfig
    ]
):
    def __init__(
        self,
        num_parallels: int,
        device: str,
    ) -> None:
        self.num_parallels = num_parallels
        self.device = device
        if self.device == "cuda":
            self.device_count = torch.cuda.device_count()
        self.cache: list[DummyUplinkPackage] = []
        self.manager = mp.Manager()
        self.stop_event = self.manager.Event()

    def uplink_package(self) -> list[DummyUplinkPackage]:
        return self.cache

    def get_client_config(self, cid: int) -> DummyClientConfig:
        return DummyClientConfig(cid=cid)

    def prepare_uplink_package_buffer(self) -> DummyUplinkPackage:
        return DummyUplinkPackage(cid=-1, message="", tensor=torch.zeros(1))

    @staticmethod
    def worker(
        config: DummyClientConfig,
        payload: DummyDownlinkPackage,
        device: str,
        stop_event: threading.Event,
        *,
        shm_buffer: DummyUplinkPackage | None = None,
    ) -> DummyUplinkPackage:
        _ = stop_event
        _ = device

        new_tensor = torch.rand(1)
        new_message = payload.message + "<client_to_server>"

        package = DummyUplinkPackage(
            cid=config.cid,
            tensor=new_tensor,
            message=new_message,
        )

        assert shm_buffer is not None
        assert isinstance(shm_buffer.tensor, torch.Tensor) and isinstance(
            package.tensor, torch.Tensor
        )
        shm_buffer.tensor.copy_(package.tensor)
        package.tensor = SHMHandle()
        return package


@pytest.mark.parametrize("num_parallels", [1, 2])
@pytest.mark.parametrize("cid_list", [[], [42], [0, 1, 2]])
def test_process_pool_client_trainer(
    num_parallels: int,
    cid_list: list[int],
) -> None:
    trainer = DummyProcessPoolClientTrainer(
        num_parallels=num_parallels,
        device="cpu",
    )

    dummy_payload = DummyDownlinkPackage(message="<server_to_client>")

    trainer.local_process(dummy_payload, cid_list)

    assert len(trainer.cache) == len(cid_list)

    cached_cids = sorted([p.cid for p in trainer.cache])
    expected_cids = sorted(cid_list)
    assert cached_cids == expected_cids

    for package in trainer.cache:
        assert package.message == "<server_to_client><client_to_server>"
        assert isinstance(package.tensor, torch.Tensor)
        assert not isinstance(package.tensor, SHMHandle)

from dataclasses import dataclass
from typing import TypeVar

from scarlet.core.server_handler import BaseServerHandler

UplinkPackage = TypeVar("UplinkPackage")
DownlinkPackage = TypeVar("DownlinkPackage")


@dataclass
class DummyUplinkPackage:
    cid: int
    data: float


@dataclass
class DummyDownlinkPackage:
    global_model: float


class DummyServerHandler(BaseServerHandler[DummyUplinkPackage, DummyDownlinkPackage]):
    def __init__(self, global_round: int) -> None:
        self.round = 0
        self.global_round = global_round
        self.buffer: list[DummyUplinkPackage] = []
        self.model_value = 0.0

    def downlink_package(self) -> DummyDownlinkPackage:
        return DummyDownlinkPackage(global_model=self.model_value)

    def sample_clients(self) -> list[int]:
        return [0, 1, 2]

    def if_stop(self) -> bool:
        return self.round >= self.global_round

    def global_update(self, buffer: list[DummyUplinkPackage]) -> None:
        if not buffer:
            return
        total_data = sum(pkg.data for pkg in buffer)
        self.model_value = total_data / len(buffer)
        self.round += 1

    def load(self, payload: DummyUplinkPackage) -> bool:
        self.buffer.append(payload)
        if len(self.buffer) >= 3:
            self.global_update(self.buffer)
            self.buffer = []
            return True
        return False


def test_server_handler_implementation() -> None:
    handler = DummyServerHandler(global_round=2)

    downlink = handler.downlink_package()
    assert isinstance(downlink, DummyDownlinkPackage)
    assert downlink.global_model == 0.0

    clients = handler.sample_clients()
    assert isinstance(clients, list)
    assert clients == [0, 1, 2]

    assert handler.load(DummyUplinkPackage(cid=0, data=1.0)) is False
    assert handler.load(DummyUplinkPackage(cid=1, data=2.0)) is False

    assert handler.load(DummyUplinkPackage(cid=2, data=3.0)) is True

    # (1.0 + 2.0 + 3.0) / 3 = 2.0
    assert handler.model_value == 2.0
    assert handler.round == 1

    assert len(handler.buffer) == 0

    assert handler.if_stop() is False

    handler.load(DummyUplinkPackage(cid=0, data=1.0))
    handler.load(DummyUplinkPackage(cid=1, data=1.0))
    handler.load(DummyUplinkPackage(cid=2, data=1.0))

    assert handler.round == 2
    assert handler.if_stop() is True

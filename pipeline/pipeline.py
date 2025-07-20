import logging
from typing import Protocol, TypeVar

from blazefl.core import BaseServerHandler, ProcessPoolClientTrainer
from torch.utils.tensorboard.writer import SummaryWriter


class SummarizableBaseServerHandler(BaseServerHandler, Protocol):
    round: int

    def get_summary(self) -> dict[str, float]: ...


CommonServerHandler = TypeVar(
    "CommonServerHandler", bound=SummarizableBaseServerHandler
)
CommonClientTrainer = TypeVar("CommonClientTrainer", bound=ProcessPoolClientTrainer)


class CommonPipeline:
    def __init__(
        self,
        handler: CommonServerHandler,
        trainer: CommonClientTrainer,
        writer: SummaryWriter,
    ) -> None:
        self.handler = handler
        self.trainer = trainer
        self.writer = writer

    def main(self) -> None:
        while not self.handler.if_stop():
            assert hasattr(self.handler, "round")
            round_ = self.handler.round
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package()

            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package()

            # server side
            for pack in uploads:
                self.handler.load(pack)

            summary = self.handler.get_summary()
            for key, value in summary.items():
                self.writer.add_scalar(key, value, round_)
            formatted_summary = ", ".join(f"{k}: {v:.3f}" for k, v in summary.items())
            logging.info(f"round: {round_}, {formatted_summary}")

        logging.info("done!")

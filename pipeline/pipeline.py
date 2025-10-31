import logging
from typing import Protocol

from blazefl.core import BaseServerHandler, ThreadPoolClientTrainer

import wandb


class SummarizableBaseServerHandler(BaseServerHandler, Protocol):
    round: int

    def get_summary(self) -> dict[str, float]: ...


class CommonPipeline:
    def __init__(
        self,
        handler: SummarizableBaseServerHandler,
        # trainer: CommonClientTrainer,
        trainer: ThreadPoolClientTrainer,
        run: wandb.Run,
    ) -> None:
        self.handler = handler
        self.trainer = trainer
        self.run = run

    def main(self) -> None:
        while not self.handler.if_stop():
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
            self.run.log(summary, step=round_)
            formatted_summary = ", ".join(f"{k}: {v:.3f}" for k, v in summary.items())
            logging.info(f"round: {round_}, {formatted_summary}")

        logging.info("done!")

import logging

from torch.utils.tensorboard.writer import SummaryWriter

from algorithm.dsfl import DSFLClientTrainer, DSFLServerHandler


class DSFLPipeline:
    def __init__(
        self,
        handler: DSFLServerHandler,
        trainer: DSFLClientTrainer,
        writer: SummaryWriter,
    ) -> None:
        self.handler = handler
        self.trainer = trainer
        self.writer = writer

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
            for key, value in summary.items():
                self.writer.add_scalar(key, value, round_)
            formatted_str = (
                "{ "
                + ", ".join("'{}': {:.3f}".format(k, v) for k, v in summary.items())
                + " }"
            )
            logging.info(f"Round {round_}: {formatted_str}")

        logging.info("Done!")

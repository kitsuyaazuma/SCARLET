from torch.utils.tensorboard.writer import SummaryWriter

from algorithm.scarlet import SCARLETClientTrainer, SCARLETServerHandler
from pipeline.dsfl import DSFLPipeline


class SCARLETPipeline(DSFLPipeline):
    def __init__(
        self,
        handler: SCARLETServerHandler,
        trainer: SCARLETClientTrainer,
        writer: SummaryWriter,
    ) -> None:
        self.handler = handler
        self.trainer = trainer
        self.writer = writer

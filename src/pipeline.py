import logging
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from algorithm import (
    DSFLServerHandler,
    DSFLParallelClientTrainer,
    SCARLETServerHandler,
    SCARLETParallelClientTrainer,
    COMETServerHandler,
    COMETParallelClientTrainer,
    SelectiveFDServerHandler,
    SelectiveFDParallelClientTrainer,
    CFDServerHandler,
    CFDParallelClientTrainer,
    CentralizedServerHandler,
)
from dataclasses import dataclass

from algorithm.individual import IndividualParallelClientTrainer


@dataclass
class BasePipeline:
    writer: SummaryWriter

    def _post_init__(self):
        self.round = 0
        self.cumulative_cost = 0.0
        self.uplink_cost = 0.0
        self.downlink_cost = 0.0

    def get_byte_size(self, tensor: torch.Tensor) -> int:
        return tensor.element_size() * tensor.nelement()


@dataclass
class DSFLPipeline(BasePipeline):
    handler: DSFLServerHandler
    trainer: DSFLParallelClientTrainer

    def __post_init__(self):
        super()._post_init__()
        self.handler.num_clients = self.trainer.num_clients

    def main(self):
        while self.handler.if_stop is False:
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package

            downlink_cost = 0.0
            for b in broadcast:
                if isinstance(b, torch.Tensor):
                    downlink_cost += self.get_byte_size(b) * len(sampled_clients)
            self.cumulative_cost += downlink_cost

            self.trainer.local_process(payload=broadcast, id_list=sampled_clients)
            uploads = self.trainer.uplink_package

            uplink_cost = 0.0
            for pack in uploads:
                self.handler.load(pack)
                for p in pack:
                    if isinstance(p, torch.Tensor):
                        uplink_cost += self.get_byte_size(p)
            self.cumulative_cost += uplink_cost

            server_loss, server_top1_acc, server_top5_acc = self.handler.evaluate()
            server_acc = (
                server_top5_acc
                if self.trainer.dataset.private_task == "cifar100"
                else server_top1_acc
            )
            logging.info(
                f"Round {self.round:>3}, Loss {server_loss:.4f}, "
                f"Test Accuracy {server_acc:.4f}, "
                f"Cost {(self.cumulative_cost / (1024**3)):.4f} GB"
            )
            logging.info(
                f"Uplink {(uplink_cost / (1024**2)):.4f} MB, "
                f"Downlink {(downlink_cost / (1024**2)): 4f} MB"
            )
            self.writer.add_scalar("Cost", self.cumulative_cost, self.round)
            self.writer.add_scalar("Uplink", uplink_cost, self.round)
            self.writer.add_scalar("Downlink", downlink_cost, self.round)

            self.writer.add_scalar("Loss/Server", server_loss, self.round)
            self.writer.add_scalar("Top1Accuracy/Server", server_top1_acc, self.round)
            self.writer.add_scalar("Top5Accuracy/Server", server_top5_acc, self.round)

            self.round += 1


@dataclass
class SCARLETPipeline(DSFLPipeline):
    handler: SCARLETServerHandler
    trainer: SCARLETParallelClientTrainer


@dataclass
class COMETPipeline(DSFLPipeline):
    handler: COMETServerHandler
    trainer: COMETParallelClientTrainer

    def main(self):
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package

            downlink_cost = 0.0
            for i, b in enumerate(broadcast):
                if isinstance(b, torch.Tensor):
                    if self.handler.sample_ratio == 1.0 and i == 0:
                        # NOTE: When sample_ratio is 1.0, we can determine the best
                        # centroid also on the server side.
                        # For fair comparison, devide the cost by the number of clusters
                        downlink_cost += (
                            self.get_byte_size(b)
                            * len(sampled_clients)
                            / self.handler.num_clusters
                        )
                    else:
                        downlink_cost += self.get_byte_size(b) * len(sampled_clients)
            self.cumulative_cost += downlink_cost

            self.trainer.local_process(payload=broadcast, id_list=sampled_clients)
            uploads = self.trainer.uplink_package

            uplink_cost = 0.0
            for pack in uploads:
                self.handler.load(pack)
                for p in pack:
                    if isinstance(p, torch.Tensor):
                        uplink_cost += self.get_byte_size(p)
            self.cumulative_cost += uplink_cost

            server_loss, server_top1_acc, server_top5_acc = self.handler.evaluate()
            server_acc = (
                server_top5_acc
                if self.trainer.dataset.private_task == "cifar100"
                else server_top1_acc
            )
            logging.info(
                f"Round {self.round:>3}, Loss {server_loss:.4f}, "
                f"Test Accuracy {server_acc:.4f}, "
                f"Cost {(self.cumulative_cost / (1024**3)):.4f} GB"
            )
            logging.info(
                f"Uplink {(uplink_cost / (1024**2)):.4f} MB, "
                f"Downlink {(downlink_cost / (1024**2)): 4f} MB"
            )
            self.writer.add_scalar("Cost", self.cumulative_cost, self.round)
            self.writer.add_scalar("Uplink", uplink_cost, self.round)
            self.writer.add_scalar("Downlink", downlink_cost, self.round)
            self.writer.add_scalar("Loss/Server", server_loss, self.round)
            self.writer.add_scalar("Top1Accuracy/Server", server_top1_acc, self.round)
            self.writer.add_scalar("Top5Accuracy/Server", server_top5_acc, self.round)

            self.round += 1


@dataclass
class SelectiveFDPipeline(DSFLPipeline):
    handler: SelectiveFDServerHandler
    trainer: SelectiveFDParallelClientTrainer


@dataclass
class CFDPipeline(DSFLPipeline):
    handler: CFDServerHandler
    trainer: CFDParallelClientTrainer


@dataclass
class IndividualPipeline(BasePipeline):
    trainer: IndividualParallelClientTrainer
    global_round: int

    def __post_init__(self):
        super()._post_init__()

    def main(self):
        while self.round < self.global_round:
            self.trainer.local_process()
            self.trainer.uplink_package

            self.round += 1


@dataclass
class CentralizedPipeline(BasePipeline):
    handler: CentralizedServerHandler

    def __post_init__(self):
        super()._post_init__()

    def main(self):
        while self.handler.if_stop is False:
            self.handler.global_update(buffer=[])

            server_loss, server_top1_acc, server_top5_acc = self.handler.evaluate()
            self.writer.add_scalar("Loss/Server", server_loss, self.round)
            self.writer.add_scalar("Top1Accuracy/Server", server_top1_acc, self.round)
            self.writer.add_scalar("Top5Accuracy/Server", server_top5_acc, self.round)
            logging.info(
                f"Round {self.round:>3}, Loss {server_loss:.4f}, "
                f"Test Accuracy {server_top1_acc:.4f}, "
            )

            self.round += 1

import signal
import threading
from multiprocessing.pool import ApplyResult
from typing import Protocol, TypeVar

import torch.multiprocessing as mp
from tqdm import tqdm

from core.utils import process_tensors_in_object, reconstruct_from_shared_memory

UplinkPackage = TypeVar("UplinkPackage")
DownlinkPackage = TypeVar("DownlinkPackage", contravariant=True)
ClientConfig = TypeVar("ClientConfig")


class BaseClientTrainer(Protocol[UplinkPackage, DownlinkPackage]):
    """
    Abstract base class for serial client training in federated learning.

    This class defines the interface for training clients in a serial manner,
    where each client is processed one after the other.

    Raises:
        NotImplementedError: If the methods are not implemented in a subclass.
    """

    def uplink_package(self) -> list[UplinkPackage]:
        """
        Prepare the data package to be sent from the client to the server.

        Returns:
            list[UplinkPackage]: A list of data packages prepared for uplink
            transmission.
        """
        ...

    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        """
        Process the downlink payload from the server for a list of client IDs.

        Args:
            payload (DownlinkPackage): The data package received from the server.
            cid_list (list[int]): A list of client IDs to process.

        Returns:
            None
        """
        ...


class ProcessPoolClientTrainer(
    BaseClientTrainer[UplinkPackage, DownlinkPackage],
    Protocol[UplinkPackage, DownlinkPackage, ClientConfig],
):
    """
    Abstract base class for parallel client training in federated learning.

    This class extends SerialClientTrainer to enable parallel processing of clients,
    allowing multiple clients to be trained concurrently.

    Attributes:
        num_parallels (int): Number of parallel processes to use for client training.
        device (str): The primary device to use for computation (e.g., "cpu", "cuda").
        device_count (int): The number of available CUDA devices, if `device` is "cuda".
        cache (list[UplinkPackage]): Cache to store uplink packages from clients.

    Raises:
        NotImplementedError: If the abstract methods are not implemented in a subclass.
    """

    num_parallels: int
    device: str
    device_count: int
    cache: list[UplinkPackage]
    stop_event: threading.Event

    def get_client_config(self, cid: int) -> ClientConfig:
        """
        Retrieve the configuration for a given client ID.

        Args:
            cid (int): Client ID.

        Returns:
            ClientConfig: The configuration for the specified client.
        """
        ...

    def get_client_device(self, cid: int) -> str:
        """
        Retrieve the device to use for processing a given client.

        Args:
            cid (int): Client ID.

        Returns:
            str: The device to use for processing the client.
        """
        if self.device == "cuda":
            return f"cuda:{cid % self.device_count}"
        return self.device

    def prepare_uplink_package_buffer(self) -> UplinkPackage:
        raise NotImplementedError

    @staticmethod
    def worker(
        config: ClientConfig,
        payload: DownlinkPackage,
        device: str,
        stop_event: threading.Event,
        *,
        shm_buffer: UplinkPackage | None = None,
    ) -> UplinkPackage:
        """
        Process a single client's training task.

        This method is executed by each worker process in the pool.
        It handles loading client configuration and payload, performing
        the client-specific operations, and returning the result.

        Args:
            config (ClientConfig):
                The client's configuration data.
            payload (DownlinkPackage):
                The downlink payload from the server.
            device (str): Device to use for processing (e.g., "cpu", "cuda:0").
            stop_event (threading.Event): Event to signal stopping the worker.
            shm_buffer (UplinkPackage | None):
                Optional shared memory buffer for the uplink package.

        Returns:
            UplinkPackage:
                The uplink package containing the client's results.
        """
        ...

    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        """
        Manage the parallel processing of clients.

        This method distributes the processing of multiple clients across
        parallel processes, handling data saving, loading, and caching.

        Args:
            payload (DownlinkPackage): The data package received from the server.
            cid_list (list[int]): A list of client IDs to process.

        Returns:
            None
        """
        process_tensors_in_object(payload, mode="move")

        shm_buffers = {}
        for cid in cid_list:
            buffer = self.prepare_uplink_package_buffer()
            process_tensors_in_object(buffer, mode="move")
            shm_buffers[cid] = buffer

        self.stop_event.clear()
        pool = mp.Pool(
            processes=self.num_parallels,
            initializer=signal.signal,
            initargs=(signal.SIGINT, signal.SIG_IGN),
        )
        try:
            jobs: list[ApplyResult] = []
            for cid in cid_list:
                config = self.get_client_config(cid)
                device = self.get_client_device(cid)
                jobs.append(
                    pool.apply_async(
                        self.worker,
                        (
                            config,
                            payload,
                            device,
                            self.stop_event,
                        ),
                        kwds={
                            "shm_buffer": shm_buffers.get(cid),
                        },
                    )
                )

            for i, job in enumerate(tqdm(jobs, desc="Client", leave=False)):
                result = job.get()
                cid = cid_list[i]

                package = reconstruct_from_shared_memory(result, shm_buffers[cid])
                self.cache.append(package)
        finally:
            self.stop_event.set()
            pool.close()
            pool.join()

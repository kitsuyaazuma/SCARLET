from typing import Any, Protocol

from torch.utils.data import DataLoader, Dataset


class DatasetProvider(Protocol):
    @property
    def num_classes(self) -> int: ...

    @property
    def public_train_size(self) -> int: ...

    def get_dataset(self, type_: Any, cid: int | None) -> Dataset: ...

    def get_dataloader(
        self,
        type_: Any,
        cid: int | None,
        batch_size: int | None = None,
        generator: Any | None = None,
    ) -> DataLoader: ...

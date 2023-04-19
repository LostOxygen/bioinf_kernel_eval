"""library module for dataset implementations and helper functions"""

from typing import Any, Iterator, Tuple, Union
from itertools import cycle
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import IterableDataset

class StreamingDataset(IterableDataset[Any]):
    """
    Dataset class implementation which streams random image data from a given path
    """

    def __init__(self, data_path: str, device: str = "cpu"):
        super().__init__()
        self.data_path = data_path
        self.device = device
        self.data = np.load(data_path, mmap_mode="r")
        self.num_samples = len(self.data)

    def __len__(self) -> Union[int, None]:
        return self.num_samples

    def __iter__(self) -> Iterator[Tensor]:
        return cycle(self.load_data())

    def load_data(self) -> Tuple[Tensor, Tensor]:
        """loads a random image from the data path with its according label"""
        for i in range(self.num_samples):
            yield torch.tensor(self.data[i][0]), torch.tensor(self.data[i][1])

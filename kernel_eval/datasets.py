"""library module for dataset implementations and helper functions"""
from typing import Any, Iterator, Union
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import IterableDataset


def create_1gb_random_array():
    # Create a 1GB array of random values
    array_size = 1024**3 // 4  # Each element is a 32-bit float, so 4 bytes
    arr = np.random.rand(array_size).astype(np.float32)

    # Save the array to disk as a binary file
    np.save("1gb_array.npy", arr)


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
        """loads a random image from the data path with its according label"""
        for i in range(self.num_samples):
            yield torch.tensor(self.data[i])


"""library module for dataset implementations and helper functions"""
from typing import Any, List
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class StreamingDataset(Dataset[Any]):
    """
    Dataset class implementation which streams random image data from a given path
    """

    def __init__(self, data_paths: List[str], transform: transforms=None):
        super().__init__()
        # data_paths need to be a list of paths to the actual numpy data arrays
        self.data_paths = data_paths
        self.data = np.load(data_paths, mmap_mode="r")
        self.num_samples = len(self.data)
        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tensor:
        # load and convert the numpy data to a tensor
        data_item = torch.tensor(torch.from_numpy(self.data[idx])).to(self.device)
        label = torch.tensor(torch.from_numpy(self.data[idx])).to(self.device)
        return data_item, label

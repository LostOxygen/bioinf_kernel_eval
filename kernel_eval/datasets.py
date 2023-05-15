"""library module for dataset implementations and helper functions"""
import os
from typing import Any, List
from glob import glob
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import webdataset as wds
from tqdm import tqdm

import tarfile
import io
import random
import torch
from torch.utils.data import Dataset, DataLoader

def process_data(data_paths: List[str], data_out: str) -> None:
    """
    Helper function to load the data from the given paths with their corresponding labels and
    saves them as a webdataset tar file containing pytorch tensors.

    Parameters:
        data_paths: List[str] - list of paths to the data files
        data_out: str - path to save the processed data

    Returns: 
        None
    """
    if not os.path.isdir(data_out):
        os.mkdir(data_out)

    # to keep track of the total file count for unique keys in the dataset
    train_key_counter = 0
    test_key_counter = 0

    # count the total number of files for the progress bar
    total_files = 0
    for data_path in data_paths:
        temp_file_list = glob(data_path+"data*.npy")
        total_files += len(temp_file_list)

    # create a progress bar with tqdm
    pbar = tqdm(total=total_files)

    pos = 0
    neg = 0

    # creates two tar files for train and test data, where 80% of the data is used for training
    with (
        wds.TarWriter(data_out+"train_data.tar") as train_sink,
        wds.TarWriter(data_out+"test_data.tar") as test_sink,
    ):
        for data_path in data_paths:
            file_list = glob(data_path+"data*.npy")
            labels = np.load(data_path+"label.npy")
            pos +=np.count_nonzero(labels >= 0)
            neg +=np.count_nonzero(labels < 0)

            for idx, file in enumerate(file_list):
                curr_data = np.load(file)
                # pad the data to 400x400, so every image has the same resulting size
                curr_data = np.pad(curr_data, ((0, 400-curr_data.shape[0]),
                                               (0, 400-curr_data.shape[1]),
                                               (0, 0)), mode="constant", constant_values=0)
                # move the channel dimension to the first dimension für PyTorch
                curr_data = np.moveaxis(curr_data, -1, 0)

                curr_label = torch.Tensor([labels[idx]])

                if idx < len(file_list)*0.8:
                    train_sink.write({
                        "__key__": f"sample{train_key_counter:06d}",
                        "data.pyd": torch.from_numpy(curr_data).float(),
                        "label.pyd": curr_label.long(),
                    })
                    train_key_counter += 1
                else:
                    test_sink.write({
                        "__key__": f"sample{test_key_counter:06d}",
                        "data.pyd": torch.from_numpy(curr_data).float(),
                        "label.pyd": curr_label.long(),
                    })
                    test_key_counter += 1
                pbar.update(1)
    pbar.close()
    print("pos", pos)
    print("neg", neg)


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
        data_item = torch.tensor(torch.from_numpy(self.data[idx]))
        label = torch.tensor(torch.from_numpy(self.data[idx]))
        return data_item, label


class CustomDataset(Dataset):
    def __init__(self, root_dirs):
        self.data_files = []
        self.labels = []
        label_files = []

        for root_dir in root_dirs:
            result = glob(root_dir+ '*.npy')
            files = [file for file in result if 'label.npy' not in file]
            self.data_files.extend(files)

            # all label.npy file paths
            tmp = root_dir + 'label.npy'
            label_files.append(tmp)

        # concatenate all in labels
        all_labels = []
        for file in label_files:
                all_labels.append(np.load(file))
        self.labels = np.concatenate(all_labels)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        label = self.labels[idx]
        sample = {'data': torch.from_numpy(data), 'label': torch.from_numpy(label)}
        return sample

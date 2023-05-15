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

    # creates two tar files for train and test data, where 80% of the data is used for training
    with (
        wds.TarWriter(data_out+"train_data.tar") as train_sink,
        wds.TarWriter(data_out+"test_data.tar") as test_sink,
    ):
        for data_path in data_paths:
            file_list = glob(data_path+"data*.npy")
            labels = np.load(data_path+"label.npy")

            for idx, file in enumerate(file_list):
                curr_data = np.load(file)
                # move the channel dimension to the first dimension for PyTorch
                curr_data = np.moveaxis(curr_data, -1, 0)
                curr_label = torch.tensor(labels[idx])

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


class SingleFileLoader(Dataset[Any]):
    """
    Dataset class implementation which loads a single image at a time
    """

    def __init__(self, data_paths: List[str], is_train: bool, transform: transforms=None):
        super().__init__()
        # data_paths need to be a list of paths to the actual numpy data arrays
        self.data_paths = data_paths
        self.num_samples = 0
        self.data = []

        for data_path in data_paths:
            file_list = glob(data_path+"data*.npy")
            labels = np.load(data_path+"label.npy")
            data_in_current_folder = []

            for idx, file in enumerate(file_list):
                curr_file_path = file
                curr_label = torch.tensor(labels[idx])

                entry = []
                entry.append(curr_file_path)
                entry.append(curr_label)

                data_in_current_folder.append(entry)

            #seperate for each folder and only take what you need here (depending on test or not)
            num_samples_in_folder = len(data_in_current_folder)

            #add to toal num_of_samples for __len__() function
            self.num_samples += num_samples_in_folder

            #split data in 80/20
            # Calculate the split index based on the ratio
            split_index = int(num_samples_in_folder * 0.8)
            
            if is_train:
                self.data.append(data_in_current_folder[:split_index])
            else:
                self.data.append(data_in_current_folder[split_index:])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tensor:
        # load and convert the numpy data to a tensor
        data_entry_with_label = self.data[idx]
        data_np = data_entry_with_label[0]
        label_tensor = data_entry_with_label[1]

        data_tensor = torch.from_numpy(np.moveaxis(data_np, -1, 0)).float()
        
        return data_tensor, label_tensor

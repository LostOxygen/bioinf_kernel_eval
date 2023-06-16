"""library module for dataset implementations and helper functions"""
import os
import random
from typing import Any, List
from glob import glob
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import webdataset as wds
from tqdm import tqdm
from enum import Enum

from .utils import augment_images, normalize_spectral_data


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


class SingleFileDatasetLoadingOptions(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class SingleFileDataset(Dataset[Any]):
    """
    Dataset class implementation which loads a single image at a time
    """

    def __init__(self, data_paths: List[str], loading_option: SingleFileDatasetLoadingOptions,
                 transform: transforms=None, augment: bool = False, normalize: bool = False):
        super().__init__()
        # data_paths need to be a list of paths to the actual numpy data arrays
        self.data_paths: List[str] = data_paths
        self.num_samples: int = 0
        self.data: List = []
        self.loading_option: SingleFileDatasetLoadingOptions = loading_option
        self.augment: bool = augment
        self.normalize: bool = normalize
        self.transform: torch.transforms = transform

        train_ratio = 0.8
        test_val_ratio = 0.1

        for data_path in data_paths:
            file_list = glob(data_path+"data*.npy")
            # sort the files according to their name index
            file_list = sorted(file_list, key=lambda x: int(x.split("data")[1].split(".npy")[0]))
            # load the corresponding labels
            labels = np.load(data_path+"label.npy")
            data_in_current_folder = []

            # build tuples for each file_path and it's corresponding label
            for idx, file in enumerate(file_list):
                curr_label = torch.tensor(labels[idx]).long()
                # tuple of (file_path: str, label: torch.Tensor)
                data_in_current_folder.append((file, curr_label))

            random.shuffle(data_in_current_folder)

            # take the number of the files in the current folder, since for train/test-splitting
            # we only want to take certain amount of files from each folder
            num_samples_in_folder = len(data_in_current_folder)

            num_train = int(train_ratio * num_samples_in_folder)
            num_tv = int(test_val_ratio * num_samples_in_folder)

            if self.loading_option == SingleFileDatasetLoadingOptions.TRAIN:
                training_data_current_folder = data_in_current_folder[:num_train]
                # add to total num_of_samples for __len__() function
                self.num_samples += len(training_data_current_folder)
                # append the trainingset tuples to the data list
                for data_entry in training_data_current_folder:
                    self.data.append(data_entry)

            elif self.loading_option == SingleFileDatasetLoadingOptions.VALIDATION:
                validation_data_current_folder = data_in_current_folder[num_train:num_train+num_tv]
                # add to total num_of_samples for __len__() function
                self.num_samples += len(validation_data_current_folder)
                # append the testset tuples to the data list
                for data_entry in validation_data_current_folder:
                    self.data.append(data_entry)

            elif self.loading_option == SingleFileDatasetLoadingOptions.TEST:
                test_data_current_folder = data_in_current_folder[num_train + num_tv:]
                # add to total num_of_samples for __len__() function
                self.num_samples += len(test_data_current_folder)
                # append the testset tuples to the data list
                for data_entry in test_data_current_folder:
                    self.data.append(data_entry)


    def __len__(self) -> int:
        return self.num_samples


    def __getitem__(self, idx: int) -> Tensor:
        # load the image from the corresponding file_path and it's corresponding label
        data_np = np.load(self.data[idx][0])
        label_tensor = self.data[idx][1]

        # move the channel dimension to the first dimension for PyTorch  (H,W,C) -> (C,H,W)
        data_np = np.moveaxis(data_np, -1, 0)

        # convert to tensor
        data_t = torch.from_numpy(data_np).float()

        if self.normalize:
            # normalize the data
            data_t = normalize_spectral_data(data_t)

        if self.augment:
            # apply augmentation to the images
            data_t = augment_images(data_t, size=224)

        return data_t, label_tensor

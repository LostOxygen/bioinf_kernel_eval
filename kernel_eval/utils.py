"""utility library for various functions"""
import os
import random
from enum import Enum
from glob import glob
from datetime import datetime
from typing import List, Tuple, Optional, Union
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt


class VisualizeSign(Enum):
    POSITIVE = 1
    ABSOLUTE = 2
    NEGATIVE = 3
    ALL = 4


def save_model(model_path: str, model_name: str, depthwise: bool,
               batch_size: int, lr: float, epochs: int, model: nn.Module) -> None:
    """
    Helper function to save the model under the specified model_path
    e.g. -> /models/vgg16_depthwise
    
    Parameters:
    model_path: str - path to save the model
    model_name: str - name of the model
    depthwise: bool - flag to indicate if the model is depthwise separable
    batch_size: int - batch size used for training
    lr: float - learning rate used for training
    epochs: int - number of epochs used for training
    model: nn.Module - model to save

    Returns:
        None
    """
    print("[ Saving Model ]")
    state = {
        "model": model.state_dict()
    }
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # create the correct model name string
    model_name += f"_{batch_size}bs_{lr}lr_{epochs}ep"
    model_name += f"{'_depthwise' if depthwise else ''}"

    torch.save(state, model_path+model_name)


def load_model(model_path: str, model_name: str, depthwise: bool,
               batch_size: int, lr: float, epochs: int,
               model: nn.Module) -> nn.Module:
    """
    Helper function to load the model from the specified model_path
    e.g. -> /models/vgg16_depthwise
    
    Parameters:
    model_path: str - path to load the model from
    model_name: str - name of the model
    depthwise: bool - flag to indicate if the model is depthwise separable
    batch_size: int - batch size used for training
    lr: float - learning rate used for training
    epochs: int - number of epochs used for training
    model: nn.Module - model to load the weights into

    Returns:
        model: nn.Module - the model with the loaded weights and parameters
    """
    # create the correct model name string
    model_name += f"_{batch_size}bs_{lr}lr_{epochs}ep"
    model_name += f"{'_depthwise' if depthwise else ''}"

    model_file = model_path+model_name
    if os.path.isfile(model_file):
        model_state = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_state["model"], strict=True)

    return model


def create_1gb_random_array() -> None:
    """helper function to generate a random numpy array for testing"""
    # Create a 1GB array of random values
    array_size = 1024**3 // 4  # Each element is a 32-bit float, so 4 bytes
    arr = np.random.rand(array_size).astype(np.float32)

    # Save the array to disk as a binary file
    np.save("1gb_array.npy", arr)


def log_metrics(train_acc: float, test_acc: torch.Tensor, model_name: str,
                f1_score: float, precision: float, recall: float) -> None:
    """
    Logs score and loss for a model over epochs and saves the log under ./logs/model_name.log
    Parameters:
        scores: torch.Tensor with the current scoreof a given model
        loss: torch.Tensor with the current loss of a given model
        epoch: current epoch
        model_name: the name of the model
        f1_score: the f1 score of the model
        precision: the precision of the model
        recall: the recall of the model
    Returns:
        None
    """
    if not os.path.exists("logs/"):
        os.mkdir("logs/")

    try:
        with open(f"./logs/{model_name}.log", encoding="utf-8", mode="a") as log_file:
            log_file.write(f"{datetime.now().strftime('%A, %d. %B %Y %I:%M%p')}" \
                           f" - train_acc: {train_acc} - test_acc: {test_acc}" \
                           f" - f1_score: {f1_score} - precision: {precision}" \
                           f" - recall: {recall}\n")
    except OSError as error:
        print(f"Could not write logs into /logs/{model_name}.log - error: {error}")


def plot_metrics(train_acc: List[float], train_loss: List[float], model_name: str) -> None:
    """
    Creates plots of the metrics using matplotlib and saves them under ./plots/model_name.png
    Parameters:
        train_acc: list of training accuracies
        train_loss: list of training losses
        model_name: the name of the model
    Returns:
        None
    """
    if not os.path.exists("plots/"):
        os.mkdir("plots/")

    try:
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        fig.suptitle(f"Training Metrics - {model_name}")
        ax[0].plot(train_loss)
        ax[0].set_title("Training Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[1].plot(train_acc)
        ax[1].set_title("Training Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")

        plt.savefig(f"./plots/{model_name}.png")
        plt.close()

    except OSError as error:
        print(f"Could not write plots into /plots/{model_name}.png - error: {error}")


def count_files(data_paths: List[str], train_test_split: float) -> Tuple[int, int]:
    """
    Helper function to count all files in a list of paths and return the number of train and 
    test files for a give train/test percentage split
    Parameters:
        path: list of paths to count the files in
        train_test_split: percentage of files to use for training
    Returns:
        train_files: number of train files
        test_files: number of test files
    """
    total_files = 0
    for data_path in data_paths:
        temp_file_list = glob(data_path+"data*.npy")
        total_files += len(temp_file_list)

    return int(total_files*train_test_split), int(total_files*(1-train_test_split))


def augment_images(img: torch.Tensor, size: int) -> torch.Tensor:
    """
    Apply augmentions to a given image in the following order: RandomResizedCrop
    Parameters:
        img [CxHxW]: torch.Tensor - image to apply augmentations to
        size: int - pixel size of the resulting image
    
    Returns:
        img: torch.Tensor - augmented image
    """
    if len(img.shape) >= 4:
        # when the images are already stacked to a batch, there is no need to
        # apply augmentations anymore
        return img

    channels, height, width = img.shape
    target_h, target_w = size, size
    cropped_resized_image = torch.zeros((channels, size, size))

    i = random.randint(0, height - target_h)
    j = random.randint(0, width - target_w)

    for channel in range(channels):
        cropped_resized_image[channel] = img[channel, i:i+target_h, j:j+target_w]

    return cropped_resized_image


def normalize_spectral_data(img: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the image spectral data wrt. the amidi-band spectral peak (359)
    Parameters:
        img [CxHxW]: torch.Tensor - image to apply normalization over all channels to
    
    Returns:
        img: torch.Tensor - normalized image
    """
    # normalization by peak
    spectral_mean = torch.mean(img, (1, 2))

    amidi_range = 10
    est_peak = 359
    # look for the peak in the amidi band area (359 +- 10)
    amidi_peak_interval = spectral_mean[(est_peak - amidi_range): (est_peak + amidi_range)]
    # take the actual peak value
    peak_point = torch.max(amidi_peak_interval)
    img /= peak_point

    # substract mean along each channel
    mean = torch.mean(img, dim=(1, 2), keepdim=True)
    img -= mean

    return img


def normalize_attribute(
    attr: np.ndarray,
    sign: str,
    outlier_perc: Union[int, float] = 2,
    reduction_axis: Optional[int] = None,
):
    attr_combined = attr
    if reduction_axis is not None:
        attr_combined = np.sum(attr, axis=reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign[sign] == VisualizeSign.ALL:
        threshold = cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.POSITIVE:
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = cumulative_sum_threshold(
            attr_combined, 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.NEGATIVE:
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign[sign] == VisualizeSign.ABSOLUTE:
        attr_combined = np.abs(attr_combined)
        threshold = cumulative_sum_threshold(
            attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return normalize_scale(attr_combined, threshold)


def normalize_scale(attr: np.ndarray, scale_factor: float):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)


def cumulative_sum_threshold(values: np.ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile in range(0, 101), (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]

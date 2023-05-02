"""utility library for various functions"""
import os
import torch
from torch import nn
import numpy as np
from datetime import datetime


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

# Metric train_acc, test_acc, date
# date, model_name, train_acc, test_acc

def log_metrics(train_acc: float, test_acc: torch.Tensor, model_name: str) -> None:
    """
    Logs score and loss for a model over epochs and saves the log under ./logs/model_name.log
    Parameters:
        scores: torch.Tensor with the current scoreof a given model
        loss: torch.Tensor with the current loss of a given model
        epoch: current epoch
        model_name: the name of the model
    Returns:
        None
    """
    if not os.path.exists("logs/"):
        os.mkdir("logs/")

    try:
        with open(f"./logs/{model_name}.log", encoding="utf-8", mode="a") as log_file:
            log_file.write(f"{datetime.now().strftime('%A, %d. %B %Y %I:%M%p')}" f" - train_acc: {train_acc} - test_acc: {test_acc}\n")
    except OSError as error:
        print(f"Could not write logs into /logs/{model_name}.log - error: {error}")
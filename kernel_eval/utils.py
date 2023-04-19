"""utility library for various functions"""
import os
import torch
from torch import nn

def save_model(model_path: str, model_name: str,
               depthwise: bool, model: nn.Module) -> None:
    """
    Helper function to save the model unter the specified model_path
    e.g. -> /models/vgg16_depthwise
    
    Parameters:
    model_path: str - path to save the model
    model_name: str - name of the model
    depthwise: bool - flag to indicate if the model is depthwise separable
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

    torch.save(state, model_path+f"{model_name}{'_depthwise' if depthwise else ''}")

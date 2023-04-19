"""main hook to start both the training, testing and evaluation procedures"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import time
import socket
import datetime
import argparse
import os
from typing import Final
import torch
import torchsummary
import torchvision
import torchvision.transforms as transforms

from kernel_eval.models import resnet34
from kernel_eval.models import vgg11, vgg13, vgg16, vgg19
from kernel_eval.datasets import StreamingDataset
from kernel_eval.train import train_model
from kernel_eval.utils import save_model

DATA_PATH: Final[str] = "/prodi/hpcmem/dave/spots_ftir"
MODEL_OUTPUT_PATH: Final[str] = "./models/"


def main(gpu: int, batch_size: int, epochs: int, model: str, depthwise: bool) -> None:
    """
    Main function to start the training, testing and evaluation procedures
        
    Parameters:
        gpu: int - specifies which gpu to use. If None, cpu is used
        batch_size: int - specifies the training batch size
        epochs: int - specifies the number of training epochs
    
    Returns:
        None
    """
    # set up devices and print system information
    start = time.perf_counter() # start timer
    if gpu == -1 or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = f"cuda:{gpu}"

    print("\n\n\n"+"#"*60)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {torch.get_num_threads()} CPU cores with "
          f"{os.cpu_count()} threads and "
          f"{torch.cuda.device_count()} GPUs on {socket.gethostname()}")
    print(f"## Using: {device}")
    print(f"## Batch Size: {batch_size}")
    print(f"## Epochs: {epochs}")
    print(f"## Model: {model}")
    print(f"## Depthwise: {depthwise}")
    print("#"*60)
    print()


    # ---------------- Create/Load Datasets ----------------
    #data = StreamingDataset(DATA_PATH, device=device)

    #CIFAR10 Test
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    cifar10_data = torchvision.datasets.CIFAR10(DATA_PATH, download=True, transform=transform)

    data = torch.utils.data.DataLoader(cifar10_data,
                                            batch_size=4,
                                            shuffle=True,
                                            num_workers=2)

    in_channels = 3 # TODO: extact input channels from data
    (width, height) = (32, 32) # TODO: extract width and height from data

    # ---------------- Load and Train Models ---------------
    match model:
        case "vgg11": model = vgg11(in_channels=in_channels, depthwise=depthwise)
        case "vgg13": model = vgg13(in_channels=in_channels, depthwise=depthwise)
        case "vgg16": model = vgg16(in_channels=in_channels, depthwise=depthwise)
        case "vgg19": model = vgg19(in_channels=in_channels, depthwise=depthwise)
        case "resnet34": model = resnet34(in_channels=in_channels, depthwise=depthwise)
        case _: raise ValueError(f"Model {model} not supported")

    torchsummary.summary(model, (in_channels, width, height))

    # -------- Test Models and evaluate kernels ------------
    model = train_model(model, data, epochs, device)
    save_model(model, MODEL_OUTPUT_PATH, model, depthwise)

    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="sets the train device var", type=int, default=-1)
    parser.add_argument("--batch_size", "-bs", help="specifies batch size", type=int, default=128)
    parser.add_argument("--epochs", "-e", help="specifies the train epochs", type=int, default=100)
    parser.add_argument("--model", "-m", help="specifies the model arch", type=str, default="vgg11")
    parser.add_argument("--depthwise", "-d", help="enables depthwise conv",
                        action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))

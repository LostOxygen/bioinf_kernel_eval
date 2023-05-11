"""main hook to start both the training, testing and evaluation procedures"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import time
import socket
import datetime
import argparse
import os
from typing import Final, List
import torch
import torchsummary
import torchvision

import torchvision.transforms as transforms

from kernel_eval.models import vgg11, vgg13, vgg16, vgg19, resnet34, SmolNet
from kernel_eval.train import train_model, test_model


# DATA_PATHS: Final[List[str]] = ["/prodi/hpcmem/spots_ftir/LC704/",
#                                "/prodi/hpcmem/spots_ftir/BC051111/",
#                                "/prodi/hpcmem/spots_ftir/CO1002b/",
#                                "/prodi/hpcmem/spots_ftir/CO1004/",
#                                "/prodi/hpcmem/spots_ftir/CO1801a/",
#                                "/prodi/hpcmem/spots_ftir/CO722/",
#                                "/prodi/hpcmem/spots_ftir/LC704/"]

DATA_PATHS: Final[List[str]] = [""]

# DATA_OUT: Final[str] = "/prodi/hpcmem/spots_ftir/data_out/"
DATA_OUT: Final[str] = "/data_out/"

MODEL_OUTPUT_PATH: Final[str] = "./models/"


def main(gpu: int, batch_size: int, epochs: int, model_type: str,
         depthwise: bool, learning_rate: float) -> None:
    """
    Main function to start the training, testing and evaluation procedures
        
    Parameters:
        gpu: int - specifies which gpu to use. If None, cpu is used
        batch_size: int - specifies the training batch size
        epochs: int - specifies the number of training epochs
        model_type: str - specifies the model architecture
        depthwise: bool - enables depthwise convolutions
        eval: bool - evaluates the model without training
    
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
    print(f"## Learning Rate: {learning_rate}")
    print(f"## Epochs: {epochs}")
    print(f"## Model: {model_type}")
    print(f"## Depthwise: {depthwise}")
    print("#"*60)
    print()

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    # load a single image to get the input shape
    # train data has the shape (batch_size, channels, width, height) -> (BATCH_SIZE, 442, 400, 400)
    print("[ creating model ]")
    images, labels  = next(iter(trainloader))
    print("label shape ",labels.shape)
    print("labels",labels)
    exit()

    in_channels = images.shape[1]  # should be 442
    (width, height) = (images.shape[2], images.shape[3])  # should be 400x400


    # ---------------- Load and Train Models ---------------
    match model_type:
        case "smol": model = SmolNet(in_channels=in_channels, depthwise=depthwise,
                                     num_classes=10, is_cifar=True)
        case "vgg11": model = vgg11(in_channels=in_channels, depthwise=depthwise,
                                    num_classes=10,is_cifar=True)
        case "vgg13": model = vgg13(in_channels=in_channels, depthwise=depthwise,
                                    num_classes=10, is_cifar=True)
        case "vgg16": model = vgg16(in_channels=in_channels, depthwise=depthwise,
                                    num_classes=10, is_cifar=True)
        case "vgg19": model = vgg19(in_channels=in_channels, depthwise=depthwise,
                                    num_classes=10, is_cifar=True)
        case "resnet34": model = resnet34(in_channels=in_channels, depthwise=depthwise,
                                          num_classes=10, is_cifar=True)
        case _: raise ValueError(f"Model {model} not supported")


    torchsummary.summary(model, (in_channels, width, height), device="cpu")
    model = model.to(device)

    print("[ train model ]")
    model, train_accuracy = train_model(model, trainloader, learning_rate, epochs, device, model_type)


    # -------- Test Models and Evaluate Kernels ------------
    print("[ evaluate model ]")
    test_accuracy = test_model(model, testloader, batch_size, device)

    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")

    print("\n\n\n"+"#"*40)
    print(f"## Model: {model_type}")
    print(f"## Train-Accuracy: {train_accuracy}")
    print(f"## Test-Accuracy: {test_accuracy}")
    print(f"## Computation time: {duration:0.4f} hours")
    print("#"*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="sets the train device var", type=int, default=0)
    parser.add_argument("--batch_size", "-bs", help="specifies batch size", type=int, default=32)
    parser.add_argument("--epochs", "-e", help="specifies the train epochs", type=int, default=100)
    parser.add_argument("--learning_rate", "-lr", help="specifies the learning rate",
                        type=float, default=0.001)
    parser.add_argument("--model_type", "-m", help="specifies the model architecture",
                        type=str, default="vgg11")
    parser.add_argument("--depthwise", "-d", help="enables depthwise conv",
                        action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))


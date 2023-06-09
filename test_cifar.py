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
from torchvision import transforms

from kernel_eval.models import vgg11, vgg13, vgg16, vgg19, resnet34, SmolNet
from kernel_eval.train_cifar import train_model, test_model
from kernel_eval.utils import plot_metrics

DATA_OUT: Final[str] = "./data/"

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


    # ---------------- Create/Load Datasets ----------------

    print("[ loading training data ]")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=DATA_OUT, train=True,
                                        download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATA_OUT, train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    # load a single image to get the input shape
    # train data has the shape (batch_size, channels, width, height) -> (BATCH_SIZE, 442, 400, 400)
    print("[ creating model ]")
    tmp_data, _ = next(iter(train_loader))
    in_channels = tmp_data.shape[1]  # should be 442
    (width, height) = (tmp_data.shape[2], tmp_data.shape[3])  # should be 400x400


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
    model, train_accuracy, train_accs, train_losses = train_model(model, train_loader,
                                                                  learning_rate, epochs, device)

    model_name = model_type + f"CIFAR_{batch_size}bs_{learning_rate}lr_{epochs}ep"
    model_name += f"{'_depthwise' if depthwise else ''}"

    plot_metrics(train_acc=train_accs, train_loss=train_losses, model_name=model_name)


    # -------- Test Models and Evaluate Kernels ------------
    print("[ evaluate model ]")
    test_accuracy = test_model(model, test_loader, device)

    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.

    print("\n\n\n"+"#"*40)
    print(f"## Model: {model_type}")
    print(f"## Train-Accuracy: {train_accuracy}")
    print(f"## Test-Accuracy: {test_accuracy}")
    print(f"## Computation time: {duration:0.4f} hours")
    print("#"*40)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="sets the train device var", type=int, default=0)
    parser.add_argument("--batch_size", "-bs", help="specifies batch size", type=int, default=128)
    parser.add_argument("--epochs", "-e", help="specifies the train epochs", type=int, default=100)
    parser.add_argument("--learning_rate", "-lr", help="specifies the learning rate",
                        type=float, default=0.001)
    parser.add_argument("--model_type", "-m", help="specifies the model architecture",
                        type=str, default="vgg11")
    parser.add_argument("--depthwise", "-d", help="enables depthwise conv",
                        action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))

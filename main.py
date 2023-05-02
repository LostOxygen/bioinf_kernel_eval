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
import webdataset as wds

from kernel_eval.models import vgg11, vgg13, vgg16, vgg19, resnet34, SmolNet
from kernel_eval.datasets import process_data
from kernel_eval.train import train_model, test_model
from kernel_eval.utils import save_model, load_model

DATA_PATHS: Final[List[str]] = ["/prodi/hpcmem/spots_ftir/LC704/",
                               "/prodi/hpcmem/spots_ftir/BC051111/",
                               "/prodi/hpcmem/spots_ftir/CO1002b/",
                               "/prodi/hpcmem/spots_ftir/CO1004/",
                               "/prodi/hpcmem/spots_ftir/CO1801a/",
                               "/prodi/hpcmem/spots_ftir/CO722/",
                               "/prodi/hpcmem/spots_ftir/LC704/"]

DATA_OUT: Final[str] = "/prodi/hpcmem/spots_ftir/data_out/"

MODEL_OUTPUT_PATH: Final[str] = "./models/"


def main(gpu: int, batch_size: int, epochs: int, model_type: str,
         depthwise: bool, eval_only: bool, learning_rate: float) -> None:
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


    # ---------------- Create/Load Datasets ----------------
    if not os.path.isfile(DATA_OUT+"train_data.tar") \
        or not os.path.isfile(DATA_OUT+"test_data.tar"):
        print("[ saving train/test data and labels ]")
        process_data(DATA_PATHS, DATA_OUT)

    print("[ loading training data ]")
    train_data = wds.WebDataset(
        DATA_OUT+"train_data.tar").shuffle(100).decode().to_tuple("data.pyd", "label.pyd")

    train_loader = wds.WebLoader((train_data.batched(batch_size)), batch_size=None, num_workers=2)

    # load a single image to get the input shape
    # train data has the shape (batch_size, channels, width, height) -> (BATCH_SIZE, 442, 400, 400)
    print("[ creating model ]")
    tmp_data, _ = next(iter(train_loader))
    in_channels = tmp_data.shape[1]  # should be 442
    (width, height) = (tmp_data.shape[2], tmp_data.shape[3])  # should be 400x400


    # ---------------- Load and Train Models ---------------
    match model_type:
        case "smol": model = SmolNet(in_channels=in_channels, depthwise=depthwise, num_classes=2)
        case "vgg11": model = vgg11(in_channels=in_channels, depthwise=depthwise, num_classes=2)
        case "vgg13": model = vgg13(in_channels=in_channels, depthwise=depthwise, num_classes=2)
        case "vgg16": model = vgg16(in_channels=in_channels, depthwise=depthwise, num_classes=2)
        case "vgg19": model = vgg19(in_channels=in_channels, depthwise=depthwise, num_classes=2)
        case "resnet34": model = resnet34(in_channels=in_channels,
                                          depthwise=depthwise, num_classes=2)
        case _: raise ValueError(f"Model {model} not supported")

    torchsummary.summary(model, (in_channels, width, height), device="cpu")
    model = model.to(device)

    if not eval_only:
        print("[ train model ]")
        model = train_model(model, train_loader, learning_rate, epochs, batch_size, device)
        save_model(MODEL_OUTPUT_PATH, model_type, depthwise,
                   batch_size, learning_rate, epochs, model)

    del train_loader


    # -------- Test Models and Evaluate Kernels ------------
    test_data = wds.WebDataset(
        DATA_OUT+"test_data.tar").shuffle(100).decode().to_tuple("data.pyd", "label.pyd")
    test_loader = wds.WebLoader((test_data.batched(batch_size)), batch_size=None, num_workers=1)

    if eval_only:
        model = load_model(MODEL_OUTPUT_PATH, model_type, depthwise,
                           batch_size, learning_rate, epochs, model)

    print("[ evaluate model ]")
    test_model(model, test_loader, batch_size, device)

    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")



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
    parser.add_argument("--eval_only", "-ev", help="evaluates the model without training",
                        action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))

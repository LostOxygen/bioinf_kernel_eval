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

from torch.utils.data import DataLoader

from kernel_eval.models import vgg11, vgg13, vgg16, vgg19, resnet34, SmolNet
from kernel_eval.datasets import SingleFileLoader
from kernel_eval.train import train_model, test_model
from kernel_eval.utils import (save_model, load_model, log_metrics,
                               count_files, plot_metrics, augment_images)

DATA_PATHS: Final[List[str]] = ["/prodi/hpcmem/spots_ftir/LC704/",
                               "/prodi/hpcmem/spots_ftir/BC051111/",
                               "/prodi/hpcmem/spots_ftir/CO1002b/",
                               "/prodi/hpcmem/spots_ftir/CO1004/",
                               "/prodi/hpcmem/spots_ftir/CO1801a/",
                               "/prodi/hpcmem/spots_ftir/CO722/"]

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

    # count the train and test files for train/test split of 80%/20%
    trainset_length, testset_length = count_files(DATA_PATHS, 0.8)

    print("[ loading training data ]")

    #train_data = wds.WebDataset(DATA_OUT+"train_data.tar").shuffle(1000).decode()
    #train_data = train_data.to_tuple("data.pyd", "label.pyd")

    train_data = SingleFileLoader(data_paths=DATA_PATHS, is_train=True)

    #train_loader = wds.WebLoader((train_data.batched(batch_size)), batch_size=None, num_workers=2)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    
    #train_loader.length = trainset_length // batch_size    

    # load a single image to get the input shape
    # train data has the shape (batch_size, channels, width, height) -> (BATCH_SIZE, 442, 400, 400)
    print("[ creating model ]")
    tmp_data, _ = next(iter(train_loader))
    tmp_data = augment_images(tmp_data, size=224)
    in_channels = tmp_data.shape[1]  # should be 442
    (height, width) = (tmp_data.shape[2], tmp_data.shape[3])  # should be 224x224 or whatever


    # ---------------- Load and Train Models ---------------
    match model_type:
        case "smol": model = SmolNet(in_channels=in_channels, depthwise=depthwise, num_classes=1)
        case "vgg11": model = vgg11(in_channels=in_channels, depthwise=depthwise, num_classes=1)
        case "vgg13": model = vgg13(in_channels=in_channels, depthwise=depthwise, num_classes=1)
        case "vgg16": model = vgg16(in_channels=in_channels, depthwise=depthwise, num_classes=1)
        case "vgg19": model = vgg19(in_channels=in_channels, depthwise=depthwise, num_classes=1)

        case "resnet34": model = resnet34(in_channels=in_channels,
                                          depthwise=depthwise, num_classes=1)
        case _: raise ValueError(f"Model {model} not supported")

    torchsummary.summary(model, (in_channels, height, width), device="cpu")
    model = model.to(device)

    model_name = model_type + f"_{batch_size}bs_{learning_rate}lr_{epochs}ep"
    model_name += f"{'_depthwise' if depthwise else ''}"

    if not eval_only:
        print("[ train model ]")
        model, train_accuracy, train_accs, train_losses = train_model(model, train_loader,
                                                                      learning_rate, epochs, device)
        save_model(MODEL_OUTPUT_PATH, model_type, depthwise,
                   batch_size, learning_rate, epochs, model)

    del train_loader


    # -------- Test Models and Evaluate Kernels ------------
    #test_data = wds.WebDataset(DATA_OUT+"test_data.tar").shuffle(1000).decode()
    #test_data = test_data.to_tuple("data.pyd", "label.pyd")
    #test_loader = wds.WebLoader((test_data.batched(batch_size)), batch_size=None, num_workers=1)
    #test_loader.length = testset_length // batch_size

    
    test_data = SingleFileLoader(data_paths=DATA_PATHS, is_train=False)

    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=2)
    

    if eval_only:
        train_accuracy = 0.0
        model = load_model(MODEL_OUTPUT_PATH, model_type, depthwise,
                           batch_size, learning_rate, epochs, model)

    print("[ evaluate model ]")
    test_accuracy = test_model(model, test_loader, device)


    log_metrics(train_acc=train_accuracy, test_acc=test_accuracy, model_name=model_name)
    plot_metrics(train_acc=train_accs, train_loss=train_losses, model_name=model_name)

    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="sets the train device var", type=int, default=0)
    parser.add_argument("--batch_size", "-bs", help="specifies batch size", type=int, default=4)
    parser.add_argument("--epochs", "-e", help="specifies the train epochs", type=int, default=100)
    parser.add_argument("--learning_rate", "-lr", help="specifies the learning rate",
                        type=float, default=0.1)
    parser.add_argument("--model_type", "-m", help="specifies the model architecture",
                        type=str, default="vgg11")
    parser.add_argument("--depthwise", "-d", help="enables depthwise conv",
                        action="store_true", default=False)
    parser.add_argument("--eval_only", "-ev", help="evaluates the model without training",
                        action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))

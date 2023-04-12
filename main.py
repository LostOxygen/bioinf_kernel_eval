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

DATA_PATH: Final[str] = "./data/"
MODEL_OUTPUT_PATH: Final[str] = "./models/"


def main(gpu: int, batch_size: int, epochs: int) -> None:
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
    print("#"*60)
    print()


    # ---------------- Create/Load Datasets ----------------
    # TODO: Create/Load Datasets

    # ---------------- Load and Train Models ---------------
    # TODO: Load and Train Models

    # -------- Test Models and evaluate kernels ------------
    # TODO: Test Models and evaluate kernels

    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="sets the train device var", type=int, default=-1)
    parser.add_argument("--batch_size", "-bs", help="specifies batch size", type=int, default=128)
    parser.add_argument("--epochs", "-e", help="specifies the train epochs", type=int, default=100)
    args = parser.parse_args()
    main(**vars(args))

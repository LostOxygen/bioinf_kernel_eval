"""main hook to start both the training, testing and evaluation procedures"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import socket
import datetime
import os
from glob import glob
from typing import Final, List

import torch
from torch.utils.data import DataLoader
from captum.attr import LRP
from captum.attr import visualization as viz
import numpy as np
from matplotlib import pyplot as plt

from kernel_eval.models import vgg11, vgg13, vgg16, vgg19, resnet34
from kernel_eval.datasets import SingleFileDataset
from kernel_eval.datasets import SingleFileDatasetLoadingOptions
from kernel_eval.utils import load_model, augment_images, normalize_spectral_data

DATA_PATHS: Final[List[str]] = ["/prodi/hpcmem/spots_ftir/LC704/",
                                "/prodi/hpcmem/spots_ftir/BC051111/",
                                "/prodi/hpcmem/spots_ftir/CO1002b/",
                                "/prodi/hpcmem/spots_ftir/CO1004/",
                                "/prodi/hpcmem/spots_ftir/CO1801a/",
                                "/prodi/hpcmem/spots_ftir/CO722/"]

DATA_OUT: Final[str] = "/prodi/hpcmem/spots_ftir/data_out/"

MODEL_OUTPUT_PATH: Final[str] = "./models/"


def main() -> None:
    """
    Load all pre-trained models under {MODEL_OUTPUT_PATH} and evaluate them on the test set to 
    visualize the Layer Relevance Propagation (LRP).

    Parameters:
        None

    Returns:
        None
    """
    device = "cpu"

    print("\n\n\n"+"#"*60)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {torch.get_num_threads()} CPU cores with "
          f"{os.cpu_count()} threads and "
          f"{torch.cuda.device_count()} GPUs on {socket.gethostname()}")
    print(f"## Using: {device}")
    print("#"*60)
    print()

    # ---------------- Create/Load Datasets ----------------
    print("[ loading data ]")
    test_data = SingleFileDataset(data_paths=DATA_PATHS,
                                  loading_option=SingleFileDatasetLoadingOptions.TEST,
                                  augment=True,
                                  normalize=True)

    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=2)

    # load a single image to get the input shape
    # train data has the shape (batch_size, channels, width, height) -> (BATCH_SIZE, 442, 400, 400)
    print("[ creating model ]")
    tmp_data, _ = next(iter(test_loader))
    in_channels = tmp_data.shape[1]

    # ---------------- Load and Train Models ---------------
    model_files = glob(MODEL_OUTPUT_PATH+"*") # load all models saved under the path

    for model_file in model_files:
        model_file = model_file.split("/")[-1]
        depthwise = "depthwise" in model_file
        model_type = model_file.split("_")[0]
        batch_size = int(model_file.split("_")[1][:-2])
        learning_rate = float(model_file.split("_")[2][:-2])
        epochs = int(model_file.split("_")[3][:-2])

        match model_type:
            case "vgg11": model = vgg11(in_channels=in_channels, depthwise=depthwise, num_classes=1)
            case "vgg13": model = vgg13(in_channels=in_channels, depthwise=depthwise, num_classes=1)
            case "vgg16": model = vgg16(in_channels=in_channels, depthwise=depthwise, num_classes=1)
            case "vgg19": model = vgg19(in_channels=in_channels, depthwise=depthwise, num_classes=1)

            case "resnet34": model = resnet34(in_channels=in_channels,
                                            depthwise=depthwise, num_classes=1)
            case _: raise ValueError(f"Model {model} not supported")

        model = model.to(device)

        model = load_model(MODEL_OUTPUT_PATH, model_type, depthwise,
                           batch_size, learning_rate, epochs, model)
        model.eval()

        # ---------------- Evaluate Models ----------------
        lrp_model = LRP(model)
        img, label = next(iter(test_loader))
        attribution = lrp_model.attribute(img, target=label).cpu().detach().numpy()
        # move the channel dimension to the last dimension for numpy (C,H,W) -> (H,W,C)
        img = np.moveaxis(img.cpu().detach().numpy(), 1, -1)
        attribution = np.moveaxis(attribution, -1, 0)

        vis_types = ["heat_map", "original_image"]
        vis_signs = ["all", "all"] # "positive", "negative", or "all" to show both
        # positive attribution indicates that the presence of the area increases the pred. score
        # negative attribution indicates distractor areas whose absence increases the pred. score

        _ = viz.visualize_image_attr_multiple(attribution, img, vis_types, vis_signs,
                                              ["Attribution", "Image"], show_colorbar = True)
        plt.savefig(f"./plots/{model_file}_lrp.png")
        plt.close()


if __name__ == "__main__":
    main()

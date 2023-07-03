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
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from kernel_eval.models import vgg11, vgg13, vgg16, vgg19, resnet34
from kernel_eval.datasets import SingleFileDataset
from kernel_eval.datasets import SingleFileDatasetLoadingOptions
from kernel_eval.utils import load_model, normalize_attribute

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
    if not os.path.exists("plots/"):
        os.mkdir("plots/")

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

    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=1)

    # load a single image to get the input shape
    # train data has the shape (batch_size, channels, width, height) -> (BATCH_SIZE, 442, 400, 400)
    print("[ creating model ]")
    tmp_data, _ = next(iter(test_loader))
    in_channels = tmp_data.shape[1]

    # ---------------- Load and Train Models ---------------
    model_files: List[str] = glob(MODEL_OUTPUT_PATH+"*") # load all models saved under the path
    # filter deothwsie models out since they will be compared separately
    model_files = list(filter(lambda key: "depthwise" not in key, model_files))

    for model_file in model_files:
        # obtain the model type and hyperparameters from the model file name
        model_file = model_file.split("/")[-1]
        model_type = model_file.split("_")[0]
        batch_size = int(model_file.split("_")[1][:-2])
        learning_rate = float(model_file.split("_")[2][:-2])
        epochs = int(model_file.split("_")[3][:-2])

        # create the normal model
        match model_type:
            case "vgg11": model_normal = vgg11(in_channels=in_channels,
                                               depthwise=False, num_classes=1)
            case "vgg13": model_normal = vgg13(in_channels=in_channels,
                                               depthwise=False, num_classes=1)
            case "vgg16": model_normal = vgg16(in_channels=in_channels,
                                               depthwise=False, num_classes=1)
            case "vgg19": model_normal = vgg19(in_channels=in_channels,
                                               depthwise=False, num_classes=1)
            case "resnet34": model_normal = resnet34(in_channels=in_channels,
                                            depthwise=False, num_classes=1)
            case _: raise ValueError(f"Model {model_type} not supported")

        # create the depthwise counterpart
        match model_type:
            case "vgg11": model_depth = vgg11(in_channels=in_channels,
                                               depthwise=True, num_classes=1)
            case "vgg13": model_depth = vgg13(in_channels=in_channels,
                                              depthwise=True, num_classes=1)
            case "vgg16": model_depth = vgg16(in_channels=in_channels,
                                              depthwise=True, num_classes=1)
            case "vgg19": model_depth = vgg19(in_channels=in_channels,
                                              depthwise=True, num_classes=1)
            case "resnet34": model_depth = resnet34(in_channels=in_channels,
                                                    depthwise=True, num_classes=1)
            case _: raise ValueError(f"Model {model_type} not supported")

        # load the models
        model_normal = model_normal.to(device)
        model_normal = load_model(MODEL_OUTPUT_PATH, model_type, False,
                                  batch_size, learning_rate, epochs, model_normal)
        model_normal.eval()
        model_normal.zero_grad()


        model_depth = model_depth.to(device)
        model_depth = load_model(MODEL_OUTPUT_PATH, model_type, True,
                                 batch_size, learning_rate, epochs, model_depth)
        model_depth.eval()
        model_depth.zero_grad()

        # ---------------- Evaluate Models ----------------
        lrp_model_normal = LRP(model_normal)
        lrp_model_depth = LRP(model_depth)
        img, _ = next(iter(test_loader))
        img.requires_grad = True

        print(f"[ analyze model: {model_file} ]")
        attribution_normal = lrp_model_normal.attribute(img, target=None).cpu().detach().numpy()[0]
        attribution_depth = lrp_model_depth.attribute(img, target=None).cpu().detach().numpy()[0]

        img = img.cpu().detach().numpy()[0]
        img = np.mean(img, axis=0)

        # add the attributions of every channel dimension up
        attribution_normal = np.sum(attribution_normal, axis=0)
        attribution_normal = np.expand_dims(attribution_normal, axis=0)
        attribution_depth = np.sum(attribution_depth, axis=0)
        attribution_depth = np.expand_dims(attribution_depth, axis=0)

        # move the channel dimension to the last dimension for numpy (C,H,W) -> (H,W,C)
        # shape is (442, 244, 244) -> (244, 244, 442)
        attribution_normal = np.moveaxis(attribution_normal, 0, -1)
        attribution_depth = np.moveaxis(attribution_depth, 0, -1)
        attribution_normal = normalize_attribute(attribution_normal, "ALL", 2, 2)
        attribution_depth = normalize_attribute(attribution_depth, "ALL", 2, 2)

        cmap = LinearSegmentedColormap.from_list(
            "RdWhGn", ["red", "white", "green"]
        )

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        #axis_separator = make_axes_locatable(axes[2])
        #colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     orientation="vertical", ax=axes,
                     label="Attribution Values")

        axes[0].grid()
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].grid()
        axes[1].imshow(attribution_normal, cmap=cmap, vmin=-1, vmax=1)
        axes[1].set_title("Attribution Normal")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[2].grid()
        axes[2].imshow(attribution_depth, cmap=cmap, vmin=-1, vmax=1)
        axes[2].set_title("Attribution Depthwise")
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        fig.savefig(f"./plots/{model_file}_lrp.png")
        plt.close()

    print("[ finished LRP analysis ]")

if __name__ == "__main__":
    main()

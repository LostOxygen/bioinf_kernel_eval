"""main hook to start both the training, testing and evaluation procedures"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import socket
import datetime
import os
import argparse
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

MODEL_OUTPUT_PATH: Final[str] = "./models/"


def main(has_cancer: bool) -> None:
    """
    Load all pre-trained models under {MODEL_OUTPUT_PATH} and evaluate them on the test set to 
    visualize the Layer Relevance Propagation (LRP).

    Parameters:
        has_cancer: bool - type of cancer example to visualize (positive or negative)

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
    print(f"## Visualizing LRP for cancer={has_cancer}")
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
    print("[ creating models ]")
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
        img: torch.Tensor = None
        # search for an example of the given type
        for _, (curr_img, curr_label) in enumerate(test_loader):
            if bool(curr_label) == has_cancer:
                img = curr_img
                break

        assert img is not None, f"Could not find an example with cancer={has_cancer}"
        img.requires_grad = True

        print(f"[ analyze model: {model_file} ]")
        attribution_normal = lrp_model_normal.attribute(img, target=None).cpu().detach().numpy()[0]
        attribution_depth = lrp_model_depth.attribute(img, target=None).cpu().detach().numpy()[0]

        img = img.cpu().detach().numpy()[0]
        input_spectral = np.copy(img)
        img = np.mean(img, axis=0)

        # specral data
        input_spectral = input_spectral.mean(axis=(1, 2))
        attribution_spectral_normal = attribution_normal.mean(axis=(1, 2))
        attribution_spectral_depth = attribution_depth.mean(axis=(1, 2))

        # normalize the spectra data
        peak_interval = input_spectral[339:379]
        peak_point = np.max(peak_interval)
        input_spectral = input_spectral / peak_point

        peak_interval = attribution_spectral_normal[339:379]
        peak_point = np.max(peak_interval)
        attribution_spectral_normal = attribution_spectral_normal / peak_point

        peak_interval = attribution_spectral_depth[339:379]
        peak_point = np.max(peak_interval)
        attribution_spectral_depth = attribution_spectral_depth / peak_point

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
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        mpl.rcParams.update({"font.size": 20})
        mpl.rcParams.update({"axes.titlesize": 20})
        mpl.rcParams.update({"axes.labelsize": 15})

        # create and save the normal LRP analysis
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle(f"LRP Analysis - Model: {model_type} - Cancer: {has_cancer}")
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     orientation="vertical", ax=axes,
                     label="Attribution Values")
        plt.rc("font", size=10)          # controls default text sizes
        axes[0][0].grid()
        axes[0][0].imshow(img)
        axes[0][0].set_title("Input Image")
        axes[0][0].set_yticks([])
        axes[0][0].set_xticks([])
        axes[0][1].grid()
        axes[0][1].imshow(attribution_normal, cmap=cmap, vmin=-1, vmax=1)
        axes[0][1].set_title("Attribution Normal")
        axes[0][1].set_xticks([])
        axes[0][1].set_yticks([])
        axes[0][2].grid()
        axes[0][2].imshow(attribution_depth, cmap=cmap, vmin=-1, vmax=1)
        axes[0][2].set_title("Attribution Depthwise")
        axes[0][2].set_xticks([])
        axes[0][2].set_yticks([])

        axes[1][0].grid()
        axes[1][0].plot(input_spectral)
        axes[1][0].set_title("Spectral Input")
        axes[1][1].grid()
        axes[1][1].plot(attribution_spectral_normal)
        axes[1][1].set_title("Spectral Attribution Normal")
        axes[1][2].grid()
        axes[1][2].plot(attribution_spectral_depth)
        axes[1][2].set_title("Spectral Attribution Depthwise")

        plt.savefig(f"./plots/{model_file}_lrp_C_{has_cancer}.png")
        plt.close()

    print("[ finished LRP analysis ]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--has_cancer", "-hc", type=bool, default=False,
                        help="choose between negative/positive cancer examples",
                        action="store_true")
    args = parser.parse_args()
    main(**vars(args))

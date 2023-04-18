"""library module for different neural network architectures and helper classes"""

from typing import cast, Dict, List, Union
import torch
from torch import nn, Tensor


class DepthwiseSeparableConvolution(nn.Module):
    """
    Depthwise separable convolution layer.
    """

    def __init__(self, n_in, kernels_per_layer, n_out):
        super().__init__()
        self.depthwise = nn.Conv2d(n_in, n_in * kernels_per_layer, kernel_size=3,
                                   padding=1, groups=n_in)
        self.pointwise = nn.Conv2d(
            n_in * kernels_per_layer, n_out, kernel_size=1)

    def forward(self, x_val):
        x_val = self.depthwise(x_val)
        x_val = self.pointwise(x_val)
        return x_val


__all__ = [
    "VGG", "vgg11", "vgg13", "vgg16", "vgg19"
]

vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
              512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256,
              "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(vgg_cfg: List[Union[str, int]], batch_norm: bool = False,
                 depthwise: bool = False) -> nn.Sequential:
    """
    Function to create the layers of the VGG neural network architecture.
    
    Parameters:
        vgg_cfg: list of integers and strings defining the architecture
        batch_norm: boolean indicating whether to use batch normalization
        depthwise: boolean indicating whether to use depthwise separable convolution

    Returns:
        layers: nn.Sequential object containing the composed layers
    """
    layers = nn.Sequential()
    in_channels = 3
    for v in vgg_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            if depthwise:
                conv2d = DepthwiseSeparableConvolution(in_channels, 1, v)
            else:
                conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = v

    return layers


class VGG(nn.Module):
    """
    Implementation of VGG Neural Network Architecture.
    Code adapted from: https://github.com/Lornatang/VGG-PyTorch/blob/main/model.py
    """

    def __init__(self, vgg_cfg: List[Union[str, int]], batch_norm: bool = False,
                 num_classes: int = 1000, depthwise: bool = False) -> None:
        super().__init__()
        self.features = _make_layers(vgg_cfg, batch_norm, depthwise)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


def vgg11(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg11"], False, **kwargs)

    return model


def vgg13(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg13"], False, **kwargs)

    return model


def vgg16(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg16"], False, **kwargs)

    return model


def vgg19(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg19"], False, **kwargs)

    return model

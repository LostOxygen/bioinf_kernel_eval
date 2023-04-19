"""library module for different neural network architectures and helper classes"""

from typing import cast, Dict, List, Union
import torch
from torch import nn, Tensor


class DepthwiseSeparableConvolution(nn.Module):
    """
    Depthwise separable convolution layer.
    """

    def __init__(self, n_in: int, kernels_per_layer: int, n_out: int, kernel_size: int = 3):
        super().__init__()
        # TODO: In some cases the kernel_size, stride and padding has to be changed in the original 
        # unchanged networks (in restnet) should this be also changed in the depthwise seperable 
        # convulution
        self.depthwise = nn.Conv2d(n_in, n_in * kernels_per_layer, kernel_size=kernel_size,
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


class VGG(nn.Module):
    """
    Implementation of VGG Neural Network Architecture.
    Code adapted from: https://github.com/Lornatang/VGG-PyTorch/blob/main/model.py
    """

    def __init__(self, vgg_cfg: List[Union[str, int]], batch_norm: bool = False,
                 num_classes: int = 1000, depthwise: bool = False, in_channels: int = 3) -> None:
        super().__init__()
        self.features = _make_layers(vgg_cfg, batch_norm, depthwise, in_channels)

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

    def forward(self, x_val: Tensor) -> Tensor:
        out = self.features(x_val)
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

    def _make_layers(vgg_cfg: List[Union[str, int]], batch_norm: bool = False,
                 depthwise: bool = False, in_channels: int = 3) -> nn.Sequential:
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
        for v in vgg_cfg:
            if v == "M":
                layers.append(nn.MaxPool2d((2, 2), (2, 2)))
            else:
                v = cast(int, v)
                if depthwise:
                    # initially in_channels is set to a specific value and will be overwritten by
                    # v using the dictionary values for the desired channels for VGG
                    conv2d = DepthwiseSeparableConvolution(n_in = in_channels, 
                                                           kernels_per_layer = 1, n_out = v,
                                                           kernel_size=3)
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

class BasicBlock(nn.Module):
    """
    Implementation of the basic building block for ResNet
    Code adapted from: https://github.com/jarvislabsai/blog/blob/master/build_resnet34_pytorch/Building%20Resnet%20in%20PyTorch.ipynb
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, depthwise: bool = False):
        super().__init__()
        if depthwise:
            self.conv1 = DepthwiseSeparableConvolution(n_in=inplanes, kernels_per_layer=1, 
                                                       n_out=planes, kernel_size=3)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                        padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if depthwise:
            self.conv2 = DepthwiseSeparableConvolution(n_in=planes, kernels_per_layer=1,
                                                       n_out=planes, kernel_size=3)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                        padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    Implementation of ResNet Neural Network Architecture.
    Code adapted from: https://github.com/jarvislabsai/blog/blob/master/build_resnet34_pytorch/Building%20Resnet%20in%20PyTorch.ipynb
    """

    def __init__(self, block: BasicBlock, layers: list[int], num_classes: int = 1000, 
                 depthwise: bool = False, in_channels: int = 3):
        super().__init__()
        
        self.inplanes = 64

        if(depthwise):
            self.conv1 = DepthwiseSeparableConvolution(n_in = in_channels, kernels_per_layer = 1,
                                                       n_out = self.inplanes, kernel_size = 7)
        else:
            self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layers(block, 64, layers[0], depthwise = depthwise)
        self.layer2 = self._make_layers(block, 128, layers[1], stride=2, depthwise = depthwise)
        self.layer3 = self._make_layers(block, 256, layers[2], stride=2, depthwise = depthwise)
        self.layer4 = self._make_layers(block, 512, layers[3], stride=2, depthwise = depthwise)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 , num_classes)


    def _make_layers(self, block: BasicBlock, planes: int, blocks: int, stride: int = 1,
                     depthwise: bool = False) -> nn.Sequential:
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, depthwise))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x_val: Tensor) -> Tensor:
        out = self.conv1(x_val)           # 224x224
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)         # 112x112

        out = self.layer1(out)          # 56x56
        out = self.layer2(out)          # 28x28
        out = self.layer3(out)          # 14x14
        out = self.layer4(out)          # 7x7

        out = self.avgpool(out)         # 1x1
        out = torch.flatten(out, 1)     # remove 1 X 1 grid and make vector of tensor shape 
        out = self.fc(out)

        return out

def resnet34(**kwargs) -> ResNet:
    layers=[3, 4, 6, 3]
    model = ResNet(BasicBlock(), layers, **kwargs)
    return model
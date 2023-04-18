"""library module for different neural network architectures and helper classes"""
import torch
from torch import nn
from torch.nn import functional as F

class DepthwiseSeparableConvolution(nn.Module):
    """
    Depthwise separable convolution layer.
    """
    def __init__(self, n_in, kernels_per_layer, n_out):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise = nn.Conv2d(n_in, n_in * kernels_per_layer, kernel_size=3,
                                   padding=1, groups=n_in)
        self.pointwise = nn.Conv2d(n_in * kernels_per_layer, n_out, kernel_size=1)

    def forward(self, x_val):
        x_val = self.depthwise(x_val)
        x_val = self.pointwise(x_val)
        return x_val

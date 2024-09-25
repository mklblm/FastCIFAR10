### Fast CIFAR10 Architecture 
# Based on implementations by:
# David C. Page: https://github.com/davidcpage/cifar10-fast
# Thomas Germer: https://github.com/99991/cifar10-fast-simple

import torch
from torch import nn
import torch.nn.functional as F


## Ghost BatchNorm Implementation by Thomas Germer
# Designed by Hoffer, Hubara, Soudry: https://arxiv.org/abs/1705.08741

class GhostBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)

        running_mean = torch.zeros(num_features * num_splits)
        running_var = torch.ones(num_features * num_splits)

        self.weight.requires_grad = False
        self.num_splits = num_splits
        self.register_buffer("running_mean", running_mean)
        self.register_buffer("running_var", running_var)

    def train(self, mode=True):
        if (self.training is True) and (mode is False):
            # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(
                self.running_mean.view(self.num_splits, self.num_features), dim=0
            ).repeat(self.num_splits)
            self.running_var = torch.mean(
                self.running_var.view(self.num_splits, self.num_features), dim=0
            ).repeat(self.num_splits)
        return super().train(mode)

    def forward(self, input):
        n, c, h, w = input.shape
        if self.training or not self.track_running_stats:
            assert n % self.num_splits == 0, f"Batch size ({n}) must be divisible by num_splits ({self.num_splits}) of GhostBatchNorm"
            return F.batch_norm(
                input.view(-1, c * self.num_splits, h, w),
                self.running_mean,
                self.running_var,
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits),
                True,
                self.momentum,
                self.eps,
            ).view(n, c, h, w)
        else:
            return F.batch_norm(
                input,
                self.running_mean[: self.num_features],
                self.running_var[: self.num_features],
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )


## Fast CIFAR10 Resnet "Bag of Tricks" architecture by David C. Page
# http://web.archive.org/web/20201123223831/https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/

class FastCIFAR_BoT(nn.Module):
    def __init__(self, first_layer_weights: torch.Tensor, celu_alpha: float, 
                 input_shape: int, hidden_units: int, output_shape: int, output_scale: float):
        super().__init__()

        ## First layer contains patch whitened initialised weights and is frozen.
        conv1_out_shape = first_layer_weights.size(0)
        conv1 = nn.Conv2d(in_channels=input_shape, out_channels=conv1_out_shape, 
                          kernel_size=3, padding=1, bias=False)
        conv1.weight.data = first_layer_weights
        conv1.weight.requires_grad = False

        self.conv1 = conv1

        # From here, modified original arch:
        # Added GhostBatchNorm, add CELU, Move max pool2d to second layer.
        # Prep layer
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=conv1_out_shape, out_channels=hidden_units,
                      kernel_size=1, padding=0, bias=False),
            GhostBatchNorm(hidden_units, num_splits=16),
            nn.CELU(alpha=celu_alpha),
        )

        # Layer 1 with residual connections and 2 res sequences
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=(hidden_units*2),
                      kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GhostBatchNorm((hidden_units*2), num_splits=16),
            nn.CELU(alpha=celu_alpha),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=(hidden_units*2), out_channels=(hidden_units*2),
                      kernel_size=3, padding=1, bias=False),
            GhostBatchNorm((hidden_units*2), num_splits=16),
            nn.CELU(alpha=0.3),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=(hidden_units*2), out_channels=(hidden_units*2),
                      kernel_size=3, padding=1, bias=False),
            GhostBatchNorm((hidden_units*2), num_splits=16),
            nn.CELU(alpha=celu_alpha),
        )


        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=(hidden_units*2), out_channels=(hidden_units*4),
                      kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GhostBatchNorm(hidden_units*4, num_splits=16),
            nn.CELU(alpha=celu_alpha),
        )

        # Setup of Layer 3 etc is the same as layer 1 etc.
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=(hidden_units*4), out_channels=(hidden_units*8),
                      kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GhostBatchNorm(hidden_units*8, num_splits=16),
            nn.CELU(alpha=celu_alpha),
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(in_channels=(hidden_units*8), out_channels=(hidden_units*8),
                      kernel_size=3, padding=1, bias=False),
            GhostBatchNorm(hidden_units*8, num_splits=16),
            nn.CELU(alpha=celu_alpha),
        )
        self.res4 = nn.Sequential(
            nn.Conv2d(in_channels=(hidden_units*8), out_channels=(hidden_units*8),
                      kernel_size=3, padding=1, bias=False),
            GhostBatchNorm(hidden_units*8, num_splits=16),
            nn.CELU(alpha=celu_alpha),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(in_features=(hidden_units*8),out_features=output_shape, 
                      bias=False),
        )

        # Scale parameter for final layer
        self.output_scale = output_scale


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.prep(x)
        x = self.layer1(x)
        x = x + self.res2(self.res1(x))         # Residual connection
        x = self.layer2(x)
        x = self.layer3(x)
        x = x + self.res4(self.res3(x))         # Residual connection
        x = self.classifier(x)
        x = torch.mul(x, self.output_scale)

        return x
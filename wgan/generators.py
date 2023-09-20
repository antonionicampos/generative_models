import torch

from torch import nn


class DeepConvolutionalGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(DeepConvolutionalGenerator, self).__init__()

        self.linear = nn.Linear(latent_dim, 4 * 4 * 4 * 64)
        self.batchnorm = nn.BatchNorm1d(4 * 4 * 4 * 64)
        self.relu = nn.ReLU()

        self.upsampling1 = nn.ConvTranspose2d(
            4 * 64,
            128,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=False,
        )
        nn.init.kaiming_normal_(self.upsampling1.weight)
        self.batchnorm1 = nn.BatchNorm2d(128)

        self.upsampling2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=5,
            stride=2,
            padding=1,
            bias=False,
        )
        nn.init.kaiming_normal_(self.upsampling2.weight)
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.upsampling3 = nn.ConvTranspose2d(
            64,
            1,
            kernel_size=5,
            stride=2,
            padding=1,
            bias=False,
        )
        nn.init.kaiming_normal_(self.upsampling3.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.batchnorm(self.linear(x)))

        x = x.view(-1, 4 * 64, 4, 4)

        x = self.relu(self.batchnorm1(self.upsampling1(x)))
        x = self.relu(self.batchnorm2(self.upsampling2(x)))
        x = self.sigmoid(self.upsampling3(x))
        return x[..., 2:-1, 2:-1]

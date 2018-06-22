import torch
import torch.nn as nn
import torch.functional as F
from spectral_norm import spectral_norm


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        pass

    def forward(self, *input):
        pass


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.image_size = config.image_size
        self.z_dim = config.z_dim
        self.g_dim = config.g_dim
        self.kernel_size = config.kernel_size
        layers = []
        channel = self.g_dim * 8
        layers.append(nn.ConvTranspose2d(self.z_dim, channel, self.kernel_size, 2))
        layers.append(nn.BatchNorm2d(channel))
        layers.append(nn.ReLU(inplace=True))

        # UpSampling
        for i in config.num_layers:
            layers.append(nn.ConvTranspose2d(channel, channel / 2, self.kernel_size, 2))
            layers.append(nn.BatchNorm2d(channel / 2))
            layers.append(nn.ReLU(inplace=True))
            channel /= 2

        self.output = nn.Sequential(
            nn.ConvTranspose2d(self.g_dim, 3, self.kernel_size, 2),
            nn.Tanh()
        )
        self.main = nn.Sequential(*layers)

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        pass

    def forward(self, *input):
        pass

class ResidualBlockA(nn.Module):
    """Residual Block with Instance Normalization"""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlockA, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out)
        )

    def forward(self, x):
        return x + self.main(x)

class ResidualBlockB(nn.Module):
    """Residual Block with Spectral Norm + Instance Normalization"""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlockB, self).__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out),
            spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out)
        )

    def forward(self, x):
        return x + self.main(x)
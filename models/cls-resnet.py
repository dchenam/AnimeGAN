import torch
import torch.nn as nn
import torch.functional as F
from spectral_norm import spectral_norm


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        layers = []
        channel = config.g_dim
        self.encoder = nn.Sequential(
            spectral_norm(nn.Linear(config.label_dim, config.encoding_dim)),
            nn.ReLU()
        )
        """Generator using the ResNet Architecture"""
        layers.append(
            nn.ConvTranspose2d(config.z_dim + config.encoding_dim, channel, kernel_size=4, stride=1, padding=0,
                               bias=False))
        layers.append(nn.InstanceNorm2d(channel))
        layers.append(nn.ReLU())
        # Downsampling Layers
        for i in range(2):
            layers.append(nn.Conv2d(channel, channel * 2, kernel_size=4, stride=1, padding=0))
            layers.append(nn.InstanceNorm2d(channel * 2))
            layers.append(nn.ReLU())
            channel = channel * 2

        # Bottleneck Layers
        for i in range(self.config.bottleneck_layers):
            layers.append(ResidualBlock(channel, channel))

        # Upsampling Layers
        for i in range(2):
            layers.append(
                nn.ConvTranspose2d(channel, channel // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(channel // 2))
            layers.append(nn.ReLU())
            channel = channel // 2

        layers.append(
            nn.ConvTranspose2d(config.g_dim, 3, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Tanh())

        self.upsampler = nn.Sequential(*layers)

        # for m in self.modules():
        #     if isinstance(m, nn.ConvTranspose2d):
        #         m.weight.data.normal_(0.0, 0.02)

    def forward(self, noise, label):
        label = self.encoder(label)
        net = torch.cat([noise, label], dim=1)
        net = net.view(net.size(0), net.size(1), 1, 1)
        net = self.upsampler(net)
        return net  # 64 x 64 x 3


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        layers = []
        channel = config.d_dim
        self.encoder = nn.Sequential(
            nn.Linear(config.label_dim, self.config.encoding_dim),
            nn.ReLU()
        )

        layers.append(nn.Conv2d(3, channel, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(channel))
        layers.append(nn.ReLU())

        for i in range(config.d_layers):
            layers.append(
                nn.Conv2d(channel, channel * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(channel * 2))
            layers.append(nn.ReLU())
            channel = channel * 2
        self.downsampler = nn.Sequential(*layers)

        self.output = nn.Sequential(
            nn.Conv2d(channel + self.config.encoding_dim, channel, 1, 1, bias=False),
            nn.ReLU(), nn.Conv2d(channel, 1, 4, 1),
            nn.Sigmoid()
        )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(0.0, 0.02)

    def forward(self, image, label):
        label = self.encoder(label)
        net = self.downsampler(image)
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.repeat(1, 1, net.size(2), net.size(3))
        net = torch.cat([net, label], dim=1)
        net = self.output(net)
        return net.view(net.size(0), net.size(1))


class ResidualBlock(nn.Module):
    """Residual Block with Instance Normalization"""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out)
        )

    def forward(self, x):
        return x + self.main(x)

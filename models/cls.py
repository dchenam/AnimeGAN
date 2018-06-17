import torch
import torch.nn as nn
import torch.functional as F


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        layers = []
        channel = config.g_dim * 8
        self.encoder = nn.Sequential(
            nn.Linear(config.label_dim, config.encoding_dim),
            nn.ReLU()
        )
        """Generator using the DC-GAN Architecture"""
        layers.append(nn.ConvTranspose2d(config.z_dim + config.encoding_dim, channel, 4, 1, 0))
        layers.append(nn.BatchNorm2d(channel))
        layers.append(nn.ReLU())

        for i in range(config.g_layers):
            layers.append(nn.ConvTranspose2d(channel, channel // 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channel // 2))
            layers.append(nn.ReLU(inplace=True))
            channel = channel // 2

        layers.append(nn.ConvTranspose2d(config.g_dim, 3, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.upsampler = nn.Sequential(*layers)

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

        layers.append(nn.Conv2d(3, channel, 4, 2, 1))
        layers.append(nn.BatchNorm2d(channel))
        layers.append(nn.ReLU())

        for i in range(config.d_layers):
            layers.append(nn.Conv2d(channel, channel * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channel * 2))
            layers.append(nn.ReLU())
            channel = channel * 2
        self.downsampler = nn.Sequential(*layers)

        self.output = nn.Sequential(
            nn.Conv2d(channel + self.config.encoding_dim, channel, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel, 1, 4, 1),
            nn.Sigmoid()
        )

    def forward(self, image, label):
        label = self.encoder(label)
        net = self.downsampler(image)
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.repeat(1, 1, net.size(2), net.size(3))
        net = torch.cat([net, label], dim=1)
        net = self.output(net)
        return net.view(net.size(0), net.size(1))

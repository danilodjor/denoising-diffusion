import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_encoding import time_embedding


class SelfAttention(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.ln1 = nn.LayerNorm([num_channels])
        self.mha = nn.MultiheadAttention(
            embed_dim=num_channels, num_heads=4, batch_first=True
        )
        self.ln2 = nn.LayerNorm([num_channels])
        self.ffn = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels),
        )

    def forward(self, x):
        img_size = x.shape[2:]

        x = x.flatten(start_dim=2).permute(0, 2, 1)
        x_ln = self.ln1(x)
        x = x + self.mha(x_ln, x_ln, x_ln, need_weights=False)[0]
        x = x + self.ffn(self.ln2(x))

        return torch.unflatten(x.swapaxes(1, 2), 2, img_size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()

        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.layers(x))
        else:
            return self.layers(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=256):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.time_emb_layer = nn.Sequential(
            nn.SiLU(), nn.Linear(time_emb_dim, out_channels)
        )

    def forward(self, x, t):
        x = self.downsample(x)
        pe = self.time_emb_layer(t)

        return x + pe[:, :, None, None]


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.time_emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.conv(x1)

        pe = self.time_emb_layer(t)

        return x1 + pe[:, :, None, None]


class UNet(nn.Module):
    def __init__(
        self, num_in_channels=3, num_out_channels=3, time_emb_dim=256, device="cuda"
    ):
        super().__init__()

        self.device = device
        self.time_emb_dim = time_emb_dim

        # Encoder
        self.input_head = DoubleConv(num_in_channels, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)

        # Convolutions
        self.conv1 = DoubleConv(256, 512)
        self.conv2 = DoubleConv(512, 512)
        self.conv3 = DoubleConv(512, 256)

        # Decoder
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)

        # Output head
        self.prediction_head = nn.Conv2d(64, num_out_channels, kernel_size=1)

    def forward(self, x, t):
        t = time_embedding(t, self.time_emb_dim, self.device)

        x1 = self.input_head(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.conv1(x4)
        x4 = self.conv2(x4)
        x4 = self.conv3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        return self.prediction_head(x)

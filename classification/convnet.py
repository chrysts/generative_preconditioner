import torch.nn as nn
from modified_pytorchmodule import Conv2d_fw, Linear_fw, BatchNorm2d_fw

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv_block_fast(in_channels, out_channels):
    return nn.Sequential(
        Conv2d_fw(in_channels, out_channels, 3, padding=1),
        BatchNorm2d_fw(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet_MAML(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block_fast(x_dim, hid_dim),
            conv_block_fast(hid_dim, hid_dim),
            conv_block_fast(hid_dim, hid_dim),
            conv_block_fast(hid_dim, z_dim),
        )

        self.out_channels = 1600

    def forward(self, x):

        x = self.encoder(x)
        return x.view(x.size(0), -1)





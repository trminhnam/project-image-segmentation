import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.GELU()):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512],
        activation="gelu",
    ):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        activation_fn = self.get_activation(activation)
        # downpart of the U-Net
        for feature in features:
            self.downs.append(DoubleConvBlock(in_channels, feature, activation_fn))
            in_channels = feature

        # uppart of the U-Net
        for feature in features[::-1]:
            self.ups.extend(
                [
                    nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2),
                    DoubleConvBlock(feature * 2, feature, activation_fn),
                ]
            )

        self.bottleneck = DoubleConvBlock(features[-1], features[-1] * 2, activation_fn)
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.apply(init_weights)

    def forward(self, x):
        skip_connections = []

        # go down the U-Net and save skip connections
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # go through the bottleneck
        x = self.bottleneck(x)

        # reverse the skip connections
        skip_connections = skip_connections[::-1]

        # go up the U-Net
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # upsample
            skip_connection = skip_connections[idx // 2]

            # restore the size of the skip connection if the downsampled image is not divisible by 2
            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            x = torch.cat((skip_connection, x), dim=1)  # concatenate
            x = self.ups[idx + 1](x)  # double conv block

        return self.output_conv(x)

    def get_activation(self, activation: str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise NotImplementedError(f"Activation {activation} is not supported")


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


if __name__ == "__main__":
    x = torch.randn((3, 1, 158, 158))
    model = UNet(in_channels=1, out_channels=1, activation="gelu")
    preds = model(x)
    assert preds.shape == x.shape
    print("Success!")

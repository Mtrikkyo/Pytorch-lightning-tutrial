#!.venv/bin/python3


from turtle import forward
import torch
import torch.nn as nn


class ToyModel(nn.Module):
    """CNN-basedModel"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        num_class: int = 10,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            
        )

        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.residual_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifar = nn.Linear(in_features=out_channels, out_features=num_class)

    def forward(self, x):
        res = x

        x = self.first_conv(x)
        x = nn.ReLU()(x)
        x = self.dw_conv(x)
        x = nn.ReLU()(x)
        x = self.pw_conv(x)
        x = nn.ReLU()(x)

        res = self.residual_conv(res)
        res = nn.ReLU()(res)

        x += res
        x = self.gap(x)
        x = x.view(-1)
        x = self.classifar(x)

        return x


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms.v2 as v2
    from torchvision.datasets import MNIST
    from torchinfo import summary

    model = ToyModel(in_channels=1)
    # dataset = MNIST("data", train=True, transform=v2.ToTensor())
    # dataloader = DataLoader(dataset, batch_size=32)
    summary(model, (1, 28, 28))

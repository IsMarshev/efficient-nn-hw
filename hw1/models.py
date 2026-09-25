"""The FP32, sequential CNN used throughout Homework 1."""

import torch
from torch import nn


class SmallCNN(nn.Sequential):
    """A 100-class network for square RGB inputs whose side is a multiple of 16."""

    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, 64, 5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 100),
        )


def make_model() -> SmallCNN:
    return SmallCNN().float().eval()


if __name__ == "__main__":
    model = make_model()
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 100)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

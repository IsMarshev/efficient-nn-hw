"""Explicit CIFAR ResNet-18 reconstructions. See protocol.md for provenance."""
import torch
from torch import nn
from .initialization import zero_init_, identity_init_

METHODS = ('zero', 'kaiming', 'xavier', 'rezero', 'fixup', 'skipinit', 'constnet', 'constnet_star', 'zero_no_bn')

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, method):
        super().__init__()
        self.method = method
        bn = method in ('zero', 'kaiming', 'xavier', 'constnet', 'constnet_star')
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=method == 'skipinit')
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=method == 'skipinit')
        self.bn1 = nn.BatchNorm2d(cout) if bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(cout) if bn else nn.Identity()
        self.shortcut = nn.Identity()
        if cin != cout or stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                          nn.BatchNorm2d(cout) if bn else nn.Identity())
        self.relu = nn.ReLU(inplace=False)
        if method in ('rezero', 'skipinit'):
            self.alpha = nn.Parameter(torch.zeros(1))
        if method == 'skipinit':
            self.merge_bias = nn.Parameter(torch.zeros(1))
        if method in ('fixup', 'zero_no_bn'):
            self.bias1a = nn.Parameter(torch.zeros(1))
            self.bias1b = nn.Parameter(torch.zeros(1))
            self.bias2a = nn.Parameter(torch.zeros(1))
            self.bias2b = nn.Parameter(torch.zeros(1))
            self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if self.method in ('fixup', 'zero_no_bn'):
            out = self.conv1(x + self.bias1a)
            out = self.relu(out + self.bias1b)
            out = self.conv2(out + self.bias2a) * self.scale + self.bias2b
            return self.relu(out + self.shortcut(x + self.bias1a))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if hasattr(self, 'alpha'):
            out = self.alpha * out
        out = out + self.shortcut(x)
        if hasattr(self, 'merge_bias'):
            out = out + self.merge_bias
        return self.relu(out)

class ResNet18(nn.Module):
    def __init__(self, method='zero', num_classes=10):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f'Unknown method {method}')
        self.method = method
        bn = method in ('zero', 'kaiming', 'xavier', 'constnet', 'constnet_star')
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=method == 'skipinit')
        self.bn1 = nn.BatchNorm2d(64) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=False)
        cin = 64
        for i, cout in enumerate((64, 128, 256, 512), 1):
            group = [BasicBlock(cin, cout, 1 if i == 1 else 2, method), BasicBlock(cout, cout, 1, method)]
            setattr(self, f'layer{i}', nn.Sequential(*group))
            cin = cout
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        if method in ('fixup', 'zero_no_bn'):
            self.stem_bias = nn.Parameter(torch.zeros(1))
            self.fc_bias = nn.Parameter(torch.zeros(1))
        self.initialize()

    @torch.no_grad()
    def initialize(self):
        method = self.method
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if method in ('zero', 'zero_no_bn'):
                    zero_init_(layer.weight)
                elif method in ('constnet', 'constnet_star'):
                    layer.weight.zero_()
                    if isinstance(layer, nn.Conv2d):
                        layer.weight[:, :, layer.kernel_size[0]//2, layer.kernel_size[1]//2] = 1 / layer.in_channels
                elif method == 'xavier':
                    nn.init.xavier_normal_(layer.weight)
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in' if method == 'skipinit' or isinstance(layer, nn.Linear) else 'fan_out',
                                            nonlinearity='linear' if isinstance(layer, nn.Linear) else 'relu')
                if layer.bias is not None:
                    layer.bias.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.reset_running_stats()
                layer.weight.fill_(1)
                layer.bias.zero_()
        for layer in self.modules():
            if isinstance(layer, BasicBlock):
                if method in ('zero', 'zero_no_bn', 'fixup', 'constnet', 'constnet_star'):
                    layer.conv2.weight.zero_()
                if method == 'fixup':
                    layer.conv1.weight.mul_(8**-0.5)
        if method in ('fixup', 'constnet', 'constnet_star'):
            self.fc.weight.zero_()

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x + self.stem_bias if hasattr(self, 'stem_bias') else x)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x + self.fc_bias if hasattr(self, 'fc_bias') else x)


def create_model(method='zero', seed=0):
    # Model allocation/initialization cannot consume the data or process RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ResNet18(method)

@torch.no_grad()
def inspect_model(method='zero'):
    model = create_model(method).eval()
    shapes, hooks = [], []
    for name, module in model.named_modules():
        if name and not list(module.children()):
            hooks.append(module.register_forward_hook(lambda m, i, o, n=name: shapes.append(
                {'name': n, 'type': type(m).__name__, 'input': list(i[0].shape), 'output': list(o.shape)})))
    model(torch.zeros(1, 3, 32, 32))
    for hook in hooks:
        hook.remove()
    return {'method': method, 'parameters': sum(p.numel() for p in model.parameters()),
            'model': str(model), 'shapes': shapes}

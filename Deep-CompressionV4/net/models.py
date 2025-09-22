import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prune import PruningModule, MaskedLinear


class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = linear(120, 84)
        self.fc2 = linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 120)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class MaskedConv2d(nn.Conv2d):
    """Conv2d layer equipped with a pruning mask."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight, dtype=self.weight.dtype))

    def forward(self, x):
        weight = self.weight * self.mask
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def prune(self, threshold):
        with torch.no_grad():
            threshold_tensor = torch.as_tensor(threshold, dtype=self.weight.dtype, device=self.weight.device)
            pruned_mask = torch.where(self.weight.abs() < threshold_tensor, torch.zeros_like(self.mask), self.mask)
            self.mask.copy_(pruned_mask)
            self.weight.mul_(self.mask)

    def prune_with_scores(self, scores, threshold):
        with torch.no_grad():
            score_tensor = scores.to(self.weight.device, dtype=self.weight.dtype)
            threshold_tensor = torch.as_tensor(threshold, dtype=score_tensor.dtype, device=score_tensor.device)
            pruned_mask = torch.where(score_tensor < threshold_tensor, torch.zeros_like(self.mask), self.mask)
            self.mask.copy_(pruned_mask)
            self.weight.mul_(self.mask)


_VGG_CONFIGS = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

_VGG_ARCHITECTURES = {
    'vgg11': ('A', False),
    'vgg11_bn': ('A', True),
    'vgg13': ('B', False),
    'vgg13_bn': ('B', True),
    'vgg16': ('D', False),
    'vgg16_bn': ('D', True),
    'vgg19': ('E', False),
    'vgg19_bn': ('E', True),
}

SUPPORTED_VGG_ARCHS = tuple(_VGG_ARCHITECTURES.keys())


def _make_vgg_layers(cfg, batch_norm=False, mask=False):
    layers = []
    in_channels = 3
    conv_cls = MaskedConv2d if mask else nn.Conv2d

    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = conv_cls(in_channels, v, kernel_size=3, padding=1, bias=True)
            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


class CifarVGG(PruningModule):
    """CIFAR-10 VGG backbone mirroring the standalone training project."""

    def __init__(self, features, num_classes=10, init_weights=True, mask=False):
        super().__init__()
        self.features = features

        linear_cls = MaskedLinear if mask else nn.Linear

        self.classifier = nn.Sequential(
            nn.Dropout(),
            linear_cls(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            linear_cls(512, 512),
            nn.ReLU(True),
            linear_cls(512, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, MaskedConv2d)):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, (nn.Linear, MaskedLinear)):
                module.weight.data.normal_(0, 0.01)
                if module.bias is not None:
                    module.bias.data.zero_()


def build_cifar_vgg(arch='vgg19', mask=False, num_classes=10, init_weights=True):
    """Instantiate a masked CIFAR-10 VGG variant."""

    try:
        cfg_key, batch_norm = _VGG_ARCHITECTURES[arch]
    except KeyError as exc:
        raise ValueError(f"Unsupported VGG architecture '{arch}'.") from exc

    features = _make_vgg_layers(_VGG_CONFIGS[cfg_key], batch_norm=batch_norm, mask=mask)
    return CifarVGG(features, num_classes=num_classes, init_weights=init_weights, mask=mask)


class Fire(nn.Module):
    """SqueezeNet Fire module with optional pruning masks."""

    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        expand1x1_channels: int,
        expand3x3_channels: int,
        *,
        mask: bool = False,
    ) -> None:
        super().__init__()
        conv_cls = MaskedConv2d if mask else nn.Conv2d

        self.squeeze = conv_cls(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = conv_cls(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = conv_cls(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x)),
        ], dim=1)


class MaskedSqueezeNet(PruningModule):
    """Masked SqueezeNet backbone tuned for CIFAR-10."""

    def __init__(self, num_classes: int = 10, *, mask: bool = False, init_weights: bool = True) -> None:
        super().__init__()
        conv_cls = MaskedConv2d if mask else nn.Conv2d

        self.features = nn.Sequential(
            conv_cls(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64, mask=mask),
            Fire(128, 16, 64, 64, mask=mask),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128, mask=mask),
            Fire(256, 32, 128, 128, mask=mask),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192, mask=mask),
            Fire(384, 48, 192, 192, mask=mask),
            Fire(384, 64, 256, 256, mask=mask),
            Fire(512, 64, 256, 256, mask=mask),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            conv_cls(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, MaskedConv2d)):
                if module is self.classifier[1]:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


def build_cifar_squeezenet(version: str = '1.1', *, mask: bool = False, num_classes: int = 10, init_weights: bool = True) -> MaskedSqueezeNet:
    """Instantiate a masked SqueezeNet variant configured for CIFAR-10."""

    if version != '1.1':
        raise ValueError(f"Unsupported SqueezeNet version '{version}'. Only '1.1' is available.")

    return MaskedSqueezeNet(num_classes=num_classes, mask=mask, init_weights=init_weights)


__all__ = [
    'LeNet',
    'LeNet_5',
    'MaskedConv2d',
    'MaskedSqueezeNet',
    'Fire',
    'SUPPORTED_VGG_ARCHS',
    'CifarVGG',
    'build_cifar_vgg',
    'build_cifar_squeezenet',
]

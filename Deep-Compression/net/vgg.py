import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape), requires_grad=False)

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MaskedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(MaskedLinear, self).__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape), requires_grad=False)

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True, mask=False):
        super(VGG, self).__init__()
        self.features = features
        
        Linear = MaskedLinear if mask else nn.Linear
        
        self.classifier = nn.Sequential(
            Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return nn.functional.log_softmax(x, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, MaskedConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or isinstance(m, MaskedLinear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, mask=False):
    layers = []
    in_channels = 3
    
    Conv2d = MaskedConv2d if mask else nn.Conv2d

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], **kwargs), **kwargs)
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True, **kwargs), **kwargs)
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], **kwargs), **kwargs)
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True, **kwargs), **kwargs)
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], **kwargs), **kwargs)
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True, **kwargs), **kwargs)
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], **kwargs), **kwargs)
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True, **kwargs), **kwargs)
    return model

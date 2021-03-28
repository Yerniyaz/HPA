import pretrainedmodels
import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

import configs as config


class Net(nn.Module):
    def __init__(self, backbone, num_classes=19):
        super(Net, self).__init__()

        if 'efficientnet' in backbone:
            self.model = EfficientNet.from_pretrained(backbone)
            self.model._conv_stem = nn.Conv2d(4, 48, 3, stride=2, padding=1, bias=False, padding_mode='zeros')
            self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_classes, bias=True)

        elif 'mobilenet' in backbone:
            self.model = models.mobilenet_v2(pretrained=True)
            self.model.classifier[1] = nn.Linear(1280, num_classes)

        elif 'resnet' in backbone:
            self.model = torch.hub.load('pytorch/vision:v0.6.0', backbone, pretrained=True)
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features,
                                      out_features=num_classes, bias=True)

        else:
            self.model = pretrainedmodels.__dict__[backbone]()
            if 'resnext' in backbone:
                self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
            else:
                assert (self.model.input_size[0] == config.img_w or self.model.input_size[1] == config.img_h), \
                    'wrong img size, expected: {}'.format(self.model.input_size[0])
            self.model.last_linear = nn.Linear(in_features=self.model.last_linear.in_features,
                                               out_features=num_classes, bias=True)

    def forward(self, x):
        out = self.model(x)

        return out

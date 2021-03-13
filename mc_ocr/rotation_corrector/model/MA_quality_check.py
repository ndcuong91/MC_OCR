import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from model import resnet


class MA_quality_check_res34(nn.Module):
    def __init__(self, classNum, pretrained=True):
        super(MA_quality_check_res34, self).__init__()

        self.base = resnet.resnet34(pretrained=pretrained)
        self.num_att = classNum

        # print ((self.base))
        # exit()

        self.classifier = nn.Linear(512, self.num_att)
        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)

        # if self.drop_pool5:
        # if self.training:
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        return x


class MA_quality_check_res50(nn.Module):
    def __init__(self, n_class, pretrained=True):
        super(MA_quality_check_res50, self).__init__()

        self.base = resnet.resnet50(pretrained=pretrained)
        self.num_att = n_class

        # print ((self.base))
        # exit()

        self.classifier = nn.Linear(2048, self.num_att)
        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)

        # if self.drop_pool5:
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        return x


class MA_quality_check_res50_mt(nn.Module):
    def __init__(self, classNum, pretrained=True):
        super(MA_quality_check_res50_mt, self).__init__()

        self.base = resnet.resnet50(pretrained=pretrained)
        self.num_att = classNum

        self.task = nn.Linear(2048, self.num_att)
        init.normal(self.task.weight, std=0.001)
        init.constant(self.task.bias, 0)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)

        x = F.dropout(x, p=0.5, training=self.training)
        a = self.task(x)
        return a

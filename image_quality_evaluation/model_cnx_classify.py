# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F


class cnx_classify(nn.Module):
    def __init__(self,dropout, num_classes=1024,input_size = 224):
        super(cnx_classify, self).__init__()
        self.size_cal = int(input_size/32)
        #(size - 1)/stride + 1
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        #(size - 1)/stride + 1
        #(size) = size
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),# 112x112 32 , 160
            conv_dw(32, 64, 1),# 112x112 64 ,160
            conv_dw(64, 128, 2),# 56x56 128 ,80
            conv_dw(128, 128, 1),# 56x56 128,80
            conv_dw(128, 256, 2),# 28x28 256, 40
            conv_dw(256, 512, 2),# 14x14 512, 20
            conv_dw(512, 512, 1),# 14x14 512, 20
            conv_dw(512, 1024, 2),#7x7x1024, 10
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
            # nn.Dropout(dropout),
            # nn.Linear(1024, 512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            # nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            # nn.Linear(256, num_classes),

        )

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, self.size_cal,1)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
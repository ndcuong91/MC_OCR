# borrowed from "https://github.com/marvis/pytorch-mobilenet"
import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary


class cnx_classify_idcard(nn.Module):
    def __init__(self, dropout, num_classes=1024, input_size=224):
        super(cnx_classify_idcard, self).__init__()
        self.size_cal = int(input_size / 32)

        # (size - 1)/stride + 1
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        # (size - 1)/stride + 1
        # (size) = size
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
            conv_bn(3, 32, 2),  # 112x112  32 , 160
            conv_dw(32, 64, 1),  # 112x112 64 ,160
            conv_dw(64, 128, 2),  # 56x56 128 ,80
            conv_dw(128, 128, 1),  # 56x56 128,80
            conv_dw(128, 256, 2),  # 28x28 256, 40
            conv_dw(256, 512, 2),  # 14x14 512, 20
            conv_dw(512, 512, 1),  # 14x14 512, 20
            conv_dw(512, 1024, 2),  # 7x7x1024, 10
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 2),
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 2),
        )

        self.fc3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 2),
        )

        self.fc4 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 2),
        )

        self.fc5 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 2),
        )

        self.fc6 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 2),
        )

        self.fc7 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, self.size_cal, 1)
        x = x.view(-1, 1024)
        overlight = self.fc1(x)
        dark = self.fc2(x)
        blur = self.fc3(x)
        dirtycover = self.fc4(x)
        lackofconner = self.fc5(x)
        photocopy = self.fc6(x)
        capviascreen = self.fc7(x)
        return [overlight, dark, blur, dirtycover, lackofconner, photocopy, capviascreen]


class cnx_classify_idcard_2(nn.Module):
    def __init__(self, dropout, num_classes=1024, input_size=224):
        super(cnx_classify_idcard_2, self).__init__()
        self.size_cal = int(input_size / 64)

        # (size - 1)/stride + 1
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        # (size - 1)/stride + 1
        # (size) = size
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
            conv_bn(3, 32, 2),  # 112x112  32 , 160
            conv_dw(32, 64, 1),  # 112x112 64 ,160
            conv_dw(64, 128, 2),  # 56x56 128 ,80
            conv_dw(128, 128, 1),  # 56x56 128,80
            conv_dw(128, 256, 2),  # 28x28 256, 40
            conv_dw(256, 512, 2),  # 14x14 512, 20
            conv_dw(512, 512, 1),  # 14x14 512, 20
            conv_dw(512, 1024, 2),  # 7x7x1024, 10
            conv_dw(1024, 2048, 2),  # 4x4x2048, 5
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 2),
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 2),
        )

        self.fc3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 2),
        )

        self.fc4 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 2),
        )

        self.fc5 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 2),
        )

        self.fc6 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 2),
        )

        self.fc7 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        x = self.model(x)
        # print('befor pooling', x.shape)
        x = F.avg_pool2d(x, self.size_cal, 1)
        # print('after pooling', x.shape)
        x = x.view(-1, 2048)
        # print('after reshape', x.shape)
        overlight = self.fc1(x)
        dark = self.fc2(x)
        blur = self.fc3(x)
        dirtycover = self.fc4(x)
        lackofconner = self.fc5(x)
        photocopy = self.fc6(x)
        capviascreen = self.fc7(x)
        return [overlight, dark, blur, dirtycover, lackofconner, photocopy, capviascreen]


class cnx_classify_recapture(nn.Module):
    def __init__(self, dropout, num_classes=1024, input_size=224):
        super(cnx_classify_recapture, self).__init__()
        self.size_cal = int(input_size / 60)

        # (size - 1)/stride + 1
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        # (size - 1)/stride + 1
        # (size) = size
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
            conv_bn(3, 32, 2),  # 112x112  32 , 160
            conv_dw(32, 64, 1),  # 112x112 64 ,160
            conv_dw(64, 128, 2),  # 56x56 128 ,80
            conv_dw(128, 128, 1),  # 56x56 128,80
            conv_dw(128, 256, 2),  # 28x28 256, 40
            conv_dw(256, 512, 2),  # 14x14 512, 20
            conv_dw(512, 512, 1),  # 14x14 512, 20
            conv_dw(512, 1024, 2),  # 7x7x1024, 10
            conv_dw(1024, 2048, 2),  # 4x4x2048, 5
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        x = self.model(x)
        # print(x.shape)
        x = F.avg_pool2d(x, self.size_cal, 1)
        # print(x.shape)
        x = x.view(-1, 2048)
        # print(x.shape)
        recaptured = self.fc1(x)
        return recaptured


class cnx1MB_classify_recapture(nn.Module):
    def __init__(self, dropout, num_classes=1024, input_size=224):
        super(cnx1MB_classify_recapture, self).__init__()
        self.size_cal = int(input_size / 30)

        # (size - 1)/stride + 1
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        # (size - 1)/stride + 1
        # (size) = size
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
            conv_bn(3, 32, 2),  # 112x112  32 , 160
            conv_dw(32, 64, 1),  # 112x112 64 ,160
            conv_dw(64, 128, 2),  # 56x56 128 ,80
            conv_dw(128, 128, 1),  # 56x56 128,80
            conv_dw(128, 256, 2),  # 28x28 256, 40
            conv_dw(256, 512, 2),  # 14x14 512, 20
            conv_dw(512, 512, 1),  # 14x14 512, 20
            conv_dw(512, 1024, 2),  # 7x7x1024, 10
            # conv_dw(1024, 2048, 2),  # 4x4x2048, 5
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        x = self.model(x)
        # print(x.shape)
        x = F.avg_pool2d(x, self.size_cal, 1)
        # print(x.shape)
        x = x.view(-1, 1024)
        # print(x.shape)
        recaptured = self.fc1(x)
        return recaptured


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5, input_size=None, linear=True):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.linear = linear
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        # Final convolution is initialized differently from the rest
        if self.linear:
            final_conv = nn.Linear(1024, 2)
        else:
            final_conv = nn.Conv2d(1024, 2, kernel_size=1)

        self.list_1 = []
        self.list_2 = []
        for i in range(self.num_classes):
            if not self.linear:
                classifier = nn.Sequential(
                    Fire(512, 64, 512, 512),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Dropout(p=dropout),
                    final_conv
                )
            else:
                classifier = nn.Sequential(
                    Fire(512, 64, 512, 512),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                final_conv1 = nn.Sequential(
                    nn.Dropout(p=dropout),
                    final_conv
                )
                self.list_2.append(final_conv1)
            self.list_1.append(classifier)

        self.list_1 = nn.ModuleList(self.list_1)
        self.list_2 = nn.ModuleList(self.list_2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        list_2 = []
        for i in range(self.num_classes):
            out = self.list_1[i](x)
            out = out.view(-1, out.shape[1])
            if self.linear:
                out = self.list_2[i](out)
            list_2.append(out)
        return list_2


class cnx_classify_idcard_4(nn.Module):
    def __init__(self, dropout, num_classes=1024, input_size=224):
        super(cnx_classify_idcard_4, self).__init__()
        self.size_cal = int(input_size / 64)
        self.num_classes = num_classes

        # (size - 1)/stride + 1
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

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
            conv_bn(3, 32, 2),  # 112x112  32 , 160
            conv_dw(32, 64, 1),  # 112x112 64 ,160
            conv_dw(64, 128, 2),  # 56x56 128 ,80
            conv_dw(128, 128, 1),  # 56x56 128,80
            conv_dw(128, 256, 2),  # 28x28 256, 40
            conv_dw(256, 512, 2),  # 14x14 512, 20
            conv_dw(512, 512, 1),  # 14x14 512, 20
            # conv_dw(512, 1024, 2),  # 7x7x1024, 10
        )
        self.list_fc = []
        self.list_conv = []
        for i in range(self.num_classes):
            conv = nn.Sequential(
                conv_dw(512, 1024, 2),  # 7x7x1024, 10
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.list_conv.append(conv)
            fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(1024, 2),
            )
            self.list_fc.append(fc)
        self.list_conv = nn.ModuleList(self.list_conv)
        self.list_fc = nn.ModuleList(self.list_fc)

    def forward(self, x):
        x = self.model(x)
        list_tensor_out = []
        for i in range(self.num_classes):
            out = self.list_conv[i](x)
            out = out.view(-1, out.shape[1])
            out = self.list_fc[i](out)
            list_tensor_out.append(out)
        return list_tensor_out


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        self.len_wide = len(num_classes)
        n = int((depth - 4) / 6)
        k = widen_factor
        filter = [16, 16 * k, 32 * k, 64 * k]

        def conv3x3(in_planes, out_planes, stride=1):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.conv1 = conv3x3(3, filter[0], stride=1)
        self.layer1 = self._wide_layer(wide_basic, filter[1], n, stride=2)
        self.layer2 = self._wide_layer(wide_basic, filter[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, filter[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(filter[3], momentum=0.9)

        self.linear = nn.ModuleList([nn.Sequential(
            nn.Linear(filter[3], num_classes[0]),
            nn.Softmax(dim=1))])

        # attention modules
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])

        for j in range(self.len_wide):
            if j < self.len_wide - 1:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.linear.append(nn.Sequential(nn.Linear(filter[3], num_classes[j + 1]),
                                                 nn.Softmax(dim=1)))
            for i in range(3):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))

        for i in range(3):
            if i < 2:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

    def conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True),
        )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, k=0):
        g_encoder = [0] * 4

        atten_encoder = [0] * self.len_wide
        for i in range(self.len_wide):
            atten_encoder[i] = [0] * 4
        for i in range(self.len_wide):
            for j in range(4):
                atten_encoder[i][j] = [0] * 3

        # shared encoder
        g_encoder[0] = self.conv1(x)
        g_encoder[1] = self.layer1(g_encoder[0])
        g_encoder[2] = self.layer2(g_encoder[1])
        g_encoder[3] = F.relu(self.bn1(self.layer3(g_encoder[2])))

        # apply attention modules
        for j in range(4):
            if j == 0:
                atten_encoder[k][j][0] = self.encoder_att[k][j](g_encoder[0])
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[0]
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1])
                atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2)
            else:
                atten_encoder[k][j][0] = self.encoder_att[k][j](
                    torch.cat((g_encoder[j], atten_encoder[k][j - 1][2]), dim=1))
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[j]
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1])
                if j < 3:
                    atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2)

        pred = F.avg_pool2d(atten_encoder[k][-1][-1], 8)
        pred = pred.view(pred.size(0), -1)

        out = self.linear[k](pred)
        return out

    def model_fit(self, x_pred, x_output, num_output):
        # convert a single label into a one-hot vector
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # apply cross-entropy loss
        loss = x_output_onehot * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Quality Checking Training With Pytorch')
    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('--num_worker', default=10, type=int)

    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--max_epoch', default=1000, type=float)
    parser.add_argument('--validation_epochs', default=1, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--input_size', default=72, type=float)
    arg = parser.parse_args()

    # model = cnx_classify_idcard_4(dropout=arg.dropout,
    #                               num_classes=4,
    #                               input_size=arg.input_size)
    # summary(model, input_size=(3, arg.input_size, arg.input_size), device='cpu')
    # img = torch.rand((1, 3, arg.input_size, arg.input_size))
    # valpre = model(img)
    # print(valpre)
    # print()
    data_class = [1, 2, ]
    model = WideResNet(depth=28, widen_factor=4, num_classes=data_class)
    summary(model, input_size=(3, arg.input_size, arg.input_size), device='cpu')
    img = torch.rand((1, 3, arg.input_size, arg.input_size))
    valpre = model(img, k=0)
    print(valpre)

    # se = SELayer(512)
    # summary(se, input_size=(512, 10, 10), device='cpu')
    #
    # sebot = SEBottleneck
    # model = SqueezeNet(num_classes=2)
    # summary(model, input_size=(3, arg.input_size, arg.input_size), device='cpu')
    # img = torch.rand((1, 3, arg.input_size, arg.input_size))
    # valpre = model(img)
    # print(valpre)

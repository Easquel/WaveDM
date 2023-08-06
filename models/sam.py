import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import pdb
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



# import common

###### Layer
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out


class irnn_layer(nn.Module):
    def __init__(self, in_channels):
        super(irnn_layer, self).__init__()
        self.left_weight = nn.Parameter(torch.tensor(1.0), True)
        self.right_weight = nn.Parameter(torch.tensor(1.0), True)
        self.up_weight = nn.Parameter(torch.tensor(1.0), True)
        self.down_weight = nn.Parameter(torch.tensor(1.0), True)

        self.zuoshang_weight = nn.Parameter(torch.tensor(1.0), True)
        self.zuoxia_weight = nn.Parameter(torch.tensor(1.0), True)
        self.youshang_weight = nn.Parameter(torch.tensor(1.0), True)
        self.youxia_weight = nn.Parameter(torch.tensor(1.0), True)

    def forward(self, x):
        _, _, H, W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_zuoshang = x.clone()
        top_zuoxia = x.clone()
        top_youshang = x.clone()
        top_youxia = x.clone()

        for i in range(H - 1):
            top_down[:, :, i + 1, :] = F.relu(
                top_down[:, :, i, :].clone() * self.down_weight + top_down[:, :, i + 1, :], inplace=False)
            top_up[:, :, -(i + 2), :] = F.relu(
                top_up[:, :, -(i + 1), :].clone() * self.up_weight + top_up[:, :, -(i + 2), :], inplace=False)

            top_zuoxia[:, :, i + 1, 1:W] = F.relu(
                top_zuoxia[:, :, i, 0:W-1].clone() * self.down_weight + top_zuoxia[:, :, i + 1, 1:W], inplace=False)

            top_youxia[:, :, i + 1, 0:W-1] = F.relu(
                top_youxia[:, :, i, 1:W ].clone() * self.down_weight + top_youxia[:, :, i + 1, 0:W-1], inplace=False)

        for i in range(W - 1):
            top_right[:, :, :, i + 1] = F.relu(
                top_right[:, :, :, i].clone() * self.right_weight + top_right[:, :, :, i + 1], inplace=False)
            top_left[:, :, :, -(i + 2)] = F.relu(
                top_left[:, :, :, -(i + 1)].clone() * self.left_weight + top_left[:, :, :, -(i + 2)], inplace=False)

            top_zuoshang[:, :, 1:H,i + 1] = F.relu(
                top_zuoshang[:, :,0:H-1 ,i ].clone() * self.down_weight + top_zuoshang[:, :, 1:H,i + 1], inplace=False)

            top_youshang[:, :, 0:H- 1,i + 1 ] = F.relu(
                top_youshang[:, :, 1:H,i ].clone() * self.down_weight + top_youshang[:, :, 0:H - 1,i + 1], inplace=False)


        return (top_up, top_right, top_down, top_left,top_zuoxia,top_youxia,top_zuoshang,top_youshang)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        #self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        #self.relu1 = nn.ReLU()
        #self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        #self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 8, kernel_size=1, padding=0, stride=1)
        self.relu3 = nn.ReLU()
        self.ca = ChannelAttention(8)
        self.sa = SpatialAttention()

    def forward(self, x):
        #out = self.conv1(x)
        #out = self.relu1(out)
        #out = self.conv2(out)
        #out = self.relu2(out)
        out = self.conv3(x)
        out = self.relu3(out)
        out = self.ca(out) * out
        out = self.sa(out) * out

        return out


class SAM(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels, self.out_channels)
        self.relu1 = nn.ReLU(True)

        self.conv1 = nn.Conv2d(64 , self.out_channels , kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.out_channels * 8, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(self.out_channels * 8, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        if self.attention:
            wight = self.attention_layer(x)
        out = self.conv1(x)
        top_up, top_right, top_down, top_left, top_zuoxia, top_youxia, top_zuoshang, top_youshang = self.irnn1(out)

        # direction attention
        if self.attention:
            top_up.mul(wight[:, 0:1, :, :])
            top_right.mul(wight[:, 1:2, :, :])
            top_down.mul(wight[:, 2:3, :, :])
            top_left.mul(wight[:, 3:4, :, :])

            top_zuoxia.mul(wight[:, 4:5, :, :])
            top_youxia.mul(wight[:, 5:6, :, :])
            top_zuoshang.mul(wight[:, 6:7, :, :])
            top_youshang.mul(wight[:, 7:8, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left,top_zuoxia, top_youxia, top_zuoshang, top_youshang], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left,top_zuoxia, top_youxia, top_zuoshang, top_youshang = self.irnn2(out)

        # direction attention
        if self.attention:
            top_up.mul(wight[:, 0:1, :, :])
            top_right.mul(wight[:, 1:2, :, :])
            top_down.mul(wight[:, 2:3, :, :])
            top_left.mul(wight[:, 3:4, :, :])

            top_zuoxia.mul(wight[:, 4:5, :, :])
            top_youxia.mul(wight[:, 5:6, :, :])
            top_zuoshang.mul(wight[:, 6:7, :, :])
            top_youshang.mul(wight[:, 7:8, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left,top_zuoxia, top_youxia, top_zuoshang, top_youshang], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        #mask = self.sigmod(self.conv_out(out))
        mask = self.conv_out(out)
        #pdb.set_trace()
        return mask

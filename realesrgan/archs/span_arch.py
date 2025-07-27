import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        self.conv1 = conv(num_feat, f, 1)
        self.conv_f = conv(f, f, 1)
        self.conv_max = conv(f, f, 3, padding=1)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, 3, padding=1)
        self.conv3_ = conv(f, f, 3, padding=1)
        self.conv4 = conv(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.p = p

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.conv_max(v_max)
        c3 = self.conv3(v_range)
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3 + cf))
        m = self.sigmoid(c4)
        if self.p > 0:
            m = F.dropout(m, p=self.p, training=self.training)
        return x * m

class SPAN(nn.Module):
    def __init__(self, num_feat=64, num_conv=32, act_type='lrelu'):
        super(SPAN, self).__init__()
        self.num_feat = num_feat
        self.num_conv = num_conv
        if act_type == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

        self.conv_list = [nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1) for _ in range(self.num_conv)]
        self.conv_list = nn.ModuleList(self.conv_list)
        self.conv_last = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1)
        self.esa = ESA(self.num_feat)

    def forward(self, x):
        res = x
        for i in range(self.num_conv):
            x = self.act(self.conv_list[i](x))
        x = self.conv_last(x)
        x = self.esa(x)
        return x + res

@ARCH_REGISTRY.register()
class SPANNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=2, act_type='lrelu', **kwargs):
        super(SPANNet, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.scale = upscale

        self.conv_first = nn.Conv2d(self.num_in_ch, self.num_feat, 3, 1, 1)
        self.span = SPAN(self.num_feat, self.num_conv, act_type)
        self.conv_last = nn.Conv2d(self.num_feat, self.num_out_ch * (self.scale ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

        if act_type == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.act(self.span(x))
        x = self.pixel_shuffle(self.conv_last(x))
        return x

# ------------------------------------------------------------------------
# Modified from CGNet (https://github.com/Ascend-Research/CascadedGaze)
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=0, stide=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stide, padding=padding, groups=nin,
                                   bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class UpsampleWithFlops(nn.Upsample):
    def __init__(self, size=None, scale_factor=None, mode='bicubic', align_corners=None):
        super(UpsampleWithFlops, self).__init__(size, scale_factor, mode, align_corners)
        self.__flops__ = 0

    def forward(self, input):
        self.__flops__ += input.numel()
        return super(UpsampleWithFlops, self).forward(input)


class GlobalContextExtractor(nn.Module):
    def __init__(self, c, kernel_sizes=[3, 3, 5], strides=[3, 3, 5], padding=0, bias=False):
        super(GlobalContextExtractor, self).__init__()
        self.depthwise_separable_convs = nn.ModuleList([
            depthwise_separable_conv(c, c, kernel_size, padding, stride, bias)
            for kernel_size, stride in zip(kernel_sizes, strides)
        ])

    def forward(self, x):
        outputs = []
        for conv in self.depthwise_separable_convs:
            outputs.append(F.gelu(conv(x)))
        return outputs


class CascadedGazeBlock(nn.Module):
    def __init__(self, c, GCE_Conv=3, DW_Expand=2, FFN_Expand=2, drop_out_rate=0):
        super().__init__()
        self.dw_channel = c * DW_Expand
        self.GCE_Conv = GCE_Conv

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1,
                                padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel,
                                kernel_size=3, padding=1, stride=1, groups=self.dw_channel,
                               bias=True)

        if self.GCE_Conv == 4:
            self.GCE = GlobalContextExtractor(c=c, kernel_sizes=[3, 3, 5], strides=[2, 3, 4])
            self.project_out = nn.Conv2d(int(self.dw_channel*2.5), c, kernel_size=1)
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=int(self.dw_channel*2.5), out_channels=int(self.dw_channel*2.5), kernel_size=1, padding=0, stride=1,
                        groups=1, bias=True))
        else:
            self.GCE = GlobalContextExtractor(c=c, kernel_sizes=[3, 3], strides=[2, 1])
            self.project_out = nn.Conv2d(self.dw_channel*2, c, kernel_size=1)
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=self.dw_channel*2, out_channels=self.dw_channel*2, kernel_size=1, padding=0, stride=1,
                        groups=1, bias=True))

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.channel_attention = ChannelAttention(c)

    def forward(self, inp):
        x = inp
        b, c, h, w = x.shape
        self.upsample = UpsampleWithFlops(size=(h, w), mode='bicubic')
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)

        x_1, x_2 = x.chunk(2, dim=1)
        if self.GCE_Conv == 4:
            x1, x2, x3 = self.GCE(x_1 + x_2)
            x = torch.cat([x, self.upsample(x1), self.upsample(x2), self.upsample(x3)], dim=1)
        else:
            x1, x2 = self.GCE(x_1 + x_2)
            x = torch.cat([x, self.upsample(x1), self.upsample(x2)], dim=1)

        x = self.sca(x) * x
        x = self.project_out(x)

        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class NAFBlock0(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class DWT(nn.Module):

    def __init__(self):
        super(DWT, self).__init__()
        haar_weights = torch.tensor([
            [[[0.5, 0.5],
              [0.5, 0.5]]],
            [[[-0.5, -0.5],
              [0.5, 0.5]]],
            [[[-0.5, 0.5],
              [-0.5, 0.5]]],
            [[[0.5, -0.5],
              [-0.5, 0.5]]]
        ])
        self.register_buffer('weight', haar_weights)

    def forward(self, x):
        # x: [B, 1, H, W]，输出： [B, 4, H//2, W//2]
        return F.conv2d(x, self.weight, stride=2)


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        haar_weights = torch.tensor([
            [[[0.5, 0.5],
              [0.5, 0.5]]],
            [[[-0.5, -0.5],
              [0.5, 0.5]]],
            [[[-0.5, 0.5],
              [-0.5, 0.5]]],
            [[[0.5, -0.5],
              [-0.5, 0.5]]]
        ])
        self.register_buffer('weight', haar_weights)

    def forward(self, x):
        return F.conv_transpose2d(x, self.weight, stride=2)


class SCNet(nn.Module):
    def __init__(self, img_channel=1, width=70, middle_blk_num=8, enc_blk_nums=[4, 4, 6, 8],
                 dec_blk_nums=[2, 2, 2, 4], GCE_CONVS_nums=[4, 4, 3, 3]):
        super().__init__()
        self.dwt = DWT()
        self.idwt = IDWT()

        self.intro = nn.Conv2d(in_channels=4, out_channels=width, kernel_size=3, padding=1, stride=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=4, kernel_size=3, padding=1, stride=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.Sequential()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.enc_channels = []

        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(
                    *[CascadedGazeBlock(chan, GCE_Conv=GCE_CONVS_nums[i]) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            self.enc_channels.append(chan)
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock0(chan) for _ in range(middle_blk_num)]
        )

        for i in range(len(dec_blk_nums)):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock0(chan) for _ in range(dec_blk_nums[i])]
                )
            )

        for c in reversed(self.enc_channels):
            self.skip_convs.append(
                nn.Sequential(
                    nn.Conv2d(c, c, 3, padding=1),
                    nn.GELU()
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape

        pad_factor = 2 * self.padder_size
        mod_pad_h = (pad_factor - H % pad_factor) % pad_factor
        mod_pad_w = (pad_factor - W % pad_factor) % pad_factor
        inp_pad = F.pad(inp, (0, mod_pad_w, 0, mod_pad_h))

        x = self.dwt(inp_pad)
        _, _, h_d, w_d = x.size()
        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            enc_skip = self.skip_convs[i](enc_skip)
            if x.shape != enc_skip.shape:
                x = F.interpolate(x, size=enc_skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        out = self.idwt(x)
        out = out[:, :, :inp_pad.size(2), :inp_pad.size(3)]
        out = out + inp_pad
        out = out[:, :, :H, :W]
        out = torch.clamp(out, 0, 1)
        return out


class SCNetLocal(Local_Base, SCNet):
    def __init__(self, *args, train_size=(1, 1, 220, 220), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        SCNet.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        mul = 1.5
        base_size = (int(H * mul), int(W * mul))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    from arch_util import measure_inference_speed

    img_channel = 1
    width = 84
    enc_blks = [6, 8, 10, 12]
    middle_blk_num = 24
    dec_blks = [4, 6, 8, 10]
    GCE_CONVS_nums = [4, 4, 4, 4]

    inp_shape = (1, 220, 220)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(0)
    data = torch.randn((1, *inp_shape))
    print("Device：", device)

    print("\n---------- SCNet ----------")
    net = SCNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                                                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, GCE_CONVS_nums=GCE_CONVS_nums)
    measure_inference_speed(net.to(device), (data.to(device),), max_iter=100, log_interval=50)

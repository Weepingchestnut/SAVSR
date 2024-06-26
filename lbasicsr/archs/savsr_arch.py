import math
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lbasicsr.archs.arch_util import make_layer
from lbasicsr.metrics.runtime import VSR_runtime_test
from lbasicsr.utils.registry import ARCH_REGISTRY


# ---------------------------------------------
# Omni-dimensional Scale-attention Convolution
# ---------------------------------------------
class ScaleAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(ScaleAttention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):     # x: torch.Size([batch, C=16, 1, 1])
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        # torch.Size([batch, C=64, 1, 1]) --> torch.Size([batch, C*h*w, 1, 1])
        return channel_attention

    def get_filter_attention(self, x):      # x: torch.Size([batch, C=16, 1, 1])
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        # torch.Size([batch, C=64, 1, 1]) --> torch.Size([batch, C=64, 1, 1])
        return filter_attention

    def get_spatial_attention(self, x):     # x: torch.Size([batch, C=16, 1, 1])
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        # torch.Size([batch, k*k, 1, 1]) --> torch.Size([batch, 1, 1, 1, k, k])
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):      # x: torch.Size([batch, C=16, 1, 1])
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        # torch.Size([batch, kernel_num, 1, 1]) --> torch.Size([batch, kernel_num, 1, 1, 1, 1])
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, scale_vector):
        # x = self.avgpool(x)     # torch.Size([batch, C(64), 1, 1])
        scale_vector = self.fc(scale_vector)          # torch.Size([batch, C'(16), 1, 1])
        scale_vector = self.bn(scale_vector)
        scale_vector = self.relu(scale_vector)
        return self.func_channel(scale_vector), self.func_filter(scale_vector), self.func_spatial(scale_vector), self.func_kernel(scale_vector)


class OSConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=8):
        super(OSConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = ScaleAttention(in_planes, out_planes, kernel_size, groups=groups,
                                        reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)      # torch.Size([kernel_num=4, C, C//groups, k, k])
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common
        
        # add scale routing weights -----------------------------
        self.scale_routing = nn.Sequential(
            nn.Linear(in_planes+2, in_planes*2),
            nn.ReLU(True),
            nn.Linear(in_planes*2, in_planes),
            nn.ReLU(True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # -------------------------------------------------------

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x, scale):
        batch_size, in_planes, height, width = x.size()
        
        # generate scale routing
        scale_h = torch.ones(1, 1).to(x.device) / scale[0]
        scale_w = torch.ones(1, 1).to(x.device) / scale[1]
        scale_info = torch.cat((scale_h, scale_w), 1).repeat(batch_size, 1)
        scale_info = self.scale_routing(torch.cat([scale_info, self.avgpool(x).view(batch_size, -1)], dim=1))    # torch.Size([bs, C_in=64, 1, 1])
        
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        # channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(scale_info.view(batch_size, self.in_planes, 1, 1))
        # ca: torch.Size([batch, C(64), 1, 1]), fa: torch.Size([batch, C(64), 1, 1]),
        # sa: torch.Size([batch, 1, 1, 1, k, k]), ka: torch.Size([batch, kernel_num, 1, 1, 1, 1])

        # batch_size, in_planes, height, width = x.size()
        x = x * channel_attention       # torch.Size([batch, C(64), h, w]) * torch.Size([batch, C(64), 1, 1])
        x = x.reshape(1, -1, height, width)     # torch.Size([1, batch*C, h, w])
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        # torch.Size([1, kernel_num, 1, 1, k, k]) * torch.Size([1, kernel_num, C, C, k, k]) * torch.Size([1, kernel_num, C, C, k, k])
        # = torch.Size([batch, kernel_num, C, C, k, k])

        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        # sum: torch.Size([batch, C, C, k, k]) --> view([batch*C_out, C_in, k, k])

        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        # output: torch.Size([1, batch*C_out, h, w])

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))     # torch.Size([batch, C_out, h, w])
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x, scale):
        return self._forward_impl(x, scale)
    
    
class OSAdapt(nn.Module):
    def __init__(self, channels, ratio=4):
        super(OSAdapt, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, channels//ratio, 3, 1, 1),
            nn.BatchNorm2d(channels//ratio),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            
            nn.Conv2d(channels//ratio, channels//ratio, 3, 1, 1),
            nn.BatchNorm2d(channels//ratio),
            nn.ReLU(True),
            nn.Conv2d(channels//ratio, channels//ratio, 3, 1, 1),
            nn.BatchNorm2d(channels//ratio),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels//ratio, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.adapt = OSConv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, scale):
        mask = self.mask(x)
        adapted = self.adapt(x, scale)
        
        return x + adapted * mask


class STAUpsample(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False, st_ksize: int = 5):
        super(STAUpsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels
        self.st_ksize = st_ksize
        
        # feat to spatio-temporal kernel
        self.kernel_conv = nn.Sequential(
            nn.Conv2d(channels, channels * st_ksize**2, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True))

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels // 8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels // 8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),  # 1x1 conv
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)      # 1x1 conv
        self.st_offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)   # 1x1 conv
        
        # scale feat and spatio-temporal feat fusion
        self.fusion = nn.Conv2d(2 * channels, channels, 1)
    
    def grid_sample(self, x: torch.Tensor, offset, scale: Union[tuple, int, float]) -> torch.Tensor:
        # generate grids
        b, _, h, w = x.size()
        # H, W = round(scale[0] * h), round(scale[1] * w)
        # H, W = int(scale[0] * h), int(scale[1] * w)
        H, W = get_HW(h, w, scale)
        # grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='xy')
        # grid = torch.stack(grid, dim=-1).to(x.device)
        grid = np.meshgrid(range(W), range(H))
        grid = np.stack(grid, axis=-1).astype(np.float64)
        grid = torch.Tensor(grid).to(x.device)

        # project into LR space
        grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale[1] - 0.5
        grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale[0] - 0.5

        # normalize to [-1, 1]
        grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
        grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1
        grid = grid.permute(2, 0, 1).unsqueeze(0)
        grid = grid.expand([b, -1, -1, -1])

        # add offsets
        offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
        offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
        grid = grid + torch.cat((offset_0, offset_1), 1)
        grid = grid.permute(0, 2, 3, 1)

        # sampling
        output = F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)
        # UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0.
        # Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.

        return output
    
    def sta_conv(self, feat: torch.Tensor, kernel: torch.Tensor):
        channels = feat.size(1)
        b, kernels, h, w = kernel.size()
        pad = (self.st_ksize - 1) // 2
        
        feat = F.pad(feat, (pad, pad, pad, pad), mode='replicate')
        feat = feat.unfold(2, self.st_ksize, 1).unfold(3, self.st_ksize, 1)
        feat = feat.permute(0, 2, 3, 1, 5, 4).contiguous()
        feat = feat.reshape(b, h, w, channels, -1)
        
        kernel = kernel.permute(0, 2, 3, 1).reshape(b, h, w, channels, self.st_ksize, self.st_ksize)
        kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(b, h, w, channels, -1)
        
        feat_out = torch.sum(feat * kernel, -1)
        feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        
        return feat_out

    def forward(self, x: torch.Tensor, scale: Union[tuple, int, float], st_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # implicitly alignment feature (st_feat) --> spatio-temporal adaptive filters
        kernel_warp = self.kernel_conv(st_feat)
        sta_feat = self.sta_conv(x, kernel_warp)

        # (1) coordinates in LR space =======================================================================
        # coordinates in HR space
        # H, W = round(h * scale[0]), round(w * scale[1])
        # H, W = int(h * scale[0]), int(w * scale[1])
        H, W = get_HW(h, w, scale)
        coor_hr = [torch.arange(0, H, 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, W, 1).unsqueeze(0).float().to(x.device)]

        # coordinates in LR space ==> R(x), R(y)
        coor_h = ((coor_hr[0] + 0.5) / scale[0]) - (torch.floor((coor_hr[0] + 0.5) / scale[0] + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale[1]) - (torch.floor((coor_hr[1] + 0.5) / scale[1] + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, W]).unsqueeze(0) / scale[1],    # 1 x H x W
            torch.ones_like(coor_h).expand([-1, W]).unsqueeze(0) / scale[0],
            coor_h.expand([-1, W]).unsqueeze(0),                                # 1 x H x W
            coor_w.expand([H, -1]).unsqueeze(0)
        ), 0).unsqueeze(0)
        # ===================================================================================================

        # (2) predict filters and offsets ==============================================
        embedding = self.body(input)        # torch.Size([1, 64, 75, 75])
        # offsets
        offset = self.offset(embedding)     # torch.Size([1, 2, 75, 75])
        st_offset = self.st_offset(embedding)

        # filters
        routing_weights = self.routing(embedding)                                           # torch.Size([1, 4, H, W])
        routing_weights = routing_weights.view(self.num_experts, H * W).transpose(0, 1)     # (H*W) * n    torch.Size([5625, 4])

        weight_compress = self.weight_compress.view(self.num_experts, -1)                   # torch.Size([n=4, Cout=8, Cin=64, ks=1, ks=1]) --> torch.Size([4, 512])
        weight_compress = torch.matmul(routing_weights, weight_compress)                    # torch.Size([225, 512])
        weight_compress = weight_compress.view(1, H, W, self.channels // 8, self.channels)  # torch.Size([1, H, W, 8, 64])

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, H, W, self.channels, self.channels // 8)      # torch.Size([1, H, W, 64, 8])
        # ===============================================================================

        # (3) grid sample & spatially varying filtering ===================================
        # grid sample
        fea0 = self.grid_sample(x, offset, scale)           # b * c * H * W
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)     # b * H * W * c * 1

        # spatially varying filtering
        fea = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        fea = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), fea).squeeze(-1)
        fea = fea.permute(0, 3, 1, 2) + fea0
        # =================================================================================
        
        sta_feat = self.grid_sample(sta_feat, st_offset, scale)
        out = self.fusion(torch.cat((sta_feat, fea), dim=1))

        return out


class ResidualBlock(nn.Module):

    def __init__(self, num_feat=64, num_frame=3, act=nn.LeakyReLU(0.2, True), use_osconv=False):
        super(ResidualBlock, self).__init__()
        self.nfe = num_feat
        self.nfr = num_frame
        self.act = act
        self.use_osconv = use_osconv

        self.conv0 = nn.Sequential(*[nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
                                     for _ in range(num_frame)])
        
        if use_osconv:
            self.osconv = OSConv2d(num_feat * num_frame, num_feat, kernel_size=3, stride=1, padding=1)
        else:
            # 1x1 conv to reduce dim
            self.conv1 = nn.Conv2d(num_feat * num_frame, num_feat, kernel_size=1, stride=1)
        self.conv2 = nn.Sequential(*[nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1)
                                     for _ in range(num_frame)])

    def forward(self, input: List):     # input [feat, scale]
        x, scale = input[0], input[1]

        x1 = [self.act(self.conv0[i](x[i])) for i in range(self.nfr)]

        merge = torch.cat(x1, dim=1)
        # base = self.act(self.conv1(merge))

        if self.use_osconv:
            base = self.act(self.osconv(merge, scale))
        else:
            base = self.act(self.conv1(merge))

        x2 = [torch.cat([base, i], 1) for i in x1]
        x2 = [self.act(self.conv2[i](x2[i])) for i in range(self.nfr)]

        return [[torch.add(x[i], x2[i]) for i in range(self.nfr)], scale]


class WindowUnit_l1(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 win_size=3,        # slid window size
                 num_block=4):
        super().__init__()
        self.nf = num_feat
        self.act = nn.LeakyReLU(0.2, True)

        # center frame conv
        self.conv_c = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # support frame conv
        self.conv_sup = nn.Conv2d(num_in_ch * (win_size - 1), num_feat, kernel_size=3, stride=1, padding=1)
        
        # self.blocks = nn.Sequential(*[ResidualBlock(num_feat, win_size, self.act, use_osconv=True) for i in range(num_block)])
        rb_list = []
        for i in range(num_block):
            if i < 1:
                rb_list.append(ResidualBlock(num_feat, 3, self.act, use_osconv=False))
            else:
                rb_list.append(ResidualBlock(num_feat, 3, self.act, use_osconv=True))
        self.blocks = nn.Sequential(*rb_list)
        
        self.merge = nn.Conv2d(3 * num_feat, num_feat, kernel_size=3, stride=1, padding=1)

    def forward(self, x, h_past, scale):
        b, t, c, h, w = x.size()

        # center frame in slide window
        x_c = x[:, t // 2]
        # the index of support frame
        sup_index = list(range(t))
        sup_index.pop(t // 2)
        # support frame
        x_sup = x[:, sup_index]
        x_sup = x_sup.reshape(b, (t - 1) * c, h, w)
        # hidden feature of support frame and center frame
        h_sup = self.act(self.conv_sup(x_sup))
        h_c = self.act(self.conv_c(x_c))
        # merge center frame, support fram and previous frame feats
        h_feat = [h_c, h_sup, h_past]
        h_feat = self.blocks([h_feat, scale])[0]  # after some residual block
        # h_feat = self.blocks(h_feat, scale)
        h_feat = self.merge(torch.cat(h_feat, dim=1))

        return h_feat


class WindowUnit_l2(nn.Module):
    def __init__(self,
                 num_feat=64,
                 win_size=5,
                 slid_win=3,
                 num_block=2):
        super().__init__()
        self.nf = num_feat
        self.ws = win_size
        self.sw = slid_win
        self.act = nn.LeakyReLU(0.2, True)

        # hidden feature conv
        self.conv_h = nn.Sequential(*[nn.Conv2d(self.nf*2, self.nf, kernel_size=3, stride=1, padding=1)
                                      for i in range(win_size)])
        self.blocks = nn.Sequential(*[ResidualBlock(self.nf, self.sw, self.act, use_osconv=True) for i in range(num_block)])
        self.merge = nn.Conv2d(self.sw * self.nf, self.nf*2, kernel_size=3, stride=1, padding=1)

    def forward(self, input: List):
        x, scale = input[0], input[1]
        # feature fusion
        h_feat = [self.act(self.conv_h[i](x[i])) for i in range(self.ws)]       # list[torch.Size([bs, C, h, w]), ..., tensor]
        
        if len(h_feat) == 1:
            out_feat = h_feat
        else:
            out_feat = []
        for i in range(self.ws - self.sw + 1):
            # idx = [i + win_i for win_i in range(self.sw)]
            sw_feat = self.blocks([h_feat[i:i + self.sw], scale])[0]
            # sw_feat = self.blocks(h_feat[i:i + self.sw], scale)
            sw_feat = self.merge(torch.cat(sw_feat, dim=1))
            out_feat.append(sw_feat)

        return [out_feat, scale]


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

        Args:
            num_feat (int): Channel number of intermediate features.
            squeeze_factor (int): Channel squeeze factor. Default: 16.
            res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x


@ARCH_REGISTRY.register()
class SAVSR(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 num_frame=7,
                 slid_win=3,
                 fusion_win=5,
                 interval=0,
                 w1_num_block=4,
                 w2_num_block=2,
                 n_resgroups=4,
                 n_resblocks=8,
                 downsample_scale=2,
                 center_frame_idx=None,
                 ):
        super(SAVSR, self).__init__()
        self.scale = (4, 4)
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.num_frame = num_frame
        
        # iteration window size
        if interval == 0:
            self.iter_win = num_frame
        else:
            if self.center_frame_idx % 2 == 0:
                self.iter_win = self.center_frame_idx + 1
            else:
                self.iter_win = self.center_frame_idx + 2
        self.slid_win = slid_win
        self.interval = interval
        
        self.num_feat = num_feat
        self.downsample_scale = downsample_scale

        # ------ 1. Implicit Alignment Module --------------------------------------------------------------------------------------
        self.f2p_win = WindowUnit_l1(num_in_ch, num_feat, win_size=slid_win, num_block=w1_num_block)
        self.p2f_win = WindowUnit_l1(num_in_ch, num_feat, win_size=slid_win, num_block=w1_num_block)
        # Pyramid fusion
        self.h_win = nn.Sequential(
            *[WindowUnit_l2(num_feat, win_size=(self.iter_win - self.slid_win + 1) - 2*i, slid_win=fusion_win, num_block=w2_num_block) 
              for i in range((self.iter_win - fusion_win + 1)//2)])
        self.h_win_act = nn.LeakyReLU(0.2, True)
        self.h_win_conv_h = nn.Conv2d(num_feat*2, num_feat, kernel_size=3, stride=1, padding=1)

        # ------ 2. Image Reconstruction ---------------------------------------------------------------
        self.RG = nn.ModuleList(
            [ResidualGroup(num_feat, num_block=n_resblocks, squeeze_factor=16, res_scale=1)
             for _ in range(n_resgroups)])
        self.K = 1
        self.adapt = nn.ModuleList([OSAdapt(num_feat) for _ in range(n_resgroups // self.K)])
        self.gamma = nn.Parameter(torch.ones(1))
        self.conv_last = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)

        # ------ 3. Arbitrary-Scale Upsampling ---------------------------------------------------------
        self.upsample = STAUpsample(num_feat)
        self.tail = nn.Conv2d(num_feat, num_in_ch, kernel_size=3, stride=1, padding=1, bias=True)

    def set_scale(self, scale: Union[tuple, float, int]):
        self.scale = scale
    
    @staticmethod
    def frame_sample(frame: torch.Tensor, num_frame: int = 9, interval: int = 1):
        if interval == 0:
            return frame, frame
        
        center_frame_idx = num_frame // 2
        # index = torch.tensor([i for i in range(num_frame)]).to(frame.device)
        index = [i for i in range(num_frame)]
        if center_frame_idx % 2 == 0:
            forward_idx = index[1::(interval+1)]                # [_, 1, _, 3, [4], 5, _, 7, _]     no center frame
            forward_idx.insert(center_frame_idx//2, center_frame_idx)
            backward_idx = index[::(interval+1)]                # [0, _, 2, _, (4), _, 6, _, 8]     have center frame
        else:
            forward_idx = index[::(interval+1)]                 # [0, _, 2, _, 4, [5], 6, _, 8, _, 10]      no center frame
            forward_idx.insert(center_frame_idx//2 + 1, center_frame_idx)
            backward_idx = index[1::(interval+1)]               # [_, 1, _, 3, _, (5), _, 7, _, 9, __]      have center frame
            if len(forward_idx) != len(backward_idx):
                backward_idx.append(forward_idx[-1])
                backward_idx.insert(0, forward_idx[0])

        return torch.index_select(frame, dim=1, index=torch.tensor(forward_idx).to(frame.device)), \
            torch.index_select(frame, dim=1, index=torch.tensor(backward_idx).to(frame.device))
    
    @staticmethod
    def generate_it(x: torch.Tensor, t: int=0, slid_win: int = 3):
        index = np.array([t - slid_win//2 + i for i in range(slid_win)])
        # index = np.clip(index, 0, iter_win - 1).tolist()
        # print(index)
        it = x[:, index, ...]

        return it
    
    def pad_spatial(self, x, multiple: int = 2):
        """Apply padding spatially.

        Since the OSAdapt module requires that the resolution is a multiple of 2, 
        we apply padding to the input LR images if their resolution is not divisible by 2.

        Args:
            x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()

        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        # padding
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        return x.view(n, t, c, h + pad_h, w + pad_w)

    def forward(self, x):
        b, t, c, h_input, w_input = x.size()
        H, W = get_HW(h_input, w_input, self.scale)
        
        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        x = self.pad_spatial(x)

        x_forward, x_backward = self.frame_sample(x, t, interval=self.interval)

        # ------ 1. Implicit Alignment Module ------------------------------------------------------------------------------------
        h_f2p_list = []  # hidden feature (future to past)
        h_p2f_list = []  # hidden feature (past to future)
        ht_f2p = torch.zeros((b, self.num_feat, x.shape[-2], x.shape[-1]), dtype=torch.float, device=x.device)
        ht_p2f = torch.zeros((b, self.num_feat, x.shape[-2], x.shape[-1]), dtype=torch.float, device=x.device)
        
        for idx in range(self.iter_win - self.slid_win + 1):  # [0, 1, 2, 3, 4, 5, 6]
            # backward (future --> past): [0, 1, 2, 3, (4, 5, 6)] --> [0, 1, 2, (3, 4, 5), 6] --> [0, 1, (2, 3, 4), 5, 6] --> ......
            cur_t = self.iter_win - 1 - self.slid_win // 2 - idx
            it = self.generate_it(x_backward, cur_t, self.slid_win)
            ht_f2p = self.f2p_win(it, ht_f2p, self.scale)
            h_f2p_list.insert(0, ht_f2p)
            
            # forward (past --> future): [(0, 1, 2), 3, 4, 5, 6] --> [0, (1, 2, 3), 4, 5, 6] --> [0, 1, (2, 3, 4), 5, 6] --> ......
            cur_t = idx + self.slid_win // 2
            it = self.generate_it(x_forward, cur_t, self.slid_win)
            ht_p2f = self.p2f_win(it, ht_p2f, self.scale)
            h_p2f_list.append(ht_p2f)

        h_feat = [torch.cat([h_f2p_list[i], h_p2f_list[i]], dim=1) for i in range(self.iter_win - self.slid_win + 1)]
        h_feat = self.h_win([h_feat, self.scale])[0][0]        # list[torch.Size([1, 128, 100, 100])]
        h_feat = self.h_win_act(self.h_win_conv_h(h_feat))
        align_feat = h_feat

        # ------ 2. Image Reconstruction Module -------------------------------------------------
        share_source = h_feat
        for i, rg in enumerate(self.RG):
            h_feat = rg(h_feat)
            if (i + 1) % self.K == 0:
                h_feat = self.adapt[i](h_feat, self.scale)
            h_feat = h_feat + self.gamma * share_source
        h_feat = self.conv_last(h_feat)
        h_feat += share_source

        # ------ 3. Arbitrary-Scale Upsampling -------------------------------------------------
        sr = self.upsample(h_feat[..., :h_input, :w_input], self.scale, align_feat[..., :h_input, :w_input])
        sr = self.tail(sr)  # 3x3Conv+bias
        sr = sr + F.interpolate(x_center, size=(H, W), mode='bilinear', align_corners=False)
        # --------------------------------------------------------------------------------------

        return sr


def get_HW_round(h, w, scale: tuple):
    return round(h * scale[0]), round(w * scale[1])

def get_HW_int(h, w, scale: tuple):
    return int(h * scale[0]), int(w * scale[1])

get_HW = get_HW_round
# ------ for fvcore ------
# get_HW = get_HW_int


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_frame = 7
    slid_win = 3
    fusion_win = 5
    
    """
    ----------------------->
    [0, 1, 2, 3, 4, 5, 6]       num_frame = 7
    
    [(0, 1, 2), 3, 4, 5, 6]     slid_win = 3
        [(0, 1, 2, 3, 4)]       fusion_win = 5
    """
    
    scale = (4, 4)
    model = SAVSR(
        num_frame=num_frame, 
        slid_win=slid_win, 
        fusion_win=fusion_win,
        interval=0,
        w1_num_block=4, 
        w2_num_block=2
    ).to(device)
    model.set_scale(scale)
    model.eval()    # needed, maybe bn bug
    
    input = torch.rand(1, num_frame, 3, 180, 320).to(device)

    with torch.no_grad():
        out = model(input)
    print(out.shape)

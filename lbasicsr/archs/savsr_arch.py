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


# class ResidualBlock(nn.Module):

#     def __init__(self, num_feat=64, num_frame=3, act=nn.LeakyReLU(0.2, True), use_osconv=False):
#         super(ResidualBlock, self).__init__()
#         self.nfe = num_feat
#         self.nfr = num_frame
#         self.act = act
#         self.use_osconv = use_osconv

#         self.conv0 = nn.Sequential(*[nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
#                                      for _ in range(num_frame)])
        
#         if use_osconv:
#             self.osconv = OSConv2d(num_feat * num_frame, num_feat, kernel_size=3, stride=1, padding=1)
#         else:
#             # 1x1 conv to reduce dim
#             self.conv1 = nn.Conv2d(num_feat * num_frame, num_feat, kernel_size=1, stride=1)
#         self.conv2 = nn.Sequential(*[nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1)
#                                      for _ in range(num_frame)])

#     def forward(self, input: List):     # input [feat, scale]
#         x, scale = input[0], input[1]

#         x1 = [self.act(self.conv0[i](x[i])) for i in range(self.nfr)]

#         merge = torch.cat(x1, dim=1)
#         # base = self.act(self.conv1(merge))

#         if self.use_osconv:
#             base = self.act(self.osconv(merge, scale))
#         else:
#             base = self.act(self.conv1(merge))

#         x2 = [torch.cat([base, i], 1) for i in x1]
#         x2 = [self.act(self.conv2[i](x2[i])) for i in range(self.nfr)]

#         return [[torch.add(x[i], x2[i]) for i in range(self.nfr)], scale]


class ResidualBlock(nn.Module):
    """
    先仿照 OVSR 中的 PFRB 构建多帧隐特征的残差融合块
    """

    def __init__(self, num_feat=64, num_frame=3, act=nn.LeakyReLU(0.2, True), use_osconv=False):
        super(ResidualBlock, self).__init__()
        self.nfe = num_feat
        self.nfr = num_frame
        self.act = act
        self.use_osconv = use_osconv

        self.conv0 = nn.Sequential(*[nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
                                     for _ in range(num_frame)])
        # 1x1 conv to reduce dim
        self.conv1 = nn.Conv2d(num_feat * num_frame, num_feat, kernel_size=1, stride=1)
        if use_osconv:
            self.osconv = OSConv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(*[nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1)
                                     for _ in range(num_frame)])

    def forward(self, input: List):     # input [feat, scale]
        x, scale = input[0], input[1]

        x1 = [self.act(self.conv0[i](x[i])) for i in range(self.nfr)]

        merge = torch.cat(x1, dim=1)
        base = self.act(self.conv1(merge))

        if self.use_osconv:
            # 518: add residual ------------
            res = self.act(self.osconv(base, scale))
            base = res + base
            # ------------------------------

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


# 540_3_0 asvsr add STAU
@ARCH_REGISTRY.register()
class SAVSR(nn.Module):
    def __init__(self,
                 num_in_ch: int = 3,
                 num_feat: int = 64,
                 num_frame: int = 7,
                 slid_win: int = 3,
                 fusion_win: int = 5,
                 interval: int = 0,
                 w1_num_block: int = 4,
                 w2_num_block: int = 2,
                 n_resgroups: int = 4,
                 n_resblocks: int = 8,
                 downsample_scale: int = 2,
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

# get_HW = get_HW_round
# ------ for fvcore ------
get_HW = get_HW_int


if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    repetitions = 300

    num_frame = 7
    slid_win = 3
    fusion_win = 5
    
    """
    ----------------------->
    [0, 1, 2, 3, 4, 5, 6]       num_frame = 7
    
    [(0, 1, 2), 3, 4, 5, 6]     slid_win = 3
        [(0, 1, 2, 3, 4)]       fusion_win = 5
    """
    
    # ----------------------------------------
    # problem scale and shape: x1.2, 480x585
    # ----------------------------------------
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
    
    # ------ torch profile -------------------------
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("model_inference"):
            out = model(input)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # ------ Runtime ------------------------------
    VSR_runtime_test(model, input, scale)

    print(
        "Model have {:.3f}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000.0))

    with torch.no_grad():
        print(flop_count_table(FlopCountAnalysis(model, input), activations=ActivationCountAnalysis(model, input)))
        out = model(input)
    print(out.shape)


"""
STAGE:2023-12-21 21:59:44 792942:792942 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2023-12-21 21:59:59 792942:792942 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2023-12-21 21:59:59 792942:792942 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference         0.50%      71.937ms       100.00%       14.283s       14.283s       0.000us         0.00%     223.570ms     223.570ms       1.44 Kb     -14.08 Mb      20.94 Gb      -3.55 Gb             1  
                                      aten::convolution         0.04%       6.088ms        90.64%       12.947s      18.184ms       0.000us         0.00%     127.800ms     179.494us           0 b           0 b       7.18 Gb           0 b           712  
                                     aten::_convolution         0.05%       7.440ms        90.60%       12.941s      18.176ms       0.000us         0.00%     127.800ms     179.494us           0 b           0 b       7.18 Gb           0 b           712  
                                           aten::conv2d         0.04%       5.643ms        90.66%       12.949s      18.187ms       0.000us         0.00%     124.232ms     174.483us           0 b           0 b       7.18 Gb     225.01 Mb           712  
                                aten::cudnn_convolution        79.68%       11.381s        90.46%       12.922s      18.148ms      84.745ms        43.02%     106.426ms     149.475us           0 b           0 b       7.18 Gb       7.18 Gb           712  
cudnn_infer_ampere_scudnn_winograd_128x128_ldg1_ldg4...         0.00%       0.000us         0.00%       0.000us       0.000us      69.762ms        35.41%      69.762ms     181.672us           0 b           0 b           0 b           0 b           384  
                                              aten::cat         0.04%       5.923ms         1.98%     282.768ms       1.071ms      24.713ms        12.55%      25.362ms      96.068us           0 b           0 b       6.44 Gb       6.44 Gb           264  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      22.407ms        11.37%      22.407ms      51.510us           0 b           0 b           0 b           0 b           435  
                                             aten::add_         0.05%       6.706ms         0.07%      10.231ms      15.961us      21.186ms        10.75%      21.441ms      33.449us           0 b           0 b           0 b           0 b           641  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us      19.346ms         9.82%      19.346ms     101.821us           0 b           0 b           0 b           0 b           190  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 14.283s
Self CUDA time total: 196.992ms

Warm up ...

Testing ...

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [01:09<00:00,  4.30it/s]

Average Runtime: 232.156 ms

Model have 11.477M parameters in total
| module                           | #parameters or shape   | #flops     | #activations   |
|:---------------------------------|:-----------------------|:-----------|:---------------|
| model                            | 11.477M                | 1.223T     | 2.936G         |
|  gamma                           |  (1,)                  |            |                |
|  f2p_win                         |  2.44M                 |  0.462T    |  0.627G        |
|   f2p_win.conv_c                 |   1.792K               |   0.498G   |   18.432M      |
|    f2p_win.conv_c.weight         |    (64, 3, 3, 3)       |            |                |
|    f2p_win.conv_c.bias           |    (64,)               |            |                |
|   f2p_win.conv_sup               |   3.52K                |   0.995G   |   18.432M      |
|    f2p_win.conv_sup.weight       |    (64, 6, 3, 3)       |            |                |
|    f2p_win.conv_sup.bias         |    (64,)               |            |                |
|   f2p_win.blocks                 |   2.324M               |   0.428T   |   0.571G       |
|    f2p_win.blocks.0              |    0.345M              |    99.09G  |    0.129G      |
|    f2p_win.blocks.1              |    0.66M               |    0.11T   |    0.147G      |
|    f2p_win.blocks.2              |    0.66M               |    0.11T   |    0.147G      |
|    f2p_win.blocks.3              |    0.66M               |    0.11T   |    0.147G      |
|   f2p_win.merge                  |   0.111M               |   31.85G   |   18.432M      |
|    f2p_win.merge.weight          |    (64, 192, 3, 3)     |            |                |
|    f2p_win.merge.bias            |    (64,)               |            |                |
|  p2f_win                         |  2.44M                 |  0.462T    |  0.627G        |
|   p2f_win.conv_c                 |   1.792K               |   0.498G   |   18.432M      |
|    p2f_win.conv_c.weight         |    (64, 3, 3, 3)       |            |                |
|    p2f_win.conv_c.bias           |    (64,)               |            |                |
|   p2f_win.conv_sup               |   3.52K                |   0.995G   |   18.432M      |
|    p2f_win.conv_sup.weight       |    (64, 6, 3, 3)       |            |                |
|    p2f_win.conv_sup.bias         |    (64,)               |            |                |
|   p2f_win.blocks                 |   2.324M               |   0.428T   |   0.571G       |
|    p2f_win.blocks.0              |    0.345M              |    99.09G  |    0.129G      |
|    p2f_win.blocks.1              |    0.66M               |    0.11T   |    0.147G      |
|    p2f_win.blocks.2              |    0.66M               |    0.11T   |    0.147G      |
|    p2f_win.blocks.3              |    0.66M               |    0.11T   |    0.147G      |
|   p2f_win.merge                  |   0.111M               |   31.85G   |   18.432M      |
|    p2f_win.merge.weight          |    (64, 192, 3, 3)     |            |                |
|    p2f_win.merge.bias            |    (64,)               |            |                |
|  h_win.0                         |  2.517M                |  0.113T    |  0.114G        |
|   h_win.0.conv_h                 |   0.369M               |   21.234G  |   18.432M      |
|    h_win.0.conv_h.0              |    73.792K             |    4.247G  |    3.686M      |
|    h_win.0.conv_h.1              |    73.792K             |    4.247G  |    3.686M      |
|    h_win.0.conv_h.2              |    73.792K             |    4.247G  |    3.686M      |
|    h_win.0.conv_h.3              |    73.792K             |    4.247G  |    3.686M      |
|    h_win.0.conv_h.4              |    73.792K             |    4.247G  |    3.686M      |
|   h_win.0.blocks                 |   1.779M               |   70.314G  |   88.474M      |
|    h_win.0.blocks.0              |    0.889M              |    35.157G |    44.237M     |
|    h_win.0.blocks.1              |    0.889M              |    35.157G |    44.237M     |
|   h_win.0.merge                  |   0.369M               |   21.234G  |   7.373M       |
|    h_win.0.merge.weight          |    (128, 320, 3, 3)    |            |                |
|    h_win.0.merge.bias            |    (128,)              |            |                |
|  h_win_conv_h                    |  73.792K               |  4.247G    |  3.686M        |
|   h_win_conv_h.weight            |   (64, 128, 3, 3)      |            |                |
|   h_win_conv_h.bias              |   (64,)                |            |                |
|  RG                              |  2.53M                 |  0.145T    |  0.251G        |
|   RG.0                           |   0.632M               |   36.127G  |   62.669M      |
|    RG.0.residual_group           |    0.595M              |    34.003G |    58.983M     |
|    RG.0.conv                     |    36.928K             |    2.123G  |    3.686M      |
|   RG.1                           |   0.632M               |   36.127G  |   62.669M      |
|    RG.1.residual_group           |    0.595M              |    34.003G |    58.983M     |
|    RG.1.conv                     |    36.928K             |    2.123G  |    3.686M      |
|   RG.2                           |   0.632M               |   36.127G  |   62.669M      |
|    RG.2.residual_group           |    0.595M              |    34.003G |    58.983M     |
|    RG.2.conv                     |    36.928K             |    2.123G  |    3.686M      |
|   RG.3                           |   0.632M               |   36.127G  |   62.669M      |
|    RG.3.residual_group           |    0.595M              |    34.003G |    58.983M     |
|    RG.3.conv                     |    36.928K             |    2.123G  |    3.686M      |
|  adapt                           |  1.318M                |  10.957G   |  20.507M       |
|   adapt.0                        |   0.329M               |   2.739G   |   5.127M       |
|    adapt.0.mask                  |    14.115K             |    0.612G  |    1.44M       |
|    adapt.0.adapt                 |    0.315M              |    2.127G  |    3.687M      |
|   adapt.1                        |   0.329M               |   2.739G   |   5.127M       |
|    adapt.1.mask                  |    14.115K             |    0.612G  |    1.44M       |
|    adapt.1.adapt                 |    0.315M              |    2.127G  |    3.687M      |
|   adapt.2                        |   0.329M               |   2.739G   |   5.127M       |
|    adapt.2.mask                  |    14.115K             |    0.612G  |    1.44M       |
|    adapt.2.adapt                 |    0.315M              |    2.127G  |    3.687M      |
|   adapt.3                        |   0.329M               |   2.739G   |   5.127M       |
|    adapt.3.mask                  |    14.115K             |    0.612G  |    1.44M       |
|    adapt.3.adapt                 |    0.315M              |    2.127G  |    3.687M      |
|  conv_last                       |  36.928K               |  2.123G    |  3.686M        |
|   conv_last.weight               |   (64, 64, 3, 3)       |            |                |
|   conv_last.bias                 |   (64,)                |            |                |
|  upsample                        |  0.121M                |  23.121G   |  1.287G        |
|   upsample.weight_compress       |   (4, 8, 64, 1, 1)     |            |                |
|   upsample.weight_expand         |   (4, 64, 8, 1, 1)     |            |                |
|   upsample.kernel_conv.0         |   0.104M               |   5.898G   |   92.16M       |
|    upsample.kernel_conv.0.weight |    (1600, 64, 1, 1)    |            |                |
|    upsample.kernel_conv.0.bias   |    (1600,)             |            |                |
|   upsample.body                  |   4.48K                |   4.011G   |   0.118G       |
|    upsample.body.0               |    0.32K               |    0.236G  |    58.982M     |
|    upsample.body.2               |    4.16K               |    3.775G  |    58.982M     |
|   upsample.routing.0             |   0.26K                |   0.236G   |   3.686M       |
|    upsample.routing.0.weight     |    (4, 64, 1, 1)       |            |                |
|    upsample.routing.0.bias       |    (4,)                |            |                |
|   upsample.offset                |   0.13K                |   0.118G   |   1.843M       |
|    upsample.offset.weight        |    (2, 64, 1, 1)       |            |                |
|    upsample.offset.bias          |    (2,)                |            |                |
|   upsample.st_offset             |   0.13K                |   0.118G   |   1.843M       |
|    upsample.st_offset.weight     |    (2, 64, 1, 1)       |            |                |
|    upsample.st_offset.bias       |    (2,)                |            |                |
|   upsample.fusion                |   8.256K               |   7.55G    |   58.982M      |
|    upsample.fusion.weight        |    (64, 128, 1, 1)     |            |                |
|    upsample.fusion.bias          |    (64,)               |            |                |
|  tail                            |  1.731K                |  1.593G    |  2.765M        |
|   tail.weight                    |   (3, 64, 3, 3)        |            |                |
|   tail.bias                      |   (3,)                 |            |                |
torch.Size([1, 3, 720, 1280])
"""

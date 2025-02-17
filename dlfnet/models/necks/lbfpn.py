import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from ..registry import NECKS


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4, kernel_size=7):
        super(CBAM, self).__init__()
        # channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca

        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        x = x * sa
        return x


@NECKS.register_module
class LBFPN(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                num_outs,
                start_level=0,  
                end_level=-1, 
                add_extra_convs=False, 
                no_norm_on_lateral=False, 
                conv_cfg=None,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='bilinear', align_corners=False),
                cfg=None):
        super(LBFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.bifpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for i in range(num_outs):
            self.bifpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False))
            
        self.cbam_modules = nn.ModuleList([
            CBAM(out_channels) for _ in range(self.num_ins)
        ])

        self.fusion_weights = nn.Parameter(torch.ones(num_outs, 4))


    
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) >= len(self.in_channels)

        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        for i in range(self.num_ins):
            weight = torch.sigmoid(self.fusion_weights[i])

            if i>0:
                downsampled = F.interpolate(laterals[i - 1], size=laterals[i].shape[2:], **self.upsample_cfg)
                laterals[i] = weight[0] * laterals[i] + weight[1] * downsampled

            if i < len(laterals) - 1:
                upsampled = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], **self.upsample_cfg)
                laterals[i] = weight[2] * laterals[i] + weight[3] * upsampled

            laterals[i] = self.bifpn_convs[i](laterals[i])

            # CRBlock
            laterals[i] = self.cbam_modules[i](laterals[i]) + laterals[i] 

        outs = tuple(laterals)

        if self.add_extra_convs and self.num_outs > len(outs):
            for i in range(self.num_outs - len(outs)):
                outs = outs + (F.max_pool2d(outs[-1], kernel_size=1, stride=2),)

        return outs
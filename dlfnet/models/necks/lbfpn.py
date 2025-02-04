import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from ..registry import NECKS


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4, kernel_size=7):
        super(CBAM, self).__init__()
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        x = x * sa
        return x

# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=4, kernel_size=7):
#         super(CBAM, self).__init__()
#         # 通道注意力模块
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#         # 空间注意力模块
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # 通道注意力
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = self.sigmoid(avg_out + max_out)
#         x = x * out

#         # 空间注意力
#         max_pool = torch.max(x, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x, dim=1, keepdim=True)
#         sa = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
#         x = x * sa
#         return x


@NECKS.register_module
class LBFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                #  refine_layer,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 no_norm_on_lateral=False,
                 use_attention=False,  
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(type='Xavier',
                               layer='Conv2d',
                               distribution='uniform'),
                 cfg=None):
        super(LBFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        self.use_attention = use_attention  # 保存参数
        # self.refine_layer = refine_layer

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
        # self.fusion_convs = nn.ModuleList()

        # 构建 lateral 和 BiFPN 层
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
            



        # 如果使用注意力机制，添加 CBAM 模块
        if self.use_attention:
            self.cbam_modules = nn.ModuleList([
                # CBAM(out_channels) for _ in range(refine_layer)
                CBAM(out_channels) for _ in range(num_outs)
            ])

        # 初始化融合权重
        # self.fusion_weights = nn.Parameter(torch.ones(num_outs, 2))
        self.fusion_weights = nn.Parameter(torch.ones(num_outs, 4))


    
    def forward(self, inputs):
        """Forward function."""
        # for x in inputs:
        #     print(x.shape)
        assert len(inputs) >= len(self.in_channels)

        # 如果输入特征层多于期望数，删除多余部分
        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # for i in range(len(self.in_channels)):
        #     print(i, inputs[i].shape)

        # 1. 构建 lateral 特征
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 2. 构建双向特征融合
        for i in range(len(self.bifpn_convs)):
            # 融合上采样和下采样特征
            weight = torch.sigmoid(self.fusion_weights[i])
            # weight = torch.softmax(self.fusion_weights[i],dim=0)

            if i>0:
                downsampled = F.interpolate(laterals[i - 1], size=laterals[i].shape[2:], **self.upsample_cfg)
                laterals[i] = weight[0] * laterals[i] + weight[1] * downsampled

            if i < len(laterals) - 1:
                upsampled = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], **self.upsample_cfg)
                # laterals[i] = weight[0] * laterals[i] + weight[1] * upsampled
                laterals[i] = weight[2] * laterals[i] + weight[3] * upsampled

            # if i > 0 and i < len(self.in_channels) - 1:
            #     downsampled = F.interpolate(laterals[i - 1], size=laterals[i].shape[2:], **self.upsample_cfg)
            #     upsampled = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], **self.upsample_cfg)
            #     laterals[i] = weight[0] * laterals[i] + weight[1] * downsampled + weight[2] * upsampled
            # elif i > 0:
            #     downsampled = F.interpolate(laterals[i - 1], size=laterals[i].shape[2:], **self.upsample_cfg)
            #     laterals[i] = weight[0] * laterals[i] + weight[1] * downsampled
            # else:
            #     upsampled = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], **self.upsample_cfg)
            #     laterals[i] = weight[0] * laterals[i] + weight[1] * upsampled

            # BiFPN卷积层
            laterals[i] = self.bifpn_convs[i](laterals[i])

            # CRBlock
            if self.use_attention:
                # laterals[i] = self.cbam_modules[i](laterals[i])  
                laterals[i] = self.cbam_modules[i](laterals[i]) + laterals[i] 

        # 3. 构建输出特征
        outs = tuple(laterals)

        # 添加额外的卷积层（如果需要）
        if self.add_extra_convs and self.num_outs > len(outs):
            for i in range(self.num_outs - len(outs)):
                outs = outs + (F.max_pool2d(outs[-1], kernel_size=1, stride=2),)

        return outs
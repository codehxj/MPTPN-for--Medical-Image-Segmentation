# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .Vit import VisionTransformer, Reconstruct
from .pixlevel import PixLevelModule


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

# def correct_dim(x):
#     x = torch.unsqueeze(x, dim=2)
#     x = torch.unsqueeze(x, dim=3)
#     return x

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        #=====================================================
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(int(in_channels * 1.5), out_channels, nb_Conv, activation)
        # =====================================================
        # =====================================================
        # self.pixModule = PixLevelModule(out_channels)
        # self.nConvs = _make_nConv(int(in_channels * 1.5), in_channels // 2, nb_Conv, activation)
        # =====================================================

    def forward(self, x, skip_x):
        up = self.up(x) #[2,512,28,28]
        skip_x_att = self.pixModule(skip_x) #[2,512,28,28]
        x = torch.cat([skip_x_att, up], dim=1)  #[2,1024,28,28] dim 1 is the channel dimension
        return self.nConvs(x)

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # 通过一个线性层将4C降为2C
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C

        # 解析输入图像的分辨率，即输入图像的长宽
        H, W = self.input_resolution
        # 解析输入图像的维度
        B, L, C = x.shape
        # 判断L是否与H * W一致，如不一致会报错
        assert L == H * W, "input feature has wrong size"
        # 判断输入图像的长宽是否可以被二整除，因为我们是通过2倍来进行下采样的
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        """
        #B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # 将xreshape为维度：(B, H, W, C)
        # 切片操作，通过切片操作将将相邻的2*2的patch进行拼接
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x.permute(0, 3, 1, 2)  # B H/2*W/2 4*C  ->  B, 4*C, H/2, W/2

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.dim = dim
        # self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.expand = nn.Linear(dim//4, dim//2, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // 4)

    def forward(self, x):
        """
        x: B, H*W, C

        D, H, W = self.input_resolution
        x = x.flatten(2).transpose(1, 2)
        x = self.expand(x)
        B, L, C = x.shape
        # assert L == D * H * W, "input feature has wrong size"
        """
        C = x.shape[1]
        x = x.permute(0, 2, 3, 1)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // 4)


        x = self.norm(x)
        x = self.expand(x)
        x = x.permute(0, 3, 1, 2)
        return x

class LViT_decoder(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        #self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))    #!!!!!!in_channels * 9
        # ===============原本模型================================================
        self.outc = nn.Conv2d(192, n_classes, kernel_size=(1, 1), stride=(1, 1))    #!!!!!!in_channels * 9
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.up4 = UpblockAttention(768, 384, nb_Conv=2)
        self.up3 = UpblockAttention(384, 192, nb_Conv=2)
        # ======================================================================
        # ===============无PatchMerging和Expanding===============================
        # self.outc = nn.Conv2d(64, n_classes, kernel_size=(1, 1), stride=(1, 1))  # !!!!!!in_channels * 9
        # self.last_activation = nn.Sigmoid()  # if using BCELoss
        # self.up4 = UpblockAttention(256, 128, nb_Conv=2)
        # self.up3 = UpblockAttention(128, 64, nb_Conv=2)
        # =======================================================================
        # ================无Up模块================================
        # self.outc = nn.Conv2d(576, n_classes, kernel_size=(1, 1), stride=(1, 1))  # !!!!!!in_channels * 9
        # self.last_activation = nn.Sigmoid()  # if using BCELoss
        # self.up4 = UpblockAttention(768, 384, nb_Conv=2)
        # self.up3 = UpblockAttention(384, 192, nb_Conv=2)
        # =======================================================================
        # self.patchMerging1_2 = PatchMerging(input_resolution=(256, 256), dim=64)  # [2,64,224,224]->[2,128,112,112]
        # self.patchMerging1_3 = PatchMerging(input_resolution=(128, 128), dim=128)  # [2,128,112,112]->[2,256,56,56]
        #
        # self.patchExpand2_1 = PatchExpand(input_resolution=(128, 128), dim=128)  # [2,128,112,112]->[2,64,224,224]
        # self.patchMerging2_3 = PatchMerging(input_resolution=(128, 128), dim=128)  # [2,128,112,112]->[2,256,56,56]
        #
        # self.patchExpand3_2 = PatchExpand(input_resolution=(64, 64), dim=256)  # [2,256,56,56]->[2,128,112,112]
        # self.patchExpand3_1 = PatchExpand(input_resolution=(128, 128), dim=128)  # [2,128,112,112]->[2,64,224,224]
        #
        # self.patchExpand21 = PatchExpand(input_resolution=(128, 128), dim=384)  # [2,384,112,112]->[2,192,224,224]
        # self.patchExpand32 = PatchExpand(input_resolution=(64, 64), dim=768)  # [2,768,56,56]->[2,384,112,112]
        # self.patchExpand31 = PatchExpand(input_resolution=(128, 128), dim=384)  # [2,384,112,112]->[2,192,224,224]
        self.patchMerging1_2 = PatchMerging(input_resolution=(224, 224), dim=64)  # [2,64,224,224]->[2,128,112,112]
        self.patchMerging1_3 = PatchMerging(input_resolution=(112, 112), dim=128)  # [2,128,112,112]->[2,256,56,56]
        self.patchExpand2_1 = PatchExpand(input_resolution=(112, 112), dim=128)  # [2,128,112,112]->[2,64,224,224]
        self.patchMerging2_3 = PatchMerging(input_resolution=(112, 112), dim=128)  # [2,128,112,112]->[2,256,56,56]
        self.patchExpand3_2 = PatchExpand(input_resolution=(56, 56), dim=256)  # [2,256,56,56]->[2,128,112,112]
        self.patchExpand3_1 = PatchExpand(input_resolution=(112, 112), dim=128)  # [2,128,112,112]->[2,64,224,224]
        self.patchExpand2 = PatchExpand(input_resolution=(112, 112), dim=128)  # [8,128,112,112]->[8,64,224,224]
        self.patchExpand21 = PatchExpand(input_resolution=(56, 56), dim=384)  # [2,384,112,112]->[2,192,224,224]
        self.patchExpand32 = PatchExpand(input_resolution=(28, 28), dim=768)  # [2,768,56,56]->[2,384,112,112]
        self.patchExpand31 = PatchExpand(input_resolution=(56, 56), dim=384)  # [2,384,112,112]->[2,192,224,224]

    def forward(self, x1, x2, x3):
        # =================原本模型=============================
        x1_2 = self.patchMerging1_2(x1)
        x1_3 = self.patchMerging1_3(x1_2)

        x2_1 = self.patchExpand2_1(x2)
        x2_3 = self.patchMerging2_3(x2)

        x3_2 = self.patchExpand3_2(x3)
        x3_1 = self.patchExpand3_1(x3_2)

        x1 = torch.cat([x1, x2_1, x3_1], dim=1)  # [2,192,224,224]
        x2 = torch.cat([x1_2,x2,x3_2], dim=1) #[2,384,112,112]
        x3 = torch.cat([x1_3,x2_3,x3], dim=1) #[2,768,56,56]
        x_up = self.up4(x3, x2)  # [2,192,112,112]
        x = self.up3(x_up, x1)
        # =====================================================
        # ============无patchMerging和patchexpanding============
        # x_up = self.up4(x3, x2)  # [2,192,112,112]
        # x = self.up3(x_up, x1)
        # =====================================================
        # ===============无up模块===============================
        # x1_2 = self.patchMerging1_2(x1)
        # x1_3 = self.patchMerging1_3(x1_2)
        #
        # x2_1 = self.patchExpand2_1(x2)
        # x2_3 = self.patchMerging2_3(x2)
        #
        # x3_2 = self.patchExpand3_2(x3)
        # x3_1 = self.patchExpand3_1(x3_2)
        #
        # x1 = torch.cat([x1, x2_1, x3_1], dim=1)  # [2,192,224,224]
        # x2 = torch.cat([x1_2, x2, x3_2], dim=1)  # [2,384,112,112]
        # x3 = torch.cat([x1_3, x2_3, x3], dim=1)  # [2,768,56,56]
        # x2 = self.patchExpand21(x2)
        # x3 = self.patchExpand32(x3)
        # x3 = self.patchExpand31(x3)
        # x = torch.cat([x1,x2,x3], dim=1) #[2,576,256,256]
        #=======================================================

        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x)) #[2, 1, 224, 224]
        else:
            logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
        return logits

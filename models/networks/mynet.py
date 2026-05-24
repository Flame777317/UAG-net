import sys
import math
import random
import torch
import numbers
import numpy as np
import torch.nn as nn
from torch.nn import init
from itertools import repeat
from functools import partial
from einops import rearrange
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import collections.abc as container_abcs
from torch.nn.modules.module import Module
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
# from timm.models._efficientnet_blocks import SqueezeExcite as SE
from .restormer_arch import TransformerBlock
from .basic_modules import *
from .newnet import *

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False), nn.PReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)*x
        x = self.sa(x)*x
        return x

class Head(nn.Module):
    def __init__(self, embed_dim=24, drop_path=0.1):
        r""" """
        super().__init__()
        self.embed_dim = embed_dim
        self.block = nn.Sequential(
            nn.Conv2d(1, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim, 3, padding=1)
        )
        alpha_0 = 1e-2
        self.alpha = nn.Parameter(
            alpha_0 * torch.ones((1, self.embed_dim, 1, 1)), requires_grad=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.alpha * self.block(x))
        return x

class Tail(nn.Module):
    def __init__(self, embed_dim=24):
        r""" """
        super().__init__()
        self.embed_dim = embed_dim
        self.block = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, 1, 3, padding=1)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class RB(nn.Module):
    def __init__(self, channel, relu_slope, use_HIN=False):
        super(RB, self).__init__()
        self.use_HIN = use_HIN
        if use_HIN:
            self.norm = nn.InstanceNorm2d(channel // 2,
                                          affine=True)  # 与批量归一化(Batch Normalization)类似，但在每个样本的基础上进行归一化，而不是在整个批次上进行
        self.conv_1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out)) + x

        return out

def A(x):
    '''
    对输入的批量单通道图像进行零填充，扩大其尺寸，然后计算其过采样的二维傅里叶变换，最后对结果进行适当的归一化处理。
    x : batch data shape: (batch, 1, imsize, imsize)
    result : OverSampled Fourier Transform of x shape:(batch, 1, 2*imsize, 2*imsize)
    '''
    imsize1 = x.shape[2]
    imsize2 = imsize1*2
    pad_num = int((imsize2-imsize1)/2)
    pad = torch.nn.ZeroPad2d((pad_num, pad_num, pad_num, pad_num))
    x = pad(x)
    oversamp_x_fft = torch.fft.fft2(x)*(1/imsize2)*(imsize1/imsize2)

    return oversamp_x_fft

def AT(x):
    '''
    x: OverSampled Fourier Transform of original data shape:(batch, 1, 2*imsize, 2*imsize)
    result:Inverse Fourier Transform and then left multiply (batch,1,imsize,imsize)
    '''
    imsize2 = x.shape[2]
    imsize1 = int(imsize2/2)
    crop_num = int((imsize2-imsize1)/2)
    ifftx = torch.real(torch.fft.ifft2(x)*imsize2*(imsize1/imsize2))
    oversampMTx = ifftx[:, :, crop_num:-crop_num, crop_num:-crop_num]
    return oversampMTx

def A_CDP(x, SamplingRate, mask):
    '''
    相位编码衍射测量矩阵的前向操作
    输入 x (batch_size, 1, imsize, imsize)
              ↓ 重复(repeat)
    x -> (batch_size, SamplingRate, imsize, imsize)
              ↓ 应用掩膜(mask * x)
    编码后的 x
              ↓ 对每个采样率的图像进行傅里叶变换
    Ax -> (batch_size, SamplingRate, imsize, imsize) (复数)
              ↓ 返回 Ax

    CDP measurement matrix
    x: batch data shape: (batch, 1, imsize, imsize)
    mask: uniform masks shape: (1, SamplingRate, imsize, imsize)
    '''
    imsize = x.shape[2]   # 128
    # batch_size,SamplingRate,imsize,imsize
    x = x.repeat(1, SamplingRate, 1, 1)
    x = mask*x
    Ax = torch.zeros_like(x, dtype=torch.complex64).to(x.device)   #biploar_mask
    # Ax = torch.zeros_like(x).to(x.device)   # batchsize*4*128*128 complex
    for i in range(SamplingRate):
        Ax[:, i, :, :] = torch.fft.fft2(
            x[:, i, :, :])*(1/imsize)   # 128*128_complex
    return Ax   # batchsize*4*128*128_complex

def At_CDP(Ax, SamplingRate, mask):
    '''
    输入 Ax (batch_size, SamplingRate, imsize, imsize)
            ↓ 对每个采样率的数据执行逆傅里叶变换
    Atx (batch_size, SamplingRate, imsize, imsize)
            ↓ 计算掩膜的复共轭 mask_
            ↓ 元素级相乘 mask_ * Atx
            ↓ 在采样率维度上求和
            ↓ 乘以归一化因子 imsize1
    Atx_sum (batch_size, imsize, imsize)
            ↓ 提取实部
            ↓ 调整形状为 (batch_size, 1, imsize, imsize)
    输出 Atx (batch_size, 1, imsize, imsize)

    CDP measurement inverse matrix
    Ax: OverSampled Fourier Transform of original data shape:(batch_size, SamplingRate, imsize, imsize)

    '''
    B, C, imsize1, imsize2 = Ax.shape  # 128
    Atx = torch.zeros_like(Ax)  # batch_size*SamplingRate*128*128
    for i in range(SamplingRate):
        Atx[:, i, :, :] = torch.fft.ifft2(Ax[:, i, :, :])   # 128*128
    mask_ = torch.conj(mask)   # batch_size * SamplingRate * 128 * 128_complex
    # batch_size * 1 * 128 * 128_complex
    Atx = torch.sum(mask_ * Atx, axis=1)*imsize1
    return torch.real(Atx).reshape(B, 1, imsize1, imsize2)  # 128*128

def Make_mask(mask_matrix):
    mask = torch.exp(1j * 2 * torch.pi * mask_matrix)
    return mask

def Poisson_noise_torch(Mx, alpha):
    '''
    add Possion noise
    '''
    device = Mx.device
    norm = torch.abs(Mx)  # |Ax|
    alpha = torch.Tensor([alpha]).to(device)  # noise level
    B, C, w, h = norm.shape
    intensity_noise = alpha/255*norm*torch.randn(B, C, w, h).to(device)
    y = norm ** 2 + intensity_noise
    y = y*(y > 0)
    y = torch.sqrt(y+1e-5)
    return y

def Gaussian_noise_torch(Mx, SNR):
    device = Mx.device
    norm = torch.abs(Mx)
    B, C, w, h = norm.shape
    noise = torch.randn(B, C, w, h).to(device)  # generate noise data which follws N(0,1)
    noise = noise-torch.mean(noise)  # mean = 0
    norm_power = torch.linalg.norm(norm ** 2) ** 2 / torch.prod(torch.tensor(norm.size()))
    noise_variance = norm_power/torch.pow(torch.tensor(10.0), (SNR/10))
    intensity_noise = (torch.sqrt(noise_variance) / torch.std(noise))*noise
    y = norm ** 2 + intensity_noise
    y = y*(y > 0)
    y = torch.sqrt(y+1e-5)
    return y


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, 4*dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(4*dim, dim, kernel_size=1, padding=0),
        )
        self.se = SELayer(dim, reduction=16)

    def forward(self, x):
        short_cut = x
        x = self.block(x)
        x = self.se(x)
        x = x + short_cut
        return x

class UBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down1 = nn.Sequential(
            # CBAM(dim),
            TransformerBlock(dim),
            nn.Conv2d(dim, dim*2, kernel_size=2, stride=2),
            LayerNorm(2*dim, eps=1e-6, data_format="channels_first"),
        )
        self.down2 = nn.Sequential(
            # CBAM(2*dim),
            TransformerBlock(2*dim),
            nn.Conv2d(2*dim, 4*dim, kernel_size=2, stride=2),
            LayerNorm(4*dim, eps=1e-6, data_format="channels_first"),
        )
        self.down3 = nn.Sequential(
            # CBAM(4*dim),
            TransrformerBlock(4*dim),
            nn.Conv2d(4*dim, 8*dim, kernel_size=2, stride=2),
            LayerNorm(8*dim, eps=1e-6, data_format="channels_first"),
        )
        self.mid_f = TransformerBlock(8*dim)
        self.soft_thr = nn.Parameter(torch.full((8*dim, 1, 1), 0.005))
        self.mid_b =  TransformerBlock(8*dim)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8*dim, out_channels=4*dim, kernel_size=2, stride=2),
            LayerNorm(4*dim, eps=1e-6, data_format="channels_first"),
            # CBAM(4*dim),
            TransformerBlock(4*dim),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4*dim, out_channels=2*dim, kernel_size=2, stride=2),
            LayerNorm(2*dim, eps=1e-6, data_format="channels_first"),
            # CBAM(2*dim),
            TransformerBlock(2*dim),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2*dim, out_channels=dim, kernel_size=2, stride=2),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            # CBAM(dim),
            TransformerBlock(dim),
        )

    def forward(self, x, xlb):
        if xlb is not None:
            xk1 = self.down1(x) + xlb[2]
            xk2 = self.down2(xk1) + xlb[1]
            xk3 = self.down3(xk2) + xlb[0]
        else:
            xk1 = self.down1(x)
            xk2 = self.down2(xk1)
            xk3 = self.down3(xk2)
        x_in = self.mid_f(xk3)
        x_m = torch.mul(torch.sign(x_in), F.relu(torch.abs(x_in) - self.soft_thr))
        xk4 = self.mid_b(x_m)
        xk5 = self.up1(xk4) + xk2
        xk6 = self.up2(xk5) + xk1
        x = self.up3(xk6) + x
        return x, [xk4,xk5,xk6]


class UBlock_t(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down1 = TransformerBlock(dim)

        self.down2 =  nn.Sequential(TransformerBlock(dim))
        self.down3 =  nn.Sequential(TransformerBlock(dim))

        self.mid_f =  nn.Sequential(TransformerBlock(dim))
        self.soft_thr = nn.Parameter(torch.full((dim, 1, 1), 0.005))
        self.mid_b =  nn.Sequential(TransformerBlock(dim))
        self.up1 = nn.Sequential(TransformerBlock(dim))
        self.up2 = nn.Sequential(TransformerBlock(dim))
        self.up3 = nn.Sequential(TransformerBlock(dim))

    def forward(self, x, xlb):
        if xlb is not None:
            xk1 = self.down1(x) + xlb[2]
            xk2 = self.down2(xk1) + xlb[1]
            xk3 = self.down3(xk2) + xlb[0]
        else:
            xk1 = self.down1(x)
            xk2 = self.down2(xk1)
            xk3 = self.down3(xk2)
        x_in = self.mid_f(x)
        x_m = torch.mul(torch.sign(x_in), F.relu(torch.abs(x_in) - self.soft_thr))
        xk4 = self.mid_b(x_m)
        xk5 = self.up1(xk4) + xk2
        xk6 = self.up2(xk5) + xk1
        x = self.up3(x_m) + x
        return x, [xk4,xk5,xk6]

class Denoise_Block(nn.Module):
    def __init__(self,
                 measurement_type):
        super(Denoise_Block, self).__init__()
        self.measurement_type = measurement_type
        self.tau = nn.Parameter(torch.Tensor([0.5]))
        self.fis = nn.Parameter(torch.Tensor([1]))
        if measurement_type == "CDPs":
            self.A = A_CDP
            self.AT = At_CDP
        else:
            self.A = A
            self.AT = AT
        self.conv_forward = Head(32)
        self.unet_layer = UBlock(32)
        self.conv_backward = Tail(32)

    def forward(self, x_in, b, rate, mask, x_last, xlb,layernum,y,t=1):
        device = x_in.device;
        if layernum == 0:
            z = self.A(x_in, rate, mask);
            r = x_in - self.tau * self.AT(z - b * (z / (torch.abs(z) + 1e-8)), rate, mask)
        else:
            z = self.A(y, rate, mask)
            r = y - self.tau * self.AT(z-b*(z/(torch.abs(z)+1e-8)),rate,mask)
        x_f = self.conv_forward(r) + x_last
        x_m, xfb = self.unet_layer(x_f, xlb)
        x_b = self.conv_backward(x_m) + r
        t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y_next = x_b + ((t - 1) / t_next) * (x_b - x_in) * self.fis

        return x_b, x_m, xfb,y_next,t_next

class AUV_Net(nn.Module):
    def __init__(self, layer_num=3, rate=4, measurement_type="CDPs"):
        super(AUV_Net, self).__init__()

        self.layer_num = layer_num
        self.rate = int(rate)
        self.measurement_type = measurement_type

        if measurement_type == "CDPs":
            self.A = A_CDP
            self.AT = At_CDP
        else:
            self.A = A
            self.AT = AT

        block_list = []
        for i in range(self.layer_num):
            block_list.append(Denoise_Block("CDPs"))
        self.denoise_stage = nn.ModuleList(block_list)

        self.mid = nn.Sequential(
            nn.Conv2d(self.layer_num, 32, kernel_size=3, padding=1),
            LayerNorm(32, eps=1e-6, data_format="channels_first"),
            # CBAM(32),
            TransformerBlock(32)
        )
        self.feature_merge = UBlock(64)
        self.post = nn.Sequential(
            LayerNorm(64, eps=1e-6, data_format="channels_first"),
            # CBAM(64),
            TransformerBlock(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            # CBAM(32),
            TransformerBlock(32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, mask, b):  # shape (batch_size, C=1, H=96, W=96)
        inter_xk = []
        if self.measurement_type == "CDPs":
            xfb = None
            x_last = torch.zeros_like(x)
            y = x
            t = 1
            for i in range(self.layer_num):
                x, x_last, xfb,y_next,t_next= self.denoise_stage[i](x,b,self.rate,mask,x_last,xfb,i,y,t)
                inter_xk.append(x)
                y = y_next
                t = t_next
        inter_xk = torch.cat(inter_xk, dim=1)
        inter_xk = self.mid(inter_xk)
        xk = torch.cat([x_last, inter_xk], dim=1)
        xk, _ = self.feature_merge(xk, None)
        xk = self.post(xk)
        # xk = x
        return xk

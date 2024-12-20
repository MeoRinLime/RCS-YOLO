import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                           bias=False)),
        ('bn', nn.BatchNorm2d(num_features=out_channels))
    ]))

class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=8):
        super().__init__()
        internal_neurons = max(input_channels // reduction_ratio, 1)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(input_channels, internal_neurons, 1),
            nn.SiLU(),
            nn.Conv2d(internal_neurons, input_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y.view(b, c, 1, 1)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    
    # 重塑张量以进行通道重排
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    
    return x

class RepVGG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        
        self.nonlinearity = nn.SiLU()
        
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, dilation=dilation,
                                       groups=groups, bias=True, padding_mode=padding_mode)
        else:
            padding_11 = padding - kernel_size // 2
            self.rbr_identity = (nn.BatchNorm2d(in_channels) 
                               if out_channels == in_channels and stride == 1 
                               else None)
            self.rbr_dense = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, 
                                   stride=stride, padding=padding, groups=groups, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels))
            ]))
            self.rbr_1x1 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, 1, 
                                   stride=stride, padding=padding_11, groups=groups, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels))
            ]))

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(x)))
        
        identity = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.nonlinearity(self.se(self.rbr_dense(x) + self.rbr_1x1(x) + identity))
    
    # 保持 get_equivalent_kernel_bias 方法不变
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight  # 已经通过 OrderedDict 命名为 'conv'
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fusevggforward(self, x):
        return self.nonlinearity(self.rbr_dense(x))

class SR(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c1_ = c1 // 2
        c2_ = c2 // 2
        self.repconv = RepVGG(c1_, c2_)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat([x1, self.repconv(x2)], dim=1)
        out = channel_shuffle(out, 2)
        return out

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def get_equivalent_kernel_bias(self):
        return self.repconv.get_equivalent_kernel_bias()

class RCSOSA(nn.Module):
    def __init__(self, c1, c2, n=1, se=True, e=0.5):
        super().__init__()
        n_ = n // 2
        c_ = make_divisible(int(c1 * e), 8)
        
        self.conv1 = RepVGG(c1, c_)
        self.sr1 = nn.ModuleList([SR(c_, c_) for _ in range(n_)])
        self.sr2 = nn.ModuleList([SR(c_, c_) for _ in range(n_)])
        self.conv3 = RepVGG(int(c_ * 3), c2)
        
        self.se = SEBlock(c2) if se else None

    def forward(self, x):
        x1 = self.conv1(x)
        
        x2 = x1
        for sr in self.sr1:
            x2 = sr(x2)
            
        x3 = x2
        for sr in self.sr2:
            x3 = sr(x3)
        
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv3(x)
        return self.se(x) if self.se is not None else x

def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor

if __name__ == '__main__':
    # 测试代码
    model = RCSOSA(256, 256)
    input_tensor = torch.randn(2, 256, 13, 13)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

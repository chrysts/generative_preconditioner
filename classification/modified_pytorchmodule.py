
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn.utils.weight_norm import WeightNorm

class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features, bias = True):
        super(Linear_fw, self).__init__(in_features, out_features, bias)
        self.weight.fast = None
        self.bias_bool = bias
        if bias:
            self.bias.fast = None

    def forward(self, x):
        if self.bias_bool:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.linear(x, self.weight.fast, self.bias.fast)
            else:
                out = super(Linear_fw, self).forward(x)
        else:
            if self.weight.fast is not None :
                out = F.linear(x, self.weight.fast)
            else:
                out = super(Linear_fw, self).forward(x)
        return out

class Linear_fwNoBias(nn.Linear):
    def __init__(self, in_features, out_features):
        super(Linear_fwNoBias, self).__init__(in_features, out_features, bias=False)
        self.weight.fast = None

    def forward(self, x):
        if self.weight.fast is not None :
            out = F.linear(x, self.weight.fast, bias=None)
        else:
            out = super(Linear_fwNoBias, self).forward(x)
        return out


class DistLinear(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(DistLinear, self).__init__(in_features, out_features, bias=False)
        self.weight.fast = None
        L_norm = torch.norm(self.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.weight.data)
        self.weight.data = self.weight.data.div(L_norm + 1e-12)

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-12)

        if self.weight.fast is not None:
            L_norm = torch.norm(self.weight.fast, p=2, dim=1).unsqueeze(1).expand_as(self.weight.fast)
            self.weight.fast = self.weight.fast.div(L_norm +  1e-12)
            out = F.linear(x_normalized, self.weight.fast, bias=None)
        else:
            L_norm = torch.norm(self.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.weight.data)
            self.weight.data = self.weight.data.div(L_norm +  1e-12)
            out = super(DistLinear, self).forward(x_normalized)
        return out

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True, groups=1):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding, groups=self.groups)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding, groups=self.groups)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out



class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out
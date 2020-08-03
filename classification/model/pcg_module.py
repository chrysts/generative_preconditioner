import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable


class PCG(nn.Module):
    def __init__(self, num_filters=64, kernel_size=3, num_plastic=300, num_mix=5):
        super(PCG, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_plastic = num_plastic
        self.num_mix = num_mix


        self.uu_3 = nn.Sequential(nn.Linear(self.num_plastic, self.num_filters * self.kernel_size),
                                  nn.ReLU(),
                                  nn.Linear(self.num_filters * self.kernel_size, self.num_mix+ self.num_mix*self.num_filters * self.kernel_size)
                                  )
        self.vv_3 = nn.Sequential(nn.Linear(self.num_plastic, self.num_filters * self.kernel_size),
                                  nn.ReLU(),
                                  nn.Linear(self.num_filters * self.kernel_size,
                                            self.num_mix + self.num_mix * self.num_filters * self.kernel_size)
                                  )
        self.bb_3 = nn.Sequential(nn.Linear(self.num_plastic, self.num_filters),
                                  nn.ReLU(),
                                  nn.Linear(self.num_filters,  self.num_mix + self.num_mix* self.num_filters)
                                  )

        self.uu_4 = nn.Sequential(nn.Linear(self.num_plastic, self.num_filters * self.kernel_size),
                                  nn.ReLU(),
                                  nn.Linear(self.num_filters * self.kernel_size,
                                            self.num_mix + self.num_mix *self.num_filters * self.kernel_size)
                                  )
        self.vv_4 = nn.Sequential(nn.Linear(self.num_plastic, self.num_filters * self.kernel_size),
                                  nn.ReLU(),
                                  nn.Linear(self.num_filters * self.kernel_size,
                                            self.num_mix + self.num_mix *self.num_filters * self.kernel_size)
                                  )
        self.bb_4 = nn.Sequential(nn.Linear(self.num_plastic, self.num_filters),
                                  nn.ReLU(),
                                  nn.Linear(self.num_filters, self.num_mix+ self.num_mix*self.num_filters)
                                  )


        self.context_params = torch.zeros(size=[self.num_plastic], requires_grad=True, device="cuda")


        for param in self.parameters():
            self.init_layer(param)

    def reset(self):
        self.context_params = self.context_params.detach() * 0.
        self.context_params.requires_grad = True

    def init_layer(self, L):
        # Initialization using fan-in
        if isinstance(L, nn.Conv2d):
            n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
            L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
        elif isinstance(L, nn.BatchNorm2d):
            L.weight.data.fill_(1)
            L.bias.data.fill_(0)
        elif isinstance(L, nn.BatchNorm2d):
            L.weight.data.fill_(1)
            L.bias.data.fill_(0)
        elif isinstance(L, nn.Linear):
            torch.nn.init.kaiming_uniform_( L.weight, nonlinearity='linear')

    def forward(self, context_params):
        if self.num_mix <= 1:
            conv3_uv, conv3_b = self.assemble_w_b(self.uu_3, self.vv_3, self.bb_3, context_params)
            conv4_uv, conv4_b = self.assemble_w_b(self.uu_4, self.vv_4, self.bb_4, context_params)
        else:
            conv3_uv, conv3_b = self.assemble_w_b_multi(self.uu_3, self.vv_3, self.bb_3, context_params)
            conv4_uv, conv4_b = self.assemble_w_b_multi(self.uu_4, self.vv_4, self.bb_4, context_params)

        return [conv3_uv, conv3_b, conv4_uv, conv4_b]


    def assemble_w_b(self, uu_func, vv_func, bb_func, lat):

        uu = uu_func(lat)
        vv = vv_func(lat)
        bb = bb_func(lat)

        wu_ext = uu.unsqueeze(-1)
        wv_ext_t = vv.unsqueeze(-1).transpose(0, 1)
        model
        conv_uv = torch.mm(wu_ext, wv_ext_t)
        conv_b = bb

        return F.relu(conv_uv), F.relu(conv_b)


    def assemble_w_b_multi(self, uu_func, vv_func, bb_func, lat):

        uu_all = uu_func(lat)
        vv_all = vv_func(lat)
        bb_all = bb_func(lat)

        mixture_coeff_uu = F.softmax(uu_all[:self.num_mix])
        mixture_coeff_vv = F.softmax(vv_all[:self.num_mix])
        mixture_coeff_bb = F.softmax(bb_all[:self.num_mix])

        uu = uu_all[self.num_mix:].view(self.num_mix, -1)
        uu = uu * mixture_coeff_uu.unsqueeze(-1)
        uu = uu.sum(0)

        vv = vv_all[self.num_mix:].view(self.num_mix, -1)
        vv = vv * mixture_coeff_vv.unsqueeze(-1)
        vv = vv.sum(0)

        bb = bb_all[self.num_mix:].view(self.num_mix, -1)
        bb = bb * mixture_coeff_bb.unsqueeze(-1)
        bb = bb.sum(0)

        wu_ext = uu.unsqueeze(-1)
        wv_ext_t = vv.unsqueeze(-1).transpose(0, 1)

        conv_uv = torch.mm(wu_ext, wv_ext_t)
        conv_b = bb

        return F.relu(conv_uv), F.relu(conv_b)

import torch
import torch.nn as nn
from convnet import ConvNet_MAML, Linear_fw
from torch.autograd import Variable
import torch.nn.functional as F


class MAML_PCG(nn.Module):
    def __init__(self, n_way, n_shot, train_lr=0.1, noise_rate=0.):
        super().__init__()
        self.cnn = ConvNet_MAML()
        self.classifier = Linear_fw(self.cnn.out_channels, n_way)
        self.train_lr = train_lr
        self.n_way = n_way
        self.n_shot = n_shot
        self.idxs = [8, 9, 12, 13] #idxs of parameters location.
        self.noise_rate = noise_rate


    def forward(self, input, query, pcg, inner_update_num=2, train=False):

        fast_parameters = []
        noises = []
        for param in self.parameters():
            param.fast = None
            fast_parameters.append(param)
            noises.append(torch.zeros_like(param).normal_(0, self.noise_rate))

        y_a_i = torch.arange(self.n_way).repeat(self.n_shot)
        y_a_i = y_a_i.type(torch.cuda.LongTensor)

        grad_support = self.run_inner_step(input, y_a_i, fast_parameters, create_graph=True, detach=False)

        pcg.reset()

        for ii in range(inner_update_num*2): # 2 forwards and backwards
            jj = 0
            precond = pcg(pcg.context_params)
            for k, weight in enumerate(self.parameters()):
                weight.fast = None

                if k in self.idxs:
                    precond[jj] = precond[jj].view(-1).view(*weight.size())
                    weight.fast = weight -  self.train_lr*(grad_support[k]+ noises[k]) * precond[jj]
                    jj = jj + 1
                else:
                    weight.fast = weight

            grad_mask = self.run_inner_step(input, y_a_i, pcg.context_params, create_graph=True, detach=False)[0]
            pcg.context_params = -grad_mask

        query_f = self.cnn(query)
        scores = self.classifier(query_f)

        return scores


    def run_inner_step(self, input, label, parameters, create_graph=False, detach=True):
        x = self.cnn(input)
        out = self.classifier(x)
        loss = F.cross_entropy(out, label)
        grad = torch.autograd.grad(loss, parameters, create_graph=create_graph, retain_graph=True)
        if detach:
            grad = [ g.detach()  for g in grad ]

        return grad
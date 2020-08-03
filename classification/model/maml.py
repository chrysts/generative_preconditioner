import torch
import torch.nn as nn
from convnet import ConvNet_MAML, Linear_fw
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from networks.modified_pytorchmodule import DistLinear


class MAML(nn.Module):
    def __init__(self, n_way, n_shot, train_lr=0.1, noise_rate=0.):
        super().__init__()
        self.cnn = ConvNet_MAML()
        self.classifier = Linear_fw(self.cnn.out_channels, n_way)
        #self.classifier = DistLinear(self.cnn.out_channels, n_way)
        self.train_lr = train_lr
        self.n_way = n_way
        self.n_shot = n_shot
        self.noise_rate = noise_rate
        self.idx=16

    def forward(self, input, query, inner_update_num=10):

        fast_parameters = []
        noises = []
        for param in self.parameters():
            param.fast = None
            fast_parameters.append(param)
            noises.append(torch.zeros_like(param).normal_(0, self.noise_rate))

        #y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_shot ) )).cuda() #label for support data
        y_a_i = torch.arange(self.n_way).repeat(self.n_shot)
        y_a_i = y_a_i.type(torch.cuda.LongTensor)

        # y_q_i = torch.arange(self.n_way).repeat(15)
        # y_q_i = y_q_i.type(torch.cuda.LongTensor)

        for ii in range(inner_update_num):
            #grad_support = self.run_inner_step(input, y_a_i, fast_parameters)
            grad_support = self.run_inner_step(input, y_a_i, fast_parameters)
            #grad_query = self.run_inner_step(self, input, y_a_i, fast_parameters)
            #do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                if k == self.idx: #### REMOVE THIS FOR NORMAL MAML
                    if weight.fast is None:
                        weight.fast = weight - self.train_lr * (grad_support[k])# + noises[k])#.detach() #create weight.fast
                    else:
                        weight.fast = weight.fast - self.train_lr * (grad_support[k])# + noises[k])#.detach() #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                else:
                    weight.fast = weight #### REMOVE THIS FOR NORMAL MAML
                fast_parameters.append(weight.fast)

        query = self.cnn(query)
        scores = self.classifier(query)

        return scores


    def run_inner_step(self, input, label, fast_parameters):
        x = self.cnn(input)
        out = self.classifier(x)
        loss = F.cross_entropy(out, label)
        grad = torch.autograd.grad(loss, fast_parameters, create_graph=False, retain_graph=True)
        grad = [ g.detach()  for g in grad ]

        return grad
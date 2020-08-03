import argparse
import os.path as osp
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from miniimagenet import MiniImageNet
from samplers import CategoriesSampler
from model.maml_pcg import MAML_PCG
from model.pcg_module import PCG
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric
from torch.nn.utils.clip_grad import clip_grad_norm_
#from newfunc.labelsmoothing import LabelSmoothingLoss

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=1)
    parser.add_argument('--save-epoch', type=int, default=1000)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--inner-step', type=int, default=1)
    parser.add_argument('--noise-rate', type=float, default=0.0)
    parser.add_argument('--load-path', default='./results/pcg_maml/max-acc.pth')
    parser.add_argument('--data-path', default='yourdatapath')
    parser.add_argument('--gpu', default='3')


    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)


    valset = MiniImageNet('test', args.data_path)
    val_sampler = CategoriesSampler(valset.label, 1000,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    model = MAML_PCG(args.train_way, args.shot, noise_rate=args.noise_rate).cuda()
    model.load_state_dict(torch.load(args.load_path))

    pcg = PCG(num_plastic=300).cuda()

    optimizer = torch.optim.Adam(list(model.parameters()) , lr=0.001, amsgrad=False)
    optimizer_pcg = torch.optim.Adam(list(pcg.parameters()), lr=0.001, amsgrad=False)


    def save_model(name):
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
        torch.save(pcg.state_dict(), osp.join(args.save_path, name + '-pcg.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['val_loss'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()


    for epoch in range(1, args.max_epoch + 1):

        for i, batch in enumerate(val_loader, 1):
            with torch.no_grad():
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.test_way
                data_shot, data_query = data[:p], data[p:]
                label = torch.arange(args.test_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)

            logits = model(data_shot, data_query, pcg, inner_update_num=args.inner_step)
            loss = F.cross_entropy(logits, label)

            vl.add(loss.item())

            acc = count_acc(logits, label)
            va.add(acc)

            vl.add(loss.item())
            va.add(acc)
            pcg.reset()

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f} maxacc={:.4f}'.format(epoch, vl, va,trlog['max_acc']))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))



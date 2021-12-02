import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.optim import optimizer
from torch.utils import data
from torchvision import models

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.multiprocessing

from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from model.resnet_simclr import ResNetSimCLR
from simclr_original import SimCLR_origin
from lars import create_optimizer_lars

'''
gpu: rank
'''


def main(local_rank, args):

    rank = args.nr * args.gpus + local_rank

    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    device = torch.device("cuda", local_rank)
    args.device = local_rank  # TAG: notify

    dataset = ContrastiveLearningDataset(root_folder=args.data)

    train_dataset = dataset.get_dataset_for_simclr_original(args.dataset_name, args.n_views)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
    )

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    # SyncBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local_rank])
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer = create_optimizer_lars(model, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.warm_up_epoch, eta_min=args.eta_min, last_epoch=-1, verbose=True)
    model.train()
    simclr = SimCLR_origin(args=args, model=model, optimizer=optimizer, scheduler=scheduler)
    simclr.train(train_loader)

def init():
    model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    #TAG: SimCLR args
    parser.add_argument('--data', metavar='DIR', default='./dataset', help='path to dataset')
    parser.add_argument('--dataset-name', default='cifar10', help='dataset name', choices=['stl10', 'cifar10'],)
    parser.add_argument('--arch', metavar='ARCH', default='resnet50', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)',)
    parser.add_argument('--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)',)
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run',)
    parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='mini-batch size (default: 256)',)
    # parser.add_argument('--lr','--learning-rate', default=0.0003, type=float, metavar='LR', help='initial learning rate', dest='lr', )
    parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay',)
    parser.add_argument('--warm_up_epoch', default=10, type=int, help='epochs for warm up')
    parser.add_argument('--eta_min', default=1e-10, type=float, help='CosineAnnealingLR min lr')
    parser.add_argument('--seed', default=12345, type=int, help='seed for initializing training. ')
    # parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.',)
    parser.add_argument('--out_dim', default=128, type=int, help='feature dimension of the final layer(NOT the encoder!) (default: 128)')
    parser.add_argument('--log-every-n-steps', default=1, type=int, help='Log every n steps')
    parser.add_argument('--temperature', default=0.5, type=float, help='softmax temperature (default: 0.07)',)
    parser.add_argument('--n-views', default=2, type=int, metavar='N', help='Number of views for contrastive learning training.',)

    #TAG: DDP args
    parser.add_argument('--nodes', default=1, type=int, metavar='N', help='number of machines/nodes')
    parser.add_argument('--gpus', default=2, type=int, help='number of gpus per node')
    parser.add_argument('--nr', default=0, type=int, help='node id')

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes

    # args.lr = 0.3 * args.batch_size * args.world_size/256
    args.lr = 1.0

    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '21657'

    mp.spawn(main, nprocs=args.gpus, args=(args,))

if __name__ == "__main__":
    init()
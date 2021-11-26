import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from model.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


'''
gpu: rank
'''


def main(local_rank, args):

    # assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # # check if gpu training is available
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')
    #     cudnn.deterministic = True
    #     cudnn.benchmark = True
    # else:
    #     args.device = torch.device('cpu')
    #     args.gpu_index = -1

    rank = args.nr * args.gpus + local_rank

    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=args.world_size, rank=rank
    )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.device = local_rank # TAG: notify

    dataset = ContrastiveLearningDataset(root_folder=args.data)

    gmm_init_dataset = dataset.get_gmm_init_dataset(args.dataset_name)

    # TAG: n_views = 1 就是mixup情况
    #  train_dataset = datasets.get_dataset(args.dataset_name, args.n_views)
    train_dataset = dataset.get_dataset(args.dataset_name, 1)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
    )

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim).to(device)

    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    # with torch.cuda.device(args.gpu_index):
    #     simclr = SimCLR(args=args,
    #                     model=model,
    #                     optimizer=optimizer,
    #                     scheduler=scheduler)
    #     simclr.train(train_loader, gmm_init_dataset)
    # print(dist.get_rank())
    model.train()
    simclr = SimCLR(args=args, model=model,
                    optimizer=optimizer, scheduler=scheduler)

    simclr.train(train_loader, gmm_init_dataset)


def init():
    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    
    ### SimCLR Arguments# #####
    parser.add_argument(
        '-data', metavar='DIR', default='./dataset', help='path to dataset'
    )
    parser.add_argument(
        '-dataset-name',
        default='cifar10',
        help='dataset name',
        choices=['stl10', 'cifar10'],
    )
    parser.add_argument(
        '-a',
        '--arch',
        metavar='ARCH',
        default='resnet50',
        choices=model_names,
        help='model architecture: ' +
        ' | '.join(model_names) + ' (default: resnet50)',
    )
    parser.add_argument(
        '-j',
        '--workers',
        default=0,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 32)',
    )
    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        metavar='N',
        help='number of total epochs to run',
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        default=256,
        type=int,
        metavar='N',
        help='mini-batch size (default: 256), this is the total '
        'batch size of all GPUs on the current node when '
        'using Data Parallel or Distributed Data Parallel',
    )
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=0.0003,
        type=float,
        metavar='LR',
        help='initial learning rate',
        dest='lr',
    )
    parser.add_argument(
        '--wd',
        '--weight-decay',
        default=1e-4,
        type=float,
        metavar='W',
        help='weight decay (default: 1e-4)',
        dest='weight_decay',
    )
    parser.add_argument(
        '--seed', default=12345, type=int, help='seed for initializing training. '
    )
    parser.add_argument(
        '--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument(
        '--fp16-precision',
        action='store_true',
        help='Whether or not to use 16-bit precision GPU training.',
    )
    parser.add_argument(
        '--out_dim', default=128, type=int, help='feature dimension (default: 128)'
    )
    parser.add_argument(
        '--log-every-n-steps', default=1, type=int, help='Log every n steps'
    )
    parser.add_argument(
        '--temperature',
        default=0.07,
        type=float,
        help='softmax temperature (default: 0.07)',
    )
    parser.add_argument(
        '--n-views',
        default=2,
        type=int,
        metavar='N',
        help='Number of views for contrastive learning training.',
    )

    parser.add_argument(
        '--gmm_every_n_epoch',
        default=10,
        type=int,
        help='Initialize GMM every n epoch'
    )
    parser.add_argument('--gpu-index', default=5, type=int, help='Gpu index.')
    # parser.add_argument("--local_rank", default=-1, type=int)
    ###########################




    ###### GMM arguments ######
    parser.add_argument(
        '-comps',
        '--components',
        default=10,
        type=int,
        help='number of gmm\'s component',
    )
    parser.add_argument(
        '-gmm_e',
        '--gmm_epoch',
        default=3,
        type=int,
        help='number of total epochs to fit a GMM',
    )
    parser.add_argument(
        '-gmm_lr',
        '--gmm_lr',
        default=1e-3,
        type=float,
        help='GMM learning rate',
    )
    parser.add_argument(
        '-gmm_b',
        '--gmm_batch_size',
        default=512,
        type=int,
        help='batch size for GMM fitting',
    )
    parser.add_argument(
        '-rst', '--restarts', default=1, type=int, help='Restart times of fitting a GMM'
    )
    parser.add_argument(
        '-kmf',
        '--k_means_factor',
        default=1,
        type=int,
        help='K-Means initialization\'s batch  size is k_means_factor larger than gmm_batch_size',
    )
    parser.add_argument(
        '-kmi',
        '--k_means_iters',
        default=3,
        type=int,
        help='Iterations of k-means initialization of gmm\'s parameters',
    )
    parser.add_argument(
        '-gmm_w',
        '--gmm_w',
        default=1e-6,
        type=float,
        help='weight decay for gmm fitting',
    )
    parser.add_argument(
        '-gmm_lr_step',
        '--gmm_lr_step',
        default=5,
        type=int,
        help='gmm fitting learning rate scheduler',
    )
    parser.add_argument(
        '-gmm_lr_gamma',
        '--gmm_lr_gamma',
        default=0.1,
        type=float,
        help='gmm fitting learning rate scheduler gamma',
    )
    ##########################

    ### DDP arguments #########
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument(
        '-g', '--gpus', default=1, type=int, help='number of gpus per node'
    )
    parser.add_argument(
        '-nr', '--nr', default=0, type=int, help='ranking within the nodes'
    )
    ###########################

    args = parser.parse_args()
    # configuration for main init multiprocessing
    args.world_size = args.gpus * args.nodes
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23357'
    # 调用main函数，传入参数(rank,args)
    mp.spawn(main, nprocs=args.gpus, args=(args,))


if __name__ == "__main__":
    init()

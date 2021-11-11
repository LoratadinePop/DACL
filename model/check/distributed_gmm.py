import os, sys
from threading import local
import time

from torch import cuda

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import numpy as np
import argparse
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import MNIST

from torch.utils.data import DataLoader

from model.sgd_gmm import SGDGMM

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp



def main(gpu, args):
    MNIST_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307),
                             (0.3081, 0.3081, 0.3081)),
    ])

    resnet_18 = resnet18(pretrained=False, progress=True)
    in_features = resnet_18.fc.in_features
    resnet_18.fc = nn.Linear(in_features, 2)


    # local_rank = int(os.environ["LOCAL_RANK"])

    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl')

    train_set = MNIST(root="dataset",
                      train=True,
                      transform=MNIST_transform,
                      download=True)
    

    cuda.set_device(gpu)
    device = torch.device("cuda:{}".format(gpu))

    gmm = SGDGMM(10,
                 2,
                 lr=0.01,
                 batch_size=512,
                 epochs=100,
                 backbone_model=resnet_18,
                 device=device,
                 restarts=10,
                 k_means_iter=5,
                 args=args)

    gmm.fit(gpu, train_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g',
                        '--gpus',
                        default=3,
                        type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr',
                        '--nr',
                        default=0,
                        type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes      
    os.environ['CUDA_VISIBLE_DEVICES']='5,6,7'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23351'  
    mp.spawn(main, nprocs=args.gpus, args=(args,))












    # gmm = None
# if dist.get_rank() == 0:
#     gmm = SGDGMM(10,
#                 2,
#                 lr=0.01,
#                 batch_size=512,
#                 epochs=10,
#                 backbone_model=resnet_18,
#                 device=local_rank,
#                 restarts=1,
#                 k_means_iter=10)
#     print(gmm)
#     print("1 done")
#     torch.distributed.barrier()
# else:
#     print("wait")
#     torch.distributed.barrier()
#     print("other done")

# print(dist.get_rank())
# print(gmm)
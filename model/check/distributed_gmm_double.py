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
from torchvision.models import resnet18, resnet50
from torchvision.datasets import MNIST

from torch.utils.data import DataLoader

from model.sgd_gmm import SGDGMM,SGDGMMModule

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

    resnet_18 = resnet50(pretrained=False, progress=True)
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
    # device = torch.device("cuda:{}".format(gpu))
    device = torch.device("cuda", gpu)
    print(device)
    print("how many gmm")
    gmm = SGDGMM(components=10,
                 dimensions=2,
                 lr=0.01,
                 batch_size=256,
                 epochs=50,
                 backbone_model=resnet_18,
                 device=device,
                 restarts=1,
                 k_means_iter=1,
                 args=args)           
    gmm.fit(gpu, train_set)

def init(gpu, args):

    rank = args.nr * args.gpus + gpu
    # Handshake with master process
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size,
                            rank=rank)


    gmm = SGDGMMModule(10,2,1e-3)
    if dist.get_rank() == 0:
        os.environ['MASTER_PORT'] = '20369'
        os.environ['MASTER_ADDR'] = 'localhost'
        mp.spawn(main,nprocs=args.gpus, args=(args,))
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()

    gmm.load_state_dict(torch.load("./result/checkpoint/sgdgmm/gmm.ckpt"))
    # print(gmm.means)
    # print(gmm.l_diag)
    x = gmm.sample(torch.Tensor([[1,2]]))
    print(x)
    x = gmm.predict(torch.Tensor([[1,2]]))
    print(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g',
                        '--gpus',
                        default=4,
                        type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr',
                        '--nr',
                        default=0,
                        type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()


    args.world_size = args.gpus * args.nodes      
    os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29710'  
    mp.spawn(init, nprocs=args.gpus, args=(args,))
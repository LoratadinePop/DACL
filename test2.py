from datetime import time
import os
import argparse
import torch
import time
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
def main(local_rank):

    rank = local_rank

    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=4, rank=rank
    )

    if dist.get_rank() == 0:
        print("hang!")
        time.sleep(30)
        dist.barrier()
    else:
        print(f'{dist.get_rank()} waiting!')
        dist.barrier()

def init():
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23357'
    # 调用main函数，传入参数(rank,args)
    mp.spawn(main, nprocs=4, args=())

if __name__ == "__main__":
    init()

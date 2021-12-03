from functools import wraps
import logging
import os
import sys
import random
import time
from matplotlib.pyplot import get
from numpy import uint
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import image
from tqdm import tqdm
from util import reduce_tensor, save_checkpoint, get_time, get_writer, linear_warmup

import torch.distributed as dist
import torch.multiprocessing as mp

class SimCLR_origin(object):
    def __init__(self, *args, **kwargs):
        # *args tuple, positional argument
        # **kwargs dict,
        self.args = kwargs['args']
        self.model = kwargs['model'] # DDP Model
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        logging.basicConfig(
            filename=('./log/simclr_origin/training.log'),
            level=logging.DEBUG,
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    # feature: 512, 128
    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        self.writer = get_writer(log_dir="./log/simclr_origin/") if dist.get_rank() == 0 else None

        scaler = GradScaler(enabled=self.args.fp16_precision)

        n_iter = 0
        if dist.get_rank() == 0:
            logging.info(f"@rank{dist.get_rank()} Start simclr_origin training for {self.args.epochs} epochs. <{get_time()}>")

        for epoch_counter in range(1, self.args.epochs+1):

            train_loader.sampler.set_epoch(epoch_counter)

            if epoch_counter > 10:
                # lr decay
                self.scheduler.step()
            else:
                # warnup for first 10 epochs
                linear_warmup(self.optimizer, self.args.eta_min, self.args.lr, self.args.warm_up_epoch, epoch_counter)

            with tqdm(total=(len(train_loader.dataset) // train_loader.batch_size // dist.get_world_size()), ncols=None, unit='it') as _tqdm:
                _tqdm.set_description(f'simclr_origin training @{dist.get_rank()} epoch {epoch_counter}/{self.args.epochs}')

                for images, _ in train_loader:
                    images = torch.cat(images, dim=0)
                    # print(f'images {images.shape}')
                    images = images.to(self.args.device)

                    with autocast(enabled=self.args.fp16_precision):
                        features = self.model(images)
                        # print(f'features {features.shape}')
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)

                    self.optimizer.zero_grad()

                    scaler.scale(loss).backward()

                    scaler.step(self.optimizer)
                    
                    scaler.update()

                    _tqdm.set_postfix(loss=f'{loss:.4f}')
                    _tqdm.update()

                    loss_reduced = reduce_tensor(loss)
                    if dist.get_rank() == 0:
                        self.writer.add_scalar(tag="simclr_origin Training Loss", scalar_value=loss_reduced.item(), global_step=n_iter+1)
                        
                    n_iter += 1


            logging.debug(f"@rank{dist.get_rank()} Epoch: {epoch_counter}\t<{get_time()}>")

            if (epoch_counter) % 100 == 0:
                if dist.get_rank() == 0:
                    logging.info("@rank{} Training has finished. <{}>".format(dist.get_rank(), get_time()))
                    # save model checkpoints
                    checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                    save_checkpoint(
                        {
                            'epoch': self.args.epochs,
                            'arch': self.args.arch,
                            'state_dict': self.model.module.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        },
                        is_best=False,
                        filename=os.path.join('./result/checkpoint/simclr_origin', checkpoint_name),
                    )
                    logging.info(
                        f"Model checkpoint and metadata has been saved at {os.path.join('./result/checkpoint/simclr_origin', checkpoint_name)}. <{get_time()}>"
                    )
        
        if self.writer:
            self.writer.close()
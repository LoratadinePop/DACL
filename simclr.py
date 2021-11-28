from re import S
import time
from functools import wraps
import logging
import os
from matplotlib.pyplot import get
from numpy import imag, uint
import torch
from torch._C import device
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from util import reduce_tensor, save_checkpoint, get_time, get_writer
from model.sgd_gmm import SGDGMM, SGDGMMModule
import torch.multiprocessing
import torch.distributed as dist
import torch.multiprocessing as mp


def fit_gmm(local_rank, args, model, dataset):
    device = torch.device("cuda", local_rank)
    gmm = SGDGMM(
        components=args.components,
        dimensions=args.gmm_dim,
        epochs=args.gmm_epoch,
        lr=args.gmm_lr,
        w=args.gmm_w,
        batch_size=args.gmm_batch_size,
        tol=1e-6,
        restarts=args.restarts,
        max_no_improvement=20,
        k_means_factor=args.k_means_factor,
        k_means_iters=args.k_means_iters,
        # lr_step=args.gmm_lr_step,
        # lr_gamma=args.gmm_lr_gamma,
        device=device,
        backbone_model=model,
        args=args,
    )
    gmm.fit(local_rank, dataset)

class SimCLR(object):
    def __init__(self, *args, **kwargs):
        # *args tuple, positional argument
        # **kwargs dict,
        self.args = kwargs['args']
        # self.model = kwargs['model'].to(self.args.device)
        self.model = kwargs['model'] # DDP Model
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        logging.basicConfig(
            filename=('./log/simclr/training.log'),
            level=logging.DEBUG,
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    # feature: 512, 128
    def info_nce_loss(self, features):
        # labels = tenser([0, 1, 2, 3, 4, ..., 255, 0, 1, 2, ..., 255])
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        # aa = tensor(1,512) [ [0,1,...,255,0,1,...,255] ]
        aa = labels.unsqueeze(0)
        # bb = tensor(512,1) [ [0],[1],...,[255],[0],[1],...,[255] ]
        bb = labels.unsqueeze(1)
        # lables = tensor(512,512)
        # 1 0 0 0 1 0 0 0
        # 0 1 0 0 0 1 0 0
        # 0 0 1 0 0 0 1 0
        # 0 0 0 1 0 0 0 1
        # 1 0 0 0 1 0 0 0
        # 0 1 0 0 0 1 0 0
        # 0 0 1 0 0 0 1 0
        # 0 0 0 1 0 0 0 1
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        # (512, 512)
        features = F.normalize(features, dim=1) # calculate norm
        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape
        # discard the main diagonal from both: labels and similarities matrix
        # eye 对角线为1其余为0的矩阵
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        # 去掉label 和 similarity 对角线部分 即自己和自己做相似度计算
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape
        # select and combine multiple positives
        # positive pair的相似度 (512, 1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives the negatives
        # 一个positive和所有negative的相似度 去掉了x+和x+以及x+和x++的相似度(512, 510)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        # （512，511）
        logits = torch.cat([positives, negatives], dim=1)
        # 512维度的0 tensor([0,0,0,0,...,0]) 这个0就是在计算crossrentropy的时候positive logit的index（类似于分类中的label），即下公式中的class
        # loss(x,class)=−log(exp(x[class])/∑jexp(x[j]))=−x[class]+log(∑jexp(x[j]))
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, gmm_dataset):

        self.writer = get_writer(log_dir="./log/simclr/") if dist.get_rank() == 0 else None

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        # save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        if dist.get_rank() == 0:
            logging.info(f"@rank{dist.get_rank()} Start SimCLR training for {self.args.epochs} epochs. <{get_time()}>")
        # logging.info(f"Training with gpu: {not self.args.disable_cuda}. <{get_time()}>")

        for epoch_counter in range(self.args.epochs):

            train_loader.sampler.set_epoch(epoch_counter)

            torch.cuda.empty_cache()

            # FIXME: Do not put gmm on GPU, it will cause error which some tensor on cpu while others on GPU.
            gmm = SGDGMMModule(self.args.components, self.args.gmm_dim, self.args.weight_decay)
            # TAG: GPU Ver.
            # gmm = SGDGMMModule(self.args.components, self.args.gmm_dim, self.args.weight_decay, device=self.args.device)
            if epoch_counter % self.args.gmm_every_n_epoch == 0:
                if dist.get_rank() == 0:
                    os.environ['MASTER_ADDR'] = 'localhost'
                    os.environ['MASTER_PORT'] = '34151'
                    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,0'
                    # os.environ['CUDA_VISIBLE_DEVICES'] = '5,0,1'
                    # self.model.eval() # TAG: 
                    mp.spawn(fit_gmm, nprocs=self.args.gpus, args=(self.args, self.model.module, gmm_dataset))
                    dist.barrier()
                else:
                    dist.barrier()

            # GMM init decay
            if epoch_counter == 50 or epoch_counter == 100:
                self.args.gmm_every_n_epoch = self.args.gmm_every_n_epoch + 10

            gmm.load_state_dict(torch.load("./result/checkpoint/sgdgmm/gmm.pth")) # TAG: GPU Ver.
            # gmm.to(self.args.device)
            gmm.eval()
            gmm.init_mvn()

            torch.cuda.empty_cache()
            
            self.model.train()

            with tqdm(total=(len(train_loader.dataset) // train_loader.batch_size // dist.get_world_size()), ncols=None, unit='it') as _tqdm:
                _tqdm.set_description(f'SimCLR training @{dist.get_rank()} epoch {epoch_counter+1}/{self.args.epochs}')

                for images, _ in train_loader:
                    # images [0], images[1]是两次augmentation的结果
                    # print(images)
                    # 在此做mixup
                    # tmp_img = images
                    # TAG: 这里images的形式是什么？[ [],[] ]
                    images = torch.cat(images, dim=0)
                    images = images.to(self.args.device)

                    with autocast(enabled=self.args.fp16_precision):
                        # features = self.model(images)
                        # 256, 128
                        features = torch.flatten(self.model.module.backbone(images), start_dim=1) # TAG: intermediate layer feature
                        # print(features.shape)
                        # features = self.model(images) # TAG: modified
                        # TAG:256,1,128
                        # DONE: Augmentation twice!
                        # time1 = time.time()
                        with torch.no_grad():
                            sampled_features = gmm.sample(features.to(torch.device("cpu")), sample_num=2)
                        # time2 = time.time()
                        # print(f'@rank{dist.get_rank()} spend {time2-time1}s to sample features. shape{sampled_features.shape}')
                        sampled_features = sampled_features.to(self.args.device)
                        mixup_coefficient = torch.distributions.uniform.Uniform(0.9, 1.0).sample()
                        mixup_aug1 = mixup_coefficient * features + (1 - mixup_coefficient) * sampled_features[:,0,:]
                        mixup_aug2 = mixup_coefficient * features + (1 - mixup_coefficient) * sampled_features[:,1,:]
                        feat1 = self.model.module.mlp(mixup_aug1)
                        feat2 = self.model.module.mlp(mixup_aug2)
                        features = torch.cat((feat1, feat2), dim=0)
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
                        self.writer.add_scalar(tag="SimCLR Training Loss", scalar_value=loss_reduced.item(), global_step=n_iter+1)

                    n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= self.args.warm_up_epoch-1:
                self.scheduler.step()

            logging.debug(f"@rank{dist.get_rank()} Epoch: {epoch_counter+1}\t<{get_time()}> loss: {loss.item()}")

        if dist.get_rank() == 0:
            logging.info("@rank{} Training has finished. <{}>".format(dist.get_rank(), get_time()))
            # save model checkpoints
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
            save_checkpoint(
                {
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                },
                is_best=False,
                filename=os.path.join('./result/checkpoint/simclr', checkpoint_name),
            )
            logging.info(
                f"Model checkpoint and metadata has been saved at {os.path.join('./result/checkpoint/simclr', checkpoint_name)}. <{get_time()}>"
            )
        
        if self.writer:
            self.writer.close()
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from pycave.bayes.gmm import GMM

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
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
        # 把 aa 和 bb 的元素一一对比，大小为bb行aa列
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        # (512, 512)
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
        # 512维度的0 tensor([0,0,0,0,...,0]) 这个0就是在计算crossrentropy的时候positive logit的index 下公式中的class
        # loss(x,class)=−log(exp(x[class])/∑jexp(x[j]))=−x[class]+log(∑jexp(x[j]))
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            # # 每一个epoch初始化一个GMM
            # print("GMM initialization")
            # gmm = GMM(num_components=16, num_features=32, covariance='diag')
            # # 传入一个batch的数据好像可以
            # history = gmm.fit(train_loader)
            # print("GMM Done")
            for images, _ in tqdm(train_loader):
                # images [0], images[1]是两次augmentation的结果
                print(images)
                # 在此做mixup
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

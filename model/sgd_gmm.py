from abc import ABC
import copy
import torch
from torch.autograd.grad_mode import F
import torch.utils
from torch._C import device
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from model.cluster import minibatch_k_means
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from util import save_plot, reduce_tensor, get_writer
import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
mvn = torch.distributions.multivariate_normal.MultivariateNormal

class SGDGMMModule(nn.Module):
    """Implementation of a standard GMM as a PyTorch nn module."""
    def __init__(self, components, dimensions, w, device=None):
        super().__init__()
        self.k = components
        self.d = dimensions
        self.device = device
        self.soft_weights = nn.Parameter(torch.zeros(self.k))
        self.soft_max = torch.nn.Softmax(dim=0)
        self.means = nn.Parameter(torch.rand(self.k, self.d))
        self.l_diag = nn.Parameter(torch.zeros(self.k, self.d))
        self.l_lower = nn.Parameter(torch.zeros(self.k, self.d * (self.d - 1) // 2))
        self.d_idx = torch.eye(self.d, device=self.device).to(torch.bool)
        self.l_idx = torch.tril_indices(self.d, self.d, -1, device=self.device)
        self.w = w * torch.eye(self.d, device=device)

    @property
    def L(self):
        L = torch.zeros(self.k, self.d, self.d, device=self.device)
        L[:, self.d_idx] = torch.exp(self.l_diag)
        L[:, self.l_idx[0], self.l_idx[1]] = self.l_lower
        return L

    @property
    def covars(self):
        return torch.matmul(self.L, torch.transpose(self.L, -2, -1))

    def forward(self, data):
        weights = self.soft_max(self.soft_weights)
        # None 的作用就是在相应的位置上增加了一个维度，在这个维度上只有一个元素
        # # 512, 1, 2
        # x_tmp = x[:, None, :]
        log_resp = mvn(loc=self.means, scale_tril=self.L).log_prob(data[:, None, :])
        log_resp += torch.log(weights)
        log_prob = torch.logsumexp(log_resp, dim=1)
        return -1 * torch.sum(log_prob)

    '''
    Predict which component data belongs to.
    input: a batch of dataset [data,label]
    return: a batch of component index which data belongs to (512,)
    '''
    def predict(self, input):
        log_prob_batch = mvn(loc=self.means, scale_tril=self.L).log_prob(input[:, None, :])  # 512,10
        component_batch = torch.max(log_prob_batch, 1)[1]
        return component_batch

    # def sample(self, input, sample_num=1):
    #     # input = input.to(self.device)
    #     component = self.predict(input)
    #     multivar_normal = [mvn(loc=self.means[idx], scale_tril=self.L[idx]) for idx in range(self.k)]
    #     sample_list = []
    #     # FIXME: Wondering if there is a batch sample way to improve performance.
    #     for idx in range(len(component)):
    #         sample_list.append([multivar_normal[component[idx]].rsample().detach().numpy().tolist() for _ in range(sample_num)])
    #     return torch.Tensor(sample_list)

    def init_mvn(self):
        self.multivar_normal = [mvn(loc=self.means[idx], scale_tril=self.L[idx]) for idx in range(self.k)]

    '''
    Given input_data, sample {sample_num} samples for each datapoint from thier corresponding Gaussian component.
    input_data: [batch_size, feature_dim]
    sample_num: default = 1
    return: [batch_size, sample_num, feature_dim]
    Return [ [ [feature1],[feature2] ],[ [],[] ],[ [],[] ]... ]
    '''
    def sample(self, input, sample_num=1):
        component = self.predict(input)
        sample_list = []
        for idx in range(len(component)):
            sample_list.append(torch.cat([ self.multivar_normal[component[idx]].rsample().unsqueeze(0) for _ in range(sample_num) ],dim=0).unsqueeze(0))
        return torch.Tensor(torch.cat(sample_list, dim=0))

class SGDGMM(ABC):
    """ABC for fitting a PyTorch nn-based GMM."""

    def __init__(
        self,
        components,
        dimensions,
        epochs=100,
        lr=1e-3,
        w=1e-6,
        batch_size=64,
        tol=1e-6,
        restarts=1,
        max_no_improvement=20,
        k_means_factor=1,
        k_means_iters=2,
        # lr_step=5,
        # lr_gamma=0.1,
        device=None,
        backbone_model=None,
        args=None,
    ):
        self.args = args
        self.k = components
        self.d = dimensions
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol
        self.lr = lr
        self.w = w
        # self.lr_step = lr_step
        # self.lr_gamma = lr_gamma
        self.restarts = restarts
        self.k_means_factor = k_means_factor
        self.k_means_iters = k_means_iters
        self.max_no_improvement = max_no_improvement
        self.backbone_model = backbone_model  # backbone model to encode the data

        self.device = torch.device('cpu') if not device else device
        self.backbone_model = self.backbone_model.backbone.to(self.device)
        self.backbone_model.eval()

    @property
    def means(self):
        return self.module.means.detach()

    @property
    def covars(self):
        return self.module.covars.detach()

    def reg_loss(self, n, n_total):
        l = ((n / n_total) * self.w / torch.diagonal(self.module.module.covars, dim1=-1, dim2=-2))
        return l.sum()

    """Fit the GMM to dataset."""
    def fit(self, local_rank, data, val_data=None, verbose=True, interval=1):
        rank = self.args.nr * self.args.gpus + local_rank  # global rank
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.args.world_size,
            rank=rank,
        )
        self.writer = get_writer('./log/sgdgmm/') if dist.get_rank() == 0 else None

        init_loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size * self.k_means_factor,
            num_workers=self.args.workers,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=self.args.world_size, rank=rank)

        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.args.workers,
            shuffle=False,
            sampler=train_sampler,
            pin_memory=False,
            drop_last=True,
        )

        best_loss = float('inf')

        # BUG: When using DDP training, should the reg loss consider ONLY the dataset slice on every GPU OR the whole dataset?
        # n_total = int(((len(data) / self.args.world_size) // self.batch_size) * self.batch_size)
        n_total = int(len(data))
        # print(f"n_total = {n_total} in rank {dist.get_rank()}")

        for restart_epoch in range(self.restarts):

            self.module = SGDGMMModule(self.k, self.d, self.w, self.device)
            self.module.to(self.device)

            if dist.get_rank() == 0:
                print(f'GMM fitting restart {restart_epoch+1}/{self.restarts}, with @rank0 initializing GMM parameter first')
                self.init_params(init_loader)

            # TAG: Wrap it with DDP
            self.module = nn.parallel.DistributedDataParallel(self.module, device_ids=[local_rank])
            self.optimiser = torch.optim.Adam(params=self.module.parameters(), lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=90, eta_min= 1e-6, last_epoch=-1, verbose=True)
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     self.optimiser,
            #     milestones=[self.lr_step, self.lr_step + 30, self.lr_step + 50, self.lr_step + 60],
            #     gamma=self.lr_gamma,
            # ) # TAG: change here!

            # DONE: Reduce all process's data
            train_loss_curve = []
            if val_data:
                val_loss_curve = []

            prev_loss = float('inf')
            if val_data:
                best_val_loss = float('inf')
                no_improvement_epochs = 0

            for epoch in range(self.epochs):
                train_loader.sampler.set_epoch(epoch)

                with tqdm(total=(len(train_loader.dataset) // train_loader.batch_size // dist.get_world_size()), ncols=None, unit='it') as _tqdm:
                    _tqdm.set_description(f'GMM fitting rst {restart_epoch+1}/{self.restarts} @rank{dist.get_rank()} epoch {epoch+1}/{self.epochs}')
                    
                    train_loss = 0.0
                    # for j, d in enumerate(loader):
                    # d = [datas, labels]
                    # d = [a.to(self.device) for a in d]
                    # data_count = 0
                    for data, _ in train_loader:

                        data = data.to(self.device)
                        if self.backbone_model:
                            with torch.no_grad():
                                # FIXME: Change backbone model to eval mode to remove BN & dropout layer etc...
                                data = self.backbone_model(data)
                                data = torch.flatten(data, start_dim=1) # TAG: backbone resnet needs to flatten

                        self.optimiser.zero_grad()
                        loss = self.module(data)
                        train_loss += loss.item()
                        
                        n = data.shape[0]
                        # data_count = data_count + n #TAG:
                        loss += self.reg_loss(n, n_total)
                        loss.backward()
                        self.optimiser.step()

                        _tqdm.set_postfix(loss=f'{loss.item():.4f}')
                        _tqdm.update()

                if epoch >= 9:
                    self.scheduler.step()

                # DONE: reduce train loss curve
                # print(f"rank {dist.get_rank()} train_loss before reduce: {train_loss}")
                train_loss = reduce_tensor(torch.Tensor([train_loss]).to(self.device)).item()
                # print(f"rank {dist.get_rank()} train_loss after reduce: {train_loss}")

                train_loss_curve.append(train_loss)

                if val_data:
                    val_loss = self.score_batch(val_data)
                    val_loss = reduce_tensor(
                        torch.Tensor([val_loss]).to(self.device)
                    ).item()
                    val_loss_curve.append(val_loss)


                # DONE: Since train & val loss have all been reduces, only rank_0 print infomation.
                if dist.get_rank() == 0:
                    self.writer.add_scalar(tag=f'GMM Training Loss/{restart_epoch}', scalar_value=train_loss, global_step=epoch+1)
                    if verbose and epoch % interval == 0:
                        if val_data:
                            print('Epoch {}, Train Loss: {}, Val Loss :{}'.format(epoch, train_loss, val_loss))
                        else:
                            print('Epoch {}, Loss: {}'.format(epoch, train_loss))

                    if val_data:
                        if val_loss < best_val_loss:
                            no_improvement_epochs = 0
                            best_val_loss = val_loss
                        else:
                            no_improvement_epochs += 1
                        if no_improvement_epochs > self.max_no_improvement:
                            print('No improvement in val loss for {} epochs. Early Stopping at {}'.format( self.max_no_improvement, val_loss))
                            break

                    if abs(train_loss - prev_loss) < self.tol:
                        print('Training loss converged within tolerance at {}'.format(train_loss))
                        break

                prev_loss = train_loss

            if val_data:
                score = val_loss
            else:
                score = train_loss

            if score < best_loss:
                best_module = copy.deepcopy(self.module.module)
                best_loss = score
                best_train_loss_curve = train_loss_curve
                if val_data:
                    best_val_loss_curve = val_loss_curve

        self.module.module = best_module
        self.train_loss_curve = best_train_loss_curve
        if val_data:
            self.val_loss_curve = best_val_loss_curve

        # TAG: Only rank 0 need to save the training curve.
        if dist.get_rank() == 0:
            x_index = np.arange(1, len(self.train_loss_curve) + 1, 1)
            save_plot(x_index, self.train_loss_curve, 
            'epoch', 'train_loss', 'Train loss curve',
            './result/training/sgdgmm/GMM_train_loss_curve')

            if val_data:
                x_index = np.arange(1, len(self.val_loss_curve) + 1, 1)
                save_plot(x_index, self.train_loss_curve, 
                'epoch', 'val_loss', 'Val loss curve',
                './result/training/sgdgmm/GMM_val_loss_curve')

            # Save model on CPU
            torch.save(self.module.module.cpu().state_dict(),"./result/checkpoint/sgdgmm/gmm.pth",)

            if self.writer:
                self.writer.close()

    def score(self, data):
        with torch.no_grad():
            return self.module(data)

    def score_batch(self, dataset):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True
        )
        log_prob = 0
        for j, d in enumerate(loader):
            d = [a.to(self.device) for a in d]
            log_prob += self.score(d).item()
        return log_prob

    """
    Initialize SGDGMMModule's parameters using mini-batch k-menas with ONLY 1 GPU training supported.
    Note: When using DistributedDataParallel, You must initialize module parameters before using DDP to wrap it.
    """

    def init_params(self, loader):
        counts, centroids = minibatch_k_means(
            loader,
            self.k,
            max_iters=self.k_means_iters,
            tol=1e-3,
            device=self.device,
            backbone_model=self.backbone_model,
        )
        self.module.soft_weights.data = torch.log(counts / counts.sum())
        self.module.means.data = centroids
        self.module.l_diag.data = nn.Parameter(
            torch.zeros(self.k, self.d, device=self.device)
        )
        self.module.l_lower.data = torch.zeros(
            self.k, self.d * (self.d - 1) // 2, device=self.device
        )
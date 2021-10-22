import os
os.system('export DISPLAY=:0.0')
from abc import ABC
import copy
import torch
import torch.distributions as dist
from torch.distributions import multivariate_normal
import torch.nn as nn
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import numpy as np
import time
from models.cluster import minibatch_k_means

mvn = dist.multivariate_normal.MultivariateNormal


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

        self.l_lower = nn.Parameter(
            torch.zeros(self.k,
                        self.d * (self.d - 1) // 2))

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

        # data = [data, label]
        # 512, 2  [512]
        x = data[0]

        # [10]
        weights = self.soft_max(self.soft_weights)

        # None 的作用就是在相应的位置上增加了一个维度，在这个维度上只有一个元素
        #
        dis_tmp = mvn(loc=self.means, scale_tril=self.L)
        # # 512, 1, 2
        # x_tmp = x[:, None, :]
        # # what_if = dis_tmp.log_prob(x)
        # predict_tmp = dis_tmp.log_prob(x[:, None, :])
        log_resp = mvn(loc=self.means, scale_tril=self.L).log_prob(x[:,
                                                                     None, :])
        log_resp += torch.log(weights)

        log_prob = torch.logsumexp(log_resp, dim=1)

        return -1 * torch.sum(log_prob)

    '''
    Predict which component data belongs to.
    input: a batch of dataset [data,label]
    return: a batch of component index which data belongs to (512,)
    '''

    def predict(self, input):
        # a1 = self.means
        # a2 = self.L
        # print(a1)
        # print(a2)

        data = input[0]  # 512,2
        log_prob_batch = mvn(loc=self.means, scale_tril=self.L).log_prob(
            data[:, None, :])  # 512,10
        component_batch = torch.max(log_prob_batch, 1)[1]
        return component_batch

    def sample(self, input, sample_num=1):
        component = self.predict(input)
        multivar_normal = [
            mvn(loc=self.means[idx], scale_tril=self.L[idx])
            for idx in range(self.k)
        ]
        sample_list = []
        for idx in range(len(component)):
            sample_list.append([
                multivar_normal[component[idx]].rsample()
                .detach()
                .cpu()
                .numpy()
                .tolist() for t in range(sample_num)
            ])
        return torch.Tensor(sample_list)


class BaseSGDGMM(ABC):
    """ABC for fitting a PyTorch nn-based GMM."""
    def __init__(
        self,
        components,
        dimensions,
        epochs=100,
        lr=1e-3,
        batch_size=64,
        tol=1e-6,
        restarts=1,
        max_no_improvement=20,
        k_means_factor=1,
        w=1e-6,
        k_means_iters=2,
        lr_step=5,
        lr_gamma=0.1,
        device=None,
        model=None,
    ):
        self.k = components
        self.d = dimensions
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = 1e-6
        self.lr = lr
        self.w = w
        self.restarts = restarts
        self.k_means_factor = k_means_factor
        self.k_means_iters = k_means_iters
        self.max_no_improvement = max_no_improvement
        # bacobone model to encode the data
        self.model = model

        if not device:
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.module = self.module.to(device)
        # print(self.module)
        self.model = self.model.to(device)
        # print(self.model)

        self.optimiser = torch.optim.Adam(params=self.module.parameters(),
                                          lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimiser, milestones=[lr_step, lr_step + 5], gamma=lr_gamma)

    @property
    def means(self):
        return self.module.means.detach()

    @property
    def covars(self):
        return self.module.covars.detach()

    def sample(self, input_data, sample_num=1):
        # [512] component indexs
        # components = self.module.predict(input_data)
        # print(components)
        samples = self.module.sample(input_data, sample_num=sample_num)
        print(samples.shape)

    def reg_loss(self, n, n_total):
        l = ((n / n_total) * self.w /
             torch.diagonal(self.module.covars, dim1=-1, dim2=-2))
        return l.sum()

    def fit(self, data, val_data=None, verbose=True, interval=1):
        """Fit the GMM to data."""
        n_total = len(data)
        # x_test = data

        init_loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size * self.k_means_factor,
            num_workers=0,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )
        # real dataloader
        loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )

        best_loss = float('inf')
        # iter for { restarts } times to get a more robust model
        for q in range(self.restarts):
            print('Restart: {}'.format(q))
            # using mini-batch k-means to initialize GMM's parameters
            # pass
            self.init_params(init_loader)
            train_loss_curve = []
            if val_data:
                val_loss_curve = []

            prev_loss = float('inf')
            if val_data:
                best_val_loss = float('inf')
                no_improvement_epochs = 0

            # mini-batch training for { epoch } epoch
            for i in range(self.epochs):
                train_loss = 0.0
                for j, d in enumerate(loader):
                    # d = [datas, labels]
                    d = [a.to(self.device) for a in d]

                    if self.model:
                        with torch.no_grad():
                            d[0] = self.model(d[0])

                    self.optimiser.zero_grad()
                    loss = self.module(d)
                    train_loss += loss.item()
                    n = d[0].shape[0]
                    loss += self.reg_loss(n, n_total)
                    loss.backward()
                    self.optimiser.step()

                    # test area ##########################
                    sample = self.sample(d,sample_num=2)

                train_loss_curve.append(train_loss)

                if val_data:
                    val_loss = self.score_batch(val_data)
                    val_loss_curve.append(val_loss)

                self.scheduler.step()

                if verbose and i % interval == 0:
                    if val_data:
                        print('Epoch {}, Train Loss: {}, Val Loss :{}'.format(
                            i, train_loss, val_loss))
                    else:
                        print('Epoch {}, Loss: {}'.format(i, train_loss))

                if val_data:
                    if val_loss < best_val_loss:
                        no_improvement_epochs = 0
                        best_val_loss = val_loss
                    else:
                        no_improvement_epochs += 1

                    if no_improvement_epochs > self.max_no_improvement:
                        print(
                            'No improvement in val loss for {} epochs. Early Stopping at {}'
                            .format(self.max_no_improvement, val_loss))
                        break

                if abs(train_loss - prev_loss) < self.tol:
                    print('Training loss converged within tolerance at {}'.
                          format(train_loss))
                    break

                prev_loss = train_loss

            if val_data:
                score = val_loss
            else:
                score = train_loss

            if score < best_loss:
                best_model = copy.deepcopy(self.module)
                best_loss = score
                best_train_loss_curve = train_loss_curve
                if val_data:
                    best_val_loss_curve = val_loss_curve

        self.module = best_model
        self.train_loss_curve = best_train_loss_curve
        '''
        Plot training curve
        '''
        #################
        print('plot')
        x_index = np.arange(1, len(self.train_loss_curve) + 1, 1)
        fig, ax = plt.subplots()
        print(x_index)
        print(self.train_loss_curve)
        ax.plot(x_index, self.train_loss_curve)
        ax.set(xlabel='epoch', ylabel='train_loss', title='Train loss curve')
        ax.grid()
        plt.show()
        plt.savefig("./result/training/train_loss_curve_{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        #################

        if val_data:
            self.val_loss_curve = best_val_loss_curve

    def score(self, data):
        with torch.no_grad():
            return self.module(data)

    def score_batch(self, dataset):
        loader = data_utils.DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       num_workers=0,
                                       pin_memory=True)

        log_prob = 0

        for j, d in enumerate(loader):
            d = [a.to(self.device) for a in d]
            log_prob += self.score(d).item()

        return log_prob

    def init_params(self, loader):
        counts, centroids = minibatch_k_means(
            loader,
            self.k,
            max_iters=self.k_means_iters,
            device=self.device,
            model=self.model,
        )
        self.module.soft_weights.data = torch.log(counts / counts.sum())
        self.module.means.data = centroids
        self.module.l_diag.data = nn.Parameter(
            torch.zeros(self.k, self.d, device=self.device))
        self.module.l_lower.data = torch.zeros(self.k,
                                               self.d * (self.d - 1) // 2,
                                               device=self.device)


class SGDGMM(BaseSGDGMM):
    """Concrete implementation of class to fit a standard GMM with SGD."""
    def __init__(
        self,
        components,
        dimensions,
        epochs=100,
        lr=1e-3,
        batch_size=64,
        tol=1e-6,
        w=1e-3,
        device=None,
        model=None,
        restarts=1,
        k_means_iter=1,
        k_means_factor=1,
    ):
        self.module = SGDGMMModule(components, dimensions, w, device)
        super().__init__(
            components,
            dimensions,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            w=w,
            tol=tol,
            device=device,
            model=model,
            restarts=restarts,
            k_means_iters=k_means_iter,
            k_means_factor=k_means_factor,
        )

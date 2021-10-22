import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from plotting import plot_covariance
from sgd_gmm import SGDGMM

colors1 = ['red', 'blue', 'green']
colors2 = ['green', 'red', 'blue']


def check_sgd_gmm(D, K, N, plot=False, device=None):
    if not device:
        device = torch.device('cpu')

    means = (np.random.rand(K, D) * 20) - 10
    q = (2 * np.random.randn(K, D, D))
    covars = np.matmul(q.swapaxes(1, 2), q)

    X = np.empty((N, K, D))

    for i in range(K):
        X[:, i, :] = np.random.multivariate_normal(
            mean=means[i, :],
            cov=covars[i, :, :],
            size=N
        )

    # X_data1 = torch.Tensor(X.reshape(-1, D).astype(np.float32))
    X_data = [torch.Tensor(X.reshape(-1, D).astype(np.float32))]

    gmm = SGDGMM(K, D, lr=0.01, device=device, batch_size=1024)
    gmm.fit(X_data)

    if plot:
        fig, ax = plt.subplots()

        for i in range(K):
            # datasets distribution
            sc = ax.scatter(X[:, i, 0], X[:, i, 1], alpha=0.2, marker='x', label='Cluster {}'.format(i))
            print(sc.get_facecolor())
            plot_covariance(means[i, :], covars[i, :, :], ax, color=colors1[i])

        # 5,2
        # temp_means = gmm.means
        # 5, 2, 2
        # temp_covars = gmm.covars

        sc = ax.scatter(gmm.means[:, 0], gmm.means[:, 1], marker='*', label='Fitted Gaussians')

        for i in range(K):
            print(sc.get_facecolor()[0])
            plot_covariance(gmm.means[i, :], gmm.covars[i, :, :], ax, color=colors2[i])

        ax.legend()
        plt.show()


if __name__ == '__main__':
    sns.set()
    D = 2
    K = 3
    N = 2000
    device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")
    print(device)
    check_sgd_gmm(D, K, N, plot=True, device=device)

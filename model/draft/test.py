from matplotlib.pyplot import xlim
from numpy.lib.function_base import cov
import numpy as np
import torch

means = torch.randn([10, 2])
covar = torch.tensor([[[1., 0.], [0., 1.]], [[1., 0.], [0., 1.]],
                      [[1., 0.], [0., 1.]], [[1., 0.], [0., 1.]],
                      [[1., 0.], [0., 1.]], [[1., 0.], [0., 1.]],
                      [[1., 0.], [0., 1.]], [[1., 0.], [0., 1.]],
                      [[1., 0.], [0., 1.]], [[1., 0.], [0., 1.]]])

dis_tmp = torch.distributions.multivariate_normal.MultivariateNormal(
    loc=means, covariance_matrix=covar)
# data = torch.tensor([1, 2])
# data1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
# print(data)
# print(dis_tmp.log_prob(data))
# print(data1)
# print(dis_tmp.log_prob(data1[:, None, :]))

# print(dis_tmp)

dis = [
    torch.distributions.multivariate_normal.MultivariateNormal(
        loc=means[i], covariance_matrix=covar[i]) for i in range(10)
]
# print(dis)
idx = [1, 3, 5, 7, 8, 2, 4, 1, 5, 2, 4, 2, 1]
print(len(idx))
sample_n = 2
xlist = []
for i in idx:
    xlist.append([ dis[i].rsample().numpy().tolist()  for j in range(sample_n)])
print(torch.Tensor(xlist).shape)
print("~~~~~~")
for i in range(1):
    print(i)